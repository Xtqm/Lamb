"""
JED-M2 Native Mamba Selective State Space Model (SSM) in Pure PyTorch.
Implements the continuous-time SSM discretized via zero-order hold (ZOH)
with input-dependent selection mechanism and constant-memory recurrent step mode.

STRICT BUILD RULE: 100% native PyTorch without external compiled C++/CUDA packages.
"""

import math
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from jed_m2.config import JEDM2Config


class MambaBlock(nn.Module):
    """
    Pure PyTorch Mamba Selective State Space Model (SSM) Block.

    Features:
    - Input-dependent continuous-to-discrete parameter projection (Δ, B, C).
    - Causal 1D depthwise convolution for local context mixing.
    - Zero-order hold (ZOH) discretization for stable recurrence.
    - Autoregressive step(...) method for O(1) constant-memory recurrent inference.
    - Exact mathematical equivalence between sequence forward and iterative step mode.
    """

    def __init__(self, config: JEDM2Config):
        super().__init__()
        self.config = config
        self.d_model: int = config.hidden_dim
        self.d_state: int = config.mamba_d_state
        self.d_conv: int = config.mamba_d_conv
        self.expand: int = config.mamba_expand
        self.d_inner: int = config.mamba_inner_dim
        self.dt_rank: int = config.mamba_dt_rank

        self.dt_min: float = 0.001
        self.dt_max: float = 0.1
        self.dt_init_floor: float = 1e-4

        # 1. In-projection for expanding input into u and gate z: [B, L, D] -> [B, L, 2 * d_inner]
        self.in_proj = nn.Linear(self.d_model, 2 * self.d_inner, bias=False)

        # 2. 1D Causal depthwise convolution on u: [B, d_inner, L] -> [B, d_inner, L]
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=self.d_conv,
            bias=True,
            groups=self.d_inner,
            padding=0,
        )

        # 3. Input-dependent projection from convolved u to (Δ_raw, B, C)
        # [B, L, d_inner] -> [B, L, dt_rank + 2 * d_state]
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + 2 * self.d_state, bias=False)

        # 4. Delta projection: [B, L, dt_rank] -> [B, L, d_inner]
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)

        # 5. Continuous SSM parameters: A (stored as log) and D (skip connection)
        A = torch.arange(1, self.d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.A_log._no_weight_decay = True

        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.D._no_weight_decay = True

        # 6. Out-projection: [B, L, d_inner] -> [B, L, D]
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=False)

        # Initialize dt_proj weights and bias
        self._init_dt_proj()

    def _init_dt_proj(self):
        """Initializes dt_proj weights and bias according to standard Mamba scaling."""
        dt_init_std = self.dt_rank**-0.5
        nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)

        # Initialize dt bias to inv_softplus(dt) for log-spaced dt values
        dt = torch.exp(
            torch.rand(self.d_inner) * (math.log(self.dt_max) - math.log(self.dt_min))
            + math.log(self.dt_min)
        ).clamp(min=self.dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Sequence forward pass for Mamba SSM.

        Args:
            x: Input tensor of shape [batch_size, seq_len, hidden_dim].

        Returns:
            Output tensor of shape [batch_size, seq_len, hidden_dim].
        """
        # [B, L, D] -> [B, L, 2 * d_inner]
        batch_size, seq_len, _ = x.shape
        xz = self.in_proj(x)
        u, z = xz.chunk(2, dim=-1)  # Each: [B, L, d_inner]

        # Causal 1D Convolution with left-padding (d_conv - 1)
        # [B, L, d_inner] -> [B, d_inner, L]
        u_conv_in = u.transpose(1, 2)
        # Left-pad to preserve causality: [B, d_inner, L + d_conv - 1]
        u_conv_padded = F.pad(u_conv_in, (self.d_conv - 1, 0))
        # [B, d_inner, L]
        u_conv = self.conv1d(u_conv_padded)
        u_conv = F.silu(u_conv).transpose(1, 2)  # [B, L, d_inner]

        # Project convolved representation to SSM inputs (Δ_raw, B, C)
        # [B, L, d_inner] -> [B, L, dt_rank + 2 * d_state]
        x_dbl = self.x_proj(u_conv)
        delta_raw, B_proj, C_proj = torch.split(
            x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1
        )
        # [B, L, dt_rank] -> [B, L, d_inner]
        delta = F.softplus(self.dt_proj(delta_raw))

        # Discretize continuous parameters A and B in float32 for numerical stability
        # A: [d_inner, d_state] -> negative real eigenvalues
        A = -torch.exp(self.A_log.float())  # [d_inner, d_state]
        delta_f = delta.float()  # [B, L, d_inner]
        u_conv_f = u_conv.float()  # [B, L, d_inner]
        B_f = B_proj.float()  # [B, L, d_state]
        C_f = C_proj.float()  # [B, L, d_state]

        # Discretization via ZOH:
        # Ā = exp(Δ ⊙ A) -> [B, L, d_inner, d_state]
        deltaA = delta_f.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(0)
        A_bar = torch.exp(deltaA)

        # B̄ = Δ ⊙ B -> [B, L, d_inner, d_state]
        B_bar = delta_f.unsqueeze(-1) * B_f.unsqueeze(2)

        # Selective Scan Recurrence
        # h_t = Ā_t ⊙ h_{t-1} + B̄_t ⊙ u_conv,t
        # y_t = (h_t · C_t) + D ⊙ u_conv,t
        h = torch.zeros(
            batch_size, self.d_inner, self.d_state,
            device=x.device, dtype=torch.float32
        )
        y_list = []
        D_f = self.D.float()  # [d_inner]

        for t in range(seq_len):
            A_bar_t = A_bar[:, t]  # [B, d_inner, d_state]
            B_bar_t = B_bar[:, t]  # [B, d_inner, d_state]
            u_t = u_conv_f[:, t].unsqueeze(-1)  # [B, d_inner, 1]
            C_t = C_f[:, t].unsqueeze(1)  # [B, 1, d_state]

            # Recurrent state update: [B, d_inner, d_state]
            h = A_bar_t * h + B_bar_t * u_t

            # Output projection at time t: [B, d_inner]
            y_t = (h * C_t).sum(dim=-1) + D_f * u_conv_f[:, t]
            y_list.append(y_t)

        # Stack over sequence dimension: [B, L, d_inner]
        y = torch.stack(y_list, dim=1).to(dtype=x.dtype)

        # Gated combination with z: y ⊙ silu(z)
        y = y * F.silu(z)

        # Final out-projection: [B, L, d_inner] -> [B, L, D]
        out = self.out_proj(y)
        return out

    def step(
        self,
        x_t: torch.Tensor,
        ssm_state: Optional[torch.Tensor] = None,
        conv_state: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Single-step constant-memory autoregressive recurrent evaluation.

        Args:
            x_t: Input tensor for single timestep of shape [batch_size, hidden_dim].
            ssm_state: Hidden state tensor of shape [batch_size, d_inner, d_state] (optional).
            conv_state: Convolution buffer of shape [batch_size, d_inner, d_conv] (optional).

        Returns:
            Tuple of:
            - out_t: Output tensor of shape [batch_size, hidden_dim].
            - next_ssm_state: Updated SSM state of shape [batch_size, d_inner, d_state].
            - next_conv_state: Updated conv buffer of shape [batch_size, d_inner, d_conv].
        """
        batch_size = x_t.shape[0]

        # Initialize state buffers if not provided
        if ssm_state is None:
            ssm_state = torch.zeros(
                batch_size, self.d_inner, self.d_state,
                device=x_t.device, dtype=torch.float32
            )
        else:
            ssm_state = ssm_state.float()

        if conv_state is None:
            conv_state = torch.zeros(
                batch_size, self.d_inner, self.d_conv,
                device=x_t.device, dtype=x_t.dtype
            )

        # 1. In-projection: [B, D] -> [B, 2 * d_inner]
        xz = self.in_proj(x_t)
        u_t, z_t = xz.chunk(2, dim=-1)  # Each: [B, d_inner]

        # 2. Update convolution buffer: shift left by 1 and insert new u_t at the end
        # [B, d_inner, d_conv]
        next_conv_state = torch.roll(conv_state, shifts=-1, dims=-1)
        next_conv_state[:, :, -1] = u_t

        # 3. Compute 1D causal convolution for single timestep:
        # conv1d weight shape: [d_inner, 1, d_conv] -> squeeze to [d_inner, d_conv]
        conv_weight = self.conv1d.weight.squeeze(1)  # [d_inner, d_conv]
        conv_bias = self.conv1d.bias  # [d_inner]

        # Dot product across kernel dimension: sum_k(conv_state[:, :, k] * weight[:, k]) + bias
        # [B, d_inner]
        u_conv_t = (next_conv_state * conv_weight.unsqueeze(0)).sum(dim=-1) + conv_bias
        u_conv_t = F.silu(u_conv_t)  # [B, d_inner]

        # 4. Project convolved representation to SSM inputs (Δ_raw, B, C)
        # [B, d_inner] -> [B, dt_rank + 2 * d_state]
        x_dbl = self.x_proj(u_conv_t)
        delta_raw, B_proj, C_proj = torch.split(
            x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1
        )
        # [B, dt_rank] -> [B, d_inner]
        delta = F.softplus(self.dt_proj(delta_raw))

        # 5. Discretize in float32
        A = -torch.exp(self.A_log.float())  # [d_inner, d_state]
        delta_f = delta.float()  # [B, d_inner]
        u_conv_f = u_conv_t.float()  # [B, d_inner]
        B_f = B_proj.float()  # [B, d_state]
        C_f = C_proj.float()  # [B, d_state]

        # Ā_t = exp(Δ ⊙ A): [B, d_inner, d_state]
        A_bar_t = torch.exp(delta_f.unsqueeze(-1) * A.unsqueeze(0))

        # B̄_t = Δ ⊙ B: [B, d_inner, d_state]
        B_bar_t = delta_f.unsqueeze(-1) * B_f.unsqueeze(1)

        # 6. Recurrent SSM state update: [B, d_inner, d_state]
        next_ssm_state = A_bar_t * ssm_state + B_bar_t * u_conv_f.unsqueeze(-1)

        # 7. Output projection at time t: [B, d_inner]
        D_f = self.D.float()
        C_t = C_f.unsqueeze(1)  # [B, 1, d_state]
        y_t = (next_ssm_state * C_t).sum(dim=-1) + D_f * u_conv_f

        # Cast back to input precision
        y_t = y_t.to(dtype=x_t.dtype)

        # Gated combination with z_t: y_t ⊙ silu(z_t)
        y_t = y_t * F.silu(z_t)

        # 8. Out-projection: [B, d_inner] -> [B, D]
        out_t = self.out_proj(y_t)

        return out_t, next_ssm_state, next_conv_state
