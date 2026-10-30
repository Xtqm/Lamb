import os
import sys
import torch
from pathlib import Path

# Ensure repo root is on sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from jed_m2.config import JEDM2Config
from jed_m2.model import JEDM2Model
from jed_m2.dataset import SimpleLAMTokenizer
from jed_m2.rl_pipeline import TrajectoryRolloutSampler

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

checkpoints = [
    ("Pretrain Baseline", "checkpoints/pretrain_unified/jed_m2_nano_pretrain_final.pt"),
    ("GRPO Step 25", "checkpoints_rl/jed_m2_grpo_step_25.pt"),
    ("GRPO Step 75", "checkpoints_rl/jed_m2_grpo_step_75.pt"),
    ("GRPO Step 100 (Final)", "checkpoints_rl/jed_m2_nano_grpo_final.pt"),
]

scenarios = ["ecommerce", "travel", "os_admin"]

print("=" * 80)
print(f"JED-M2 RL EVALUATION PROBE: 100-STEP GRPO SCALING RUN")
print(f"Device: {device} | Dtype: {dtype}")
print("=" * 80)

results = {}

for label, ckpt_path in checkpoints:
    if not os.path.exists(ckpt_path):
        print(f"Skipping {label}: {ckpt_path} not found")
        continue

    print(f"\n---> Evaluating [{label}]: {ckpt_path}")
    model = JEDM2Model(JEDM2Config())
    data = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = data.get("model_state_dict", data) if isinstance(data, dict) else data
    model.load_state_dict(sd, strict=False)
    model.to(device=device, dtype=dtype)
    model.eval()

    tokenizer = SimpleLAMTokenizer(model.config.vocab_size)
    sampler = TrajectoryRolloutSampler(model=model, tokenizer=tokenizer, device=device)

    results[label] = {}

    for s_id in scenarios:
        # Greedy evaluation (temperature=0.0)
        greedy_rollouts = sampler.sample_group_rollouts(
            scenario_id=s_id, group_size=1, temperature=0.0, max_turns=8
        )
        rg = greedy_rollouts[0]

        # Stochastic evaluation (temperature=0.7, 4 rollouts)
        stoch_rollouts = sampler.sample_group_rollouts(
            scenario_id=s_id, group_size=4, temperature=0.7, max_turns=8
        )
        stoch_success = sum(1 for r in stoch_rollouts if r.success)
        stoch_rewards = [round(r.cumulative_reward, 3) for r in stoch_rollouts]
        mean_stoch_rew = sum(stoch_rewards) / len(stoch_rewards)

        results[label][s_id] = {
            "greedy_success": rg.success,
            "greedy_reward": rg.cumulative_reward,
            "greedy_steps": rg.num_steps,
            "stoch_success_rate": stoch_success / 4.0,
            "stoch_mean_rew": mean_stoch_rew,
            "stoch_rewards": stoch_rewards,
            "greedy_actions": [f"{t.action_type.name}@({t.coordinates[0]:.2f},{t.coordinates[1]:.2f})" for t in rg.turns],
        }

        print(f"  [{s_id:<9}] Greedy: Succ={rg.success:<5} Rew={rg.cumulative_reward:+6.3f} Steps={rg.num_steps} | Stoch: Succ={stoch_success}/4 MeanRew={mean_stoch_rew:+6.3f}")
        for idx, turn in enumerate(rg.turns):
            print(f"      Turn {idx+1}: {turn.action_type.name:<8} coords=({turn.coordinates[0]:.2f}, {turn.coordinates[1]:.2f}) arg={turn.argument[:15]!r} rew={turn.reward:+.2f}")

print("\n" + "=" * 80)
print("EVALUATION MATRIX SUMMARY")
print("=" * 80)
header = f"| {'Checkpoint':<22} | {'Task':<10} | {'Greedy Succ':<11} | {'Greedy Rew':<10} | {'Steps':<5} | {'Stoch SR':<8} | {'Stoch Rew':<9} |"
print(header)
print("|" + "-" * 24 + "|" + "-" * 12 + "|" + "-" * 13 + "|" + "-" * 12 + "|" + "-" * 7 + "|" + "-" * 10 + "|" + "-" * 11 + "|")

for label in results:
    for s_id in scenarios:
        res = results[label][s_id]
        succ_str = "SUCCESS" if res["greedy_success"] else "FAIL"
        stoch_sr_str = f"{res['stoch_success_rate']*100:.0f}%"
        print(f"| {label:<22} | {s_id:<10} | {succ_str:<11} | {res['greedy_reward']:+10.3f} | {res['greedy_steps']:<5} | {stoch_sr_str:<8} | {res['stoch_mean_rew']:+9.3f} |")
