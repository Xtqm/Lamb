"""
JED-M2 Human-in-the-Loop Safety Supervisor.
Guards destructive file system operations, sensitive authentication/credential fields,
and financial transactions, automatically pausing policy execution and requiring
explicit confirmation before physical peripheral dispatch.
"""

import ctypes
import os
import re
import sys
import threading
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Pattern, Tuple, Union

from jed_m2.heads import ActionType


class SafetyCategory(str, Enum):
    """Classification of intercepted safety violations."""

    DESTRUCTIVE_FILESYSTEM = "destructive_filesystem"
    CREDENTIAL_AUTHENTICATION = "credential_authentication"
    FINANCIAL_TRANSACTION = "financial_transaction"


@dataclass
class SafetyViolation:
    """Details of an intercepted action violating safety rules."""

    category: SafetyCategory
    action_type: str
    argument: str
    matched_rule: str
    risk_description: str
    timestamp: float


class SafetySupervisor:
    """
    Deterministic rule-based safety interceptor.
    Monitors candidate agent decisions and triggers human approval workflows.
    """

    # Regex patterns for destructive filesystem operations
    DESTRUCTIVE_PATTERNS: List[Tuple[Pattern[str], str]] = [
        (re.compile(r"\b(rm\s+-[rf]+|rmdir\s+/[sq]|del\s+/[fqsa]|erase\s+/[fq])", re.IGNORECASE), "Force recursive delete command"),
        (re.compile(r"\b(format\s+[a-z]:|diskpart|mkfs|dd\s+if=)", re.IGNORECASE), "Disk format or block-level mutation command"),
        (re.compile(r"\b(remove-item\s+.*-recurse.*-force)", re.IGNORECASE), "PowerShell recursive forced deletion"),
        (re.compile(r"\b(drop\s+database|truncate\s+table|drop\s+table)\b", re.IGNORECASE), "Destructive database operation"),
        (re.compile(r"\b(del\s+\*|rm\s+\*)", re.IGNORECASE), "Wildcard mass deletion"),
    ]

    # Keywords for password, credential, and authentication fields
    CREDENTIAL_KEYWORDS: List[str] = [
        "password",
        "passwd",
        "passcode",
        "pin code",
        "secret_key",
        "api_key",
        "private_key",
        "bearer token",
        "auth token",
        "credentials",
        "social security",
        "ssn",
    ]

    # Keywords for financial and payment transactions
    FINANCIAL_KEYWORDS: List[str] = [
        "place order",
        "submit payment",
        "pay now",
        "buy now",
        "complete purchase",
        "confirm payment",
        "credit card",
        "card number",
        "expiration date",
        "cvv",
        "cvc",
        "bank transfer",
        "wire funds",
        "crypto transfer",
        "send money",
    ]

    def __init__(
        self,
        auto_approve: bool = False,
        prompt_callback: Optional[Callable[[SafetyViolation], bool]] = None,
        use_toast: bool = True,
    ):
        """
        Initializes the Safety Supervisor.

        Args:
            auto_approve: If True, logs violations without blocking (used for testing).
            prompt_callback: Optional custom callback (violation -> bool).
            use_toast: When True, uses Windows API modal prompt/notification on Windows.
        """
        self.auto_approve = auto_approve
        self.prompt_callback = prompt_callback
        self.use_toast = use_toast
        self.interception_history: List[SafetyViolation] = []
        self._lock = threading.Lock()

    def check_safety(
        self,
        action_type: Union[ActionType, int, str],
        argument: str = "",
        context_text: str = "",
    ) -> Optional[SafetyViolation]:
        """
        Evaluates an action proposal against safety rules.

        Args:
            action_type: Primitive action type.
            argument: Action argument string (e.g. text typed or command).
            context_text: Optional DOM or active element text surrounding target.

        Returns:
            SafetyViolation object if intercepted, or None if safe.
        """
        act_str = action_type.name if hasattr(action_type, "name") else str(action_type)
        combined_text = f"{argument} {context_text}".lower().strip()
        timestamp = time.time()

        # 1. Guard destructive filesystem mutations
        for pattern, desc in self.DESTRUCTIVE_PATTERNS:
            if pattern.search(argument) or pattern.search(combined_text):
                return SafetyViolation(
                    category=SafetyCategory.DESTRUCTIVE_FILESYSTEM,
                    action_type=act_str,
                    argument=argument,
                    matched_rule=pattern.pattern,
                    risk_description=desc,
                    timestamp=timestamp,
                )

        # 2. Guard credential and authentication fields
        for kw in self.CREDENTIAL_KEYWORDS:
            if kw in combined_text:
                return SafetyViolation(
                    category=SafetyCategory.CREDENTIAL_AUTHENTICATION,
                    action_type=act_str,
                    argument=argument,
                    matched_rule=kw,
                    risk_description=f"Action interacts with credential/secret keyword '{kw}'",
                    timestamp=timestamp,
                )

        # 3. Guard financial transactions
        for kw in self.FINANCIAL_KEYWORDS:
            if kw in combined_text:
                return SafetyViolation(
                    category=SafetyCategory.FINANCIAL_TRANSACTION,
                    action_type=act_str,
                    argument=argument,
                    matched_rule=kw,
                    risk_description=f"Action targets financial transaction keyword '{kw}'",
                    timestamp=timestamp,
                )

        return None

    def authorize_action(
        self,
        action_type: Union[ActionType, int, str],
        argument: str = "",
        context_text: str = "",
    ) -> Tuple[bool, Optional[SafetyViolation]]:
        """
        Validates safety. If safe, returns (True, None).
        If violated, logs violation and prompts human user for confirmation.

        Returns:
            Tuple of (is_authorized: bool, violation: Optional[SafetyViolation]).
        """
        violation = self.check_safety(action_type, argument, context_text)
        if violation is None:
            return True, None

        with self._lock:
            self.interception_history.append(violation)

        if self.auto_approve:
            return True, violation

        # Custom callback takes priority if provided
        if self.prompt_callback is not None:
            approved = self.prompt_callback(violation)
            return approved, violation

        # Interactive human confirmation
        approved = self._request_user_confirmation(violation)
        return approved, violation

    def _request_user_confirmation(self, violation: SafetyViolation) -> bool:
        """Prompts the user via Windows native dialog or terminal fallback."""
        prompt_title = f"JED-M2 Safety Warning: {violation.category.value}"
        prompt_message = (
            f"An autonomous action was INTERCEPTED for human authorization:\n\n"
            f"Category: {violation.category.value}\n"
            f"Action: {violation.action_type}\n"
            f"Payload: '{violation.argument}'\n"
            f"Risk: {violation.risk_description}\n\n"
            f"Do you authorize this action to execute?"
        )

        if sys.platform == "win32" and self.use_toast:
            try:
                # MB_YESNO (0x00000004) | MB_ICONWARNING (0x00000030) | MB_TOPMOST (0x00040000)
                MB_YESNO = 0x00000004
                MB_ICONWARNING = 0x00000030
                MB_TOPMOST = 0x00040000
                IDYES = 6

                result = ctypes.windll.user32.MessageBoxW(
                    0,
                    prompt_message,
                    prompt_title,
                    MB_YESNO | MB_ICONWARNING | MB_TOPMOST,
                )
                return result == IDYES
            except Exception:
                pass

        # Terminal fallback
        print("\n" + "=" * 60)
        print(f"[!] {prompt_title}")
        print(prompt_message)
        print("=" * 60)
        try:
            resp = input("Authorize action? [y/N]: ").strip().lower()
            return resp in ("y", "yes")
        except (EOFError, KeyboardInterrupt):
            return False
