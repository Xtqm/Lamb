"""
JED-M2 Synthetic Agent Trajectory Generator & Dynamic PyTorch Dataset.
Provides:
- Trajectory tuple representation: tau = (s_0, a_0, r_0, s_1, a_1, ..., s_T).
- Realistic multi-turn WebArena and Mind2Web interaction traces:
  DOM element trees, UI bounding boxes, discrete tools (CLICK, TYPE, SCROLL, NAVIGATE, API_CALL, FINISH),
  spatial (x, y) coordinates in [0, 1], and string argument tokens.
- Streaming PyTorch Dataset: AgentTrajectoryDataset.
- Dynamic Trajectory Collator: TrajectoryCollator assembling padded token sequences,
  4D causal & padding attention masks, action token index mapping, component loss masks,
  and JEPA observation transition pairs.

STRICT CONVENTION: Pure PyTorch without external compiled C++/CUDA packages or Hugging Face pipelines.
"""

from dataclasses import dataclass, field
import hashlib
import random
from typing import Any, Dict, List, Optional, Tuple, Union
import torch
from torch.utils.data import Dataset

from jed_m2.bpe_tokenizer import ActionBPETokenizer
from jed_m2.config import JEDM2Config
from jed_m2.core.features import compute_obs_features
from jed_m2.core.types import InteractionTurn, UIElement
from jed_m2.heads import ActionType


# ==============================================================================
# 1. Trajectory Data Structures
# ==============================================================================

@dataclass
class TrajectorySample:
    """Complete multi-turn agent interaction trajectory: tau = (s_0, a_0, r_0, ..., s_T)."""

    sample_id: str
    goal: str
    turns: List[InteractionTurn]
    total_reward: float = 0.0

    @property
    def length(self) -> int:
        """Returns number of turns in this trajectory."""
        return len(self.turns)


# ==============================================================================
# 2. Tokenizers: Invertible ActionBPETokenizer & LegacyHashTokenizer
# ==============================================================================

class LegacyHashTokenizer:
    """
    Legacy lightweight, deterministic MD5 hash tokenizer for JED-M2.
    Retained for backward compatibility with historical checkpoints.
    Reserves special tokens in [0..10] and hashes subwords/words into [11..vocab_size-1].
    """

    PAD_TOKEN_ID: int = 0
    BOS_TOKEN_ID: int = 1
    EOS_TOKEN_ID: int = 2
    OBS_TOKEN_ID: int = 3
    OBS_END_TOKEN_ID: int = 4
    THOUGHT_TOKEN_ID: int = 5
    THOUGHT_END_TOKEN_ID: int = 6
    ACTION_TOKEN_ID: int = 7
    ACTION_END_TOKEN_ID: int = 8
    ARG_TOKEN_ID: int = 9
    ARG_END_TOKEN_ID: int = 10

    def __init__(self, vocab_size: int = 8192):
        self.vocab_size = vocab_size
        self.special_tokens = {
            "<pad>": self.PAD_TOKEN_ID,
            "<bos>": self.BOS_TOKEN_ID,
            "<eos>": self.EOS_TOKEN_ID,
            "<obs>": self.OBS_TOKEN_ID,
            "</obs>": self.OBS_END_TOKEN_ID,
            "<thought>": self.THOUGHT_TOKEN_ID,
            "</thought>": self.THOUGHT_END_TOKEN_ID,
            "<action>": self.ACTION_TOKEN_ID,
            "</action>": self.ACTION_END_TOKEN_ID,
            "<arg>": self.ARG_TOKEN_ID,
            "</arg>": self.ARG_END_TOKEN_ID,
        }
        self.inv_special_tokens = {v: k for k, v in self.special_tokens.items()}

    def _hash_word(self, word: str) -> int:
        """Hashes arbitrary word/string into [11, vocab_size - 1]."""
        h = int(hashlib.md5(word.encode("utf-8")).hexdigest()[:8], 16)
        return 11 + (h % (self.vocab_size - 11))

    def encode(self, text: str) -> List[int]:
        """Encodes string into list of token IDs."""
        tokens: List[int] = []
        words = text.split()
        for w in words:
            if w in self.special_tokens:
                tokens.append(self.special_tokens[w])
            else:
                tokens.append(self._hash_word(w))
        return tokens

    def decode(self, tokens: List[int], skip_special_tokens: bool = False) -> str:
        """Decodes token IDs into string representation."""
        words: List[str] = []
        for t in tokens:
            if t in self.inv_special_tokens:
                if not skip_special_tokens:
                    words.append(self.inv_special_tokens[t])
            else:
                words.append(f"tok_{t}")
        return " ".join(words)


# Primary tokenizer alias: SimpleLAMTokenizer now points to ActionBPETokenizer
SimpleLAMTokenizer = ActionBPETokenizer


# ==============================================================================
# 3. Synthetic Trajectory Generator (WebArena / Mind2Web Scenarios)
# ==============================================================================

class SyntheticTrajectoryGenerator:
    """
    Synthesizes realistic multi-turn web/OS interaction trajectories modeled after
    WebArena and Mind2Web digital agent benchmarks.
    Generates DOM element trees, UI bounding boxes, discrete tools, pointer targets,
    and rationale tokens.
    """

    DOM_TAGS = ["button", "input", "a", "select", "div", "textarea", "span", "header", "nav", "footer"]
    ROLES = [
        "search_bar", "submit_btn", "nav_link", "filter_select", "item_card",
        "checkout_btn", "close_icon", "menu_item", "tab", "dialog"
    ]

    # Guardrail 2A: Bounded BPE CSS Vocabulary (Tailwind, Bootstrap, MUI, Shadcn)
    BOUNDED_CLASSES = [
        "btn", "btn-primary", "btn-secondary", "btn-outline", "btn-sm",
        "form-control", "input-field", "text-input", "select-input",
        "card", "card-body", "card-header", "modal-content", "dropdown-menu",
        "flex", "flex-col", "flex-row", "items-center", "justify-between",
        "grid", "grid-cols-2", "grid-cols-4", "gap-2", "gap-4",
        "p-2", "p-4", "px-3", "py-2", "px-4", "py-3", "m-2", "my-4",
        "text-sm", "text-base", "text-lg", "font-medium", "font-bold",
        "text-gray-700", "text-gray-900", "text-white", "bg-blue-600",
        "rounded-md", "rounded-lg", "border", "border-gray-300", "shadow-sm",
    ]

    BOUNDED_TEST_IDS = [
        "submit-btn", "search-input", "filter-select", "checkout-btn", "apply-btn",
        "confirm-btn", "item-row", "nav-link", "modal-close", "tab-active",
        "action-menu", "save-btn", "cancel-btn", "date-picker", "quantity-select",
    ]

    BOUNDED_ARIA_LABELS = [
        "Search", "Submit form", "Apply discount", "Confirm selection", "Close modal",
        "Next page", "Previous page", "Select option", "Filter results", "Checkout",
        "Delete item", "Edit details", "Save changes", "View cart", "User profile",
    ]

    compute_obs_features = staticmethod(compute_obs_features)

    # 20+ Domain Archetypes with Adversarial Near-Miss Entity Generators (Workstream 2)
    ARCHETYPES = [
        # 1. E-Commerce & Retail
        {
            "domain": "ecommerce",
            "archetype": "E-Commerce & Retail",
            "goal_fn": lambda r: f"Your expired promo code was {r['d1']} and loyalty code {r['d2']}; apply active discount code {r['target']} to order #{r['d3']} and proceed to checkout",
            "entities_fn": lambda rng: {
                "target": rng.choice(["VIP25", "FALL20", "SAVE15", "SUMMER30", "CYBER40"]),
                "d1": rng.choice(["EXPIRED10", "OLD5", "SPRING10", "WINTER5"]),
                "d2": rng.choice(["LOYALTY5", "MEMBER10", "REWARDS2"]),
                "d3": rng.choice(["88412", "99104", "77231", "66519"]),
            },
            "steps_fn": lambda e: [
                (ActionType.CLICK, "button", "cart_btn", "view_cart_button", "", "Open cart review modal"),
                (ActionType.TYPE, "input", "coupon_input", "promo_input_field", e["target"], f"Enter active promo coupon code {e['target']}"),
                (ActionType.CLICK, "button", "apply_btn", "apply_discount_button", "", "Apply promo discount code"),
                (ActionType.CLICK, "button", "checkout_btn", "proceed_checkout_button", "", "Proceed to checkout"),
                (ActionType.FINISH, "div", "checkout_btn", "order_confirmation", "success", "Confirm checkout initiation"),
            ],
        },
        # 2. Travel & Booking
        {
            "domain": "travel",
            "archetype": "Travel & Booking",
            "goal_fn": lambda r: f"Departing from {r['d1']} with transit at {r['d2']}, search destination {r['target']} checking in on {r['d3']} for guest {r['d4']}",
            "entities_fn": lambda rng: {
                "target": rng.choice(["Seattle, WA", "Austin, TX", "Denver, CO", "Boston, MA", "Miami, FL"]),
                "d1": rng.choice(["SFO", "LAX", "ORD", "JFK", "ATL"]),
                "d2": rng.choice(["DFW", "DEN", "MSP", "CLT"]),
                "d3": rng.choice(["2026-11-15", "2026-12-01", "2026-11-20", "2026-12-10"]),
                "d4": rng.choice(["Alex Chen", "Elena Rostova", "Marcus Vance", "Sarah Jenkins"]),
            },
            "steps_fn": lambda e: [
                (ActionType.NAVIGATE, "input", "nav_link", "url_address_bar", "https://travel.booking.internal/search", "Navigate to booking engine"),
                (ActionType.TYPE, "input", "search_bar", "destination_input_field", e["target"], f"Enter target destination city {e['target']}"),
                (ActionType.CLICK, "button", "submit_btn", "search_hotels_button", "", "Execute destination search"),
                (ActionType.CLICK, "button", "item_card", "select_hotel_room_button", "", "Select preferred hotel room"),
                (ActionType.FINISH, "div", "checkout_btn", "booking_confirmation_banner", "confirmed", "Confirm hotel booking"),
            ],
        },
        # 3. SaaS CRM & Sales
        {
            "domain": "crm",
            "archetype": "SaaS CRM & Sales",
            "goal_fn": lambda r: f"Review existing account {r['d1']} with ARR ${r['d2']}; create new expansion deal for client {r['target']} valued at ${r['d3']}",
            "entities_fn": lambda rng: {
                "target": rng.choice(["Datadog LLC", "Stripe Inc", "Snowflake Corp", "Cloudflare Inc", "HubSpot"]),
                "d1": rng.choice(["Acme Legacy Corp", "Global Retail Ltd", "Pied Piper"]),
                "d2": rng.choice(["25000", "40000", "15000"]),
                "d3": rng.choice(["85000", "120000", "95000", "150000"]),
            },
            "steps_fn": lambda e: [
                (ActionType.CLICK, "button", "create_btn", "new_deal_button", "", "Open deal creation form"),
                (ActionType.TYPE, "input", "deal_company_input", "company_name_field", e["target"], f"Type company account name {e['target']}"),
                (ActionType.CLICK, "button", "save_btn", "save_deal_button", "", "Save deal record to pipeline"),
                (ActionType.FINISH, "div", "deal_card", "deal_created_status", "saved", "Verify deal registered"),
            ],
        },
        # 4. Cloud Admin & DevOps
        {
            "domain": "cloud_admin",
            "archetype": "Cloud Admin & DevOps",
            "goal_fn": lambda r: f"Preserve active cluster {r['d1']} in region {r['d2']}; terminate decommissioned instance {r['target']} confirming with {r['d3']}",
            "entities_fn": lambda rng: {
                "target": rng.choice(["i-0a89bc32", "i-0f44e912", "i-0c77aa55", "i-0d3388bb"]),
                "d1": rng.choice(["k8s-prod-primary", "db-read-replica-3", "edge-ingress-pool"]),
                "d2": rng.choice(["us-east-1", "eu-central-1", "us-west-2"]),
                "d3": rng.choice(["TERMINATE", "DELETE_CONFIRM", "FORCE_SHUTDOWN"]),
            },
            "steps_fn": lambda e: [
                (ActionType.CLICK, "button", "nav_link", "instances_tab", "", "Open compute instances list"),
                (ActionType.TYPE, "input", "search_bar", "instance_filter_input", e["target"], f"Filter list by instance ID {e['target']}"),
                (ActionType.CLICK, "button", "terminate_btn", "terminate_action_button", "", "Initiate instance termination"),
                (ActionType.FINISH, "div", "status_banner", "instance_terminated_banner", "terminated", "Verify instance decommissioned"),
            ],
        },
        # 5. Productivity & Workspace
        {
            "domain": "productivity",
            "archetype": "Productivity & Workspace",
            "goal_fn": lambda r: f"Archive obsolete draft {r['d1']} in folder {r['d2']}; share project document {r['target']} with collaborator {r['d3']}",
            "entities_fn": lambda rng: {
                "target": rng.choice(["Q4 Strategy Roadmap", "Annual Security Audit", "Core Architecture Spec", "Customer Journey Map"]),
                "d1": rng.choice(["Old Notes 2024", "Meeting Scratchpad", "Temp Budget"]),
                "d2": rng.choice(["Archive/Drafts", "Personal/Temp", "Scratch/Old"]),
                "d3": rng.choice(["jordan@company.internal", "taylor@domain.org", "sam@workspace.net"]),
            },
            "steps_fn": lambda e: [
                (ActionType.CLICK, "a", "nav_link", "documents_workspace_link", "", "Open workspace files directory"),
                (ActionType.TYPE, "input", "search_bar", "document_search_input", e["target"], f"Locate document {e['target']}"),
                (ActionType.CLICK, "button", "submit_btn", "share_document_button", "", "Open share dialog"),
                (ActionType.FINISH, "div", "modal_content", "share_success_toast", "shared", "Confirm document shared"),
            ],
        },
        # 6. Fintech & Banking
        {
            "domain": "fintech",
            "archetype": "Fintech & Banking",
            "goal_fn": lambda r: f"Checking account balance is ${r['d1']}; transfer funds to recipient {r['target']} with memo {r['d2']} routing code {r['d3']}",
            "entities_fn": lambda rng: {
                "target": rng.choice(["Apex Holdings", "Vanguard Escrow", "Horizon Capital", "Starlight Partners"]),
                "d1": rng.choice(["48200.50", "73150.00", "92400.25"]),
                "d2": rng.choice(["Invoice #9901", "Legal Retainer", "Quarterly Rent"]),
                "d3": rng.choice(["021000021", "121000358", "071000013"]),
            },
            "steps_fn": lambda e: [
                (ActionType.CLICK, "button", "nav_link", "transfer_funds_tab", "", "Open transfer payments menu"),
                (ActionType.TYPE, "input", "search_bar", "recipient_name_field", e["target"], f"Enter verified recipient {e['target']}"),
                (ActionType.CLICK, "button", "submit_btn", "confirm_transfer_button", "", "Submit transfer request"),
                (ActionType.FINISH, "div", "status_banner", "transfer_receipt_card", "transferred", "Verify fund transfer receipt"),
            ],
        },
        # 7. Customer Support & Helpdesk
        {
            "domain": "customer_support",
            "archetype": "Customer Support",
            "goal_fn": lambda r: f"Closed ticket #{r['d1']} was low priority; escalate urgent incident #{r['target']} reported by user {r['d2']} category {r['d3']}",
            "entities_fn": lambda rng: {
                "target": rng.choice(["INC-4892", "INC-7703", "INC-9124", "INC-3381"]),
                "d1": rng.choice(["TIC-1022", "TIC-2041", "TIC-3099"]),
                "d2": rng.choice(["morgan.reed@tech.io", "claire.bennett@cloud.org", "arthur.dent@galaxy.net"]),
                "d3": rng.choice(["Billing Gateway", "SSO Auth Outage", "Data Sync Failure"]),
            },
            "steps_fn": lambda e: [
                (ActionType.TYPE, "input", "search_bar", "ticket_lookup_field", e["target"], f"Search for incident ticket {e['target']}"),
                (ActionType.CLICK, "button", "submit_btn", "search_ticket_button", "", "Submit ticket lookup query"),
                (ActionType.CLICK, "button", "item_card", "escalate_priority_button", "", "Escalate ticket priority to urgent"),
                (ActionType.FINISH, "div", "status_banner", "escalation_toast", "escalated", "Confirm incident escalated"),
            ],
        },
        # 8. Auth & Settings
        {
            "domain": "auth_settings",
            "archetype": "Auth & Settings",
            "goal_fn": lambda r: f"Revoke deprecated token {r['d1']}; generate new scoped service API key {r['target']} for service {r['d2']} expiration {r['d3']}",
            "entities_fn": lambda rng: {
                "target": rng.choice(["billing-sync-worker", "telemetry-collector", "backup-daemon", "auth-proxy"]),
                "d1": rng.choice(["legacy-api-key-2023", "temp-read-token", "staging-debug-key"]),
                "d2": rng.choice(["prod-payment-svc", "analytics-aggregator", "metrics-exporter"]),
                "d3": rng.choice(["90 days", "180 days", "1 year"]),
            },
            "steps_fn": lambda e: [
                (ActionType.CLICK, "a", "nav_link", "api_tokens_link", "", "Navigate to API tokens configuration"),
                (ActionType.TYPE, "input", "search_bar", "token_name_field", e["target"], f"Enter service token name {e['target']}"),
                (ActionType.CLICK, "button", "submit_btn", "generate_token_button", "", "Click generate token button"),
                (ActionType.FINISH, "div", "modal_content", "token_created_modal", "created", "Confirm key created"),
            ],
        },
        # 9. Content Portals & CMS
        {
            "domain": "content_portals",
            "archetype": "Content Portals & CMS",
            "goal_fn": lambda r: f"Disregard rejected draft {r['d1']}; publish approved article {r['target']} author {r['d2']} tag {r['d3']}",
            "entities_fn": lambda rng: {
                "target": rng.choice(["Modern Web Architecture", "AI Agent Security 2026", "Scaling Recurrent Models", "Database Sharding"]),
                "d1": rng.choice(["Draft Untitled 4", "Test Post Alpha", "Raw Notes"]),
                "d2": rng.choice(["Dr. Aris Thorne", "Dr. Maya Patel", "Prof. Liam Scott"]),
                "d3": rng.choice(["Engineering", "Research", "Systems"]),
            },
            "steps_fn": lambda e: [
                (ActionType.CLICK, "button", "nav_link", "articles_dashboard_btn", "", "Open articles list"),
                (ActionType.TYPE, "input", "search_bar", "title_search_field", e["target"], f"Search for article title {e['target']}"),
                (ActionType.CLICK, "button", "submit_btn", "publish_article_button", "", "Publish article"),
                (ActionType.FINISH, "div", "status_banner", "published_status_card", "published", "Verify article live"),
            ],
        },
        # 10. Healthcare Scheduling
        {
            "domain": "healthcare",
            "archetype": "Healthcare Scheduling",
            "goal_fn": lambda r: f"Cancel tentative checkup with Dr. {r['d1']}; book specialist consultation with Dr. {r['target']} clinic {r['d2']} slot {r['d3']}",
            "entities_fn": lambda rng: {
                "target": rng.choice(["Katherine Brooks", "Jonathan Vance", "Grace Holloway", "Raymond Castillo"]),
                "d1": rng.choice(["Oldman", "Miller", "Harrison"]),
                "d2": rng.choice(["Downtown Medical Center", "North Valley Specialty Clinic", "Westside Health"]),
                "d3": rng.choice(["10:30 AM", "2:15 PM", "4:00 PM"]),
            },
            "steps_fn": lambda e: [
                (ActionType.TYPE, "input", "search_bar", "physician_name_input", e["target"], f"Enter physician name Dr. {e['target']}"),
                (ActionType.CLICK, "button", "submit_btn", "search_physician_button", "", "Search physician schedule"),
                (ActionType.CLICK, "button", "item_card", "select_time_slot_button", "", "Select appointment slot"),
                (ActionType.FINISH, "div", "checkout_btn", "appointment_confirmed_banner", "scheduled", "Confirm medical consultation"),
            ],
        },
        # 11. Logistics & Fleet Tracking
        {
            "domain": "logistics",
            "archetype": "Logistics & Fleet Tracking",
            "goal_fn": lambda r: f"Delivered package was #{r['d1']} origin {r['d2']}; track in-transit freight container {r['target']} carrier {r['d3']}",
            "entities_fn": lambda rng: {
                "target": rng.choice(["CONT-99421-US", "CONT-88102-EU", "CONT-77319-AP", "CONT-66204-GL"]),
                "d1": rng.choice(["PKG-10492", "PKG-33910", "PKG-84912"]),
                "d2": rng.choice(["Port of Oakland", "Long Beach Terminal", "Rotterdam Hub"]),
                "d3": rng.choice(["Maersk Line", "Evergreen Marine", "Hapag-Lloyd"]),
            },
            "steps_fn": lambda e: [
                (ActionType.TYPE, "input", "search_bar", "container_search_field", e["target"], f"Enter container tracking number {e['target']}"),
                (ActionType.CLICK, "button", "submit_btn", "track_shipment_button", "", "Execute container tracking query"),
                (ActionType.SCROLL, "div", "item_card", "manifest_details_panel", "scroll_down", "Scroll shipment manifest details"),
                (ActionType.FINISH, "div", "status_banner", "customs_cleared_badge", "tracked", "Verify shipment tracking status"),
            ],
        },
        # 12. HR & Recruiting
        {
            "domain": "recruiting",
            "archetype": "HR & Recruiting",
            "goal_fn": lambda r: f"Archived applicant {r['d1']} was rejected; schedule final technical interview for candidate {r['target']} role {r['d2']} room {r['d3']}",
            "entities_fn": lambda rng: {
                "target": rng.choice(["Sophia Hernandez", "Dmitri Volkov", "Aisha Al-Mansoor", "Gabriel Santos"]),
                "d1": rng.choice(["John Doe 2023", "Test Candidate B", "Unqualified Sample"]),
                "d2": rng.choice(["Senior ML Engineer", "Distributed Systems Lead", "Principal Agent Architect"]),
                "d3": rng.choice(["Virtual Room 4", "Conference Hall B", "Executive Suite 10"]),
            },
            "steps_fn": lambda e: [
                (ActionType.TYPE, "input", "search_bar", "candidate_name_field", e["target"], f"Type candidate name {e['target']}"),
                (ActionType.CLICK, "button", "submit_btn", "filter_candidates_button", "", "Filter candidate applications"),
                (ActionType.CLICK, "button", "item_card", "schedule_interview_button", "", "Open interview scheduler"),
                (ActionType.FINISH, "div", "checkout_btn", "invitation_sent_toast", "scheduled", "Confirm candidate interview invitation"),
            ],
        },
        # 13. Real Estate Listings
        {
            "domain": "real_estate",
            "archetype": "Real Estate Listings",
            "goal_fn": lambda r: f"Past rented property was #{r['d1']} in {r['d2']}; schedule virtual property tour for listing {r['target']} budget ${r['d3']}",
            "entities_fn": lambda rng: {
                "target": rng.choice(["742 Evergreen Terrace", "1004 Mountain View Ave", "88 Ocean Boulevard", "450 Skyline Court"]),
                "d1": rng.choice(["MLS-99120", "MLS-44019", "MLS-33104"]),
                "d2": rng.choice(["Springfield", "Oakland", "San Jose"]),
                "d3": rng.choice(["1250000", "890000", "2100000"]),
            },
            "steps_fn": lambda e: [
                (ActionType.TYPE, "input", "search_bar", "address_search_input", e["target"], f"Search address {e['target']}"),
                (ActionType.CLICK, "button", "submit_btn", "search_properties_button", "", "Search properties"),
                (ActionType.CLICK, "button", "item_card", "schedule_tour_button", "", "Schedule property tour"),
                (ActionType.FINISH, "div", "status_banner", "tour_scheduled_banner", "scheduled", "Confirm tour appointment"),
            ],
        },
        # 14. University Course Registration
        {
            "domain": "education",
            "archetype": "University Course Registration",
            "goal_fn": lambda r: f"Completed prerequisite {r['d1']} grade {r['d2']}; register student for advanced course {r['target']} section {r['d3']}",
            "entities_fn": lambda rng: {
                "target": rng.choice(["CS-4820 Artificial Intelligence", "ECE-6250 Distributed Systems", "MATH-5140 Real Analysis", "CS-5300 Compilers"]),
                "d1": rng.choice(["CS-2110 OOP", "MATH-2940 Linear Algebra", "CS-3110 Data Structures"]),
                "d2": rng.choice(["A", "A+", "A-"]),
                "d3": rng.choice(["SEC-001", "SEC-002", "SEC-004"]),
            },
            "steps_fn": lambda e: [
                (ActionType.TYPE, "input", "search_bar", "course_catalog_input", e["target"], f"Lookup course {e['target']}"),
                (ActionType.CLICK, "button", "submit_btn", "search_courses_button", "", "Search course catalog"),
                (ActionType.CLICK, "button", "item_card", "add_to_schedule_button", "", "Add course to semester schedule"),
                (ActionType.FINISH, "div", "checkout_btn", "registration_confirmed_card", "enrolled", "Confirm course enrollment"),
            ],
        },
        # 15. Insurance Claims
        {
            "domain": "insurance",
            "archetype": "Insurance Claims",
            "goal_fn": lambda r: f"Expired policy was #{r['d1']} deductible ${r['d2']}; submit claim for active commercial policy {r['target']} incident #{r['d3']}",
            "entities_fn": lambda rng: {
                "target": rng.choice(["POL-9924-COMM", "POL-7718-IND", "POL-5542-FLEET", "POL-3301-PROP"]),
                "d1": rng.choice(["OLD-0021-EXP", "OLD-9481-VOID", "OLD-5510-CAN"]),
                "d2": rng.choice(["500", "1000", "2500"]),
                "d3": rng.choice(["INC-2026-9901", "INC-2026-8842", "INC-2026-7731"]),
            },
            "steps_fn": lambda e: [
                (ActionType.TYPE, "input", "search_bar", "policy_number_field", e["target"], f"Enter active policy number {e['target']}"),
                (ActionType.CLICK, "button", "submit_btn", "verify_policy_button", "", "Verify insurance policy"),
                (ActionType.CLICK, "button", "item_card", "file_claim_button", "", "Submit claim documentation"),
                (ActionType.FINISH, "div", "status_banner", "claim_received_toast", "filed", "Confirm insurance claim filing"),
            ],
        },
        # 16. Restaurant Reservations
        {
            "domain": "restaurant",
            "archetype": "Restaurant Reservations",
            "goal_fn": lambda r: f"Previous cancelled reservation was at {r['d1']} party {r['d2']}; book table at {r['target']} time {r['d3']} party of {r['d4']}",
            "entities_fn": lambda rng: {
                "target": rng.choice(["L'Atelier Bistro", "Noma Harbor", "Osteria Francescana", "Le Bernardin", "Per Se"]),
                "d1": rng.choice(["Cafe Bella", "Burger Shack", "Taco Stand"]),
                "d2": rng.choice(["2", "3", "5"]),
                "d3": rng.choice(["7:30 PM", "8:00 PM", "6:45 PM"]),
                "d4": rng.choice(["4", "6", "8"]),
            },
            "steps_fn": lambda e: [
                (ActionType.TYPE, "input", "search_bar", "restaurant_name_input", e["target"], f"Search dining venue {e['target']}"),
                (ActionType.CLICK, "button", "submit_btn", "find_tables_button", "", "Find available tables"),
                (ActionType.CLICK, "button", "item_card", "reserve_table_button", "", "Select table reservation"),
                (ActionType.FINISH, "div", "checkout_btn", "reservation_confirmed_banner", "reserved", "Confirm table booking"),
            ],
        },
        # 17. Media Streaming & Playlists
        {
            "domain": "streaming",
            "archetype": "Media Streaming",
            "goal_fn": lambda r: f"Skip disliked album {r['d1']} by artist {r['d2']}; add featured album {r['target']} to collection #{r['d3']}",
            "entities_fn": lambda rng: {
                "target": rng.choice(["Midnight Echoes", "Quantum Symphony", "Neon Horizons", "Celestial Drift"]),
                "d1": rng.choice(["Unreleased Demo", "Rough Draft 2", "Noise Experiment"]),
                "d2": rng.choice(["The Static", "Null Pointer", "Zero Signal"]),
                "d3": rng.choice(["Favorites", "Chill Weekend", "Focus Ambient"]),
            },
            "steps_fn": lambda e: [
                (ActionType.TYPE, "input", "search_bar", "media_search_input", e["target"], f"Type album title {e['target']}"),
                (ActionType.CLICK, "button", "submit_btn", "search_music_button", "", "Search music catalog"),
                (ActionType.CLICK, "button", "item_card", "add_to_playlist_button", "", "Add album to playlist"),
                (ActionType.FINISH, "div", "status_banner", "playlist_updated_toast", "saved", "Verify media playlist updated"),
            ],
        },
        # 18. Bug Trackers & Issue Boards
        {
            "domain": "bug_tracker",
            "archetype": "Bug Trackers & Issue Boards",
            "goal_fn": lambda r: f"Closed ticket was #{r['d1']} component {r['d2']}; triage critical bug #{r['target']} sprint {r['d3']} assignee {r['d4']}",
            "entities_fn": lambda rng: {
                "target": rng.choice(["BUG-9021", "BUG-8840", "BUG-7715", "BUG-6623"]),
                "d1": rng.choice(["BUG-1011", "BUG-2022", "BUG-3033"]),
                "d2": rng.choice(["css-padding", "tooltip-typo", "readme-link"]),
                "d3": rng.choice(["Sprint 42", "Sprint 43", "Q4 Hardening"]),
                "d4": rng.choice(["alex.c", "maria.g", "sam.k"]),
            },
            "steps_fn": lambda e: [
                (ActionType.TYPE, "input", "search_bar", "issue_key_input", e["target"], f"Search issue ticket {e['target']}"),
                (ActionType.CLICK, "button", "submit_btn", "search_issues_button", "", "Search issue board"),
                (ActionType.CLICK, "button", "item_card", "assign_sprint_button", "", "Assign issue to active sprint"),
                (ActionType.FINISH, "div", "status_banner", "issue_triaged_badge", "triaged", "Confirm issue triaged"),
            ],
        },
        # 19. Analytics Dashboards
        {
            "domain": "analytics",
            "archetype": "Analytics Dashboards",
            "goal_fn": lambda r: f"Baseline conversion was {r['d1']}% in segment {r['d2']}; export analytics report for metric {r['target']} period {r['d3']}",
            "entities_fn": lambda rng: {
                "target": rng.choice(["Daily Active Users", "Net Churn Rate", "Average Order Value", "Checkout Dropoff"]),
                "d1": rng.choice(["2.4", "3.1", "1.8"]),
                "d2": rng.choice(["EMEA Enterprise", "APAC Mobile", "LATAM Self-Serve"]),
                "d3": rng.choice(["Last 30 Days", "Current Quarter", "Trailing 12 Months"]),
            },
            "steps_fn": lambda e: [
                (ActionType.CLICK, "button", "nav_link", "metrics_dashboard_tab", "", "Open metrics dashboard"),
                (ActionType.TYPE, "input", "search_bar", "metric_search_field", e["target"], f"Select analytical metric {e['target']}"),
                (ActionType.CLICK, "button", "submit_btn", "export_csv_button", "", "Export metric report to CSV"),
                (ActionType.FINISH, "div", "modal_content", "export_ready_toast", "exported", "Confirm data report export"),
            ],
        },
        # 20. Inventory Management
        {
            "domain": "inventory",
            "archetype": "Inventory Management",
            "goal_fn": lambda r: f"Out-of-stock SKU was {r['d1']} aisle {r['d2']}; restock priority product SKU {r['target']} warehouse {r['d3']} quantity {r['d4']}",
            "entities_fn": lambda rng: {
                "target": rng.choice(["SKU-99402-A", "SKU-88319-B", "SKU-77201-C", "SKU-66104-D"]),
                "d1": rng.choice(["SKU-0012-DISC", "SKU-0099-OBS", "SKU-0044-OLD"]),
                "d2": rng.choice(["Aisle 4-B", "Shelf 12-C", "Bay 8"]),
                "d3": rng.choice(["Main Distribution Center", "West Coast Hub", "Northeast Depot"]),
                "d4": rng.choice(["250", "500", "1200"]),
            },
            "steps_fn": lambda e: [
                (ActionType.TYPE, "input", "search_bar", "sku_lookup_input", e["target"], f"Enter product SKU identifier {e['target']}"),
                (ActionType.CLICK, "button", "submit_btn", "search_inventory_button", "", "Search inventory stock"),
                (ActionType.CLICK, "button", "item_card", "restock_order_button", "", "Create restock purchase order"),
                (ActionType.FINISH, "div", "status_banner", "reorder_confirmed_banner", "restocked", "Confirm inventory restock order"),
            ],
        },
    ]

    # Backward compatibility alias
    SCENARIOS = ARCHETYPES

    # Realistic Web Scaffolds for DOM Grafting (Guardrail 2D)
    DOM_SCAFFOLDS = [
        "<header class='site-header bg-white shadow-sm'><nav class='flex items-center justify-between px-6 py-4'><a href='/' class='logo font-bold text-xl'>Portal</a><div class='nav-items flex gap-4 text-sm'><span>Overview</span><span>Documentation</span><span>Support</span></div></nav></header><main class='main-content container mx-auto px-4 py-8'>{content}</main><footer class='site-footer text-xs text-gray-500 text-center py-6 border-t'>&copy; 2026 Enterprise Systems Inc. All rights reserved.</footer>",
        "<div class='app-layout flex h-screen'><aside class='sidebar w-64 bg-slate-900 text-white p-4'><div class='sidebar-header font-semibold text-lg mb-6'>Dashboard</div><nav class='space-y-2'><div class='nav-entry'>Home</div><div class='nav-entry'>Settings</div></nav></aside><div class='content-panel flex-1 overflow-auto p-6'><div class='panel-header mb-4 text-xl font-bold'>Workspace</div>{content}</div></div>",
        "<div class='checkout-wrapper max-w-4xl mx-auto py-10'><div class='breadcrumb text-sm text-gray-400 mb-6'>Store &gt; Catalog &gt; Checkout</div><div class='grid grid-cols-1 md:grid-cols-3 gap-6'><div class='col-span-2 form-container'>{content}</div><div class='summary-panel bg-gray-50 p-6 rounded-lg border'><h3>Order Summary</h3><p class='text-sm'>Subtotal: $120.00</p></div></div></div>",
        "<section class='booking-view max-w-5xl mx-auto my-8'><div class='tabs flex border-b mb-6'><button class='tab-item active font-medium py-2 px-4 border-b-2 border-blue-600'>Primary</button><button class='tab-item py-2 px-4 text-gray-500'>History</button></div><div class='view-body'>{content}</div></section>",
    ]

    def __init__(
        self,
        config: JEDM2Config,
        seed: Optional[int] = None,
        tokenizer: Optional[ActionBPETokenizer] = None,
    ):
        self.config = config
        self.hidden_dim = config.hidden_dim
        self.rng = random.Random(seed)
        self.tokenizer = tokenizer or ActionBPETokenizer(config.vocab_size)

    def _generate_dom_tree(
        self,
        target_element_name: str,
        target_tag: str,
        target_role: str,
        distractor_entities: Optional[List[str]] = None,
        drop_id_class: bool = False,
    ) -> Tuple[List[UIElement], UIElement]:
        """Generates 6 to 12 realistic UI elements with bounded attributes, near-miss distractors, and optional attribute dropout."""
        num_elements = self.rng.randint(6, 10)
        elements: List[UIElement] = []

        # Target element dimensions
        target_w = self.rng.uniform(0.10, 0.25)
        target_h = self.rng.uniform(0.04, 0.08)
        target_x = self.rng.uniform(0.05, 0.95 - target_w)
        target_y = self.rng.uniform(0.05, 0.95 - target_h)
        target_bbox = (target_x, target_y, target_x + target_w, target_y + target_h)

        # Guardrail 2A: Sample realistic classes, test-ids, aria-labels from bounded pools
        target_classes = " ".join(self.rng.sample(self.BOUNDED_CLASSES, k=self.rng.randint(1, 3)))
        target_testid = self.rng.choice(self.BOUNDED_TEST_IDS)
        target_aria = self.rng.choice(self.BOUNDED_ARIA_LABELS)

        target_attrs = {
            "class": target_classes,
            "data-testid": target_testid,
            "aria-label": target_aria,
        }
        if drop_id_class:
            target_attrs["__drop_id_class__"] = True

        target_el = UIElement(
            element_id=0,
            tag=target_tag,
            role=target_role,
            text=target_element_name,
            bbox=target_bbox,
            attributes=target_attrs,
            is_clickable=(target_tag in ("button", "a") or target_role in ("button", "link")),
            is_input=(target_tag in ("input", "textarea") or target_role in ("textbox", "searchbox")),
        )
        elements.append(target_el)

        # Distractor elements with Guardrail 2A (bounded attributes) and Guardrail 2C (near-miss distractors)
        distractor_pool = list(distractor_entities) if distractor_entities else []
        self.rng.shuffle(distractor_pool)

        for i in range(1, num_elements):
            w = self.rng.uniform(0.08, 0.22)
            h = self.rng.uniform(0.03, 0.07)
            x = self.rng.uniform(0.02, 0.98 - w)
            y = self.rng.uniform(0.02, 0.98 - h)
            tag = self.rng.choice(self.DOM_TAGS)
            role = self.rng.choice(self.ROLES)

            # Guardrail 2C: Inject near-miss distractor if available
            distractor_val = None
            if distractor_pool and (i <= len(distractor_pool) or self.rng.random() < 0.4):
                dist_entity = distractor_pool[(i - 1) % len(distractor_pool)]
                text = f"{dist_entity}"
                distractor_val = dist_entity if tag in ("input", "textarea") else None
            else:
                text = f"{self.rng.choice(self.BOUNDED_ARIA_LABELS)} {i}"

            classes = " ".join(self.rng.sample(self.BOUNDED_CLASSES, k=self.rng.randint(1, 2)))
            testid = self.rng.choice(self.BOUNDED_TEST_IDS)
            aria = self.rng.choice(self.BOUNDED_ARIA_LABELS)

            attrs = {
                "class": classes,
                "data-testid": testid,
                "aria-label": aria,
            }
            if drop_id_class:
                attrs["__drop_id_class__"] = True

            elements.append(
                UIElement(
                    element_id=i,
                    tag=tag,
                    role=role,
                    text=text,
                    value=distractor_val,
                    bbox=(x, y, x + w, y + h),
                    attributes=attrs,
                    is_clickable=(tag in ("button", "a") or role in ("button", "link")),
                    is_input=(tag in ("input", "textarea") or role in ("textbox", "searchbox")),
                )
            )

        self.rng.shuffle(elements)
        return elements, target_el

    def _render_screenshot(
        self,
        elements: List[UIElement],
        target_el: UIElement,
        h: int = 224,
        w: int = 224,
    ) -> torch.Tensor:
        """
        Renders a synthetic pure-PyTorch screenshot image [3, H, W] representing the GUI canvas.
        Draws background and UI element bounding box rectangular patches.
        """
        # Background: light gray [0.92, 0.94, 0.96]
        img = torch.full((3, h, w), 0.94, dtype=torch.float32)
        img[0] = 0.92
        img[1] = 0.94
        img[2] = 0.96

        # Draw distractor UI elements
        for el in elements:
            x0 = max(0, min(w - 1, int(el.bbox[0] * w)))
            y0 = max(0, min(h - 1, int(el.bbox[1] * h)))
            x1 = max(0, min(w, int(el.bbox[2] * w)))
            y1 = max(0, min(h, int(el.bbox[3] * h)))
            if x1 <= x0 or y1 <= y0:
                continue

            tag_hash = int(hashlib.md5(f"{el.tag}_{el.role}".encode()).hexdigest()[:6], 16)
            r = ((tag_hash >> 16) & 0xFF) / 255.0 * 0.4 + 0.5  # pastel range 0.5-0.9
            g = ((tag_hash >> 8) & 0xFF) / 255.0 * 0.4 + 0.5
            b = (tag_hash & 0xFF) / 255.0 * 0.4 + 0.5

            img[0, y0:y1, x0:x1] = r
            img[1, y0:y1, x0:x1] = g
            img[2, y0:y1, x0:x1] = b

            # Darker border
            border_r, border_g, border_b = r * 0.7, g * 0.7, b * 0.7
            img[0, y0, x0:x1] = border_r
            img[1, y0, x0:x1] = border_g
            img[2, y0, x0:x1] = border_b
            img[0, y1 - 1, x0:x1] = border_r
            img[1, y1 - 1, x0:x1] = border_g
            img[2, y1 - 1, x0:x1] = border_b
            img[0, y0:y1, x0] = border_r
            img[1, y0:y1, x0] = border_g
            img[2, y0:y1, x0] = border_b
            img[0, y0:y1, x1 - 1] = border_r
            img[1, y0:y1, x1 - 1] = border_g
            img[2, y0:y1, x1 - 1] = border_b

        # Draw target element with prominent accent color
        tx0 = max(0, min(w - 1, int(target_el.bbox[0] * w)))
        ty0 = max(0, min(h - 1, int(target_el.bbox[1] * h)))
        tx1 = max(0, min(w, int(target_el.bbox[2] * w)))
        ty1 = max(0, min(h, int(target_el.bbox[3] * h)))
        if tx1 > tx0 and ty1 > ty0:
            # Vibrant accent color: [0.15, 0.45, 0.85]
            img[0, ty0:ty1, tx0:tx1] = 0.15
            img[1, ty0:ty1, tx0:tx1] = 0.45
            img[2, ty0:ty1, tx0:tx1] = 0.85

        return img

    def generate_trajectory(self, scenario_idx: Optional[int] = None, max_turns: Optional[int] = None) -> TrajectorySample:
        """
        Synthesizes a full multi-turn trajectory sample tau = (s_0, a_0, r_0, ..., s_T).
        Supports 20+ domain archetypes, Guardrail 2A (bounded BPE CSS vocabulary),
        Guardrail 2B (30% structural attribute dropout), Guardrail 2C (near-miss distractors),
        and Guardrail 2D (DOM grafting mode on 25% of synthetic turns).
        """
        if scenario_idx is None:
            scenario = self.rng.choice(self.ARCHETYPES)
        else:
            scenario = self.ARCHETYPES[scenario_idx % len(self.ARCHETYPES)]

        if "entities_fn" in scenario:
            entities = scenario["entities_fn"](self.rng)
            goal = scenario["goal_fn"](entities)
            steps = scenario["steps_fn"](entities)
            distractor_entities = [v for k, v in entities.items() if k.startswith("d")]
        else:
            entities = {}
            goal = scenario["goal"]
            steps = scenario["steps"]
            distractor_entities = []

        if max_turns is not None:
            steps = steps[:max_turns]

        sample_id = f"traj_{self.rng.randint(100000, 999999)}"
        turns: List[InteractionTurn] = []

        # Generate turns
        for turn_idx, (act_type, tag, role, el_name, arg_val, thought_tmpl) in enumerate(steps):
            # Guardrail 2B: 30% structural attribute dropout
            drop_id_class = (self.rng.random() < 0.30)

            # Guardrail 2D: DOM Grafting Mode on 25% of turns
            graft_dom = (self.rng.random() < 0.25)

            dom_elements, target_el = self._generate_dom_tree(
                target_element_name=el_name,
                target_tag=tag,
                target_role=role,
                distractor_entities=distractor_entities,
                drop_id_class=drop_id_class,
            )

            # Observation text: serialized DOM elements
            core_dom_str = " ".join(el.to_dom_string(drop_id_class=drop_id_class) for el in dom_elements[:6])
            if graft_dom:
                scaffold = self.rng.choice(self.DOM_SCAFFOLDS)
                dom_str = scaffold.replace("{content}", core_dom_str)
            else:
                dom_str = core_dom_str

            obs_text = f"Goal: {goal} | Screen State: {dom_str}"

            # Continuous observation embedding [hidden_dim] for JEPA world model
            # Derived deterministically from DOM elements and observation structure
            obs_features = compute_obs_features(obs_text, dom_elements[:6], self.hidden_dim)

            # Target coordinates
            if act_type in (ActionType.CLICK, ActionType.TYPE):
                coords = target_el.center
            elif act_type == ActionType.SCROLL:
                coords = (0.50, 0.75)  # Normalized scroll position
            else:
                coords = (0.0, 0.0)  # Non-spatial action

            # Thought rationale
            thought = f"Step {turn_idx}: {thought_tmpl}. Target element: {el_name}."

            # Synthetic screenshot image [3, H, W] for visual grounding
            img_size = getattr(self.config, "image_size", 224)
            screenshot = self._render_screenshot(dom_elements, target_el, h=img_size, w=img_size)

            turns.append(
                InteractionTurn(
                    turn_id=turn_idx,
                    dom_elements=dom_elements,
                    obs_text=obs_text,
                    obs_features=obs_features,
                    thought=thought,
                    action_type=act_type,
                    coordinates=coords,
                    argument=arg_val,
                    reward=1.0 if turn_idx == len(steps) - 1 else 0.0,
                    screenshot=screenshot,
                )
            )

        return TrajectorySample(
            sample_id=sample_id,
            goal=goal,
            turns=turns,
            total_reward=1.0,
        )


# ==============================================================================
# 4. Streaming PyTorch Dataset
# ==============================================================================

class AgentTrajectoryDataset(Dataset):
    """
    Streaming PyTorch Dataset yielding structured multi-turn agent interaction trajectories.
    Generates samples on-the-fly to eliminate monolithic dataset caching and fit
    comfortably within host system RAM and GPU VRAM constraints.
    """

    def __init__(
        self,
        config: JEDM2Config,
        virtual_size: int = 10000,
        seed: int = 42,
        max_turns: Optional[int] = None,
        tokenizer: Optional[ActionBPETokenizer] = None,
    ):
        super().__init__()
        self.config = config
        self.virtual_size = virtual_size
        self.seed = seed
        self.max_turns = max_turns
        self.tokenizer = tokenizer or ActionBPETokenizer(config.vocab_size)
        self.generator = SyntheticTrajectoryGenerator(config, seed=seed, tokenizer=self.tokenizer)

    def __len__(self) -> int:
        return self.virtual_size

    def __getitem__(self, idx: int) -> TrajectorySample:
        # Seed generator deterministically per index
        self.generator.rng.seed(self.seed + idx)
        return self.generator.generate_trajectory(
            scenario_idx=idx, max_turns=self.max_turns
        )

    def __iter__(self):
        """Allows streaming iteration over infinite trajectory stream."""
        idx = 0
        while True:
            yield self.__getitem__(idx)
            idx += 1


# ==============================================================================
# 5. Dynamic Trajectory Batch & Collator
# ==============================================================================

@dataclass
class TrajectoryBatch:
    """
    Structured container for dynamically collated multi-turn trajectory batches.
    Provides padded token IDs, causal attention masks, action token index mapping,
    component loss masks, and aligned JEPA state transition tuples.
    """

    input_ids: torch.Tensor  # [B, S]
    attention_mask: torch.Tensor  # [B, 1, S, S] causal & padding mask
    labels: torch.Tensor  # [B, S] autoregressive target token IDs (-100 for non-target)
    reason_loss_mask: torch.Tensor  # [B, S] bool mask for thought tokens
    action_indices: torch.Tensor  # [B, max_actions] sequence indices for action emissions
    action_type_targets: torch.Tensor  # [B, max_actions] ground-truth ActionType IDs
    action_type_mask: torch.Tensor  # [B, max_actions] bool mask for valid action positions
    coord_targets: torch.Tensor  # [B, max_actions, 2] ground-truth continuous coordinates
    coord_loss_mask: torch.Tensor  # [B, max_actions] bool mask for spatial actions
    arg_targets: torch.Tensor  # [B, S] ground-truth argument token IDs (-100 for non-arg)
    arg_loss_mask: torch.Tensor  # [B, S] bool mask for argument tokens
    jepa_obs_t: torch.Tensor  # [B, hidden_dim] JEPA observation at time t
    jepa_action_t: torch.Tensor  # [B] JEPA action discrete ID at time t
    jepa_obs_next: torch.Tensor  # [B, hidden_dim] JEPA target observation at time t+1
    images: Optional[torch.Tensor] = None  # [B, 3, H, W] screenshot images for visual grounding
    prompt_token_ids: Optional[torch.Tensor] = None  # [B, S_p] prompt token IDs for pointer copy
    prompt_mask: Optional[torch.Tensor] = None  # [B, S_p] boolean valid mask
    source_mask: Optional[torch.Tensor] = None  # [B] 1 for REAL, 0 for SYNTH

    def to(self, device: torch.device, dtype: Optional[torch.dtype] = None) -> "TrajectoryBatch":
        """Transfers all internal batch tensors to target device and precision."""
        images_tensor = (
            self.images.to(device=device, dtype=dtype or torch.float32)
            if self.images is not None
            else None
        )
        attn_mask_tensor = self.attention_mask.to(device=device)
        if dtype is not None and attn_mask_tensor.is_floating_point():
            attn_mask_tensor = attn_mask_tensor.to(dtype=dtype)
        prompt_ids_tensor = (
            self.prompt_token_ids.to(device=device)
            if self.prompt_token_ids is not None
            else None
        )
        prompt_mask_tensor = (
            self.prompt_mask.to(device=device)
            if self.prompt_mask is not None
            else None
        )
        source_mask_tensor = (
            self.source_mask.to(device=device)
            if self.source_mask is not None
            else None
        )
        return TrajectoryBatch(
            input_ids=self.input_ids.to(device=device),
            attention_mask=attn_mask_tensor,
            labels=self.labels.to(device=device),
            reason_loss_mask=self.reason_loss_mask.to(device=device),
            action_indices=self.action_indices.to(device=device),
            action_type_targets=self.action_type_targets.to(device=device),
            action_type_mask=self.action_type_mask.to(device=device),
            coord_targets=self.coord_targets.to(device=device, dtype=dtype or torch.float32),
            coord_loss_mask=self.coord_loss_mask.to(device=device),
            arg_targets=self.arg_targets.to(device=device),
            arg_loss_mask=self.arg_loss_mask.to(device=device),
            jepa_obs_t=self.jepa_obs_t.to(device=device, dtype=dtype or torch.float32),
            jepa_action_t=self.jepa_action_t.to(device=device),
            jepa_obs_next=self.jepa_obs_next.to(device=device, dtype=dtype or torch.float32),
            images=images_tensor,
            prompt_token_ids=prompt_ids_tensor,
            prompt_mask=prompt_mask_tensor,
            source_mask=source_mask_tensor,
        )


class TrajectoryCollator:
    """
    Dynamic Trajectory Collator for JED-M2 Large Action Model.
    Serializes variable-length multi-turn interaction traces into packed token sequences,
    computes 4D causal attention masks, maps action emission tokens, constructs
    component loss masks, and extracts JEPA transition pairs.
    """

    def __init__(
        self,
        config: JEDM2Config,
        tokenizer: Optional[Union[ActionBPETokenizer, LegacyHashTokenizer]] = None,
        max_seq_len: Optional[int] = None,
    ):
        self.config = config
        self.tokenizer = tokenizer or ActionBPETokenizer(config.vocab_size)
        self.max_seq_len = max_seq_len or config.max_seq_len

    def __call__(self, samples: List[TrajectorySample]) -> TrajectoryBatch:
        """
        Collates a list of TrajectorySample instances into a unified TrajectoryBatch.

        Args:
            samples: List of multi-turn TrajectorySample instances.

        Returns:
            Collate TrajectoryBatch ready for model ingestion and composite loss evaluation.
        """
        batch_size = len(samples)

        batch_tokens: List[List[int]] = []
        batch_reason_masks: List[List[bool]] = []
        batch_arg_masks: List[List[bool]] = []
        batch_action_indices: List[List[int]] = []
        batch_action_types: List[List[int]] = []
        batch_coords: List[List[Tuple[float, float]]] = []
        batch_is_spatial: List[List[bool]] = []

        batch_jepa_obs_t: List[torch.Tensor] = []
        batch_jepa_act_t: List[int] = []
        batch_jepa_obs_next: List[torch.Tensor] = []
        batch_images: List[torch.Tensor] = []
        batch_prompt_tokens: List[List[int]] = []

        for sample in samples:
            tokens: List[int] = [self.tokenizer.BOS_TOKEN_ID]
            reason_mask: List[bool] = [False]
            arg_mask: List[bool] = [False]
            action_indices: List[int] = []
            action_types: List[int] = []
            coords: List[Tuple[float, float]] = []
            is_spatial: List[bool] = []

            # Extract prompt tokens (turn 0 observation text)
            sample_prompt: List[int] = []
            if sample.turns:
                obs_toks_turn0 = self.tokenizer.encode(sample.turns[0].obs_text)[:128]
                sample_prompt = [self.tokenizer.OBS_TOKEN_ID] + obs_toks_turn0 + [self.tokenizer.OBS_END_TOKEN_ID]
            batch_prompt_tokens.append(sample_prompt or [self.tokenizer.BOS_TOKEN_ID])

            # Extract screenshot for visual grounding (turn 0)
            if sample.turns and sample.turns[0].screenshot is not None:
                batch_images.append(sample.turns[0].screenshot)
            else:
                img_sz = getattr(self.config, "image_size", 224)
                batch_images.append(torch.zeros(3, img_sz, img_sz, dtype=torch.float32))

            # Process each turn in trajectory
            for turn_idx, turn in enumerate(sample.turns):
                # 1. Observation sequence: <obs> ... </obs>
                tokens.append(self.tokenizer.OBS_TOKEN_ID)
                reason_mask.append(False)
                arg_mask.append(False)

                obs_toks = self.tokenizer.encode(turn.obs_text)[:32]  # Cap per-turn DOM tokens
                tokens.extend(obs_toks)
                reason_mask.extend([False] * len(obs_toks))
                arg_mask.extend([False] * len(obs_toks))

                tokens.append(self.tokenizer.OBS_END_TOKEN_ID)
                reason_mask.append(False)
                arg_mask.append(False)

                # 2. System 2 Reasoning rationale: <thought> ... </thought>
                tokens.append(self.tokenizer.THOUGHT_TOKEN_ID)
                reason_mask.append(False)
                arg_mask.append(False)

                thought_toks = self.tokenizer.encode(turn.thought)[:24]  # Cap rationale tokens
                tokens.extend(thought_toks)
                reason_mask.extend([True] * len(thought_toks))  # Loss active on thought tokens
                arg_mask.extend([False] * len(thought_toks))

                tokens.append(self.tokenizer.THOUGHT_END_TOKEN_ID)
                reason_mask.append(False)
                arg_mask.append(False)

                # 3. System 1 Action emission point: <action>
                # Record sequence position where action heads will be evaluated
                action_indices.append(len(tokens))
                action_types.append(int(turn.action_type))
                coords.append(turn.coordinates)
                is_spatial.append(turn.action_type in (ActionType.CLICK, ActionType.TYPE, ActionType.SCROLL))

                tokens.append(self.tokenizer.ACTION_TOKEN_ID)
                reason_mask.append(False)
                arg_mask.append(False)

                # Action type marker token
                act_token = self.tokenizer.encode(turn.action_type.name)[0]
                tokens.append(act_token)
                reason_mask.append(False)
                arg_mask.append(False)

                tokens.append(self.tokenizer.ACTION_END_TOKEN_ID)
                reason_mask.append(False)
                arg_mask.append(False)

                # 4. Action Argument payload: <arg> ... </arg>
                if turn.argument:
                    tokens.append(self.tokenizer.ARG_TOKEN_ID)
                    reason_mask.append(False)
                    arg_mask.append(False)

                    arg_toks = self.tokenizer.encode(turn.argument)[:16]
                    tokens.extend(arg_toks)
                    reason_mask.extend([False] * len(arg_toks))
                    arg_mask.extend([True] * len(arg_toks))  # Loss active on argument tokens

                    tokens.append(self.tokenizer.ARG_END_TOKEN_ID)
                    reason_mask.append(False)
                    arg_mask.append(False)

            # End of trajectory sequence: <eos>
            tokens.append(self.tokenizer.EOS_TOKEN_ID)
            reason_mask.append(False)
            arg_mask.append(False)

            # Bound length to max_seq_len
            tokens = tokens[:self.max_seq_len]
            reason_mask = reason_mask[:self.max_seq_len]
            arg_mask = arg_mask[:self.max_seq_len]

            # Filter action indices within max_seq_len
            valid_actions = [(idx, at, c, sp) for idx, at, c, sp in zip(action_indices, action_types, coords, is_spatial) if idx < self.max_seq_len]
            if valid_actions:
                a_idx, a_t, c_t, s_t = zip(*valid_actions)
                batch_action_indices.append(list(a_idx))
                batch_action_types.append(list(a_t))
                batch_coords.append(list(c_t))
                batch_is_spatial.append(list(s_t))
            else:
                batch_action_indices.append([0])
                batch_action_types.append([0])
                batch_coords.append([(0.0, 0.0)])
                batch_is_spatial.append([False])

            batch_tokens.append(tokens)
            batch_reason_masks.append(reason_mask)
            batch_arg_masks.append(arg_mask)

            # Extract JEPA state transition pair (s_t, a_t, s_{t+1})
            if len(sample.turns) >= 2:
                # Pair turn 0 and turn 1
                batch_jepa_obs_t.append(sample.turns[0].obs_features)
                batch_jepa_act_t.append(int(sample.turns[0].action_type))
                batch_jepa_obs_next.append(sample.turns[1].obs_features)
            else:
                # Single-turn fallback
                batch_jepa_obs_t.append(sample.turns[0].obs_features)
                batch_jepa_act_t.append(int(sample.turns[0].action_type))
                batch_jepa_obs_next.append(sample.turns[0].obs_features)

        # Compute max sequence length in batch
        max_batch_seq_len = max(len(t) for t in batch_tokens)
        max_batch_actions = max(len(a) for a in batch_action_indices)

        # Pad tokens, masks, and labels
        padded_input_ids = torch.full((batch_size, max_batch_seq_len), self.tokenizer.PAD_TOKEN_ID, dtype=torch.long)
        padded_labels = torch.full((batch_size, max_batch_seq_len), -100, dtype=torch.long)
        padded_reason_mask = torch.zeros((batch_size, max_batch_seq_len), dtype=torch.bool)
        padded_arg_mask = torch.zeros((batch_size, max_batch_seq_len), dtype=torch.bool)
        padded_arg_targets = torch.full((batch_size, max_batch_seq_len), -100, dtype=torch.long)

        # Pad action containers
        padded_action_indices = torch.zeros((batch_size, max_batch_actions), dtype=torch.long)
        padded_action_types = torch.zeros((batch_size, max_batch_actions), dtype=torch.long)
        padded_action_mask = torch.zeros((batch_size, max_batch_actions), dtype=torch.bool)
        padded_coords = torch.zeros((batch_size, max_batch_actions, 2), dtype=torch.float32)
        padded_coord_mask = torch.zeros((batch_size, max_batch_actions), dtype=torch.bool)

        for b in range(batch_size):
            seq_l = len(batch_tokens[b])
            padded_input_ids[b, :seq_l] = torch.tensor(batch_tokens[b], dtype=torch.long)
            # Autoregressive next-token prediction: labels shifted right
            padded_labels[b, :seq_l] = torch.tensor(batch_tokens[b], dtype=torch.long)

            padded_reason_mask[b, :seq_l] = torch.tensor(batch_reason_masks[b], dtype=torch.bool)
            padded_arg_mask[b, :seq_l] = torch.tensor(batch_arg_masks[b], dtype=torch.bool)
            # Argument target labels active on arg tokens
            for s_idx in range(seq_l):
                if batch_arg_masks[b][s_idx]:
                    padded_arg_targets[b, s_idx] = batch_tokens[b][s_idx]

            # Action fields
            num_act = len(batch_action_indices[b])
            padded_action_indices[b, :num_act] = torch.tensor(batch_action_indices[b], dtype=torch.long)
            padded_action_types[b, :num_act] = torch.tensor(batch_action_types[b], dtype=torch.long)
            padded_action_mask[b, :num_act] = True
            padded_coords[b, :num_act] = torch.tensor(batch_coords[b], dtype=torch.float32)
            padded_coord_mask[b, :num_act] = torch.tensor(batch_is_spatial[b], dtype=torch.bool)

        # Pad prompt tokens for pointer copy supervision
        max_batch_prompt_len = max(max(len(p) for p in batch_prompt_tokens), 1)
        padded_prompt_token_ids = torch.full((batch_size, max_batch_prompt_len), self.tokenizer.PAD_TOKEN_ID, dtype=torch.long)
        padded_prompt_mask = torch.zeros((batch_size, max_batch_prompt_len), dtype=torch.bool)
        for b in range(batch_size):
            p_len = len(batch_prompt_tokens[b])
            padded_prompt_token_ids[b, :p_len] = torch.tensor(batch_prompt_tokens[b], dtype=torch.long)
            padded_prompt_mask[b, :p_len] = True

        # 4D Causal & Padding Attention Mask: [B, 1, S, S]
        # True = participate in attention, False = masked out
        pad_mask = (padded_input_ids != self.tokenizer.PAD_TOKEN_ID)[:, None, None, :]  # [B, 1, 1, S]
        causal_mask = torch.tril(torch.ones(max_batch_seq_len, max_batch_seq_len, dtype=torch.bool))[None, None, :, :]  # [1, 1, S, S]
        attention_mask = causal_mask & pad_mask  # [B, 1, S, S]

        # Stack JEPA transitions: [B, hidden_dim]
        jepa_obs_t = torch.stack(batch_jepa_obs_t, dim=0)
        jepa_action_t = torch.tensor(batch_jepa_act_t, dtype=torch.long)
        jepa_obs_next = torch.stack(batch_jepa_obs_next, dim=0)

        # Stack screenshot images: [B, 3, H, W]
        images_tensor = torch.stack(batch_images, dim=0) if batch_images else None

        # Source domain flag: 1.0 for REAL, 0.0 for SYNTH (Guardrail 4A)
        source_flags = [
            1.0 if getattr(s, "is_real", False) or s.__class__.__name__ == "RealWebTrajectorySample" else 0.0
            for s in samples
        ]
        source_mask_tensor = torch.tensor(source_flags, dtype=torch.float32)

        return TrajectoryBatch(
            input_ids=padded_input_ids,
            attention_mask=attention_mask,
            labels=padded_labels,
            reason_loss_mask=padded_reason_mask,
            action_indices=padded_action_indices,
            action_type_targets=padded_action_types,
            action_type_mask=padded_action_mask,
            coord_targets=padded_coords,
            coord_loss_mask=padded_coord_mask,
            arg_targets=padded_arg_targets,
            arg_loss_mask=padded_arg_mask,
            jepa_obs_t=jepa_obs_t,
            jepa_action_t=jepa_action_t,
            jepa_obs_next=jepa_obs_next,
            images=images_tensor,
            prompt_token_ids=padded_prompt_token_ids,
            prompt_mask=padded_prompt_mask,
            source_mask=source_mask_tensor,
        )
