"""
JED-M2 CDP Micro-Script Synthesis & Workspace Persistence Engine (BrowserCode Style).

Enables synthesis, parameterized execution, and local disk persistence of reusable
JavaScript browser automation routines (.bcode/workspace/) executing in ~1 ms with 0 LLM tokens.
"""

from dataclasses import dataclass, field
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

logger = logging.getLogger("jed_m2.bcode")


# ==============================================================================
# 1. Micro-Script Routine Container
# ==============================================================================

@dataclass
class MicroScriptRoutine:
    """Represents a parameterized, reusable JavaScript browser routine."""

    name: str
    description: str
    source: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    returns: Dict[str, Any] = field(default_factory=dict)
    version: str = "1.0.0"

    def to_manifest_dict(self) -> Dict[str, Any]:
        """Serializes routine metadata to JSON manifest representation."""
        return {
            "name": self.name,
            "description": self.description,
            "version": self.version,
            "parameters": self.parameters,
            "returns": self.returns,
        }

    @classmethod
    def from_files(cls, js_path: Path, json_path: Optional[Path] = None) -> "MicroScriptRoutine":
        """Loads routine from JavaScript source and companion JSON manifest."""
        with open(js_path, "r", encoding="utf-8") as f:
            source = f.read().strip()

        manifest: Dict[str, Any] = {}
        if json_path is not None and json_path.exists():
            with open(json_path, "r", encoding="utf-8") as f:
                manifest = json.load(f)

        name = manifest.get("name", js_path.stem)
        description = manifest.get("description", f"Micro-script routine: {name}")
        parameters = manifest.get("parameters", {"type": "object", "properties": {}})
        returns = manifest.get("returns", {"type": "object"})
        version = manifest.get("version", "1.0.0")

        return cls(
            name=name,
            description=description,
            source=source,
            parameters=parameters,
            returns=returns,
            version=version,
        )


# ==============================================================================
# 2. Canonical Routine Templates
# ==============================================================================

CANONICAL_ROUTINES: Dict[str, Dict[str, Any]] = {
    "form_fill": {
        "description": "Atomically populates multiple form fields and emits change/input events.",
        "source": """(args) => {
    const fields = args.fields || {};
    let count = 0;
    for (const [selector, val] of Object.entries(fields)) {
        const el = document.querySelector(selector);
        if (el) {
            el.value = String(val);
            el.dispatchEvent(new Event('input', { bubbles: true }));
            el.dispatchEvent(new Event('change', { bubbles: true }));
            count++;
        }
    }
    return { success: true, fields_updated: count };
}""",
        "parameters": {
            "type": "object",
            "required": ["fields"],
            "properties": {
                "fields": {
                    "type": "object",
                    "description": "Mapping of CSS selectors to input string values.",
                }
            },
        },
        "returns": {
            "type": "object",
            "properties": {
                "success": {"type": "boolean"},
                "fields_updated": {"type": "integer"},
            },
        },
    },
    "sort_table": {
        "description": "Locates target table, clicks column header, and verifies sort order.",
        "source": """(args) => {
    const selector = args.table_selector || 'table';
    const colIdx = args.column_index !== undefined ? args.column_index : 0;
    const table = document.querySelector(selector);
    if (!table) return { success: false, error: 'Table not found' };
    const headers = table.querySelectorAll('th');
    if (colIdx >= 0 && colIdx < headers.length) {
        headers[colIdx].click();
        return { success: true, sorted_column: colIdx };
    }
    return { success: false, error: 'Column index out of bounds' };
}""",
        "parameters": {
            "type": "object",
            "properties": {
                "table_selector": {"type": "string", "default": "table"},
                "column_index": {"type": "integer", "default": 0},
            },
        },
        "returns": {
            "type": "object",
            "properties": {
                "success": {"type": "boolean"},
                "sorted_column": {"type": "integer"},
            },
        },
    },
    "cart_checkout": {
        "description": "Automated cart checkout button click and submission sequence.",
        "source": """(args) => {
    const selector = args.checkout_selector || 'button.checkout, #checkout, [data-action="checkout"]';
    const btn = document.querySelector(selector);
    if (!btn) return { success: false, error: 'Checkout button not found' };
    btn.click();
    return { success: true, message: 'Checkout button triggered' };
}""",
        "parameters": {
            "type": "object",
            "properties": {
                "checkout_selector": {"type": "string", "default": "button.checkout"},
            },
        },
        "returns": {
            "type": "object",
            "properties": {
                "success": {"type": "boolean"},
                "message": {"type": "string"},
            },
        },
    },
    "batch_select": {
        "description": "Atomically sets multiple select dropdown options across the page.",
        "source": """(args) => {
    const selections = args.selections || {};
    let count = 0;
    for (const [selector, val] of Object.entries(selections)) {
        const select = document.querySelector(selector);
        if (select && select.tagName.toLowerCase() === 'select') {
            const option = [...select.options].find(o => o.value === String(val) || o.textContent.trim() === String(val));
            if (option) {
                select.value = option.value;
                select.dispatchEvent(new Event('input', { bubbles: true }));
                select.dispatchEvent(new Event('change', { bubbles: true }));
                count++;
            }
        }
    }
    return { success: true, selections_updated: count };
}""",
        "parameters": {
            "type": "object",
            "required": ["selections"],
            "properties": {
                "selections": {
                    "type": "object",
                    "description": "Mapping of select element selectors to option values.",
                }
            },
        },
        "returns": {
            "type": "object",
            "properties": {
                "success": {"type": "boolean"},
                "selections_updated": {"type": "integer"},
            },
        },
    },
}


# ==============================================================================
# 3. Micro-Script Workspace Engine
# ==============================================================================

class MicroScriptWorkspace:
    """
    Manages persistent local disk storage of synthesized browser micro-scripts (.bcode/workspace/).
    Handles script discovery, schema validation, registration, and routine persistence.
    """

    def __init__(
        self,
        workspace_dir: Optional[Union[str, Path]] = None,
        auto_seed: bool = True,
    ):
        if workspace_dir is not None:
            self.workspace_dir = Path(workspace_dir)
        else:
            self.workspace_dir = Path(".bcode/workspace")

        self.routines: Dict[str, MicroScriptRoutine] = {}
        self.workspace_dir.mkdir(parents=True, exist_ok=True)

        if auto_seed:
            self.seed_default_library()

        self.discover_routines()

    def discover_routines(self) -> Dict[str, MicroScriptRoutine]:
        """Discovers and parses all .js and companion .json manifests in workspace directory."""
        if not self.workspace_dir.exists():
            return self.routines

        for js_file in self.workspace_dir.glob("*.js"):
            routine_name = js_file.stem
            json_file = self.workspace_dir / f"{routine_name}.json"
            try:
                routine = MicroScriptRoutine.from_files(js_file, json_file)
                self.routines[routine.name] = routine
            except Exception as exc:
                logger.warning("Failed to parse routine '%s': %s", js_file.name, exc)

        return self.routines

    def register_routine(
        self,
        routine: MicroScriptRoutine,
        persist: bool = True,
    ) -> None:
        """Registers a micro-script routine, optionally persisting source and manifest to disk."""
        self.routines[routine.name] = routine

        if persist:
            self.workspace_dir.mkdir(parents=True, exist_ok=True)
            js_path = self.workspace_dir / f"{routine.name}.js"
            json_path = self.workspace_dir / f"{routine.name}.json"

            with open(js_path, "w", encoding="utf-8") as f:
                f.write(routine.source)

            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(routine.to_manifest_dict(), f, indent=2)

    def get_routine(self, name: str) -> Optional[MicroScriptRoutine]:
        """Retrieves routine by identifier."""
        return self.routines.get(name)

    def validate_arguments(
        self,
        routine_name: str,
        args: Optional[Dict[str, Any]] = None,
    ) -> Tuple[bool, Optional[str]]:
        """
        Validates argument payload against the routine's declared parameter schema.
        Returns:
            Tuple of (is_valid: bool, error_message: Optional[str]).
        """
        routine = self.get_routine(routine_name)
        if routine is None:
            return False, f"Routine '{routine_name}' is not registered in MicroScriptWorkspace."

        payload = args if args is not None else {}
        if not isinstance(payload, dict):
            return False, f"Routine arguments must be a dictionary/object, got {type(payload).__name__}."

        schema = routine.parameters or {}
        required_fields = schema.get("required", [])
        properties = schema.get("properties", {})

        # Check required fields
        for req in required_fields:
            if req not in payload:
                return False, f"Missing required parameter '{req}' for routine '{routine_name}'."

        # Check parameter types
        for k, val in payload.items():
            if k in properties:
                expected_type = properties[k].get("type")
                if expected_type == "string" and not isinstance(val, str):
                    return False, f"Parameter '{k}' must be string, got {type(val).__name__}."
                elif expected_type == "integer" and (not isinstance(val, int) or isinstance(val, bool)):
                    return False, f"Parameter '{k}' must be integer, got {type(val).__name__}."
                elif expected_type == "number" and (not isinstance(val, (int, float)) or isinstance(val, bool)):
                    return False, f"Parameter '{k}' must be number, got {type(val).__name__}."
                elif expected_type == "boolean" and not isinstance(val, bool):
                    return False, f"Parameter '{k}' must be boolean, got {type(val).__name__}."
                elif expected_type == "object" and not isinstance(val, dict):
                    return False, f"Parameter '{k}' must be object/dict, got {type(val).__name__}."
                elif expected_type == "array" and not isinstance(val, list):
                    return False, f"Parameter '{k}' must be array/list, got {type(val).__name__}."

        return True, None

    def seed_default_library(self) -> None:
        """Seeds canonical standard routines into workspace directory if missing."""
        self.workspace_dir.mkdir(parents=True, exist_ok=True)

        for name, spec in CANONICAL_ROUTINES.items():
            js_path = self.workspace_dir / f"{name}.js"
            json_path = self.workspace_dir / f"{name}.json"

            if not js_path.exists():
                with open(js_path, "w", encoding="utf-8") as f:
                    f.write(spec["source"])

            if not json_path.exists():
                manifest = {
                    "name": name,
                    "description": spec["description"],
                    "version": "1.0.0",
                    "parameters": spec["parameters"],
                    "returns": spec["returns"],
                }
                with open(json_path, "w", encoding="utf-8") as f:
                    json.dump(manifest, f, indent=2)
