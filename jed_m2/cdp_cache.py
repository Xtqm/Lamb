"""
JED-M2 Chrome DevTools Protocol (CDP) Script Cache Engine.

Provides:
1. CachedScript and CDPScriptCache managing persistent script compilation and handle reuse.
2. Dual-tier evaluation:
   - Tier 1: In-page function closure registry on window.__jed_scripts__.
   - Tier 2: Native CDP session Runtime.compileScript / Runtime.runScript bytecode execution.
3. Automatic execution context invalidation on document navigation and transparent re-compilation.
4. Evaluation timing telemetry and cache performance metrics.
5. Graceful fallback to uncompiled evaluation and mock mode pass-through.

STRICT CONVENTION: 100% native PyTorch/Python without external compiled C++/CUDA packages.
STRICT CONVENTION: Pure 7-bit ASCII logging (no Unicode box-drawing characters).
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import logging
import time
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

logger = logging.getLogger("jed_m2.cdp_cache")


# ==============================================================================
# 1. Script Record & Cache Container
# ==============================================================================

@dataclass
class CachedScript:
    """Represents a pre-compiled, cached JavaScript evaluation routine."""

    key: str
    source: str
    in_page_name: str
    hash_digest: str
    script_id: Optional[str] = None
    call_count: int = 0
    total_time_ms: float = 0.0

    @property
    def avg_time_ms(self) -> float:
        """Average evaluation duration in milliseconds."""
        if self.call_count == 0:
            return 0.0
        return self.total_time_ms / float(self.call_count)


class CDPScriptCache:
    """
    Persistent browser script compilation and handle caching engine.
    Eliminates redundant V8 AST parsing and compilation overhead by compiling
    recurring browser evaluation routines once and invoking cached closures.
    """

    def __init__(self, enable_cdp_session: bool = True):
        self.enable_cdp_session = enable_cdp_session
        self.scripts: Dict[str, CachedScript] = {}
        self.is_compiled: bool = False
        self.current_url: Optional[str] = None
        self._cdp_session: Optional[Any] = None
        self._last_page_ref: Optional[Any] = None

        # Telemetry statistics
        self.stats: Dict[str, Any] = {
            "hits": 0,
            "misses": 0,
            "compilations": 0,
            "invalidations": 0,
            "total_eval_time_ms": 0.0,
        }

    def register(self, key: str, source: str) -> CachedScript:
        """
        Registers a JavaScript source routine under a unique key.
        The source must evaluate to a callable function accepting a single argument dictionary:
        e.g., '(args) => { ... }'.
        """
        clean_src = source.strip()
        digest = hashlib.sha256(clean_src.encode("utf-8")).hexdigest()[:16]
        in_page_name = f"__fn_{key}"

        script = CachedScript(
            key=key,
            source=clean_src,
            in_page_name=in_page_name,
            hash_digest=digest,
        )
        self.scripts[key] = script
        self.is_compiled = False
        return script

    def register_routine(self, routine: Any) -> CachedScript:
        """Registers a MicroScriptRoutine instance into the script cache."""
        name = getattr(routine, "name", str(routine))
        source = getattr(routine, "source", str(routine))
        return self.register(name, source)

    def register_workspace(self, workspace: Any) -> None:
        """Registers all discovered routines from a MicroScriptWorkspace."""
        routines = getattr(workspace, "routines", {})
        for routine in routines.values():
            self.register_routine(routine)

    def invalidate(self) -> None:
        """
        Invalidates cached script handles and resets compiled state.
        Called on navigation, page reload, or frame detachment.
        """
        for script in self.scripts.values():
            script.script_id = None
        self.is_compiled = False
        self._cdp_session = None
        self.stats["invalidations"] += 1

    def attach_cdp_session(self, page: Any) -> Optional[Any]:
        """Attempts to open and attach a direct CDP session to the page."""
        if not self.enable_cdp_session or page is None:
            return None
        if self._cdp_session is not None:
            return self._cdp_session

        try:
            if hasattr(page, "context") and hasattr(page.context, "new_cdp_session"):
                self._cdp_session = page.context.new_cdp_session(page)
                return self._cdp_session
        except Exception as exc:
            logger.debug("Could not attach CDPSession: %s. Using in-page function handles.", exc)
            self._cdp_session = None
        return None

    def ensure_compiled(self, page: Any) -> bool:
        """
        Compiles all registered scripts in the active page context.
        Injects in-page function handles into window.__jed_scripts__ (Tier 1)
        and compiles V8 scriptIds via CDP Runtime.compileScript (Tier 2).
        """
        if page is None:
            return False

        try:
            curr_url = page.url if hasattr(page, "url") else None
        except Exception:
            curr_url = None

        if self.is_compiled and curr_url == self.current_url and page is self._last_page_ref:
            return True

        self._last_page_ref = page
        self.current_url = curr_url

        # Ensure base in-page script namespace exists
        try:
            page.evaluate("window.__jed_scripts__ = window.__jed_scripts__ || {};")
        except Exception as exc:
            logger.debug("Could not initialize window.__jed_scripts__: %s", exc)
            return False

        # Compile each registered script
        cdp = self.attach_cdp_session(page)
        for key, script in self.scripts.items():
            # Tier 1: In-page function closure registration
            injection_js = f"""
            (() => {{
                try {{
                    const fn = ({script.source});
                    window.__jed_scripts__['{key}'] = fn;
                    return true;
                }} catch (e) {{
                    return false;
                }}
            }})();
            """
            try:
                page.evaluate(injection_js)
            except Exception as exc:
                logger.debug("Failed in-page registration for '%s': %s", key, exc)

            # Tier 2: Native CDP bytecode compilation if session is available
            if cdp is not None:
                try:
                    expr = f"window.__jed_scripts__['{key}'](window.__jed_arg__)"
                    res = cdp.send(
                        "Runtime.compileScript",
                        {
                            "expression": expr,
                            "sourceURL": f"__jed_cdp_{key}.js",
                            "persistScript": True,
                        },
                    )
                    script.script_id = res.get("scriptId")
                except Exception as exc:
                    logger.debug("CDP Runtime.compileScript failed for '%s': %s", key, exc)
                    script.script_id = None

        self.is_compiled = True
        self.stats["compilations"] += 1
        return True

    def evaluate(
        self,
        page: Optional[Any],
        key: str,
        args: Optional[Dict[str, Any]] = None,
        fallback_fn: Optional[Callable[[], Any]] = None,
    ) -> Any:
        """
        Evaluates a pre-compiled script by key, passing args dictionary.
        Automatically re-compiles on execution context destruction or navigation.
        """
        if page is None:
            if fallback_fn is not None:
                return fallback_fn()
            return None

        if key not in self.scripts:
            raise KeyError(f"Script key '{key}' has not been registered in CDPScriptCache.")

        script = self.scripts[key]
        call_args = args or {}
        start_t = time.perf_counter()

        # Check navigation / compilation state
        try:
            curr_url = page.url if hasattr(page, "url") else None
        except Exception:
            curr_url = None

        if not self.is_compiled or curr_url != self.current_url or page is not self._last_page_ref:
            self.stats["misses"] += 1
            self.ensure_compiled(page)
        else:
            self.stats["hits"] += 1

        result = None
        eval_success = False

        # Tier 2: Execute via native CDP Runtime.runScript if script_id is present
        if self._cdp_session is not None and script.script_id is not None:
            try:
                # Set arguments in window.__jed_arg__
                page.evaluate("arg => { window.__jed_arg__ = arg; }", call_args)
                run_res = self._cdp_session.send(
                    "Runtime.runScript",
                    {
                        "scriptId": script.script_id,
                        "awaitPromise": True,
                        "returnByValue": True,
                    },
                )
                result = run_res.get("result", {}).get("value")
                eval_success = True
            except Exception as exc:
                logger.debug("CDP Runtime.runScript failed for '%s': %s. Retrying via Tier 1.", key, exc)

        # Tier 1: Execute via in-page function closure
        if not eval_success:
            try:
                result = page.evaluate(
                    f"arg => {{ return (window.__jed_scripts__ && window.__jed_scripts__['{key}']) ? window.__jed_scripts__['{key}'](arg) : null; }}",
                    call_args,
                )
                if result is not None:
                    eval_success = True
            except Exception as exc:
                # Execution context was likely destroyed; invalidate and retry once
                logger.debug("Tier 1 evaluation failed for '%s': %s. Re-compiling...", key, exc)
                self.invalidate()
                self.ensure_compiled(page)
                try:
                    result = page.evaluate(
                        f"arg => {{ return (window.__jed_scripts__ && window.__jed_scripts__['{key}']) ? window.__jed_scripts__['{key}'](arg) : null; }}",
                        call_args,
                    )
                    eval_success = True
                except Exception as retry_exc:
                    logger.debug("Retry evaluation failed: %s", retry_exc)
                    if fallback_fn is not None:
                        return fallback_fn()
                    return None

        elapsed_ms = (time.perf_counter() - start_t) * 1000.0
        script.call_count += 1
        script.total_time_ms += elapsed_ms
        self.stats["total_eval_time_ms"] += elapsed_ms

        return result

    def get_telemetry(self) -> Dict[str, Any]:
        """Returns performance metrics and cache statistics."""
        total_calls = self.stats["hits"] + self.stats["misses"]
        hit_rate = (float(self.stats["hits"]) / float(total_calls)) if total_calls > 0 else 0.0
        avg_latency = (
            self.stats["total_eval_time_ms"] / float(total_calls) if total_calls > 0 else 0.0
        )
        return {
            "total_calls": total_calls,
            "hits": self.stats["hits"],
            "misses": self.stats["misses"],
            "hit_rate": round(hit_rate, 4),
            "compilations": self.stats["compilations"],
            "invalidations": self.stats["invalidations"],
            "total_eval_time_ms": round(self.stats["total_eval_time_ms"], 3),
            "avg_latency_ms": round(avg_latency, 3),
            "scripts": {
                k: {
                    "calls": s.call_count,
                    "avg_time_ms": round(s.avg_time_ms, 3),
                    "script_id": s.script_id,
                }
                for k, s in self.scripts.items()
            },
        }
