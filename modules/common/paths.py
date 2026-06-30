"""Central path / settings engine for WaferTools.

Two user-facing settings live here and are shared by every module:

  * results_root  -- the base folder all results are written under
                     (Richard's request: user-definable, was hard-wired to
                      <install>/results, i.e. D:\\WaferTools-main\\results).
  * project_name  -- the wafer / project label. It is auto-derived from each
                     module's input (image file name or images-folder name) and
                     used BOTH as the run-folder suffix and as the file-name
                     prefix, e.g.  results/section_counter/wafer 14_20260726_0930/
                                   wafer 14_sections.csv

Settings are persisted to <install>/wafertools_config.json so they survive
restarts. results_root may also be overridden with the WAFERTOOLS_RESULTS_ROOT
environment variable.
"""

import os
import re
import json
from datetime import datetime

# --------------------------------------------------------------------------- #
# locations
# --------------------------------------------------------------------------- #

def install_dir() -> str:
    """Repo / install root: <install>/modules/common/paths.py -> <install>."""
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _config_path() -> str:
    return os.path.join(install_dir(), "wafertools_config.json")


def _load_config() -> dict:
    try:
        with open(_config_path(), "r", encoding="utf-8") as f:
            cfg = json.load(f)
            return cfg if isinstance(cfg, dict) else {}
    except Exception:
        return {}


def _save_config(cfg: dict) -> None:
    try:
        with open(_config_path(), "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2, ensure_ascii=False)
    except Exception:
        # never let a settings-write failure break a results export
        pass


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #

# characters that are illegal in Windows file/folder names
_ILLEGAL = re.compile(r'[<>:"/\\|?*\x00-\x1f]')


def sanitize_name(name: str) -> str:
    """Make a string safe to use as a Windows/macOS file or folder name.

    Spaces are preserved on purpose (Richard's example folder is 'wafer 14_...').
    """
    if not name:
        return ""
    name = _ILLEGAL.sub("", str(name))
    name = re.sub(r"\s+", " ", name).strip()
    # Windows forbids trailing dots / spaces
    name = name.rstrip(". ")
    return name


def derive_project_name(source: str | None) -> str:
    """Derive a project/wafer label from an input file or folder path.

    A file  -> basename without extension (a leading 'original_' added by the
               section-counter uploader is stripped).
    A folder-> basename.
    """
    if not source:
        return ""
    base = os.path.basename(str(source).rstrip("/\\"))
    # the section-counter uploader saves originals as 'original_<name>'
    if base.startswith("original_"):
        base = base[len("original_"):]
    root, _ext = os.path.splitext(base)
    return sanitize_name(root or base)


# --------------------------------------------------------------------------- #
# results root
# --------------------------------------------------------------------------- #

def default_results_root() -> str:
    return os.path.join(install_dir(), "results")


def get_results_root() -> str:
    """Resolved results base: env var > config > default. Created on access."""
    root = os.environ.get("WAFERTOOLS_RESULTS_ROOT") or _load_config().get("results_root")
    root = root or default_results_root()
    return ensure_dir(os.path.abspath(os.path.expanduser(root)))


def set_results_root(path: str) -> str:
    """Persist a user-defined results base folder. Empty -> revert to default."""
    cfg = _load_config()
    path = (path or "").strip()
    if path:
        cfg["results_root"] = os.path.abspath(os.path.expanduser(path))
    else:
        cfg.pop("results_root", None)
    _save_config(cfg)
    return get_results_root()


# --------------------------------------------------------------------------- #
# project / wafer name
# --------------------------------------------------------------------------- #

def get_project_name() -> str:
    return sanitize_name(_load_config().get("project_name", ""))


def set_project_name(name: str, locked: bool = True) -> str:
    """Manually set the project name. locked=True stops auto-derivation from
    overwriting it (used by the sidebar override field)."""
    cfg = _load_config()
    name = sanitize_name(name)
    if name:
        cfg["project_name"] = name
        cfg["project_locked"] = bool(locked)
    else:
        # clearing the field releases the lock and lets auto-derive resume
        cfg.pop("project_name", None)
        cfg["project_locked"] = False
    _save_config(cfg)
    return name


def is_project_locked() -> bool:
    return bool(_load_config().get("project_locked"))


def resolve_project(source: str | None = None) -> str:
    """Return the effective project name for a run and persist it.

    If a manual override is locked, it wins. Otherwise derive from ``source``;
    fall back to the last persisted name when nothing can be derived.
    """
    cfg = _load_config()
    if cfg.get("project_locked") and cfg.get("project_name"):
        return sanitize_name(cfg["project_name"])

    derived = derive_project_name(source)
    name = derived or sanitize_name(cfg.get("project_name", ""))
    if name and name != cfg.get("project_name"):
        cfg["project_name"] = name
        _save_config(cfg)
    return name


# --------------------------------------------------------------------------- #
# run ids / dirs / file names
# --------------------------------------------------------------------------- #

def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def project_root() -> str:
    # kept for backwards compatibility; results now resolve via get_results_root()
    return install_dir()


def get_run_id(project: str | None = None) -> str:
    """Timestamped run id, prefixed with the project name when available.

    e.g. 'wafer 14_20260726_093015'  (or just '20260726_093015').
    """
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    p = sanitize_name(project if project is not None else get_project_name())
    return f"{p}_{ts}" if p else ts


def get_run_dir(module: str, base: str | None = None,
                run_id: str | None = None, project: str | None = None) -> str:
    """<results_root>/<module>/<project>_<timestamp>/  (created)."""
    root = base or get_results_root()
    rid = run_id or get_run_id(project)
    path = os.path.join(root, module, rid)
    return ensure_dir(path)


def prefixed(name: str, project: str | None = None) -> str:
    """Prefix a file name with the project label: 'wafer 14_sections.csv'."""
    p = sanitize_name(project if project is not None else get_project_name())
    return f"{p}_{name}" if p else name
