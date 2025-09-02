import os
from datetime import datetime


def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def get_run_id() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def project_root() -> str:
    # Assume running from project root; fall back to cwd
    return os.getcwd()


def get_run_dir(module: str, base: str = "results", run_id: str | None = None) -> str:
    rid = run_id or get_run_id()
    root = project_root()
    path = os.path.join(root, base, module, rid)
    ensure_dir(path)
    return path

