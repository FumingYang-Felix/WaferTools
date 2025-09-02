#!/usr/bin/env python3
import os
import sys
import time
import subprocess
from importlib import import_module


REQUIRED = {
    # module: pip name (or install command segment)
    'dash': 'dash',
    'dash_bootstrap_components': 'dash-bootstrap-components',
    'plotly': 'plotly',
    'numpy': 'numpy',
    'pandas': 'pandas',
    'PIL': 'Pillow',
    'matplotlib': 'matplotlib',
    'scipy': 'scipy',
    'skimage': 'scikit-image',
    'cv2': 'opencv-python',
    'sklearn': 'scikit-learn',
    'tifffile': 'tifffile',
    'seaborn': 'seaborn',
    'requests': 'requests',
    'PyQt5': 'PyQt5',
    'ultralytics': 'ultralytics',
    # segment-anything from GitHub
    'segment_anything': 'git+https://github.com/facebookresearch/segment-anything.git',
}


def run(cmd: list[str]) -> int:
    return subprocess.call(cmd)


def ensure_package(mod: str, spec: str) -> None:
    try:
        if mod == 'PIL':
            import_module('PIL.Image')
        else:
            import_module(mod)
        return
    except Exception:
        pass

    py = sys.executable
    print(f"[SETUP] Installing {spec} ...")
    wheels = os.path.join(os.getcwd(), 'offline', 'wheels')
    args = [py, '-m', 'pip', 'install', '--quiet']
    if os.path.isdir(wheels):
        args += ['--no-index', '--find-links', wheels]
    args.append(spec)
    rc = run(args)
    if rc != 0:
        # Fallback for torch CPU on some platforms
        if mod == 'torch':
            print("[SETUP] Retrying torch CPU wheels ...")
            run([py, '-m', 'pip', 'install', '--quiet', 'torch'])


def ensure_torch() -> None:
    try:
        import_module('torch')
        return
    except Exception:
        pass
    py = sys.executable
    print('[SETUP] Installing torch (CPU) ...')
    rc = run([py, '-m', 'pip', 'install', '--quiet', 'torch'])
    if rc != 0 and sys.platform.startswith('win'):
        # Try official CPU index url if default fails
        run([py, '-m', 'pip', 'install', '--quiet', 'torch', '--index-url', 'https://download.pytorch.org/whl/cpu'])


def ensure_dependencies() -> None:
    ensure_torch()
    for mod, spec in REQUIRED.items():
        ensure_package(mod, spec)


def venv_python() -> str | None:
    root = os.getcwd()
    if sys.platform.startswith('win'):
        cand = os.path.join(root, '.venv', 'Scripts', 'python.exe')
    else:
        cand = os.path.join(root, '.venv', 'bin', 'python')
    return cand if os.path.exists(cand) else None


def is_port_open(port: int) -> bool:
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.2)
        return s.connect_ex(('127.0.0.1', port)) == 0


def launch_app(port: int = 8050) -> None:
    env = os.environ.copy()
    env['PORT'] = str(port)
    log_path = os.path.join('/tmp' if os.name != 'nt' else os.getenv('TEMP', '.'), 'wafer_tool.log')
    py = venv_python() or sys.executable
    if is_port_open(port):
        print(f"[INFO] Port {port} already in use; not starting a new server.")
        url = f"http://127.0.0.1:{port}"
        try:
            if sys.platform == 'darwin':
                subprocess.Popen(['open', url])
            elif sys.platform.startswith('win'):
                subprocess.Popen(['cmd', '/c', 'start', '', url])
            else:
                subprocess.Popen(['xdg-open', url])
        except Exception:
            pass
        return
    with open(log_path, 'w') as log:
        proc = subprocess.Popen([py, 'app.py'], stdout=log, stderr=log, env=env)
    url = f"http://127.0.0.1:{port}"
    print(f"[INFO] Wafer Tool starting at {url} (logs: {log_path})")
    # Best effort open browser only once (delay to let server bind)
    time.sleep(2)
    try:
        if sys.platform == 'darwin':
            subprocess.Popen(['open', url])
        elif sys.platform.startswith('win'):
            subprocess.Popen(['cmd', '/c', 'start', '', url])
        else:
            subprocess.Popen(['xdg-open', url])
    except Exception:
        pass
    # Keep the launcher alive a bit so double-click users see messages
    time.sleep(1)


if __name__ == '__main__':
    # If embedded venv exists, skip installs and use it directly
    if venv_python() is None:
        try:
            ensure_dependencies()
        except Exception as e:
            print(f"[WARN] Auto-setup encountered an issue: {e}")
    launch_app()

