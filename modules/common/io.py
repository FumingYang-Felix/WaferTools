import os
import json
import shutil


def copy_to(dst_dir: str, files: list[tuple[str, str]]):
    os.makedirs(dst_dir, exist_ok=True)
    out_paths = []
    for src, new_name in files:
        if not src or not os.path.exists(src):
            continue
        dst = os.path.join(dst_dir, new_name)
        shutil.copy2(src, dst)
        out_paths.append(dst)
    return out_paths


def save_meta(dst_dir: str, meta: dict):
    os.makedirs(dst_dir, exist_ok=True)
    path = os.path.join(dst_dir, 'meta.json')
    with open(path, 'w') as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    return path

