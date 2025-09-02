import argparse
import os
import shutil
from pathlib import Path
import cv2

def write_metadata(section_dir: Path, img_name: str, height: int, width: int, resolution: float = 4.0):
    """Create metadata.txt for a section directory."""
    meta_file = section_dir / "metadata.txt"
    with meta_file.open('w') as f:
        # root dir is current section directory
        f.write('{ROOT_DIR}\t.\n')
        f.write(f'{{RESOLUTION}}\t{resolution}\n')
        # x_min y_min x_max y_max
        line = f"{img_name}\t0\t0\t{width}\t{height}\n"
        f.write(line)


def prepare_workdir(src_dir: Path, work_dir: Path, link: bool = True):
    """Populate FEABAS work directory with stitched_sections layout for single-tile images."""
    stitched_base = work_dir / "stitched_sections" / "mip0"
    stitched_base.mkdir(parents=True, exist_ok=True)

    # gather pngs
    imgs = sorted(src_dir.glob('*.png'))
    if len(imgs) == 0:
        raise RuntimeError(f"No PNG images found in {src_dir}")

    order_file = work_dir / 'section_order.txt'
    with order_file.open('w') as fo:
        for idx, img_path in enumerate(imgs):
            sec_name = img_path.stem
            fo.write(f"{idx}\t{sec_name}\n")
            sec_dir = stitched_base / sec_name
            sec_dir.mkdir(parents=True, exist_ok=True)
            dst_img = sec_dir / img_path.name
            if link:
                if not dst_img.exists():
                    os.symlink(img_path.resolve(), dst_img)
            else:
                shutil.copy2(img_path, dst_img)
            # get image shape
            img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                raise RuntimeError(f"Failed to read image {img_path}")
            height, width = img.shape[:2]
            write_metadata(sec_dir, img_path.name, height, width)
    print(f"Prepared {len(imgs)} sections under {stitched_base}")


def write_general_configs(work_dir: Path):
    cfg_dir = work_dir / 'configs'
    cfg_dir.mkdir(parents=True, exist_ok=True)
    general_cfg = cfg_dir / 'general_configs.yaml'
    if general_cfg.exists():
        return
    with general_cfg.open('w') as f:
        f.write('working_directory: ./\n')
        f.write('cpu_budget: null\n')
        f.write('parallel_framework: process\n')
        f.write('full_resolution: 4\n')
        f.write('section_thickness: 30\n')
        f.write('logging_directory: null\n')
        f.write('logfile_level: WARNING\n')
        f.write('console_level: INFO\n')
        f.write('archive_level: INFO\n')
        f.write('tensorstore_timeout: null\n')


def main():
    parser = argparse.ArgumentParser(description="Prepare FEABAS work directory for single-tile PNG dataset")
    parser.add_argument('src', help='Directory with PNG images')
    parser.add_argument('workdir', help='Target FEABAS working directory (will be created)')
    parser.add_argument('--copy', action='store_true', help='Copy images instead of symlinks')
    args = parser.parse_args()

    src_dir = Path(args.src).expanduser().resolve()
    work_dir = Path(args.workdir).expanduser().resolve()
    work_dir.mkdir(parents=True, exist_ok=True)

    write_general_configs(work_dir)
    prepare_workdir(src_dir, work_dir, link=not args.copy)

    print('Done. Now cd to the workdir and run:\n'\
          '  python -m feabas.scripts.thumbnail_main --mode downsample')

if __name__ == '__main__':
    main() 
import os
import shutil
from pathlib import Path
import cv2

def write_metadata(section_dir: Path, img_name: str, height: int, width: int, resolution: float = 4.0):
    """Create metadata.txt for a section directory."""
    meta_file = section_dir / "metadata.txt"
    with meta_file.open('w') as f:
        # root dir is current section directory
        f.write('{ROOT_DIR}\t.\n')
        f.write(f'{{RESOLUTION}}\t{resolution}\n')
        # x_min y_min x_max y_max
        line = f"{img_name}\t0\t0\t{width}\t{height}\n"
        f.write(line)


def prepare_workdir(src_dir: Path, work_dir: Path, link: bool = True):
    """Populate FEABAS work directory with stitched_sections layout for single-tile images."""
    stitched_base = work_dir / "stitched_sections" / "mip0"
    stitched_base.mkdir(parents=True, exist_ok=True)

    # gather pngs
    imgs = sorted(src_dir.glob('*.png'))
    if len(imgs) == 0:
        raise RuntimeError(f"No PNG images found in {src_dir}")

    order_file = work_dir / 'section_order.txt'
    with order_file.open('w') as fo:
        for idx, img_path in enumerate(imgs):
            sec_name = img_path.stem
            fo.write(f"{idx}\t{sec_name}\n")
            sec_dir = stitched_base / sec_name
            sec_dir.mkdir(parents=True, exist_ok=True)
            dst_img = sec_dir / img_path.name
            if link:
                if not dst_img.exists():
                    os.symlink(img_path.resolve(), dst_img)
            else:
                shutil.copy2(img_path, dst_img)
            # get image shape
            img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                raise RuntimeError(f"Failed to read image {img_path}")
            height, width = img.shape[:2]
            write_metadata(sec_dir, img_path.name, height, width)
    print(f"Prepared {len(imgs)} sections under {stitched_base}")


def write_general_configs(work_dir: Path):
    cfg_dir = work_dir / 'configs'
    cfg_dir.mkdir(parents=True, exist_ok=True)
    general_cfg = cfg_dir / 'general_configs.yaml'
    if general_cfg.exists():
        return
    with general_cfg.open('w') as f:
        f.write('working_directory: ./\n')
        f.write('cpu_budget: null\n')
        f.write('parallel_framework: process\n')
        f.write('full_resolution: 4\n')
        f.write('section_thickness: 30\n')
        f.write('logging_directory: null\n')
        f.write('logfile_level: WARNING\n')
        f.write('console_level: INFO\n')
        f.write('archive_level: INFO\n')
        f.write('tensorstore_timeout: null\n')


def main():
    parser = argparse.ArgumentParser(description="Prepare FEABAS work directory for single-tile PNG dataset")
    parser.add_argument('src', help='Directory with PNG images')
    parser.add_argument('workdir', help='Target FEABAS working directory (will be created)')
    parser.add_argument('--copy', action='store_true', help='Copy images instead of symlinks')
    args = parser.parse_args()

    src_dir = Path(args.src).expanduser().resolve()
    work_dir = Path(args.workdir).expanduser().resolve()
    work_dir.mkdir(parents=True, exist_ok=True)

    write_general_configs(work_dir)
    prepare_workdir(src_dir, work_dir, link=not args.copy)

    print('Done. Now cd to the workdir and run:\n'\
          '  python -m feabas.scripts.thumbnail_main --mode downsample')

if __name__ == '__main__':
    main() 
 
import os
import shutil
from pathlib import Path
import cv2

def write_metadata(section_dir: Path, img_name: str, height: int, width: int, resolution: float = 4.0):
    """Create metadata.txt for a section directory."""
    meta_file = section_dir / "metadata.txt"
    with meta_file.open('w') as f:
        # root dir is current section directory
        f.write('{ROOT_DIR}\t.\n')
        f.write(f'{{RESOLUTION}}\t{resolution}\n')
        # x_min y_min x_max y_max
        line = f"{img_name}\t0\t0\t{width}\t{height}\n"
        f.write(line)


def prepare_workdir(src_dir: Path, work_dir: Path, link: bool = True):
    """Populate FEABAS work directory with stitched_sections layout for single-tile images."""
    stitched_base = work_dir / "stitched_sections" / "mip0"
    stitched_base.mkdir(parents=True, exist_ok=True)

    # gather pngs
    imgs = sorted(src_dir.glob('*.png'))
    if len(imgs) == 0:
        raise RuntimeError(f"No PNG images found in {src_dir}")

    order_file = work_dir / 'section_order.txt'
    with order_file.open('w') as fo:
        for idx, img_path in enumerate(imgs):
            sec_name = img_path.stem
            fo.write(f"{idx}\t{sec_name}\n")
            sec_dir = stitched_base / sec_name
            sec_dir.mkdir(parents=True, exist_ok=True)
            dst_img = sec_dir / img_path.name
            if link:
                if not dst_img.exists():
                    os.symlink(img_path.resolve(), dst_img)
            else:
                shutil.copy2(img_path, dst_img)
            # get image shape
            img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                raise RuntimeError(f"Failed to read image {img_path}")
            height, width = img.shape[:2]
            write_metadata(sec_dir, img_path.name, height, width)
    print(f"Prepared {len(imgs)} sections under {stitched_base}")


def write_general_configs(work_dir: Path):
    cfg_dir = work_dir / 'configs'
    cfg_dir.mkdir(parents=True, exist_ok=True)
    general_cfg = cfg_dir / 'general_configs.yaml'
    if general_cfg.exists():
        return
    with general_cfg.open('w') as f:
        f.write('working_directory: ./\n')
        f.write('cpu_budget: null\n')
        f.write('parallel_framework: process\n')
        f.write('full_resolution: 4\n')
        f.write('section_thickness: 30\n')
        f.write('logging_directory: null\n')
        f.write('logfile_level: WARNING\n')
        f.write('console_level: INFO\n')
        f.write('archive_level: INFO\n')
        f.write('tensorstore_timeout: null\n')


def main():
    parser = argparse.ArgumentParser(description="Prepare FEABAS work directory for single-tile PNG dataset")
    parser.add_argument('src', help='Directory with PNG images')
    parser.add_argument('workdir', help='Target FEABAS working directory (will be created)')
    parser.add_argument('--copy', action='store_true', help='Copy images instead of symlinks')
    args = parser.parse_args()

    src_dir = Path(args.src).expanduser().resolve()
    work_dir = Path(args.workdir).expanduser().resolve()
    work_dir.mkdir(parents=True, exist_ok=True)

    write_general_configs(work_dir)
    prepare_workdir(src_dir, work_dir, link=not args.copy)

    print('Done. Now cd to the workdir and run:\n'\
          '  python -m feabas.scripts.thumbnail_main --mode downsample')

if __name__ == '__main__':
    main() 
import os
import shutil
from pathlib import Path
import cv2

def write_metadata(section_dir: Path, img_name: str, height: int, width: int, resolution: float = 4.0):
    """Create metadata.txt for a section directory."""
    meta_file = section_dir / "metadata.txt"
    with meta_file.open('w') as f:
        # root dir is current section directory
        f.write('{ROOT_DIR}\t.\n')
        f.write(f'{{RESOLUTION}}\t{resolution}\n')
        # x_min y_min x_max y_max
        line = f"{img_name}\t0\t0\t{width}\t{height}\n"
        f.write(line)


def prepare_workdir(src_dir: Path, work_dir: Path, link: bool = True):
    """Populate FEABAS work directory with stitched_sections layout for single-tile images."""
    stitched_base = work_dir / "stitched_sections" / "mip0"
    stitched_base.mkdir(parents=True, exist_ok=True)

    # gather pngs
    imgs = sorted(src_dir.glob('*.png'))
    if len(imgs) == 0:
        raise RuntimeError(f"No PNG images found in {src_dir}")

    order_file = work_dir / 'section_order.txt'
    with order_file.open('w') as fo:
        for idx, img_path in enumerate(imgs):
            sec_name = img_path.stem
            fo.write(f"{idx}\t{sec_name}\n")
            sec_dir = stitched_base / sec_name
            sec_dir.mkdir(parents=True, exist_ok=True)
            dst_img = sec_dir / img_path.name
            if link:
                if not dst_img.exists():
                    os.symlink(img_path.resolve(), dst_img)
            else:
                shutil.copy2(img_path, dst_img)
            # get image shape
            img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                raise RuntimeError(f"Failed to read image {img_path}")
            height, width = img.shape[:2]
            write_metadata(sec_dir, img_path.name, height, width)
    print(f"Prepared {len(imgs)} sections under {stitched_base}")


def write_general_configs(work_dir: Path):
    cfg_dir = work_dir / 'configs'
    cfg_dir.mkdir(parents=True, exist_ok=True)
    general_cfg = cfg_dir / 'general_configs.yaml'
    if general_cfg.exists():
        return
    with general_cfg.open('w') as f:
        f.write('working_directory: ./\n')
        f.write('cpu_budget: null\n')
        f.write('parallel_framework: process\n')
        f.write('full_resolution: 4\n')
        f.write('section_thickness: 30\n')
        f.write('logging_directory: null\n')
        f.write('logfile_level: WARNING\n')
        f.write('console_level: INFO\n')
        f.write('archive_level: INFO\n')
        f.write('tensorstore_timeout: null\n')


def main():
    parser = argparse.ArgumentParser(description="Prepare FEABAS work directory for single-tile PNG dataset")
    parser.add_argument('src', help='Directory with PNG images')
    parser.add_argument('workdir', help='Target FEABAS working directory (will be created)')
    parser.add_argument('--copy', action='store_true', help='Copy images instead of symlinks')
    args = parser.parse_args()

    src_dir = Path(args.src).expanduser().resolve()
    work_dir = Path(args.workdir).expanduser().resolve()
    work_dir.mkdir(parents=True, exist_ok=True)

    write_general_configs(work_dir)
    prepare_workdir(src_dir, work_dir, link=not args.copy)

    print('Done. Now cd to the workdir and run:\n'\
          '  python -m feabas.scripts.thumbnail_main --mode downsample')

if __name__ == '__main__':
    main() 
 
 
 
 
 
 
 
 
 
 
 
 
 