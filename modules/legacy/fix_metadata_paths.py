import pathlib, sys
from feabas import storage

root = pathlib.Path('stitched_sections/mip0')
for meta in root.glob('*/metadata.txt'):
    sec_dir = meta.parent.resolve()
    lines = storage.load_text(str(meta)).splitlines()
    if not lines:
        continue
    if lines[0].startswith('{ROOT_DIR}'):
        parts = lines[0].split('\t')
        if len(parts)>=2 and parts[1].strip()=='.':
            parts[1] = str(sec_dir)
            lines[0] = '\t'.join(parts)
            storage.save_text(str(meta), '\n'.join(lines)+'\n')
            print('fixed', meta)
print('done') 
 
 
 
 
 
 
 
 
 
 
 
 
 