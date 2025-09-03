#!/usr/bin/env python3

def extract_mst_only():
    """
    extract sections from mst_complete_fixed.txt
    """
    # read sections from mst_complete_fixed.txt
    with open('mst_raw_no_duplicates.txt', 'r') as f:
        mst_raw_line = f.read().strip()
    mst_raw_sections = set(mst_raw_line.split())
    
    # read fixed order
    with open('mst_complete_fixed.txt', 'r') as f:
        fixed_line = f.read().strip()
    fixed_sections = fixed_line.split()
    
    # extract sections from mst_raw_no_duplicates.txt, keep them in the fixed order
    mst_only_sections = []
    for section in fixed_sections:
        if section in mst_raw_sections:
            mst_only_sections.append(section)
    
    # save to new file
    with open('mst_complete_fixed_mst_only.txt', 'w') as f:
        f.write(' '.join(mst_only_sections))
    
    print(f"extracted {len(mst_only_sections)} sections")
    print("saved to mst_complete_fixed_mst_only.txt")
    
    # show extracted sections
    print("\nextracted sections:")
    for i, section in enumerate(mst_only_sections, 1):
        print(f"{i:2d}. {section}")

if __name__ == "__main__":
    extract_mst_only() 

def extract_mst_only():
    """
    extract sections from mst_complete_fixed.txt
    """
    # read sections from mst_complete_fixed.txt
    with open('mst_raw_no_duplicates.txt', 'r') as f:
        mst_raw_line = f.read().strip()
    mst_raw_sections = set(mst_raw_line.split())
    
    # read fixed order
    with open('mst_complete_fixed.txt', 'r') as f:
        fixed_line = f.read().strip()
    fixed_sections = fixed_line.split()
    
    # extract sections from mst_raw_no_duplicates.txt, keep them in the fixed order
    mst_only_sections = []
    for section in fixed_sections:
        if section in mst_raw_sections:
            mst_only_sections.append(section)
    
    # save to new file
    with open('mst_complete_fixed_mst_only.txt', 'w') as f:
        f.write(' '.join(mst_only_sections))
    
    print(f"extracted {len(mst_only_sections)} sections")
    print("saved to mst_complete_fixed_mst_only.txt")
    
    # show extracted sections
    print("\nextracted sections:")
    for i, section in enumerate(mst_only_sections, 1):
        print(f"{i:2d}. {section}")

if __name__ == "__main__":
    extract_mst_only() 
 
 
 
 
 
 
