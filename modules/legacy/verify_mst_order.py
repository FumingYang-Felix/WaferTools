#!/usr/bin/env python3

def verify_mst_order():
    """
    verify the consistency between MST raw results and corrected order
    """
    # read MST raw results
    with open('mst_raw_no_duplicates.txt', 'r') as f:
        mst_raw_line = f.read().strip()
    mst_raw_sections = mst_raw_line.split()
    
    # read corrected complete order
    with open('mst_complete_fixed.txt', 'r') as f:
        fixed_line = f.read().strip()
    fixed_sections = fixed_line.split()
    
    print(f"MST raw results length: {len(mst_raw_sections)}")
    print(f"corrected complete order length: {len(fixed_sections)}")
    
    # extract sections from corrected order, keeping the order
    extracted_sections = []
    for section in fixed_sections:
        if section in mst_raw_sections:
            extracted_sections.append(section)
    
    print(f"extracted sections: {len(extracted_sections)}")
    
    # compare two orders
    if extracted_sections == mst_raw_sections:
        print("✓ MST raw results and corrected order of MST part are completely consistent!")
        print("✓ the main tree is fine, the problem is in the insertion process")
    else:
        print("✗ found inconsistency!")
        print("MST raw results:", mst_raw_sections)
        print("extracted results:", extracted_sections)
        
        # find the first position of inconsistency
        for i, (orig, extr) in enumerate(zip(mst_raw_sections, extracted_sections)):
            if orig != extr:
                print(f"first inconsistency position {i+1}: {orig} vs {extr}")
                break
    
    # show inserted sections
    inserted_sections = [s for s in fixed_sections if s not in mst_raw_sections]
    print(f"\ninserted sections: {len(inserted_sections)}")
    print("inserted sections:", inserted_sections)

if __name__ == "__main__":
    verify_mst_order() 

def verify_mst_order():
    """
    verify the consistency between MST raw results and corrected order
    """
    # read MST raw results
    with open('mst_raw_no_duplicates.txt', 'r') as f:
        mst_raw_line = f.read().strip()
    mst_raw_sections = mst_raw_line.split()
    
    # read corrected complete order
    with open('mst_complete_fixed.txt', 'r') as f:
        fixed_line = f.read().strip()
    fixed_sections = fixed_line.split()
    
    print(f"MST raw results length: {len(mst_raw_sections)}")
    print(f"corrected complete order length: {len(fixed_sections)}")
    
    # extract sections from corrected order, keeping the order
    extracted_sections = []
    for section in fixed_sections:
        if section in mst_raw_sections:
            extracted_sections.append(section)
    
    print(f"extracted sections: {len(extracted_sections)}")
    
    # compare two orders
    if extracted_sections == mst_raw_sections:
        print("✓ MST raw results and corrected order of MST part are completely consistent!")
        print("✓ the main tree is fine, the problem is in the insertion process")
    else:
        print("✗ found inconsistency!")
        print("MST raw results:", mst_raw_sections)
        print("extracted results:", extracted_sections)
        
        # find the first position of inconsistency
        for i, (orig, extr) in enumerate(zip(mst_raw_sections, extracted_sections)):
            if orig != extr:
                print(f"first inconsistency position {i+1}: {orig} vs {extr}")
                break
    
    # show inserted sections
    inserted_sections = [s for s in fixed_sections if s not in mst_raw_sections]
    print(f"\ninserted sections: {len(inserted_sections)}")
    print("inserted sections:", inserted_sections)

if __name__ == "__main__":
    verify_mst_order() 
 
 
 
 
 
 
