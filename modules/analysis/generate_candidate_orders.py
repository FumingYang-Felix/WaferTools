#!/usr/bin/env python3
"""
generate_candidate_orders.py - generate two candidate orders, forward and reverse
"""

def generate_candidate_orders():
    """generate candidate orders"""
    
    # extract paths from analysis results
    forward_path = [
        "section_14_r01_c01", "section_42_r01_c01", "section_41_r01_c01", "section_78_r01_c01",
        "section_21_r01_c01", "section_47_r01_c01", "section_96_r01_c01", "section_84_r01_c01",
        "section_93_r01_c01", "section_94_r01_c01", "section_22_r01_c01", "section_26_r01_c01",
        "section_59_r01_c01", "section_4_r01_c01", "section_27_r01_c01", "section_10_r01_c01",
        "section_66_r01_c01", "section_73_r01_c01", "section_28_r01_c01", "section_48_r01_c01",
        "section_71_r01_c01", "section_2_r01_c01", "section_36_r01_c01", "section_102_r01_c01",
        "section_103_r01_c01", "section_82_r01_c01", "section_62_r01_c01", "section_97_r01_c01",
        "section_32_r01_c01", "section_72_r01_c01", "section_24_r01_c01", "section_40_r01_c01",
        "section_65_r01_c01", "section_43_r01_c01", "section_23_r01_c01", "section_80_r01_c01",
        "section_9_r01_c01", "section_64_r01_c01", "section_25_r01_c01", "section_74_r01_c01",
        "section_30_r01_c01", "section_29_r01_c01", "section_16_r01_c01", "section_76_r01_c01",
        "section_56_r01_c01", "section_88_r01_c01", "section_37_r01_c01", "section_12_r01_c01",
        "section_89_r01_c01", "section_35_r01_c01", "section_6_r01_c01", "section_57_r01_c01",
        "section_61_r01_c01", "section_18_r01_c01", "section_92_r01_c01", "section_13_r01_c01",
        "section_1_r01_c01", "section_63_r01_c01", "section_55_r01_c01", "section_44_r01_c01",
        "section_85_r01_c01", "section_58_r01_c01", "section_86_r01_c01", "section_67_r01_c01",
        "section_5_r01_c01", "section_99_r01_c01", "section_11_r01_c01", "section_7_r01_c01",
        "section_104_r01_c01", "section_69_r01_c01", "section_46_r01_c01", "section_53_r01_c01",
        "section_91_r01_c01", "section_87_r01_c01", "section_68_r01_c01", "section_100_r01_c01",
        "section_60_r01_c01", "section_51_r01_c01", "section_45_r01_c01", "section_105_r01_c01",
        "section_106_r01_c01"
    ]
    
    # generate reverse path
    reverse_path = forward_path[::-1]
    
    # save forward order
    with open('candidate_order_forward.txt', 'w') as f:
        for section in forward_path:
            f.write(section + '\n')
    
    # save reverse order
    with open('candidate_order_reverse.txt', 'w') as f:
        for section in reverse_path:
            f.write(section + '\n')
    
    # save complete information
    with open('candidate_orders_complete.txt', 'w') as f:
        f.write("candidate orders generation results\n")
        f.write("=" * 50 + "\n\n")
        f.write("note: since best pair only provides connection information, not direction information,\n")
        f.write("we generated two candidate orders, and need to determine the correct direction based on other information.\n\n")
        
        f.write("candidate order 1 (forward):\n")
        f.write("-" * 30 + "\n")
        for i, section in enumerate(forward_path, 1):
            f.write(f"{i:3d}. {section}\n")
        
        f.write(f"\ncandidate order 2 (reverse):\n")
        f.write("-" * 30 + "\n")
        for i, section in enumerate(reverse_path, 1):
            f.write(f"{i:3d}. {section}\n")
        
        f.write(f"\nstatistics:\n")
        f.write(f"  path length: {len(forward_path)} sections\n")
        f.write(f"  covered sections: {len(forward_path)}/100\n")
        f.write(f"  missing sections: {100 - len(forward_path)} sections\n")
        
        # find missing sections
        all_sections = set()
        for i in range(1, 107):
            all_sections.add(f"section_{i}_r01_c01")
        
        missing_sections = all_sections - set(forward_path)
        if missing_sections:
            f.write(f"  missing sections: {', '.join(sorted(missing_sections))}\n")
    
    print("candidate orders generated:")
    print("  - candidate_order_forward.txt (forward)")
    print("  - candidate_order_reverse.txt (reverse)")
    print("  - candidate_orders_complete.txt (complete information)")
    print(f"\npath length: {len(forward_path)} sections")
    print(f"coverage: {len(forward_path)}/100 = {len(forward_path)/100*100:.1f}%")

if __name__ == "__main__":
    generate_candidate_orders() 
"""
generate_candidate_orders.py - generate two candidate orders, forward and reverse
"""

def generate_candidate_orders():
    """generate candidate orders"""
    
    # extract paths from analysis results
    forward_path = [
        "section_14_r01_c01", "section_42_r01_c01", "section_41_r01_c01", "section_78_r01_c01",
        "section_21_r01_c01", "section_47_r01_c01", "section_96_r01_c01", "section_84_r01_c01",
        "section_93_r01_c01", "section_94_r01_c01", "section_22_r01_c01", "section_26_r01_c01",
        "section_59_r01_c01", "section_4_r01_c01", "section_27_r01_c01", "section_10_r01_c01",
        "section_66_r01_c01", "section_73_r01_c01", "section_28_r01_c01", "section_48_r01_c01",
        "section_71_r01_c01", "section_2_r01_c01", "section_36_r01_c01", "section_102_r01_c01",
        "section_103_r01_c01", "section_82_r01_c01", "section_62_r01_c01", "section_97_r01_c01",
        "section_32_r01_c01", "section_72_r01_c01", "section_24_r01_c01", "section_40_r01_c01",
        "section_65_r01_c01", "section_43_r01_c01", "section_23_r01_c01", "section_80_r01_c01",
        "section_9_r01_c01", "section_64_r01_c01", "section_25_r01_c01", "section_74_r01_c01",
        "section_30_r01_c01", "section_29_r01_c01", "section_16_r01_c01", "section_76_r01_c01",
        "section_56_r01_c01", "section_88_r01_c01", "section_37_r01_c01", "section_12_r01_c01",
        "section_89_r01_c01", "section_35_r01_c01", "section_6_r01_c01", "section_57_r01_c01",
        "section_61_r01_c01", "section_18_r01_c01", "section_92_r01_c01", "section_13_r01_c01",
        "section_1_r01_c01", "section_63_r01_c01", "section_55_r01_c01", "section_44_r01_c01",
        "section_85_r01_c01", "section_58_r01_c01", "section_86_r01_c01", "section_67_r01_c01",
        "section_5_r01_c01", "section_99_r01_c01", "section_11_r01_c01", "section_7_r01_c01",
        "section_104_r01_c01", "section_69_r01_c01", "section_46_r01_c01", "section_53_r01_c01",
        "section_91_r01_c01", "section_87_r01_c01", "section_68_r01_c01", "section_100_r01_c01",
        "section_60_r01_c01", "section_51_r01_c01", "section_45_r01_c01", "section_105_r01_c01",
        "section_106_r01_c01"
    ]
    
    # generate reverse path
    reverse_path = forward_path[::-1]
    
    # save forward order
    with open('candidate_order_forward.txt', 'w') as f:
        for section in forward_path:
            f.write(section + '\n')
    
    # save reverse order
    with open('candidate_order_reverse.txt', 'w') as f:
        for section in reverse_path:
            f.write(section + '\n')
    
    # save complete information
    with open('candidate_orders_complete.txt', 'w') as f:
        f.write("candidate orders generation results\n")
        f.write("=" * 50 + "\n\n")
        f.write("note: since best pair only provides connection information, not direction information,\n")
        f.write("we generated two candidate orders, and need to determine the correct direction based on other information.\n\n")
        
        f.write("candidate order 1 (forward):\n")
        f.write("-" * 30 + "\n")
        for i, section in enumerate(forward_path, 1):
            f.write(f"{i:3d}. {section}\n")
        
        f.write(f"\ncandidate order 2 (reverse):\n")
        f.write("-" * 30 + "\n")
        for i, section in enumerate(reverse_path, 1):
            f.write(f"{i:3d}. {section}\n")
        
        f.write(f"\nstatistics:\n")
        f.write(f"  path length: {len(forward_path)} sections\n")
        f.write(f"  covered sections: {len(forward_path)}/100\n")
        f.write(f"  missing sections: {100 - len(forward_path)} sections\n")
        
        # find missing sections
        all_sections = set()
        for i in range(1, 107):
            all_sections.add(f"section_{i}_r01_c01")
        
        missing_sections = all_sections - set(forward_path)
        if missing_sections:
            f.write(f"  missing sections: {', '.join(sorted(missing_sections))}\n")
    
    print("candidate orders generated:")
    print("  - candidate_order_forward.txt (forward)")
    print("  - candidate_order_reverse.txt (reverse)")
    print("  - candidate_orders_complete.txt (complete information)")
    print(f"\npath length: {len(forward_path)} sections")
    print(f"coverage: {len(forward_path)}/100 = {len(forward_path)/100*100:.1f}%")

if __name__ == "__main__":
    generate_candidate_orders() 
 
 
 
 
 
 
