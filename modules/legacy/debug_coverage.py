#!/usr/bin/env python3
"""
debug_coverage.py - debug coverage problem
"""

def check_original_coverage():
    """check coverage of original chains"""
    print("check coverage of original chains...")
    
    # original chains' sections
    original_sections = set()
    
    # extract all sections from rigid_chains_result.txt
    chains = [
        ['section_1_r01_c01', 'section_13_r01_c01'],
        ['section_39_r01_c01', 'section_15_r01_c01', 'section_52_r01_c01', 'section_3_r01_c01'],
        ['section_4_r01_c01', 'section_59_r01_c01'],
        ['section_5_r01_c01', 'section_99_r01_c01'],
        ['section_35_r01_c01', 'section_6_r01_c01'],
        ['section_104_r01_c01', 'section_7_r01_c01', 'section_11_r01_c01'],
        ['section_8_r01_c01', 'section_77_r01_c01'],
        ['section_9_r01_c01', 'section_64_r01_c01'],
        ['section_27_r01_c01', 'section_10_r01_c01'],
        ['section_12_r01_c01', 'section_89_r01_c01'],
        ['section_14_r01_c01', 'section_101_r01_c01'],
        ['section_29_r01_c01', 'section_16_r01_c01', 'section_76_r01_c01'],
        ['section_90_r01_c01', 'section_31_r01_c01', 'section_17_r01_c01', 'section_38_r01_c01'],
        ['section_92_r01_c01', 'section_18_r01_c01'],
        ['section_19_r01_c01', 'section_34_r01_c01'],
        ['section_21_r01_c01', 'section_78_r01_c01'],
        ['section_22_r01_c01', 'section_26_r01_c01'],
        ['section_80_r01_c01', 'section_23_r01_c01', 'section_43_r01_c01'],
        ['section_24_r01_c01', 'section_40_r01_c01'],
        ['section_25_r01_c01', 'section_74_r01_c01'],
        ['section_28_r01_c01', 'section_73_r01_c01'],
        ['section_72_r01_c01', 'section_32_r01_c01', 'section_97_r01_c01'],
        ['section_2_r01_c01', 'section_36_r01_c01', 'section_102_r01_c01'],
        ['section_42_r01_c01', 'section_41_r01_c01'],
        ['section_45_r01_c01', 'section_105_r01_c01'],
        ['section_46_r01_c01', 'section_69_r01_c01'],
        ['section_47_r01_c01', 'section_96_r01_c01'],
        ['section_48_r01_c01', 'section_71_r01_c01'],
        ['section_49_r01_c01', 'section_75_r01_c01'],
        ['section_50_r01_c01', 'section_70_r01_c01'],
        ['section_55_r01_c01', 'section_63_r01_c01'],
        ['section_56_r01_c01', 'section_88_r01_c01'],
        ['section_35_r01_c01', 'section_6_r01_c01', 'section_57_r01_c01', 'section_61_r01_c01'],
        ['section_58_r01_c01', 'section_85_r01_c01'],
        ['section_51_r01_c01', 'section_60_r01_c01', 'section_100_r01_c01'],
        ['section_27_r01_c01', 'section_10_r01_c01', 'section_66_r01_c01'],
        ['section_67_r01_c01', 'section_86_r01_c01'],
        ['section_82_r01_c01', 'section_103_r01_c01'],
        ['section_87_r01_c01', 'section_91_r01_c01'],
        ['section_93_r01_c01', 'section_94_r01_c01'],
    ]
    
    for chain in chains:
        original_sections.update(chain)
    
    print(f"original chains cover {len(original_sections)} sections")
    print(f"original chains' sections: {sorted(original_sections)}")
    
    return original_sections

def check_updated_coverage():
    """check coverage of updated components"""
    print("\ncheck coverage of updated components...")
    
    # updated components' sections
    updated_sections = set()
    
    # updated components
    components = [
        ['section_1_r01_c01', 'section_13_r01_c01'],
        ['section_39_r01_c01', 'section_15_r01_c01', 'section_52_r01_c01', 'section_3_r01_c01', 'section_34_r01_c01'],
        ['section_4_r01_c01', 'section_59_r01_c01'],
        ['section_5_r01_c01', 'section_99_r01_c01'],
        ['section_35_r01_c01', 'section_6_r01_c01', 'section_57_r01_c01', 'section_61_r01_c01'],
        ['section_104_r01_c01', 'section_7_r01_c01', 'section_11_r01_c01'],
        ['section_8_r01_c01', 'section_77_r01_c01'],
        ['section_9_r01_c01', 'section_64_r01_c01'],
        ['section_27_r01_c01', 'section_10_r01_c01', 'section_66_r01_c01', 'section_73_r01_c01'],
        ['section_12_r01_c01', 'section_89_r01_c01'],
        ['section_14_r01_c01', 'section_101_r01_c01'],
        ['section_29_r01_c01', 'section_16_r01_c01', 'section_76_r01_c01'],
        ['section_90_r01_c01', 'section_31_r01_c01', 'section_17_r01_c01', 'section_38_r01_c01'],
        ['section_92_r01_c01', 'section_18_r01_c01'],
        ['section_21_r01_c01', 'section_78_r01_c01'],
        ['section_22_r01_c01', 'section_26_r01_c01'],
        ['section_80_r01_c01', 'section_23_r01_c01', 'section_43_r01_c01'],
        ['section_24_r01_c01', 'section_40_r01_c01'],
        ['section_25_r01_c01', 'section_74_r01_c01'],
        ['section_72_r01_c01', 'section_32_r01_c01', 'section_97_r01_c01'],
        ['section_2_r01_c01', 'section_36_r01_c01', 'section_102_r01_c01'],
        ['section_42_r01_c01', 'section_41_r01_c01'],
        ['section_45_r01_c01', 'section_105_r01_c01'],
        ['section_46_r01_c01', 'section_69_r01_c01'],
        ['section_47_r01_c01', 'section_96_r01_c01'],
        ['section_48_r01_c01', 'section_71_r01_c01'],
        ['section_49_r01_c01', 'section_75_r01_c01'],
        ['section_50_r01_c01', 'section_70_r01_c01'],
        ['section_55_r01_c01', 'section_63_r01_c01'],
        ['section_56_r01_c01', 'section_88_r01_c01'],
        ['section_58_r01_c01', 'section_85_r01_c01'],
        ['section_51_r01_c01', 'section_60_r01_c01', 'section_100_r01_c01'],
        ['section_67_r01_c01', 'section_86_r01_c01'],
        ['section_82_r01_c01', 'section_103_r01_c01'],
        ['section_87_r01_c01', 'section_91_r01_c01'],
        ['section_93_r01_c01', 'section_94_r01_c01'],
    ]
    
    for comp in components:
        updated_sections.update(comp)
    
    print(f"updated components cover {len(updated_sections)} sections")
    print(f"updated components' sections: {sorted(updated_sections)}")
    
    return updated_sections

def compare_coverage():
    """compare coverage"""
    original = check_original_coverage()
    updated = check_updated_coverage()
    
    print(f"\ncompare coverage:")
    print(f"original chains cover {len(original)} sections")
    print(f"updated components cover {len(updated)} sections")
    
    missing = original - updated
    if missing:
        print(f"missing sections ({len(missing)}): {sorted(missing)}")
    else:
        print("no missing sections")
    
    gained = updated - original
    if gained:
        print(f"new sections ({len(gained)}): {sorted(gained)}")
    else:
        print("no new sections")

if __name__ == "__main__":
    compare_coverage() 
