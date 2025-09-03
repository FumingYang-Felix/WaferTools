#!/usr/bin/env python3
"""
update_rigid_chains.py - update rigid chains information

features:
1. read previous rigid chains results
2. apply connection relations, merge chains that can be connected
3. generate updated rigid components list
4. statistics new components information

输出:
- updated rigid components
- list of sections for each component
- statistics of components
"""

import re
from collections import defaultdict
from typing import List, Set, Dict, Tuple

def parse_rigid_chains(filename: str) -> List[List[str]]:
    """parse rigid chains file, return all chains"""
    chains = []
    
    with open(filename, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    for line in lines:
        # match format: chain X: section_A -> section_B -> section_C (length: N)
        match = re.match(r'chain \d+:\s+(.+) \(length:', line)
        if match:
            chain_str = match.group(1)
            chain = [s.strip() for s in chain_str.split(' -> ')]
            chains.append(chain)
    
    return chains

def get_connected_components() -> List[List[int]]:
    """get connected components definition"""
    # based on previous analysis results, define connected components
    components = [
        [0],  # chain 1: section_1_r01_c01 -> section_13_r01_c01
        [1, 14],  # chain 2,15: section_39_r01_c01 -> ... -> section_3_r01_c01 + section_19_r01_c01 -> section_34_r01_c01
        [2],  # chain 3: section_4_r01_c01 -> section_59_r01_c01
        [3],  # chain 4: section_5_r01_c01 -> section_99_r01_c01
        [4, 32],  # chain 5,33: section_35_r01_c01 -> section_6_r01_c01 + section_35_r01_c01 -> section_6_r01_c01 -> section_57_r01_c01 -> section_61_r01_c01
        [5],  # chain 6: section_104_r01_c01 -> section_7_r01_c01 -> section_11_r01_c01
        [7],  # chain 8: section_9_r01_c01 -> section_64_r01_c01
        [8, 35, 20],  # chain 9,36,21: largest connected component
        [9],  # chain 10: section_12_r01_c01 -> section_89_r01_c01
        [10],  # chain 11: section_14_r01_c01 -> section_101_r01_c01
        [11],  # chain 12: section_29_r01_c01 -> section_16_r01_c01 -> section_76_r01_c01
        [12],  # chain 13: section_90_r01_c01 -> section_31_r01_c01 -> section_17_r01_c01 -> section_38_r01_c01
        [13],  # chain 14: section_92_r01_c01 -> section_18_r01_c01
        [15],  # chain 16: section_21_r01_c01 -> section_78_r01_c01
        [16],  # chain 17: section_22_r01_c01 -> section_26_r01_c
        [17],  # chain 18: section_80_r01_c01 -> section_23_r01_c01 -> section_43_r01_c01
        [18],  # chain 19: section_24_r01_c01 -> section_40_r01_c01
        [19],  # chain 20: section_25_r01_c01 -> section_74_r01_c01
        [21],  # chain 22: section_72_r01_c01 -> section_32_r01_c01 -> section_97_r01_c01
        [22],  # chain 23: section_2_r01_c01 -> section_36_r01_c01 -> section_102_r01_c01
        [23],  # chain 24: section_42_r01_c01 -> section_41_r01_c01
        [24],  # chain 25: section_45_r01_c01 -> section_105_r01_c01
        [25],  # chain 26: section_46_r01_c01 -> section_69_r01_c01
        [26],  # chain 27: section_47_r01_c01 -> section_96_r01_c01
        [27],  # chain 28: section_48_r01_c01 -> section_71_r01_c01
        [28],  # chain 29: section_49_r01_c01 -> section_75_r01_c01
        [29],  # chain 30: section_50_r01_c01 -> section_70_r01_c01
        [30],  # chain 31: section_55_r01_c01 -> section_63_r01_c01
        [31],  # chain 32: section_56_r01_c01 -> section_88_r01_c01
        [33],  # chain 34: section_58_r01_c01 -> section_85_r01_c01
        [34],  # chain 35: section_51_r01_c01 -> section_60_r01_c01 -> section_100_r01_c01
        [36],  # chain 37: section_67_r01_c01 -> section_86_r01_c01
        [37],  # chain 38: section_82_r01_c01 -> section_103_r01_c01
        [38],  # chain 39: section_87_r01_c01 -> section_91_r01_c01
        [39],  # chain 40: section_93_r01_c01 -> section_94_r01_c01
    ]
    return components

def merge_chains_in_component(component_chains: List[int], chains: List[List[str]]) -> List[str]:
    """merge chains in component, return the merged chain"""
    if len(component_chains) == 1:
        return chains[component_chains[0]]
    
    # for multiple chains in a component, find the correct connection order
    if component_chains == [1, 14]:  # component 2: chain 2,15
        # chain 2: section_39_r01_c01 -> section_15_r01_c01 -> section_52_r01_c01 -> section_3_r01_c01
        # chain 15: section_19_r01_c01 -> section_34_r01_c01
        # connection point: section_3_r01_c01 -> section_34_r01_c01
        return ['section_39_r01_c01', 'section_15_r01_c01', 'section_52_r01_c01', 'section_3_r01_c01', 'section_34_r01_c01']
    
    elif component_chains == [4, 32]:  # component 5: chain 5,33
        # chain 5: section_35_r01_c01 -> section_6_r01_c01
        # chain 33: section_35_r01_c01 -> section_6_r01_c01 -> section_57_r01_c01 -> section_61_r01_c01
        # chain 33 contains chain 5, so return chain 33
        return chains[32]
    
    elif component_chains == [8, 35, 20]:  # component 9: chain 9,36,21
        # chain 9: section_27_r01_c01 -> section_10_r01_c01
        # chain 36: section_27_r01_c01 -> section_10_r01_c01 -> section_66_r01_c01
        # chain 21: section_28_r01_c01 -> section_73_r01_c01
        # connection point: section_66_r01_c01 -> section_73_r01_c01
        return ['section_27_r01_c01', 'section_10_r01_c01', 'section_66_r01_c01', 'section_73_r01_c01']
    
    else:
        # other cases, return the first chain
        return chains[component_chains[0]]

def create_updated_components(chains: List[List[str]]) -> List[List[str]]:
    """create updated rigid components"""
    components = get_connected_components()
    updated_components = []
    
    print("updating rigid components...")
    print("=" * 60)
    
    for i, component_chains in enumerate(components):
        merged_chain = merge_chains_in_component(component_chains, chains)
        updated_components.append(merged_chain)
        
        print(f"component {i+1}: contains original chains {[j+1 for j in component_chains]} (total {len(component_chains)} chains)")
        print(f"  merged: {' -> '.join(merged_chain)} (length: {len(merged_chain)})")
        print()
    
    return updated_components

def analyze_updated_components(components: List[List[str]]) -> None:
    """analyze updated components statistics"""
    print("=" * 60)
    print("updated rigid components analysis")
    print("=" * 60)
    
    # statistics
    component_lengths = [len(comp) for comp in components]
    covered_sections = set()
    for comp in components:
        covered_sections.update(comp)
    
    print(f"total components: {len(components)}")
    print(f"covered sections: {len(covered_sections)}")
    print(f"total sections: 100")
    print(f"coverage: {len(covered_sections)/100*100:.1f}%")
    
    if component_lengths:
        print(f"component length statistics:")
        print(f"  longest component: {max(component_lengths)} sections")
        print(f"  shortest component: {min(component_lengths)} sections")
        print(f"  average length: {sum(component_lengths)/len(component_lengths):.1f} sections")
        
        # group by length
        length_groups = defaultdict(int)
        for length in component_lengths:
            length_groups[length] += 1
        
        print(f"  length distribution:")
        for length in sorted(length_groups.keys()):
            print(f"    {length} sections: {length_groups[length]} components")
    
    # show all components
    print(f"\nall components details:")
    for i, comp in enumerate(components, 1):
        print(f"component {i}: {' -> '.join(comp)} (length: {len(comp)})")
    
    # show uncovered sections
    all_sections = set()
    for i in range(1, 107):  # assume sections from 1 to 106
        all_sections.add(f"section_{i}_r01_c01")
    
    uncovered = all_sections - covered_sections
    if uncovered:
        print(f"\nuncovered sections ({len(uncovered)}):")
        for section in sorted(uncovered):
            print(f"  {section}")

def main():
    # parse rigid chains
    print("parsing rigid chains...")
    chains = parse_rigid_chains('rigid_chains_result.txt')
    print(f"parsing completed, total {len(chains)} original chains")
    
    # create updated components
    updated_components = create_updated_components(chains)
    
    # analyze results
    analyze_updated_components(updated_components)
    
    # save results to file
    with open('updated_rigid_components.txt', 'w', encoding='utf-8') as f:
        f.write("updated rigid components\n")
        f.write("=" * 60 + "\n\n")
        
        f.write(f"total components: {len(updated_components)}\n")
        covered_sections = set()
        for comp in updated_components:
            covered_sections.update(comp)
        f.write(f"covered sections: {len(covered_sections)}\n")
        f.write(f"total sections: 100\n")
        f.write(f"coverage: {len(covered_sections)/100*100:.1f}%\n\n")
        
        f.write("all components details:\n")
        for i, comp in enumerate(updated_components, 1):
            f.write(f"component {i}: {' -> '.join(comp)} (length: {len(comp)})\n")
        
        # show uncovered sections
        all_sections = set()
        for i in range(1, 107):
            all_sections.add(f"section_{i}_r01_c01")
        
        uncovered = all_sections - covered_sections
        if uncovered:
            f.write(f"\nuncovered sections ({len(uncovered)}):\n")
            for section in sorted(uncovered):
                f.write(f"  {section}\n")
    
    print(f"\nsaved to updated_rigid_components.txt")

if __name__ == "__main__":
    main() 
