#!/usr/bin/env python3
"""
fix_rigid_components.py - fix rigid components merging logic
"""

def load_original_chains():
    """load original chains"""
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
    return chains

def load_chain_connections():
    """load chain connections"""
    # load chain connections from chain_connections_result.txt
    connections = [
        (2, 15),   # chain 2 and chain 15 connected: section_3_r01_c01 -> section_34_r01_c01
        (5, 33),   # chain 5 and chain 33 connected: section_6_r01_c01 -> section_57_r01_c01
        (9, 36),   # chain 9 and chain 36 connected: section_10_r01_c01 -> section_66_r01_c01
        (14, 33),  # chain 14 and chain 33 connected: section_18_r01_c01 -> section_61_r01_c01
        (24, 16),  # chain 24 and chain 16 connected: section_41_r01_c01 -> section_78_r01_c01
        (33, 5),   # chain 33 and chain 5 connected: section_35_r01_c01 -> section_6_r01_c01
        (36, 9),   # chain 36 and chain 9 connected: section_27_r01_c01 -> section_10_r01_c01
        (36, 21),  # chain 36 and chain 21 connected: section_66_r01_c01 -> section_73_r01_c01
    ]
    return connections

def find_connected_components(chains, connections):
    """find connected components"""
    # build graph
    graph = {}
    for i in range(len(chains)):
        graph[i] = []
    
    for conn in connections:
        if conn[0]-1 < len(chains) and conn[1]-1 < len(chains):
            graph[conn[0]-1].append(conn[1]-1)
            graph[conn[1]-1].append(conn[0]-1)
    
    # DFS find connected components
    visited = set()
    components = []
    
    def dfs(node, component):
        visited.add(node)
        component.append(node)
        for neighbor in graph[node]:
            if neighbor not in visited:
                dfs(neighbor, component)
    
    for node in range(len(chains)):
        if node not in visited:
            component = []
            dfs(node, component)
            components.append(component)
    
    return components

def merge_chains_in_component(chains, component_indices):
    """merge chains in component, keep all original sections"""
    if len(component_indices) == 1:
        return chains[component_indices[0]]
    
    # collect all sections
    all_sections = set()
    for idx in component_indices:
        all_sections.update(chains[idx])
    
    # different merging strategies based on component type
    if len(component_indices) == 2:
        # two chains, need to merge based on connection points
        chain1_idx, chain2_idx = component_indices
        chain1 = chains[chain1_idx]
        chain2 = chains[chain2_idx]
        
        # check if there are overlapping sections
        overlap = set(chain1) & set(chain2)
        if overlap:
            # there are overlapping sections, need to merge and remove duplicates
            result = list(chain1)
            for section in chain2:
                if section not in result:
                    result.append(section)
            return result
        else:
            # no overlapping sections, just connect
            return chain1 + chain2
    elif len(component_indices) == 3:
        # three chains (e.g. component 9)
        chain1_idx, chain2_idx, chain3_idx = component_indices
        chain1 = chains[chain1_idx]
        chain2 = chains[chain2_idx]
        chain3 = chains[chain3_idx]
        
        # collect all sections and remove duplicates
        all_sections_list = list(chain1)
        for section in chain2:
            if section not in all_sections_list:
                all_sections_list.append(section)
        for section in chain3:
            if section not in all_sections_list:
                all_sections_list.append(section)
        
        return all_sections_list
    else:
        # other cases, just merge
        result = []
        for idx in component_indices:
            for section in chains[idx]:
                if section not in result:
                    result.append(section)
        return result

def main():
    """main function"""
    print("fix rigid components merging logic")
    print("=" * 60)
    
    # load original chains
    chains = load_original_chains()
    print(f"original chains: {len(chains)}")
    
    # load chain connections
    connections = load_chain_connections()
    print(f"chain connections: {len(connections)}")
    
    # find connected components
    components = find_connected_components(chains, connections)
    print(f"connected components: {len(components)}")
    
    # merge chains in each component
    merged_components = []
    for i, component in enumerate(components):
        merged_chain = merge_chains_in_component(chains, component)
        merged_components.append(merged_chain)
        print(f"component {i+1}: {len(merged_chain)} sections")
    
    # calculate coverage
    all_sections = set()
    for chain in chains:
        all_sections.update(chain)
    
    covered_sections = set()
    for comp in merged_components:
        covered_sections.update(comp)
    
    print(f"\nstatistics:")
    print(f"total sections: {len(all_sections)}")
    print(f"covered sections: {len(covered_sections)}")
    print(f"coverage: {len(covered_sections)/len(all_sections)*100:.1f}%")
    
    # output results
    with open('fixed_rigid_components.txt', 'w') as f:
        f.write("fixed rigid components\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"total components: {len(merged_components)}\n")
        f.write(f"covered sections: {len(covered_sections)}\n")
        f.write(f"total sections: {len(all_sections)}\n")
        f.write(f"coverage: {len(covered_sections)/len(all_sections)*100:.1f}%\n\n")
        
        f.write("all components details:\n")
        for i, comp in enumerate(merged_components):
            f.write(f"component {i+1}: {' -> '.join(comp)} (length: {len(comp)})\n")
        
        f.write(f"\nuncovered sections ({len(all_sections - covered_sections)}):\n")
        uncovered = sorted(all_sections - covered_sections)
        for section in uncovered:
            f.write(f"  {section}\n")
    
    print(f"\nresults saved to fixed_rigid_components.txt")
    
    # check if all original sections are covered
    original_sections = set()
    for chain in chains:
        original_sections.update(chain)
    
    if covered_sections == original_sections:
        print("✓ success! all original sections are covered")
    else:
        missing = original_sections - covered_sections
        print(f"✗ still missing sections: {sorted(missing)}")

if __name__ == "__main__":
    main() 
