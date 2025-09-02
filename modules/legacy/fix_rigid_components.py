#!/usr/bin/env python3
"""
fix_rigid_components.py - 修复不可动组件合并逻辑
"""

def load_original_chains():
    """加载原始链"""
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
    """加载链之间的连接关系"""
    # 根据chain_connections_result.txt的实际连接关系
    connections = [
        (2, 15),   # 链2和链15连接: section_3_r01_c01 -> section_34_r01_c01
        (5, 33),   # 链5和链33连接: section_6_r01_c01 -> section_57_r01_c01
        (9, 36),   # 链9和链36连接: section_10_r01_c01 -> section_66_r01_c01
        (14, 33),  # 链14和链33连接: section_18_r01_c01 -> section_61_r01_c01
        (24, 16),  # 链24和链16连接: section_41_r01_c01 -> section_78_r01_c01
        (33, 5),   # 链33和链5连接: section_35_r01_c01 -> section_6_r01_c01
        (36, 9),   # 链36和链9连接: section_27_r01_c01 -> section_10_r01_c01
        (36, 21),  # 链36和链21连接: section_66_r01_c01 -> section_73_r01_c01
    ]
    return connections

def find_connected_components(chains, connections):
    """找到连接的组件"""
    # 构建图
    graph = {}
    for i in range(len(chains)):
        graph[i] = []
    
    for conn in connections:
        if conn[0]-1 < len(chains) and conn[1]-1 < len(chains):
            graph[conn[0]-1].append(conn[1]-1)
            graph[conn[1]-1].append(conn[0]-1)
    
    # DFS找连通分量
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
    """合并组件中的链，保留所有原始sections"""
    if len(component_indices) == 1:
        return chains[component_indices[0]]
    
    # 收集所有sections
    all_sections = set()
    for idx in component_indices:
        all_sections.update(chains[idx])
    
    # 根据组件类型进行不同的合并策略
    if len(component_indices) == 2:
        # 两个链的情况，需要根据连接点合并
        chain1_idx, chain2_idx = component_indices
        chain1 = chains[chain1_idx]
        chain2 = chains[chain2_idx]
        
        # 检查是否有重叠的sections
        overlap = set(chain1) & set(chain2)
        if overlap:
            # 有重叠，需要去重合并
            result = list(chain1)
            for section in chain2:
                if section not in result:
                    result.append(section)
            return result
        else:
            # 无重叠，直接连接
            return chain1 + chain2
    elif len(component_indices) == 3:
        # 三个链的情况（如组件9）
        chain1_idx, chain2_idx, chain3_idx = component_indices
        chain1 = chains[chain1_idx]
        chain2 = chains[chain2_idx]
        chain3 = chains[chain3_idx]
        
        # 收集所有sections并去重
        all_sections_list = list(chain1)
        for section in chain2:
            if section not in all_sections_list:
                all_sections_list.append(section)
        for section in chain3:
            if section not in all_sections_list:
                all_sections_list.append(section)
        
        return all_sections_list
    else:
        # 其他情况，简单合并
        result = []
        for idx in component_indices:
            for section in chains[idx]:
                if section not in result:
                    result.append(section)
        return result

def main():
    """主函数"""
    print("修复不可动组件合并逻辑")
    print("=" * 60)
    
    # 加载原始链
    chains = load_original_chains()
    print(f"原始链数: {len(chains)}")
    
    # 加载连接关系
    connections = load_chain_connections()
    print(f"连接关系数: {len(connections)}")
    
    # 找到连通分量
    components = find_connected_components(chains, connections)
    print(f"连通分量数: {len(components)}")
    
    # 合并每个组件中的链
    merged_components = []
    for i, component in enumerate(components):
        merged_chain = merge_chains_in_component(chains, component)
        merged_components.append(merged_chain)
        print(f"组件 {i+1}: {len(merged_chain)} sections")
    
    # 统计覆盖率
    all_sections = set()
    for chain in chains:
        all_sections.update(chain)
    
    covered_sections = set()
    for comp in merged_components:
        covered_sections.update(comp)
    
    print(f"\n统计结果:")
    print(f"总sections数: {len(all_sections)}")
    print(f"覆盖的sections数: {len(covered_sections)}")
    print(f"覆盖率: {len(covered_sections)/len(all_sections)*100:.1f}%")
    
    # 输出结果
    with open('fixed_rigid_components.txt', 'w') as f:
        f.write("修复后的不可动组件\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"总组件数: {len(merged_components)}\n")
        f.write(f"覆盖的sections数: {len(covered_sections)}\n")
        f.write(f"总sections数: {len(all_sections)}\n")
        f.write(f"覆盖率: {len(covered_sections)/len(all_sections)*100:.1f}%\n\n")
        
        f.write("所有组件详情:\n")
        for i, comp in enumerate(merged_components):
            f.write(f"组件 {i+1}: {' -> '.join(comp)} (长度: {len(comp)})\n")
        
        f.write(f"\n未覆盖的sections ({len(all_sections - covered_sections)}):\n")
        uncovered = sorted(all_sections - covered_sections)
        for section in uncovered:
            f.write(f"  {section}\n")
    
    print(f"\n结果已保存到 fixed_rigid_components.txt")
    
    # 检查是否修复了丢失的sections
    original_sections = set()
    for chain in chains:
        original_sections.update(chain)
    
    if covered_sections == original_sections:
        print("✓ 成功修复！所有原始sections都被保留")
    else:
        missing = original_sections - covered_sections
        print(f"✗ 仍有丢失的sections: {sorted(missing)}")

if __name__ == "__main__":
    main() 