#!/usr/bin/env python3
"""
update_rigid_chains.py - 更新不可动的链信息

功能:
1. 读取之前的rigid chains结果
2. 应用连接关系，将可以连接的链合并
3. 生成更新后的不可动组件列表
4. 统计新的组件信息

输出:
- 更新后的不可动组件
- 每个组件的sections列表
- 组件的统计信息
"""

import re
from collections import defaultdict
from typing import List, Set, Dict, Tuple

def parse_rigid_chains(filename: str) -> List[List[str]]:
    """解析rigid chains文件，返回所有链"""
    chains = []
    
    with open(filename, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    for line in lines:
        # 匹配格式: 链 X: section_A -> section_B -> section_C (长度: N)
        match = re.match(r'链 \d+:\s+(.+) \(长度:', line)
        if match:
            chain_str = match.group(1)
            chain = [s.strip() for s in chain_str.split(' -> ')]
            chains.append(chain)
    
    return chains

def get_connected_components() -> List[List[int]]:
    """获取连接组件的定义"""
    # 基于之前的分析结果，定义连接组件
    components = [
        [0],  # 链1: section_1_r01_c01 -> section_13_r01_c01
        [1, 14],  # 链2,15: section_39_r01_c01 -> ... -> section_3_r01_c01 + section_19_r01_c01 -> section_34_r01_c01
        [2],  # 链3: section_4_r01_c01 -> section_59_r01_c01
        [3],  # 链4: section_5_r01_c01 -> section_99_r01_c01
        [4, 32],  # 链5,33: section_35_r01_c01 -> section_6_r01_c01 + section_35_r01_c01 -> section_6_r01_c01 -> section_57_r01_c01 -> section_61_r01_c01
        [5],  # 链6: section_104_r01_c01 -> section_7_r01_c01 -> section_11_r01_c01
        [6],  # 链7: section_8_r01_c01 -> section_77_r01_c01
        [7],  # 链8: section_9_r01_c01 -> section_64_r01_c01
        [8, 35, 20],  # 链9,36,21: 最大的连接组件
        [9],  # 链10: section_12_r01_c01 -> section_89_r01_c01
        [10],  # 链11: section_14_r01_c01 -> section_101_r01_c01
        [11],  # 链12: section_29_r01_c01 -> section_16_r01_c01 -> section_76_r01_c01
        [12],  # 链13: section_90_r01_c01 -> section_31_r01_c01 -> section_17_r01_c01 -> section_38_r01_c01
        [13],  # 链14: section_92_r01_c01 -> section_18_r01_c01
        [15],  # 链16: section_21_r01_c01 -> section_78_r01_c01
        [16],  # 链17: section_22_r01_c01 -> section_26_r01_c01
        [17],  # 链18: section_80_r01_c01 -> section_23_r01_c01 -> section_43_r01_c01
        [18],  # 链19: section_24_r01_c01 -> section_40_r01_c01
        [19],  # 链20: section_25_r01_c01 -> section_74_r01_c01
        [21],  # 链22: section_72_r01_c01 -> section_32_r01_c01 -> section_97_r01_c01
        [22],  # 链23: section_2_r01_c01 -> section_36_r01_c01 -> section_102_r01_c01
        [23],  # 链24: section_42_r01_c01 -> section_41_r01_c01
        [24],  # 链25: section_45_r01_c01 -> section_105_r01_c01
        [25],  # 链26: section_46_r01_c01 -> section_69_r01_c01
        [26],  # 链27: section_47_r01_c01 -> section_96_r01_c01
        [27],  # 链28: section_48_r01_c01 -> section_71_r01_c01
        [28],  # 链29: section_49_r01_c01 -> section_75_r01_c01
        [29],  # 链30: section_50_r01_c01 -> section_70_r01_c01
        [30],  # 链31: section_55_r01_c01 -> section_63_r01_c01
        [31],  # 链32: section_56_r01_c01 -> section_88_r01_c01
        [33],  # 链34: section_58_r01_c01 -> section_85_r01_c01
        [34],  # 链35: section_51_r01_c01 -> section_60_r01_c01 -> section_100_r01_c01
        [36],  # 链37: section_67_r01_c01 -> section_86_r01_c01
        [37],  # 链38: section_82_r01_c01 -> section_103_r01_c01
        [38],  # 链39: section_87_r01_c01 -> section_91_r01_c01
        [39],  # 链40: section_93_r01_c01 -> section_94_r01_c01
    ]
    return components

def merge_chains_in_component(component_chains: List[int], chains: List[List[str]]) -> List[str]:
    """合并组件内的链，形成完整的不可动组件"""
    if len(component_chains) == 1:
        return chains[component_chains[0]]
    
    # 对于多个链的组件，需要找到正确的连接顺序
    if component_chains == [1, 14]:  # 组件2: 链2,15
        # 链2: section_39_r01_c01 -> section_15_r01_c01 -> section_52_r01_c01 -> section_3_r01_c01
        # 链15: section_19_r01_c01 -> section_34_r01_c01
        # 连接点: section_3_r01_c01 -> section_34_r01_c01
        return ['section_39_r01_c01', 'section_15_r01_c01', 'section_52_r01_c01', 'section_3_r01_c01', 'section_34_r01_c01']
    
    elif component_chains == [4, 32]:  # 组件5: 链5,33
        # 链5: section_35_r01_c01 -> section_6_r01_c01
        # 链33: section_35_r01_c01 -> section_6_r01_c01 -> section_57_r01_c01 -> section_61_r01_c01
        # 链33包含了链5，所以直接返回链33
        return chains[32]
    
    elif component_chains == [8, 35, 20]:  # 组件9: 链9,36,21
        # 链9: section_27_r01_c01 -> section_10_r01_c01
        # 链36: section_27_r01_c01 -> section_10_r01_c01 -> section_66_r01_c01
        # 链21: section_28_r01_c01 -> section_73_r01_c01
        # 连接点: section_66_r01_c01 -> section_73_r01_c01
        return ['section_27_r01_c01', 'section_10_r01_c01', 'section_66_r01_c01', 'section_73_r01_c01']
    
    else:
        # 其他情况，暂时返回第一个链
        return chains[component_chains[0]]

def create_updated_components(chains: List[List[str]]) -> List[List[str]]:
    """创建更新后的不可动组件"""
    components = get_connected_components()
    updated_components = []
    
    print("更新不可动组件...")
    print("=" * 60)
    
    for i, component_chains in enumerate(components):
        merged_chain = merge_chains_in_component(component_chains, chains)
        updated_components.append(merged_chain)
        
        print(f"组件 {i+1}: 包含原链 {[j+1 for j in component_chains]} (共{len(component_chains)}个链)")
        print(f"  合并后: {' -> '.join(merged_chain)} (长度: {len(merged_chain)})")
        print()
    
    return updated_components

def analyze_updated_components(components: List[List[str]]) -> None:
    """分析更新后的组件统计信息"""
    print("=" * 60)
    print("更新后的不可动组件分析")
    print("=" * 60)
    
    # 统计信息
    component_lengths = [len(comp) for comp in components]
    covered_sections = set()
    for comp in components:
        covered_sections.update(comp)
    
    print(f"总组件数: {len(components)}")
    print(f"覆盖的sections数: {len(covered_sections)}")
    print(f"总sections数: 100")
    print(f"覆盖率: {len(covered_sections)/100*100:.1f}%")
    
    if component_lengths:
        print(f"组件长度统计:")
        print(f"  最长组件: {max(component_lengths)} sections")
        print(f"  最短组件: {min(component_lengths)} sections")
        print(f"  平均长度: {sum(component_lengths)/len(component_lengths):.1f} sections")
        
        # 按长度分组
        length_groups = defaultdict(int)
        for length in component_lengths:
            length_groups[length] += 1
        
        print(f"  长度分布:")
        for length in sorted(length_groups.keys()):
            print(f"    {length} sections: {length_groups[length]} 个组件")
    
    # 显示所有组件
    print(f"\n所有组件详情:")
    for i, comp in enumerate(components, 1):
        print(f"组件 {i}: {' -> '.join(comp)} (长度: {len(comp)})")
    
    # 显示未覆盖的sections
    all_sections = set()
    for i in range(1, 107):  # 假设sections从1到106
        all_sections.add(f"section_{i}_r01_c01")
    
    uncovered = all_sections - covered_sections
    if uncovered:
        print(f"\n未覆盖的sections ({len(uncovered)}):")
        for section in sorted(uncovered):
            print(f"  {section}")

def main():
    # 解析rigid chains
    print("解析rigid chains...")
    chains = parse_rigid_chains('rigid_chains_result.txt')
    print(f"解析完成，共 {len(chains)} 个原始链")
    
    # 创建更新后的组件
    updated_components = create_updated_components(chains)
    
    # 分析结果
    analyze_updated_components(updated_components)
    
    # 保存结果到文件
    with open('updated_rigid_components.txt', 'w', encoding='utf-8') as f:
        f.write("更新后的不可动组件\n")
        f.write("=" * 60 + "\n\n")
        
        f.write(f"总组件数: {len(updated_components)}\n")
        covered_sections = set()
        for comp in updated_components:
            covered_sections.update(comp)
        f.write(f"覆盖的sections数: {len(covered_sections)}\n")
        f.write(f"总sections数: 100\n")
        f.write(f"覆盖率: {len(covered_sections)/100*100:.1f}%\n\n")
        
        f.write("所有组件详情:\n")
        for i, comp in enumerate(updated_components, 1):
            f.write(f"组件 {i}: {' -> '.join(comp)} (长度: {len(comp)})\n")
        
        # 显示未覆盖的sections
        all_sections = set()
        for i in range(1, 107):
            all_sections.add(f"section_{i}_r01_c01")
        
        uncovered = all_sections - covered_sections
        if uncovered:
            f.write(f"\n未覆盖的sections ({len(uncovered)}):\n")
            for section in sorted(uncovered):
                f.write(f"  {section}\n")
    
    print(f"\n结果已保存到 updated_rigid_components.txt")

if __name__ == "__main__":
    main() 