#!/usr/bin/env python3
"""
build_unidirectional_chains.py - 构建单向配对链

算法思路:
1. 从每个section开始，沿着单向配对构建链
2. 当遇到重复的section时，说明找到了连接点
3. 合并有共同sections的链
4. 统计所有可能的链组合

输出:
- 所有可能的链
- 链的连接点
- 覆盖的sections数量
"""

import re
from collections import defaultdict, deque
from typing import List, Set, Dict, Tuple

def parse_best_pairs(filename: str) -> Dict[str, List[str]]:
    """解析best pairs文件，返回每个section的最佳配对"""
    best_pairs = {}
    
    with open(filename, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    for line in lines:
        # 匹配格式: section_X -> section_Y (score: Z) [direction]
        match = re.match(r'(\w+)\s+->\s+(\w+)\s+\(score:\s+[\d.]+\)', line)
        if match:
            section1, section2 = match.groups()
            if section1 not in best_pairs:
                best_pairs[section1] = []
            best_pairs[section1].append(section2)
    
    return best_pairs

def build_forward_chain(start_section: str, best_pairs: Dict[str, List[str]], visited: Set[str]) -> List[str]:
    """从指定section开始向前构建链"""
    chain = [start_section]
    visited.add(start_section)
    current = start_section
    
    while current in best_pairs and best_pairs[current]:
        next_section = best_pairs[current][0]
        
        # 如果遇到已访问的section，说明找到了连接点
        if next_section in visited:
            print(f"  找到连接点: {next_section} (已在链中)")
            break
            
        chain.append(next_section)
        visited.add(next_section)
        current = next_section
    
    return chain

def build_backward_chain(start_section: str, best_pairs: Dict[str, List[str]], visited: Set[str]) -> List[str]:
    """从指定section开始向后构建链"""
    chain = []
    current = start_section
    
    while True:
        # 找到选择current作为最佳配对的section
        prev_section = None
        for section, pairs in best_pairs.items():
            if pairs and pairs[0] == current and section not in visited:
                prev_section = section
                break
        
        if prev_section is None:
            break
            
        # 如果遇到已访问的section，说明找到了连接点
        if prev_section in visited:
            print(f"  找到连接点: {prev_section} (已在链中)")
            break
            
        chain.insert(0, prev_section)
        visited.add(prev_section)
        current = prev_section
    
    return chain

def find_all_chains(best_pairs: Dict[str, List[str]]) -> List[List[str]]:
    """找到所有可能的链"""
    print("开始构建单向配对链...")
    print("=" * 60)
    
    all_chains = []
    processed = set()
    
    # 为每个section尝试构建链
    for start_section in best_pairs.keys():
        if start_section in processed:
            continue
            
        print(f"\n从 {start_section} 开始构建链...")
        
        # 构建完整的链（向前+向后）
        visited = set()
        backward_chain = build_backward_chain(start_section, best_pairs, visited)
        forward_chain = build_forward_chain(start_section, best_pairs, visited)
        
        # 合并链
        full_chain = backward_chain + forward_chain[1:]  # 避免重复start_section
        
        if len(full_chain) > 1:
            all_chains.append(full_chain)
            print(f"找到链: {' -> '.join(full_chain)} (长度: {len(full_chain)})")
            
            # 标记链中的所有sections为已处理
            processed.update(full_chain)
        else:
            print(f"{start_section} 无法形成链")
    
    return all_chains

def find_chain_connections(chains: List[List[str]]) -> List[Tuple[int, int, str]]:
    """找到链之间的连接点"""
    print(f"\n寻找链之间的连接点...")
    print("=" * 60)
    
    connections = []
    
    for i in range(len(chains)):
        for j in range(i + 1, len(chains)):
            chain1 = chains[i]
            chain2 = chains[j]
            
            # 找到共同的sections
            common_sections = set(chain1) & set(chain2)
            
            for common_section in common_sections:
                connections.append((i, j, common_section))
                print(f"链 {i+1} 和链 {j+1} 通过 {common_section} 连接")
    
    return connections

def merge_connected_chains(chains: List[List[str]], connections: List[Tuple[int, int, str]]) -> List[List[str]]:
    """合并有连接的链"""
    print(f"\n合并连接的链...")
    print("=" * 60)
    
    if not connections:
        print("没有找到连接的链")
        return chains
    
    # 使用并查集来合并链
    parent = list(range(len(chains)))
    
    def find(x):
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]
    
    def union(x, y):
        px, py = find(x), find(y)
        if px != py:
            parent[px] = py
    
    # 合并有连接的链
    for i, j, _ in connections:
        union(i, j)
    
    # 按组收集链
    groups = defaultdict(list)
    for i in range(len(chains)):
        groups[find(i)].append(i)
    
    # 合并每个组内的链
    merged_chains = []
    for group_indices in groups.values():
        if len(group_indices) == 1:
            # 单个链，直接添加
            merged_chains.append(chains[group_indices[0]])
        else:
            # 多个链，需要合并
            print(f"合并链组: {[i+1 for i in group_indices]}")
            merged_chain = merge_chain_group([chains[i] for i in group_indices])
            merged_chains.append(merged_chain)
    
    return merged_chains

def merge_chain_group(chain_group: List[List[str]]) -> List[str]:
    """合并一组链"""
    if len(chain_group) == 1:
        return chain_group[0]
    
    # 简单的合并策略：找到连接点，然后合并
    # 这里可以实现更复杂的合并逻辑
    all_sections = set()
    for chain in chain_group:
        all_sections.update(chain)
    
    # 按section编号排序（简化处理）
    sorted_sections = sorted(all_sections, key=lambda x: int(x.split('_')[1]))
    
    print(f"  合并后的链: {' -> '.join(sorted_sections)} (长度: {len(sorted_sections)})")
    return sorted_sections

def analyze_chains(chains: List[List[str]], best_pairs: Dict[str, List[str]]) -> None:
    """分析链的统计信息"""
    print(f"\n链分析结果")
    print("=" * 60)
    
    # 统计信息
    chain_lengths = [len(chain) for chain in chains]
    covered_sections = set()
    for chain in chains:
        covered_sections.update(chain)
    
    print(f"总链数: {len(chains)}")
    print(f"覆盖的sections数: {len(covered_sections)}")
    print(f"总sections数: {len(best_pairs)}")
    print(f"覆盖率: {len(covered_sections)/len(best_pairs)*100:.1f}%")
    
    if chain_lengths:
        print(f"链长度统计:")
        print(f"  最长链: {max(chain_lengths)} sections")
        print(f"  最短链: {min(chain_lengths)} sections")
        print(f"  平均长度: {sum(chain_lengths)/len(chain_lengths):.1f} sections")
        
        # 按长度分组
        length_groups = defaultdict(int)
        for length in chain_lengths:
            length_groups[length] += 1
        
        print(f"  长度分布:")
        for length in sorted(length_groups.keys()):
            print(f"    {length} sections: {length_groups[length]} 条链")
    
    # 显示所有链
    print(f"\n所有链详情:")
    for i, chain in enumerate(chains, 1):
        print(f"链 {i}: {' -> '.join(chain)} (长度: {len(chain)})")
    
    # 显示未覆盖的sections
    uncovered = set(best_pairs.keys()) - covered_sections
    if uncovered:
        print(f"\n未覆盖的sections ({len(uncovered)}):")
        for section in sorted(uncovered):
            print(f"  {section}")

def main():
    # 解析best pairs文件
    print("解析best pairs文件...")
    best_pairs = parse_best_pairs('top_two_pairs_for_each_section.txt')
    print(f"解析完成，共 {len(best_pairs)} 个sections")
    
    # 构建所有链
    chains = find_all_chains(best_pairs)
    
    # 找到链之间的连接
    connections = find_chain_connections(chains)
    
    # 合并连接的链
    merged_chains = merge_connected_chains(chains, connections)
    
    # 分析结果
    analyze_chains(merged_chains, best_pairs)
    
    # 保存结果到文件
    with open('unidirectional_chains_result.txt', 'w', encoding='utf-8') as f:
        f.write("单向配对链分析结果\n")
        f.write("=" * 60 + "\n\n")
        
        f.write(f"原始链数: {len(chains)}\n")
        f.write(f"合并后链数: {len(merged_chains)}\n")
        f.write(f"连接点数量: {len(connections)}\n\n")
        
        covered_sections = set()
        for chain in merged_chains:
            covered_sections.update(chain)
        f.write(f"覆盖的sections数: {len(covered_sections)}\n")
        f.write(f"总sections数: {len(best_pairs)}\n")
        f.write(f"覆盖率: {len(covered_sections)/len(best_pairs)*100:.1f}%\n\n")
        
        f.write("所有链详情:\n")
        for i, chain in enumerate(merged_chains, 1):
            f.write(f"链 {i}: {' -> '.join(chain)} (长度: {len(chain)})\n")
        
        uncovered = set(best_pairs.keys()) - covered_sections
        if uncovered:
            f.write(f"\n未覆盖的sections ({len(uncovered)}):\n")
            for section in sorted(uncovered):
                f.write(f"  {section}\n")
    
    print(f"\n结果已保存到 unidirectional_chains_result.txt")

if __name__ == "__main__":
    main() 