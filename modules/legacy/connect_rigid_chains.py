#!/usr/bin/env python3
"""
connect_rigid_chains.py - 连接不可动的链

算法思路:
1. 从rigid_chains_result.txt读取40个不可动的链
2. 提取每个链的头节点和尾节点
3. 使用best pairs数据尝试连接这些链
4. 构建链之间的连接关系
5. 尝试形成更大的连接组件

输出:
- 链之间的连接关系
- 可能的连接组件
- 无法连接的孤立链
"""

import re
from collections import defaultdict, deque
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

def get_chain_endpoints(chains: List[List[str]]) -> Dict[int, Tuple[str, str]]:
    """获取每个链的头节点和尾节点"""
    endpoints = {}
    
    print("链的头尾节点:")
    print("=" * 60)
    
    for i, chain in enumerate(chains):
        head = chain[0]
        tail = chain[-1]
        endpoints[i] = (head, tail)
        print(f"链 {i+1}: 头={head}, 尾={tail}")
    
    return endpoints

def find_chain_connections(chains: List[List[str]], best_pairs: Dict[str, List[str]]) -> Dict[int, List[Tuple[int, str, str]]]:
    """找到链之间的连接关系"""
    endpoints = get_chain_endpoints(chains)
    connections = defaultdict(list)
    
    print("\n寻找链之间的连接...")
    print("=" * 60)
    
    # 检查每个链的头尾节点是否能连接到其他链
    for i, (head, tail) in endpoints.items():
        print(f"\n检查链 {i+1} (头={head}, 尾={tail}):")
        
        # 检查头节点是否能连接到其他链
        if head in best_pairs:
            for target_section in best_pairs[head]:
                # 找到target_section属于哪个链
                target_chain = find_section_chain(target_section, chains)
                if target_chain is not None and target_chain != i:
                    print(f"  头节点 {head} -> 链 {target_chain+1} 的 {target_section}")
                    connections[i].append((target_chain, head, target_section))
        
        # 检查尾节点是否能连接到其他链
        if tail in best_pairs:
            for target_section in best_pairs[tail]:
                # 找到target_section属于哪个链
                target_chain = find_section_chain(target_section, chains)
                if target_chain is not None and target_chain != i:
                    print(f"  尾节点 {tail} -> 链 {target_chain+1} 的 {target_section}")
                    connections[i].append((target_chain, tail, target_section))
    
    return connections

def find_section_chain(section: str, chains: List[List[str]]) -> int:
    """找到section属于哪个链"""
    for i, chain in enumerate(chains):
        if section in chain:
            return i
    return None

def build_connected_components(chains: List[List[str]], connections: Dict[int, List[Tuple[int, str, str]]]) -> List[List[int]]:
    """构建连接的组件"""
    print("\n构建连接的组件...")
    print("=" * 60)
    
    # 使用BFS找到所有连接的组件
    visited = set()
    components = []
    
    for start_chain in range(len(chains)):
        if start_chain in visited:
            continue
        
        # BFS找到所有连接的链
        component = []
        queue = deque([start_chain])
        visited.add(start_chain)
        
        while queue:
            current_chain = queue.popleft()
            component.append(current_chain)
            
            # 添加所有连接的链
            for target_chain, _, _ in connections[current_chain]:
                if target_chain not in visited:
                    visited.add(target_chain)
                    queue.append(target_chain)
        
        components.append(component)
        print(f"组件 {len(components)}: 包含链 {[i+1 for i in component]} (共{len(component)}个链)")
    
    return components

def analyze_components(components: List[List[int]], chains: List[List[str]]) -> None:
    """分析连接组件的统计信息"""
    print("\n" + "=" * 60)
    print("连接组件分析")
    print("=" * 60)
    
    print(f"总组件数: {len(components)}")
    
    # 统计每个组件的大小
    component_sizes = [len(comp) for comp in components]
    if component_sizes:
        print(f"组件大小统计:")
        print(f"  最大组件: {max(component_sizes)} 个链")
        print(f"  最小组件: {min(component_sizes)} 个链")
        print(f"  平均大小: {sum(component_sizes)/len(component_sizes):.1f} 个链")
        
        # 按大小分组
        size_groups = defaultdict(int)
        for size in component_sizes:
            size_groups[size] += 1
        
        print(f"  大小分布:")
        for size in sorted(size_groups.keys()):
            print(f"    {size} 个链: {size_groups[size]} 个组件")
    
    # 显示每个组件的详细信息
    print(f"\n组件详情:")
    for i, component in enumerate(components, 1):
        total_sections = sum(len(chains[chain_idx]) for chain_idx in component)
        print(f"组件 {i}: 包含链 {[j+1 for j in component]} (共{len(component)}个链, {total_sections}个sections)")
        
        # 显示组件内的所有链
        for chain_idx in component:
            chain = chains[chain_idx]
            print(f"  链 {chain_idx+1}: {' -> '.join(chain)}")
    
    # 统计覆盖的sections
    all_covered_sections = set()
    for component in components:
        for chain_idx in component:
            all_covered_sections.update(chains[chain_idx])
    
    print(f"\n覆盖统计:")
    print(f"  覆盖的sections数: {len(all_covered_sections)}")
    print(f"  总sections数: {sum(len(chain) for chain in chains)}")
    print(f"  覆盖率: {len(all_covered_sections)/sum(len(chain) for chain in chains)*100:.1f}%")

def main():
    # 解析rigid chains
    print("解析rigid chains...")
    chains = parse_rigid_chains('rigid_chains_result.txt')
    print(f"解析完成，共 {len(chains)} 个链")
    
    # 解析best pairs
    print("解析best pairs...")
    best_pairs = parse_best_pairs('top_two_pairs_for_each_section.txt')
    print(f"解析完成，共 {len(best_pairs)} 个sections的best pairs")
    
    # 获取链的头尾节点
    endpoints = get_chain_endpoints(chains)
    
    # 寻找链之间的连接
    connections = find_chain_connections(chains, best_pairs)
    
    # 构建连接的组件
    components = build_connected_components(chains, connections)
    
    # 分析结果
    analyze_components(components, chains)
    
    # 保存结果到文件
    with open('chain_connections_result.txt', 'w', encoding='utf-8') as f:
        f.write("链连接分析结果\n")
        f.write("=" * 60 + "\n\n")
        
        f.write(f"总链数: {len(chains)}\n")
        f.write(f"总组件数: {len(components)}\n\n")
        
        f.write("链的头尾节点:\n")
        for i, (head, tail) in endpoints.items():
            f.write(f"链 {i+1}: 头={head}, 尾={tail}\n")
        
        f.write("\n链之间的连接:\n")
        for i, conns in connections.items():
            if conns:
                f.write(f"链 {i+1} 的连接:\n")
                for target_chain, from_section, to_section in conns:
                    f.write(f"  {from_section} -> 链 {target_chain+1} 的 {to_section}\n")
        
        f.write("\n连接组件详情:\n")
        for i, component in enumerate(components, 1):
            total_sections = sum(len(chains[chain_idx]) for chain_idx in component)
            f.write(f"组件 {i}: 包含链 {[j+1 for j in component]} (共{len(component)}个链, {total_sections}个sections)\n")
            for chain_idx in component:
                chain = chains[chain_idx]
                f.write(f"  链 {chain_idx+1}: {' -> '.join(chain)}\n")
    
    print(f"\n结果已保存到 chain_connections_result.txt")

if __name__ == "__main__":
    main() 