#!/usr/bin/env python3
"""
build_rigid_chains.py - 通过best pairs构建不可动的节点链

算法思路:
1. 从每个section的best pair开始
2. 如果A-B是best pair，B-C也是best pair，那么A-B-C就是不可动的链
3. 继续扩展，直到无法继续连接
4. 统计所有不可动的链

输出:
- 每个链的sections列表
- 链的长度统计
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

def find_rigid_chains(best_pairs: Dict[str, List[str]]) -> List[List[str]]:
    """找到所有不可动的节点链"""
    print("开始构建不可动节点链...")
    print("=" * 60)
    
    # 记录已处理的sections
    processed = set()
    chains = []
    
    # 为每个section尝试构建链
    for start_section in best_pairs.keys():
        if start_section in processed:
            continue
            
        print(f"\n从 {start_section} 开始构建链...")
        chain = build_chain_from_section(start_section, best_pairs, processed)
        
        if len(chain) > 1:  # 只保留长度大于1的链
            chains.append(chain)
            print(f"找到链: {' -> '.join(chain)} (长度: {len(chain)})")
        else:
            print(f"{start_section} 无法形成链")
    
    return chains

def build_chain_from_section(start_section: str, best_pairs: Dict[str, List[str]], processed: Set[str]) -> List[str]:
    """从指定section开始构建链"""
    chain = [start_section]
    processed.add(start_section)
    
    # 向前扩展
    current = start_section
    while current in best_pairs and best_pairs[current]:
        next_section = best_pairs[current][0]  # 取最佳配对
        
        # 检查是否形成循环
        if next_section in chain:
            print(f"  检测到循环: {next_section} 已在链中")
            break
            
        # 检查next_section是否也选择current作为最佳配对
        if next_section in best_pairs and best_pairs[next_section]:
            if best_pairs[next_section][0] == current:
                print(f"  确认双向连接: {current} <-> {next_section}")
                chain.append(next_section)
                processed.add(next_section)
                current = next_section
            else:
                print(f"  单向连接: {current} -> {next_section} (但{next_section}的最佳配对是{best_pairs[next_section][0]})")
                break
        else:
            print(f"  {next_section} 没有最佳配对")
            break
    
    # 向后扩展
    current = start_section
    while current in best_pairs:
        # 找到选择current作为最佳配对的section
        prev_section = None
        for section, pairs in best_pairs.items():
            if pairs and pairs[0] == current and section not in chain:
                prev_section = section
                break
        
        if prev_section is None:
            break
            
        # 检查是否形成循环
        if prev_section in chain:
            print(f"  检测到循环: {prev_section} 已在链中")
            break
            
        print(f"  向后扩展: {prev_section} -> {current}")
        chain.insert(0, prev_section)
        processed.add(prev_section)
        current = prev_section
    
    return chain

def analyze_chains(chains: List[List[str]], best_pairs: Dict[str, List[str]]) -> None:
    """分析链的统计信息"""
    print("\n" + "=" * 60)
    print("链分析结果")
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
    
    # 构建不可动链
    chains = find_rigid_chains(best_pairs)
    
    # 分析结果
    analyze_chains(chains, best_pairs)
    
    # 保存结果到文件
    with open('rigid_chains_result.txt', 'w', encoding='utf-8') as f:
        f.write("不可动节点链分析结果\n")
        f.write("=" * 60 + "\n\n")
        
        f.write(f"总链数: {len(chains)}\n")
        covered_sections = set()
        for chain in chains:
            covered_sections.update(chain)
        f.write(f"覆盖的sections数: {len(covered_sections)}\n")
        f.write(f"总sections数: {len(best_pairs)}\n")
        f.write(f"覆盖率: {len(covered_sections)/len(best_pairs)*100:.1f}%\n\n")
        
        f.write("所有链详情:\n")
        for i, chain in enumerate(chains, 1):
            f.write(f"链 {i}: {' -> '.join(chain)} (长度: {len(chain)})\n")
        
        uncovered = set(best_pairs.keys()) - covered_sections
        if uncovered:
            f.write(f"\n未覆盖的sections ({len(uncovered)}):\n")
            for section in sorted(uncovered):
                f.write(f"  {section}\n")
    
    print(f"\n结果已保存到 rigid_chains_result.txt")

if __name__ == "__main__":
    main() 