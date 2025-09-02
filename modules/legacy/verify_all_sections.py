#!/usr/bin/env python3
"""
verify_all_sections.py - 验证每个section是否都能形成链

简单验证：每个section都应该能形成至少长度为2的链
"""

import re
from typing import Dict, List, Set

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

def verify_section_chain(section: str, best_pairs: Dict[str, List[str]]) -> List[str]:
    """验证单个section是否能形成链"""
    if section not in best_pairs or not best_pairs[section]:
        return []
    
    # 获取最佳配对
    best_pair = best_pairs[section][0]
    
    # 检查最佳配对是否也有配对
    if best_pair in best_pairs and best_pairs[best_pair]:
        return [section, best_pair]
    else:
        return [section]

def main():
    # 解析best pairs文件
    print("解析best pairs文件...")
    best_pairs = parse_best_pairs('top_two_pairs_for_each_section.txt')
    print(f"解析完成，共 {len(best_pairs)} 个sections")
    
    print("\n验证每个section的链...")
    print("=" * 60)
    
    all_chains = []
    covered_sections = set()
    
    for section in sorted(best_pairs.keys()):
        chain = verify_section_chain(section, best_pairs)
        
        if len(chain) >= 2:
            all_chains.append(chain)
            covered_sections.update(chain)
            print(f"{section}: {' -> '.join(chain)} (长度: {len(chain)})")
        else:
            print(f"{section}: 无法形成链")
    
    print(f"\n统计结果:")
    print(f"总sections数: {len(best_pairs)}")
    print(f"能形成链的sections数: {len(covered_sections)}")
    print(f"覆盖率: {len(covered_sections)/len(best_pairs)*100:.1f}%")
    
    # 检查未覆盖的sections
    uncovered = set(best_pairs.keys()) - covered_sections
    if uncovered:
        print(f"\n未覆盖的sections ({len(uncovered)}):")
        for section in sorted(uncovered):
            if section in best_pairs and best_pairs[section]:
                best_pair = best_pairs[section][0]
                print(f"  {section} -> {best_pair} (但{best_pair}没有最佳配对)")
            else:
                print(f"  {section} (没有最佳配对)")
    
    # 检查是否有sections没有最佳配对
    sections_without_pairs = []
    for section in best_pairs.keys():
        if not best_pairs[section]:
            sections_without_pairs.append(section)
    
    if sections_without_pairs:
        print(f"\n没有最佳配对的sections ({len(sections_without_pairs)}):")
        for section in sections_without_pairs:
            print(f"  {section}")

if __name__ == "__main__":
    main() 