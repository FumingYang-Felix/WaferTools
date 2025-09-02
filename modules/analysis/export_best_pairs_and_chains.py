#!/usr/bin/env python3
"""
export_best_pairs_and_chains.py - 输出每个section的best pair和自动组合的链（递归融合所有best pair指向关系，严格校验不丢失section）
"""
import re
from collections import defaultdict
from typing import Dict, List, Tuple

def parse_best_pairs(filename: str) -> Dict[str, Tuple[str, float, str]]:
    """解析best pairs文件，返回每个section的最佳配对、分数和方向"""
    best_pairs = {}
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            # 匹配格式: section_X -> section_Y (score: Z) [方向]
            match = re.match(r'(section_\d+_r01_c01)\s+->\s+(section_\d+_r01_c01)\s+\(score:\s*([\d.]+)\)\s*\[(.*?)\]', line)
            if match:
                section, pair, score, direction = match.groups()
                best_pairs[section] = (pair, float(score), direction)
    return best_pairs

def section_sort_key(section: str) -> int:
    # 提取section编号用于排序
    m = re.match(r'section_(\d+)_r01_c01', section)
    return int(m.group(1)) if m else 0

def export_best_pairs(best_pairs: Dict[str, Tuple[str, float, str]], out_f):
    out_f.write("# 每个section的best pair\n")
    for section in sorted(best_pairs.keys(), key=section_sort_key):
        pair, score, direction = best_pairs[section]
        out_f.write(f"{section}   -> {pair}   (score: {score:.4f}) [{direction}]\n")
    out_f.write("\n")

def build_initial_chains(best_pairs: Dict[str, Tuple[str, float, str]]) -> List[List[str]]:
    """生成初始链，每个section出发，遇到已访问就断开"""
    visited = set()
    chains = []
    for section in sorted(best_pairs.keys(), key=section_sort_key):
        if section in visited:
            continue
        chain = [section]
        visited.add(section)
        current = section
        while True:
            next_section = best_pairs.get(current, (None, None, None))[0]
            if (not next_section or next_section in visited or next_section not in best_pairs):
                break
            chain.append(next_section)
            visited.add(next_section)
            current = next_section
        chains.append(chain)
    return chains

def merge_chains_bestpair(chains: List[List[str]], best_pairs: Dict[str, Tuple[str, float, str]], all_sections: set) -> List[List[str]]:
    """递归合并所有能通过best pair指向链头/尾的链，直到不能再合并为止，合并后严格校验section完整性"""
    changed = True
    while changed:
        changed = False
        head_map = {chain[0]: idx for idx, chain in enumerate(chains)}
        tail_map = {chain[-1]: idx for idx, chain in enumerate(chains)}
        merge_ops = []
        used = set()
        for i, chain in enumerate(chains):
            tail = chain[-1]
            # 如果有其他链的头节点是tail的best pair，则合并
            for j, other in enumerate(chains):
                if i == j:
                    continue
                other_head = other[0]
                # tail的best pair是other_head，则tail链接other链
                if best_pairs.get(tail, (None,))[0] == other_head and (i, j) not in used:
                    merge_ops.append((i, j, 'tail->head_by_bestpair'))
                    used.add((i, j))
                # other链的尾节点的best pair是chain的头节点，则other链接chain
                if best_pairs.get(other[-1], (None,))[0] == chain[0] and (j, i) not in used:
                    merge_ops.append((j, i, 'tail->head_by_bestpair'))
                    used.add((j, i))
            # 依然保留原有的尾接头、头接尾逻辑
            if tail in head_map and i != head_map[tail] and (i, head_map[tail]) not in used:
                merge_ops.append((i, head_map[tail], 'tail-head'))
                used.add((i, head_map[tail]))
            if chain[0] in tail_map and i != tail_map[chain[0]] and (tail_map[chain[0]], i) not in used:
                merge_ops.append((tail_map[chain[0]], i, 'tail-head'))
                used.add((tail_map[chain[0]], i))
        if not merge_ops:
            break
        merged = set()
        new_chains = []
        for i, j, mode in merge_ops:
            if i in merged or j in merged:
                continue
            # i尾接j头
            if chains[i][-1] == chains[j][0]:
                new_chain = chains[i] + chains[j][1:]
            # j尾接i头
            elif chains[j][-1] == chains[i][0]:
                new_chain = chains[j] + chains[i][1:]
            # i尾的best pair是j头
            elif best_pairs.get(chains[i][-1], (None,))[0] == chains[j][0]:
                new_chain = chains[i] + chains[j][1:]
            # j尾的best pair是i头
            elif best_pairs.get(chains[j][-1], (None,))[0] == chains[i][0]:
                new_chain = chains[j] + chains[i][1:]
            else:
                continue
            merged.add(i)
            merged.add(j)
            new_chains.append(new_chain)
        for idx, chain in enumerate(chains):
            if idx not in merged:
                new_chains.append(chain)
        chains = new_chains
        changed = True
    # 去重，确保每个section只出现一次
    seen = set()
    final_chains = []
    for chain in chains:
        filtered = []
        for s in chain:
            if s not in seen:
                filtered.append(s)
                seen.add(s)
        if filtered:
            final_chains.append(filtered)
    # 校验完整性
    merged_sections = set()
    for chain in final_chains:
        merged_sections.update(chain)
    missing = all_sections - merged_sections
    if missing:
        print(f"[ERROR] 缺失section: {sorted(missing, key=section_sort_key)}")
        raise RuntimeError(f"链合并后丢失section: {missing}")
    return final_chains

def export_chains(chains: List[List[str]], out_f):
    out_f.write("# 可组合的链（递归融合所有best pair指向关系，最终每个section只出现一次）\n")
    for i, chain in enumerate(chains, 1):
        out_f.write(f"链{i}: {' -> '.join(chain)}\n")
    out_f.write("\n")

def main():
    best_pairs = parse_best_pairs('top_two_pairs_for_each_section.txt')
    all_sections = set(best_pairs.keys())
    with open('best_pairs_and_chains.txt', 'w', encoding='utf-8') as out_f:
        export_best_pairs(best_pairs, out_f)
        initial_chains = build_initial_chains(best_pairs)
        merged_chains = merge_chains_bestpair(initial_chains, best_pairs, all_sections)
        export_chains(merged_chains, out_f)
    print("已输出到 best_pairs_and_chains.txt")

if __name__ == "__main__":
    main() 