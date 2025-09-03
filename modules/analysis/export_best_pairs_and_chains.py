#!/usr/bin/env python3
"""
export_best_pairs_and_chains.py - export best pairs and chains
"""
import re
from collections import defaultdict
from typing import Dict, List, Tuple

def parse_best_pairs(filename: str) -> Dict[str, Tuple[str, float, str]]:
    """parse best pairs file, return best pairs for each section"""
    best_pairs = {}
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            # match format: section_X -> section_Y (score: Z) [direction]
            match = re.match(r'(section_\d+_r01_c01)\s+->\s+(section_\d+_r01_c01)\s+\(score:\s*([\d.]+)\)\s*\[(.*?)\]', line)
            if match:
                section, pair, score, direction = match.groups()
                best_pairs[section] = (pair, float(score), direction)
    return best_pairs

def section_sort_key(section: str) -> int:
    # extract section number for sorting
    m = re.match(r'section_(\d+)_r01_c01', section)
    return int(m.group(1)) if m else 0

def export_best_pairs(best_pairs: Dict[str, Tuple[str, float, str]], out_f):
    out_f.write("# best pairs for each section\n")
    for section in sorted(best_pairs.keys(), key=section_sort_key):
        pair, score, direction = best_pairs[section]
        out_f.write(f"{section}   -> {pair}   (score: {score:.4f}) [{direction}]\n")
    out_f.write("\n")

def build_initial_chains(best_pairs: Dict[str, Tuple[str, float, str]]) -> List[List[str]]:
    """build initial chains, each section出发,遇到已访问就断开"""
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
    """recursively merge all chains that can be pointed to by best pair, until no more can be merged, and strictly verify the completeness of the sections"""
    changed = True
    while changed:
        changed = False
        head_map = {chain[0]: idx for idx, chain in enumerate(chains)}
        tail_map = {chain[-1]: idx for idx, chain in enumerate(chains)}
        merge_ops = []
        used = set()
        for i, chain in enumerate(chains):
            tail = chain[-1]
            # if there is another chain's head node is the best pair of tail, then merge
            for j, other in enumerate(chains):
                if i == j:
                    continue
                other_head = other[0]
                # if the best pair of tail is other_head, then tail links other chain
                if best_pairs.get(tail, (None,))[0] == other_head and (i, j) not in used:
                    merge_ops.append((i, j, 'tail->head_by_bestpair'))
                    used.add((i, j))
                # if the best pair of the tail of other chain is the head of chain, then other chain links chain
                if best_pairs.get(other[-1], (None,))[0] == chain[0] and (j, i) not in used:
                    merge_ops.append((j, i, 'tail->head_by_bestpair'))
                    used.add((j, i))
            # still keep the original tail-head, head-tail logic
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
            # i tail links j head
            if chains[i][-1] == chains[j][0]:
                new_chain = chains[i] + chains[j][1:]
            # j tail links i head
            elif chains[j][-1] == chains[i][0]:
                new_chain = chains[j] + chains[i][1:]
            # if the best pair of the tail of i chain is the head of j chain, then i chain links j chain
            elif best_pairs.get(chains[i][-1], (None,))[0] == chains[j][0]:
                new_chain = chains[i] + chains[j][1:]
            # if the best pair of the tail of j chain is the head of i chain, then j chain links i chain
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
    # deduplicate, ensure each section only appears once
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
    # verify completeness
    merged_sections = set()
    for chain in final_chains:
        merged_sections.update(chain)
    missing = all_sections - merged_sections
    if missing:
        print(f"[ERROR] missing sections: {sorted(missing, key=section_sort_key)}")
        raise RuntimeError(f"merged chains lost sections: {missing}")
    return final_chains

def export_chains(chains: List[List[str]], out_f):
    out_f.write("# merged chains\n")
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
    print("output to best_pairs_and_chains.txt")

if __name__ == "__main__":
    main() 
