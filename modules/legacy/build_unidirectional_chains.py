#!/usr/bin/env python3
"""
build_unidirectional_chains.py - build unidirectional chains

algorithm:
1. start from each section, build chain along unidirectional pairs
2. when a section is repeated, a connection point is found
3. merge chains with common sections
4. count all possible chain combinations

output:
- all possible chains
- connection points of chains
- covered sections
"""

import re
from collections import defaultdict, deque
from typing import List, Set, Dict, Tuple

def parse_best_pairs(filename: str) -> Dict[str, List[str]]:
    """parse best pairs file, return best pairs for each section"""
    best_pairs = {}
    
    with open(filename, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    for line in lines:
        # match format: section_X -> section_Y (score: Z) [direction]
        match = re.match(r'(\w+)\s+->\s+(\w+)\s+\(score:\s+[\d.]+\)', line)
        if match:
            section1, section2 = match.groups()
            if section1 not in best_pairs:
                best_pairs[section1] = []
            best_pairs[section1].append(section2)
    
    return best_pairs

def build_forward_chain(start_section: str, best_pairs: Dict[str, List[str]], visited: Set[str]) -> List[str]:
    """build chain forward from specified section"""
    chain = [start_section]
    visited.add(start_section)
    current = start_section
    
    while current in best_pairs and best_pairs[current]:
        next_section = best_pairs[current][0]
        
        # if a section is visited, a connection point is found
        if next_section in visited:
            print(f"  connection point found: {next_section} (already in chain)")
            break
            
        chain.append(next_section)
        visited.add(next_section)
        current = next_section
    
    return chain

def build_backward_chain(start_section: str, best_pairs: Dict[str, List[str]], visited: Set[str]) -> List[str]:
    """build chain backward from specified section"""
    chain = []
    current = start_section
    
    while True:
        # find section that selects current as best pair
        prev_section = None
        for section, pairs in best_pairs.items():
            if pairs and pairs[0] == current and section not in visited:
                prev_section = section
                break
        
        if prev_section is None:
            break
            
        # if a section is visited, a connection point is found
        if prev_section in visited:
            print(f"  connection point found: {prev_section} (already in chain)")
            break
            
        chain.insert(0, prev_section)
        visited.add(prev_section)
        current = prev_section
    
    return chain

def find_all_chains(best_pairs: Dict[str, List[str]]) -> List[List[str]]:
    """find all possible chains"""
    print("start building unidirectional chains...")
    print("=" * 60)
    
    all_chains = []
    processed = set()
    
    # try to build chain for each section
    for start_section in best_pairs.keys():
        if start_section in processed:
            continue
            
        print(f"\nstart building chain from {start_section}...")
        
        # build complete chain (forward + backward)
        visited = set()
        backward_chain = build_backward_chain(start_section, best_pairs, visited)
        forward_chain = build_forward_chain(start_section, best_pairs, visited)
        
        # merge chains
        full_chain = backward_chain + forward_chain[1:]  # avoid repeating start_section
        
        if len(full_chain) > 1:
            all_chains.append(full_chain)
            print(f"find chain: {' -> '.join(full_chain)} (length: {len(full_chain)})")
            
            # mark all sections in chain as processed
            processed.update(full_chain)
        else:
            print(f"{start_section} cannot form a chain")
    
    return all_chains

def find_chain_connections(chains: List[List[str]]) -> List[Tuple[int, int, str]]:
    """find connection points between chains"""
    print(f"\nfind connection points between chains...")
    print("=" * 60)
    
    connections = []
    
    for i in range(len(chains)):
        for j in range(i + 1, len(chains)):
            chain1 = chains[i]
            chain2 = chains[j]
            
            # find common sections
            common_sections = set(chain1) & set(chain2)
            
            for common_section in common_sections:
                connections.append((i, j, common_section))
                print(f"chain {i+1} and chain {j+1} are connected by {common_section}")
    
    return connections

def merge_connected_chains(chains: List[List[str]], connections: List[Tuple[int, int, str]]) -> List[List[str]]:
    """merge chains with connections"""
    print(f"\nmerge connected chains...")
    print("=" * 60)
    
    if not connections:
        print("no connected chains found")
        return chains
    
    # use union-find to merge chains
    parent = list(range(len(chains)))
    
    def find(x):
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]
    
    def union(x, y):
        px, py = find(x), find(y)
        if px != py:
            parent[px] = py
    
    # merge chains with connections
    for i, j, _ in connections:
        union(i, j)
    
    # collect chains by group
    groups = defaultdict(list)
    for i in range(len(chains)):
        groups[find(i)].append(i)
    
    # merge chains within each group
    merged_chains = []
    for group_indices in groups.values():
        if len(group_indices) == 1:
            # single chain, add directly
            merged_chains.append(chains[group_indices[0]])
        else:
            # multiple chains, need to merge
            print(f"merge chain group: {[i+1 for i in group_indices]}")
            merged_chain = merge_chain_group([chains[i] for i in group_indices])
            merged_chains.append(merged_chain)
    
    return merged_chains

def merge_chain_group(chain_group: List[List[str]]) -> List[str]:
    """merge a group of chains"""
    if len(chain_group) == 1:
        return chain_group[0]
    
    # simple merge strategy: find connection points, then merge
    # here can implement more complex merge logic
    all_sections = set()
    for chain in chain_group:
        all_sections.update(chain)
    
    # sort by section number (simplified processing)
    sorted_sections = sorted(all_sections, key=lambda x: int(x.split('_')[1]))
    
    print(f"  merged chain: {' -> '.join(sorted_sections)} (length: {len(sorted_sections)})")
    return sorted_sections

def analyze_chains(chains: List[List[str]], best_pairs: Dict[str, List[str]]) -> None:
    """analyze chain statistics"""
    print(f"\nchain analysis results")
    print("=" * 60)
    
    # statistics
    chain_lengths = [len(chain) for chain in chains]
    covered_sections = set()
    for chain in chains:
        covered_sections.update(chain)
    
    print(f"total chains: {len(chains)}")
    print(f"covered sections: {len(covered_sections)}")
    print(f"total sections: {len(best_pairs)}")
    print(f"coverage: {len(covered_sections)/len(best_pairs)*100:.1f}%")
    
    if chain_lengths:
        print(f"chain length statistics:")
        print(f"  longest chain: {max(chain_lengths)} sections")
        print(f"  shortest chain: {min(chain_lengths)} sections")
        print(f"  average length: {sum(chain_lengths)/len(chain_lengths):.1f} sections")
        
        # group by length
        length_groups = defaultdict(int)
        for length in chain_lengths:
            length_groups[length] += 1
        
        print(f"  length distribution:")
        for length in sorted(length_groups.keys()):
            print(f"    {length} sections: {length_groups[length]} chains")
    
    # show all chains
    print(f"\nall chains details:")
    for i, chain in enumerate(chains, 1):
        print(f"chain {i}: {' -> '.join(chain)} (length: {len(chain)})")
    
    # show uncovered sections
    uncovered = set(best_pairs.keys()) - covered_sections
    if uncovered:
        print(f"\nuncovered sections ({len(uncovered)}):")
        for section in sorted(uncovered):
            print(f"  {section}")

def main():
    # parse best pairs file
    print("parse best pairs file...")
    best_pairs = parse_best_pairs('top_two_pairs_for_each_section.txt')
    print(f"parse completed, {len(best_pairs)} sections")
    
    # build all chains
    chains = find_all_chains(best_pairs)
    
    # find connections between chains
    connections = find_chain_connections(chains)
    
    # merge connected chains
    merged_chains = merge_connected_chains(chains, connections)
    
    # analyze results
    analyze_chains(merged_chains, best_pairs)
    
    # save results to file
    with open('unidirectional_chains_result.txt', 'w', encoding='utf-8') as f:
        f.write("unidirectional chains analysis results\n")
        f.write("=" * 60 + "\n\n")
        
        f.write(f"original chains: {len(chains)}\n")
        f.write(f"merged chains: {len(merged_chains)}\n")
        f.write(f"connection points: {len(connections)}\n\n")
        
        covered_sections = set()
        for chain in merged_chains:
            covered_sections.update(chain)
        f.write(f"covered sections: {len(covered_sections)}\n")
        f.write(f"total sections: {len(best_pairs)}\n")
        f.write(f"coverage: {len(covered_sections)/len(best_pairs)*100:.1f}%\n\n")
        
        f.write("all chains details:\n")
        for i, chain in enumerate(merged_chains, 1):
            f.write(f"chain {i}: {' -> '.join(chain)} (length: {len(chain)})\n")
        
        uncovered = set(best_pairs.keys()) - covered_sections
        if uncovered:
            f.write(f"\nuncovered sections ({len(uncovered)}):\n")
            for section in sorted(uncovered):
                f.write(f"  {section}\n")
    
    print(f"\nresults saved to unidirectional_chains_result.txt")

if __name__ == "__main__":
    main() 
