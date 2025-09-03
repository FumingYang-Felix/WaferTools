#!/usr/bin/env python3
"""
build_rigid_chains.py - build rigid chains from best pairs

algorithm:
1. start from each section's best pair
2. if A-B is best pair, B-C is also best pair, then A-B-C is a rigid chain
3. continue to extend until no more connections
4. count all rigid chains

output:
- list of sections in each chain
- statistics of chain lengths
- number of covered sections
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

def find_rigid_chains(best_pairs: Dict[str, List[str]]) -> List[List[str]]:
    """find all rigid chains"""
    print("start building rigid chains...")
    print("=" * 60)
    
    # record processed sections
    processed = set()
    chains = []
    
    # try to build chain for each section
    for start_section in best_pairs.keys():
        if start_section in processed:
            continue
            
        print(f"\nstart building chain from {start_section}...")
        chain = build_chain_from_section(start_section, best_pairs, processed)
        
        if len(chain) > 1:  # only keep chains with length greater than 1
            chains.append(chain)
            print(f"find chain: {' -> '.join(chain)} (length: {len(chain)})")
        else:
            print(f"{start_section} cannot form a chain")
    
    return chains

def build_chain_from_section(start_section: str, best_pairs: Dict[str, List[str]], processed: Set[str]) -> List[str]:
    """build chain from specified section"""
    chain = [start_section]
    processed.add(start_section)
    
    # forward extension
    current = start_section
    while current in best_pairs and best_pairs[current]:
        next_section = best_pairs[current][0]  # get best pair
        
        # check if loop is formed
        if next_section in chain:
            print(f"  loop detected: {next_section} is already in chain")
            break
            
        # check if next_section also selects current as best pair
        if next_section in best_pairs and best_pairs[next_section]:
            if best_pairs[next_section][0] == current:
                print(f"  confirmed bidirectional connection: {current} <-> {next_section}")
                chain.append(next_section)
                processed.add(next_section)
                current = next_section
            else:
                print(f"  unidirectional connection: {current} -> {next_section} (but {next_section}'s best pair is {best_pairs[next_section][0]})")
                break
        else:
            print(f"  {next_section} has no best pair")
            break
    
    # backward extension
    current = start_section
    while current in best_pairs:
        # find section that selects current as best pair
        prev_section = None
        for section, pairs in best_pairs.items():
            if pairs and pairs[0] == current and section not in chain:
                prev_section = section
                break
        
        if prev_section is None:
            break
            
        # check if loop is formed
        if prev_section in chain:
            print(f"  loop detected: {prev_section} is already in chain")
            break
            
        print(f"  backward extension: {prev_section} -> {current}")
        chain.insert(0, prev_section)
        processed.add(prev_section)
        current = prev_section
    
    return chain

def analyze_chains(chains: List[List[str]], best_pairs: Dict[str, List[str]]) -> None:
    """analyze chain statistics"""
    print("\n" + "=" * 60)
    print("chain analysis results")
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
    
    # build rigid chains
    chains = find_rigid_chains(best_pairs)
    
    # analyze results
    analyze_chains(chains, best_pairs)
    
    # save results to file
    with open('rigid_chains_result.txt', 'w', encoding='utf-8') as f:
        f.write("rigid chains analysis results\n")
        f.write("=" * 60 + "\n\n")
        
        f.write(f"total chains: {len(chains)}\n")
        covered_sections = set()
        for chain in chains:
            covered_sections.update(chain)
        f.write(f"covered sections: {len(covered_sections)}\n")
        f.write(f"total sections: {len(best_pairs)}\n")
        f.write(f"coverage: {len(covered_sections)/len(best_pairs)*100:.1f}%\n\n")
        
        f.write("all chains details:\n")
        for i, chain in enumerate(chains, 1):
            f.write(f"chain {i}: {' -> '.join(chain)} (length: {len(chain)})\n")
        
        uncovered = set(best_pairs.keys()) - covered_sections
        if uncovered:
            f.write(f"\nuncovered sections ({len(uncovered)}):\n")
            for section in sorted(uncovered):
                f.write(f"  {section}\n")
    
    print(f"\nresults saved to rigid_chains_result.txt")

if __name__ == "__main__":
    main() 
