#!/usr/bin/env python3
"""
verify_all_sections.py - verify if each section can form a chain

simple verification: each section should be able to form a chain of at least length 2
"""

import re
from typing import Dict, List, Set

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

def verify_section_chain(section: str, best_pairs: Dict[str, List[str]]) -> List[str]:
    """verify if a single section can form a chain"""
    if section not in best_pairs or not best_pairs[section]:
        return []
    
    # get best pair
    best_pair = best_pairs[section][0]
    
    # check if best pair has a pair
    if best_pair in best_pairs and best_pairs[best_pair]:
        return [section, best_pair]
    else:
        return [section]

def main():
    # parse best pairs file
    print("parsing best pairs file...")
    best_pairs = parse_best_pairs('top_two_pairs_for_each_section.txt')
    print(f"parsing completed, total {len(best_pairs)} sections")
    
    print("\nverifying each section's chain...")
    print("=" * 60)
    
    all_chains = []
    covered_sections = set()
    
    for section in sorted(best_pairs.keys()):
        chain = verify_section_chain(section, best_pairs)
        
        if len(chain) >= 2:
            all_chains.append(chain)
            covered_sections.update(chain)
            print(f"{section}: {' -> '.join(chain)} (length: {len(chain)})")
        else:
            print(f"{section}: cannot form a chain")
    
    print(f"\nstatistics:")
    print(f"total sections: {len(best_pairs)}")
    print(f"sections that can form a chain: {len(covered_sections)}")
    print(f"coverage: {len(covered_sections)/len(best_pairs)*100:.1f}%")
    
    # check uncovered sections
    uncovered = set(best_pairs.keys()) - covered_sections
    if uncovered:
        print(f"\nuncovered sections ({len(uncovered)}):")
        for section in sorted(uncovered):
            if section in best_pairs and best_pairs[section]:
                best_pair = best_pairs[section][0]
                print(f"  {section} -> {best_pair} (but {best_pair} has no best pair)")
            else:
                print(f"  {section} (has no best pair)")
    
    # check if there are sections without best pairs
    sections_without_pairs = []
    for section in best_pairs.keys():
        if not best_pairs[section]:
            sections_without_pairs.append(section)
    
    if sections_without_pairs:
        print(f"\nsections without best pairs ({len(sections_without_pairs)}):")
        for section in sections_without_pairs:
            print(f"  {section}")

if __name__ == "__main__":
    main() 
