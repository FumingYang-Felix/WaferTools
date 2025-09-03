#!/usr/bin/env python3
"""
connect_rigid_chains.py - connect rigid chains

algorithm:
1. read 40 rigid chains from rigid_chains_result.txt
2. extract head and tail of each chain
3. use best pairs data to try to connect these chains
4. build connection relations between chains
5. try to form larger connected components

output:
- connection relations between chains
- possible connected components
- isolated chains that cannot be connected
"""

import re
from collections import defaultdict, deque
from typing import List, Set, Dict, Tuple

def parse_rigid_chains(filename: str) -> List[List[str]]:
    """parse rigid chains file, return all chains"""
    chains = []
    
    with open(filename, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    for line in lines:
        # match format: chain X: section_A -> section_B -> section_C (length: N)
        match = re.match(r'chain \d+:\s+(.+) \(length:', line)
        if match:
            chain_str = match.group(1)
            chain = [s.strip() for s in chain_str.split(' -> ')]
            chains.append(chain)
    
    return chains

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

def get_chain_endpoints(chains: List[List[str]]) -> Dict[int, Tuple[str, str]]:
    """get head and tail of each chain"""
    endpoints = {}
    
    print("head and tail of each chain:")
    print("=" * 60)
    
    for i, chain in enumerate(chains):
        head = chain[0]
        tail = chain[-1]
        endpoints[i] = (head, tail)
        print(f"chain {i+1}: head={head}, tail={tail}")
    
    return endpoints

def find_chain_connections(chains: List[List[str]], best_pairs: Dict[str, List[str]]) -> Dict[int, List[Tuple[int, str, str]]]:
    """find connections between chains"""
    endpoints = get_chain_endpoints(chains)
    connections = defaultdict(list)
    
    print("\nfind connections between chains...")
    print("=" * 60)
    
    # check if head and tail can connect to other chains
    for i, (head, tail) in endpoints.items():
        print(f"\nchain {i+1} (head={head}, tail={tail}):")
        
        # check if head can connect to other chains
        if head in best_pairs:
            for target_section in best_pairs[head]:
                # find which chain target_section belongs to
                target_chain = find_section_chain(target_section, chains)
                if target_chain is not None and target_chain != i:
                    print(f"  head {head} -> chain {target_chain+1} of {target_section}")
                    connections[i].append((target_chain, head, target_section))
        
        # check if tail can connect to other chains
        if tail in best_pairs:
            for target_section in best_pairs[tail]:
                # find which chain target_section belongs to
                target_chain = find_section_chain(target_section, chains)
                if target_chain is not None and target_chain != i:
                    print(f"  tail {tail} -> chain {target_chain+1} of {target_section}")
                    connections[i].append((target_chain, tail, target_section))
    
    return connections

def find_section_chain(section: str, chains: List[List[str]]) -> int:
    """find which chain section belongs to"""
    for i, chain in enumerate(chains):
        if section in chain:
            return i
    return None

def build_connected_components(chains: List[List[str]], connections: Dict[int, List[Tuple[int, str, str]]]) -> List[List[int]]:
    """build connected components"""
    print("\nbuild connected components...")
    print("=" * 60)
    
    # use BFS to find all connected components
    visited = set()
    components = []
    
    for start_chain in range(len(chains)):
        if start_chain in visited:
            continue
        
        # use BFS to find all connected chains
        component = []
        queue = deque([start_chain])
        visited.add(start_chain)
        
        while queue:
            current_chain = queue.popleft()
            component.append(current_chain)
            
            # add all connected chains
            for target_chain, _, _ in connections[current_chain]:
                if target_chain not in visited:
                    visited.add(target_chain)
                    queue.append(target_chain)
        
        components.append(component)
        print(f"component {len(components)}: contains chains {[i+1 for i in component]} (total {len(component)} chains)")
    
    return components

def analyze_components(components: List[List[int]], chains: List[List[str]]) -> None:
    """analyze connected components statistics"""
    print("\n" + "=" * 60)
    print("connected components analysis")
    print("=" * 60)
    
    print(f"total components: {len(components)}")
    
    # statistics of each component size
    component_sizes = [len(comp) for comp in components]
    if component_sizes:
        print(f"component size statistics:")
        print(f"  largest component: {max(component_sizes)} chains")
        print(f"  smallest component: {min(component_sizes)} chains")
        print(f"  average size: {sum(component_sizes)/len(component_sizes):.1f} chains")
        
        # group by size
        size_groups = defaultdict(int)
        for size in component_sizes:
            size_groups[size] += 1
        
        print(f"  size distribution:")
        for size in sorted(size_groups.keys()):
            print(f"    {size} chains: {size_groups[size]} components")
    
    # show details of each component
    print(f"\ncomponent details:")
    for i, component in enumerate(components, 1):
        total_sections = sum(len(chains[chain_idx]) for chain_idx in component)
        print(f"component {i}: contains chains {[j+1 for j in component]} (total {len(component)} chains, {total_sections} sections)")
        
        # show all chains in the component
        for chain_idx in component:
            chain = chains[chain_idx]
            print(f"  chain {chain_idx+1}: {' -> '.join(chain)}")
    
    # statistics of covered sections
    all_covered_sections = set()
    for component in components:
        for chain_idx in component:
            all_covered_sections.update(chains[chain_idx])
    
    print(f"\ncovered sections statistics:")
    print(f"  covered sections: {len(all_covered_sections)}")
    print(f"  total sections: {sum(len(chain) for chain in chains)}")
    print(f"  coverage: {len(all_covered_sections)/sum(len(chain) for chain in chains)*100:.1f}%")

def main():
    # parse rigid chains
    print("parse rigid chains...")
    chains = parse_rigid_chains('rigid_chains_result.txt')
    print(f"parse completed, {len(chains)} chains")
    
    # parse best pairs
    print("parse best pairs...")
    best_pairs = parse_best_pairs('top_two_pairs_for_each_section.txt')
    print(f"parse completed, {len(best_pairs)} sections' best pairs")
    
    # get head and tail of each chain
    endpoints = get_chain_endpoints(chains)
    
    # find connections between chains
    connections = find_chain_connections(chains, best_pairs)
    
    # build connected components
    components = build_connected_components(chains, connections)
    
    # analyze results
    analyze_components(components, chains)
    
    # save results to file
    with open('chain_connections_result.txt', 'w', encoding='utf-8') as f:
        f.write("chain connections analysis results\n")
        f.write("=" * 60 + "\n\n")
        
        f.write(f"total chains: {len(chains)}\n")
        f.write(f"total components: {len(components)}\n\n")
        
        f.write("head and tail of each chain:\n")
        for i, (head, tail) in endpoints.items():
            f.write(f"chain {i+1}: head={head}, tail={tail}\n")
        
        f.write("\nconnections between chains:\n")
        for i, conns in connections.items():
            if conns:
                f.write(f"chain {i+1} connections:\n")
                for target_chain, from_section, to_section in conns:
                    f.write(f"  {from_section} -> chain {target_chain+1} of {to_section}\n")
        
        f.write("\nconnected components details:\n")
        for i, component in enumerate(components, 1):
            total_sections = sum(len(chains[chain_idx]) for chain_idx in component)
            f.write(f"component {i}: contains chains {[j+1 for j in component]} (total {len(component)} chains, {total_sections} sections)\n")
            for chain_idx in component:
                chain = chains[chain_idx]
                f.write(f"  chain {chain_idx+1}: {' -> '.join(chain)}\n")
    
    print(f"\nresults saved to chain_connections_result.txt")

if __name__ == "__main__":
    main() 
