import re
import csv
import pandas as pd
import sys
import argparse
from collections import Counter, defaultdict
from itertools import combinations
from typing import List, Dict, Tuple, Optional


# Legacy pattern kept for callers that still expect it; main path no longer depends on it.
NAME_PAT = r'[^\s,->()]+'
SEC_PAT = re.compile(NAME_PAT)
# "A -> B" or "A -> B (score: ...)" — names may contain almost anything except "->"
ARROW_LINE_PAT = re.compile(
    r'^\s*(?:chain\d+\s*:\s*)?(.+?)\s*->\s*(.+?)(?:\s*\(|\s*$)',
    re.I,
)
SCORE_PAT = re.compile(r'\bscore\s*:\s*([0-9.]+)', re.I)


def section_sort_key(section: str):
    """Natural-ish sort: first integer in the name if present, else lexicographic."""
    nums = re.findall(r'\d+', str(section))
    if nums:
        return (0, int(nums[0]), str(section))
    return (1, 0, str(section))


def _normalize_section_id(value) -> str:
    return str(value).strip()


def find_best_pairs_from_csv(csvfile):
    best_scores = {}
    best_pairs = {}
    second_best_scores = {}
    second_best_pairs = {}
    with open(csvfile, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            fixed = _normalize_section_id(row['fixed'])
            moving = _normalize_section_id(row['moving'])
            if not fixed or not moving:
                continue
            score = float(row['score'])
            # fixed perspective
            if fixed not in best_scores or score > best_scores[fixed]:
                # if current score is higher, original best becomes second best
                if fixed in best_scores:
                    second_best_scores[fixed] = best_scores[fixed]
                    second_best_pairs[fixed] = best_pairs[fixed]
                best_scores[fixed] = score
                best_pairs[fixed] = (moving, score)
            elif fixed not in second_best_scores or score > second_best_scores[fixed]:
                # if current score is second highest
                second_best_scores[fixed] = score
                second_best_pairs[fixed] = (moving, score)
            # moving perspective
            if moving not in best_scores or score > best_scores[moving]:
                # if current score is higher, original best becomes second best
                if moving in best_scores:
                    second_best_scores[moving] = best_scores[moving]
                    second_best_pairs[moving] = best_pairs[moving]
                best_scores[moving] = score
                best_pairs[moving] = (fixed, score)
            elif moving not in second_best_scores or score > second_best_scores[moving]:
                # if current score is second highest
                second_best_scores[moving] = score
                second_best_pairs[moving] = (fixed, score)
    return best_pairs, second_best_pairs


def _chains_from_edges(edges: List[Tuple[str, str]], hub_threshold: int = 3) -> List[List[str]]:
    """Build linear chains from undirected best-pair edges (name-agnostic)."""
    graph = defaultdict(set)
    for a, b in edges:
        if not a or not b or a == b:
            continue
        graph[a].add(b)
        graph[b].add(a)
    counts = Counter(target for _, target in edges)
    hubs = {n for n, c in counts.items() if c >= hub_threshold}
    for h in hubs:
        for nbr in list(graph[h]):
            graph[nbr].discard(h)
        graph.pop(h, None)
    visited, chains = set(), []

    def walk(start):
        chain, prev, cur = [start], None, start
        visited.add(cur)
        while True:
            nxt = [x for x in graph[cur] if x != prev and x not in hubs]
            if len(nxt) != 1 or nxt[0] in visited:
                break
            prev, cur = cur, nxt[0]
            chain.append(cur)
            visited.add(cur)
        return chain

    for node, nbrs in graph.items():
        if node not in visited and len(nbrs) <= 1:
            chains.append(walk(node))
    for node in graph:
        if node not in visited:
            chains.append(walk(node))

    chains.sort(key=lambda c: section_sort_key(c[0]))
    return chains


def build_best_pair_chains_from_pairs(best_pairs: Dict[str, Tuple[str, float]],
                                      hub_threshold: int = 3) -> List[List[str]]:
    """Build chains directly from best-pair dict — works with any section ID strings."""
    edges = [(src, dst) for src, (dst, _score) in best_pairs.items()]
    return _chains_from_edges(edges, hub_threshold=hub_threshold)


def build_best_pair_chains(text, hub_threshold=3):
    """Legacy text parser: split on '->' so any section name is accepted."""
    edges = []
    for line in str(text).splitlines():
        m = ARROW_LINE_PAT.match(line)
        if not m:
            continue
        a = _normalize_section_id(m.group(1))
        b = _normalize_section_id(m.group(2))
        # drop trailing junk like "no second best pair"
        if ' ' in b and 'score' not in line.lower():
            b = b.split()[0]
        edges.append((a, b))
    return _chains_from_edges(edges, hub_threshold=hub_threshold)


def second_best_to_candidates(
    second_best_pairs: Dict[str, Tuple[str, float]],
) -> Dict[str, List[Tuple[str, float]]]:
    """Convert second-best dict into stitch_chains candidate map."""
    return {sec: [pair] for sec, pair in second_best_pairs.items()}

# ------------------------- Stitch Core -------------------------

def build_index(chains: List[List[str]]):
    """section -> (chain_id, 'head'|'tail'|'middle')"""
    idx = {}
    for cid, ch in enumerate(chains):
        for i, sec in enumerate(ch):
            pos = 'head' if i == 0 else 'tail' if i == len(ch) - 1 else 'middle'
            idx[sec] = (cid, pos)
    return idx

def merge_two(chains, cid_a, e1, cid_b, e2):
    """
    Ensure e1 is tail of A, e2 is head of B, then concatenate directly
    Reverse operations only occur at chain head↔tail
    """
    A, B = chains[cid_a], chains[cid_b]

    if A[0] == e1:      # e1 at head, reverse A
        A.reverse()
    if B[-1] == e2:     # e2 at tail, reverse B
        B.reverse()

    assert A[-1] == e1 and B[0] == e2
    chains[cid_a] = A + B
    chains[cid_b] = []          # clear merged chain, will be filtered later
    return chains

def stitch_chains(primary_chains: List[List[str]],
                  sec_candidates: Dict[str, List[Tuple[str, float]]],
                  verbose=False) -> List[List[str]]:
    """
    Round by round: high-score endpoint-endpoint edges merge first; rebuild endpoints after each success
    Returns concatenated super-chains list (internal order locked)
    """
    chains = [ch[:] for ch in primary_chains]   # deep copy

    while True:
        idx = build_index(chains)
        endpoints = {s: info for s, info in idx.items() if info[1] in ('head', 'tail')}

        # —— Generate candidate endpoint pairs, keep highest score —— #
        edges = {}
        for sec, (cid, _) in endpoints.items():
            for partner, score in sec_candidates.get(sec, []):
                if partner not in endpoints:            # other side not open endpoint
                    continue
                cid2, _ = endpoints[partner]
                if cid == cid2:                         # same chain, skip
                    continue
                edge = tuple(sorted((sec, partner)))    # undirected deduplication
                # take the higher score one
                if edge not in edges or score > edges[edge][2]:
                    edges[edge] = (sec, partner, score)

        if not edges:
            break   # no candidates → converge

        merged_this_round = False
        # —— Try concatenation by score descending order —— #
        for secA, secB, score in sorted(edges.values(), key=lambda x: -x[2]):
            idx = build_index(chains)                   # rebuild index after each merge
            if secA not in idx or secB not in idx:
                continue                                # endpoint already occupied
            cidA, posA = idx[secA]
            cidB, posB = idx[secB]
            if cidA == cidB or not chains[cidA] or not chains[cidB]:
                continue                                # same chain or already empty

            if verbose:
                print(f"[merge {score:.2f}] {secA} ({posA})  +  {secB} ({posB})")

            chains = merge_two(chains, cidA, secA, cidB, secB)
            merged_this_round = True
            break                                       # immediately enter next round

        if not merged_this_round:
            break                                       # no successful merge this round → end

    # remove empty chains
    return [ch for ch in chains if ch]

# ------------------------- Parsing Tools -------------------------

# === parse_primary / parse_secondary: name-agnostic (split on '->') ===
def parse_primary(raw: str):
    """Parse chain lines of the form 'chain1: A -> B -> C' (any section IDs)."""
    chains = []
    for line in raw.strip().splitlines():
        if '->' not in line:
            continue
        rest = line.split(':', 1)[1] if ':' in line else line
        rest = SCORE_PAT.sub('', rest)
        ids = [_normalize_section_id(p) for p in rest.split('->')]
        ids = [p for p in ids if p]
        if ids:
            chains.append(ids)
    return chains

def parse_secondary(raw: str):
    """Parse 'A -> B (score: x)' lines; accepts any section ID strings."""
    sec_dict = defaultdict(list)
    for line in raw.strip().splitlines():
        m = ARROW_LINE_PAT.match(line)
        if not m:
            continue
        a = _normalize_section_id(m.group(1))
        b = _normalize_section_id(m.group(2))
        if not a or not b or b.lower().startswith('no '):
            continue
        score = float(SCORE_PAT.search(line).group(1)) if SCORE_PAT.search(line) else 0.0
        sec_dict[a].append((b, score))
    for k, lst in sec_dict.items():
        lst.sort(key=lambda x: -x[1])
    return sec_dict

# ------------------------------------------------------------
# 1. Read CSV ➜ Endpoint pair score dictionary
# ------------------------------------------------------------
def load_pair_scores(csv_path: str) -> Dict[frozenset, float]:
    """
    Read CSV file
    Must contain columns: 'fixed', 'moving', 'score'
    Returns dict{frozenset({sec1, sec2}): max_score}
    (fixed, moving) and (moving, fixed) treated as same edge, take highest score
    """
    df = pd.read_csv(csv_path)
    score_dict: Dict[frozenset, float] = {}
    for _, row in df.iterrows():
        fixed = _normalize_section_id(row["fixed"])
        moving = _normalize_section_id(row["moving"])
        if not fixed or not moving:
            continue
        key = frozenset((fixed, moving))
        sc  = float(row["score"])
        if key not in score_dict or sc > score_dict[key]:
            score_dict[key] = sc
    return score_dict

# ------------------------------------------------------------
# 2. Evaluate best connection scheme between two super-chains
# ------------------------------------------------------------
def best_link_for_two(chainA: List[str],
                      chainB: List[str],
                      score_dict: Dict[frozenset, float],
                      fallback_depth: int = 1
                      ) -> Optional[Tuple[float, str, str, str, str]]:
    """
    考察 4×4 (=16) 端点组合：
        A:{head0, head1, tail0, tail1} × B:{head0, head1, tail0, tail1}
    返回 (score, nodeA, posA, nodeB, posB) —— 其中 pos∈{'head','tail'}
    """

    def endpoint_nodes(chain):
        """返回 [(node, 'head'/'tail'), ...]"""
        nodes = [(chain[0], 'head'), (chain[-1], 'tail')]
        if fallback_depth > 0 and len(chain) > fallback_depth:
            nodes.append((chain[fallback_depth], 'head'))
            nodes.append((chain[-1 - fallback_depth], 'tail'))
        # 去重，保持顺序
        seen, uniq = set(), []
        for n, p in nodes:
            if n not in seen:
                uniq.append((n, p))
                seen.add(n)
        return uniq

    nodesA = endpoint_nodes(chainA)
    nodesB = endpoint_nodes(chainB)

    best = None  # (score, nA, posA, nB, posB)
    for nA, posA in nodesA:
        for nB, posB in nodesB:
            if nA == nB:
                continue
            sc = score_dict.get(frozenset((nA, nB)))
            if sc is None:
                continue
            if (best is None) or (sc > best[0]):
                best = (sc, nA, posA, nB, posB)

    if best is None:
        return None

    sc, _, posA, _, posB = best
    # 把 pos 映射回真正链端（head→chain[0]，tail→chain[-1]）
    nodeA_end = chainA[0] if posA == 'head' else chainA[-1]
    nodeB_end = chainB[0] if posB == 'head' else chainB[-1]
    return sc, nodeA_end, posA, nodeB_end, posB


# ------------------------------------------------------------
# 3. Connect all super-chains into one total chain
# ------------------------------------------------------------
def link_super_chains(super_chains: List[List[str]],
                      csv_path: str,
                      verbose: bool = True,
                      fallback_depth: int = 1
                      ) -> List[str]:
    """
    super_chains :  several chains with locked order
    csv_path     :  CSV file path
    returns      :  final connected single chain (list[str])
                     if still multiple unconnectable, returns residual chain list
    """
    score_dict = load_pair_scores(csv_path)
    chains = [ch[:] for ch in super_chains]                # deep copy

    while len(chains) > 1:
        best_global = None   # (score, idA, idB, nodeA, posA, nodeB, posB)

        # —— Traverse all chain pairs, pick global highest score —— #
        for (i, A), (j, B) in combinations(enumerate(chains), 2):
            best = best_link_for_two(A, B, score_dict, fallback_depth)
            if best:
                sc, nA, posA, nB, posB = best
                if (best_global is None) or (sc > best_global[0]):
                    best_global = (sc, i, j, nA, posA, nB, posB)

        if best_global is None:
            if verbose:
                print("⚠️  No more valid endpoint pairs, stopping.")
            break

        # —— Execute highest score merge —— #
        sc, idA, idB, nA, posA, nB, posB = best_global
        if verbose:
            print(f"[LINK {sc:.2f}] chain{idA} ({posA}:{nA})  ⟶  "
                  f"chain{idB} ({posB}:{nB})")

        A, B = chains[idA], chains[idB]

        # Make nA tail of A; nB head of B
        if posA == 'head':
            A.reverse()
        if posB == 'tail':
            B.reverse()

        chains[idA] = A + B
        chains.pop(idB)      # delete merged chain

    if not chains:
        if verbose:
            print("⚠️  No chains to link (empty input).")
        return []
    return chains if len(chains) > 1 else chains[0]

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Build chains from pairwise alignment results')
    parser.add_argument('--csv', '-c', type=str, default='new_pairwise_filtered.csv',
                       help='Input CSV file path (default: new_pairwise_filtered.csv)')
    parser.add_argument('--output', '-o', type=str, default='best_pair_chains_graph.txt',
                       help='Output text file path (default: best_pair_chains_graph.txt)')
    
    args = parser.parse_args()
    
    csvfile = args.csv
    output_file = args.output
    
    print(f"Using CSV file: {csvfile}")
    print(f"Output will be saved to: {output_file}")
    
    best_pairs, second_best_pairs = find_best_pairs_from_csv(csvfile)
    sorted_sections = sorted(best_pairs.keys(), key=section_sort_key)
    # step1: print each section's best pair
    print('step1: each section\'s best pair')
    best_pair_lines = []
    for section in sorted_sections:
        moving, score = best_pairs[section]
        line = f'{section} -> {moving} (score: {score:.4f})'
        print(line)
        best_pair_lines.append(line)
    print('\nstep2: chain grouping (undirected graph method, each section appears only once)')
    # Use CSV IDs directly — no regex name whitelist
    chains = build_best_pair_chains_from_pairs(best_pairs)
    for i, chain in enumerate(chains, 1):
        print(f'chain{i}: ' + ' -> '.join(chain))
    # step3: print each section's second best pair
    print('\nstep3: each section\'s second best pair')
    second_best_lines = []
    for section in sorted_sections:
        if section in second_best_pairs:
            moving, score = second_best_pairs[section]
            line = f'{section} -> {moving} (score: {score:.4f})'
            print(line)
            second_best_lines.append(line)
        else:
            line = f'{section} -> no second best pair'
            print(line)
            second_best_lines.append(line)
    # step4: stitch chains (dict path; avoids brittle text re-parsing)
    print('\nstep4: merge chains using stitch algorithm')
    sec_candidates = second_best_to_candidates(second_best_pairs)
    super_chains = stitch_chains(chains, sec_candidates, verbose=True)
    print("\n=== Final Chains ===")
    for i, ch in enumerate(super_chains, 1):
        print(f"chain{i:02d} ({len(ch)}): " + " -> ".join(ch))
    # step5: link super chains
    print('\nstep5: connect super chains into final chain')
    final_chain = link_super_chains(super_chains, csvfile, verbose=True)
    if isinstance(final_chain, list) and len(final_chain) > 0:
        if isinstance(final_chain[0], list):
            # returns chain list
            print("\n=== Final Connected Chains ===")
            for i, ch in enumerate(final_chain, 1):
                print(f"final chain{i:02d} ({len(ch)}): " + " -> ".join(ch))
        else:
            # returns single chain
            print(f"\n=== Final Single Chain ({len(final_chain)} sections) ===")
            print(" -> ".join(final_chain))
    else:
        print("Cannot connect into final chain")
    # also write to txt
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('=' * 80 + '\n')
        f.write('STEP 1: EACH SECTION\'S BEST PAIR\n')
        f.write('=' * 80 + '\n')
        for line in best_pair_lines:
            f.write(line + '\n')
        
        f.write('\n' + '=' * 80 + '\n')
        f.write('STEP 2: CHAIN GROUPING (UNDIRECTED GRAPH METHOD)\n')
        f.write('=' * 80 + '\n')
        for i, chain in enumerate(chains, 1):
            f.write(f'chain{i}: ' + ' -> '.join(chain) + '\n')
        
        f.write('\n' + '=' * 80 + '\n')
        f.write('STEP 3: EACH SECTION\'S SECOND BEST PAIR\n')
        f.write('=' * 80 + '\n')
        for line in second_best_lines:
            f.write(line + '\n')
        
        f.write('\n' + '=' * 80 + '\n')
        f.write('STEP 4: FINAL CHAINS AFTER STITCH ALGORITHM MERGE\n')
        f.write('=' * 80 + '\n')
        for i, ch in enumerate(super_chains, 1):
            f.write(f"chain{i:02d} ({len(ch)}): " + " -> ".join(ch) + '\n')
        
        f.write('\n' + '=' * 80 + '\n')
        f.write('STEP 5: FINAL CONNECTION RESULT\n')
        f.write('=' * 80 + '\n')
        if isinstance(final_chain, list) and len(final_chain) > 0:
            if isinstance(final_chain[0], list):
                for i, ch in enumerate(final_chain, 1):
                    f.write(f"final chain{i:02d} ({len(ch)}): " + " -> ".join(ch) + '\n')
            else:
                f.write(f"final single chain ({len(final_chain)}): " + " -> ".join(final_chain) + '\n')
        else:
            f.write("cannot connect into final chain\n")
        
        f.write('\n' + '=' * 80 + '\n')
        f.write('SUMMARY\n')
        f.write('=' * 80 + '\n')
        f.write(f'Total sections processed: {len(sorted_sections)}\n')
        f.write(f'Initial chains from step 2: {len(chains)}\n')
        f.write(f'Super chains after step 4: {len(super_chains)}\n')
        if isinstance(final_chain, list) and len(final_chain) > 0:
            if isinstance(final_chain[0], list):
                f.write(f'Final chains after step 5: {len(final_chain)}\n')
                total_sections = sum(len(ch) for ch in final_chain)
                f.write(f'Total sections in final result: {total_sections}\n')
            else:
                f.write(f'Final single chain with {len(final_chain)} sections\n')
        f.write('=' * 80 + '\n')
    print(f'output saved to {output_file}')

if __name__ == '__main__':
    main() 
