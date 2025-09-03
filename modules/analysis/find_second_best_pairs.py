#!/usr/bin/env python3
"""
find_second_best_pairs.py - find second best pairs for uncovered sections
"""

def load_rigid_chains():
    """load 40 rigid chains"""
    chains = [
        ['section_1_r01_c01', 'section_13_r01_c01'],
        ['section_39_r01_c01', 'section_15_r01_c01', 'section_52_r01_c01', 'section_3_r01_c01'],
        ['section_4_r01_c01', 'section_59_r01_c01'],
        ['section_5_r01_c01', 'section_99_r01_c01'],
        ['section_35_r01_c01', 'section_6_r01_c01'],
        ['section_104_r01_c01', 'section_7_r01_c01', 'section_11_r01_c01'],
        ['section_8_r01_c01', 'section_77_r01_c01'],
        ['section_9_r01_c01', 'section_64_r01_c01'],
        ['section_27_r01_c01', 'section_10_r01_c01'],
        ['section_12_r01_c01', 'section_89_r01_c01'],
        ['section_14_r01_c01', 'section_101_r01_c01'],
        ['section_29_r01_c01', 'section_16_r01_c01', 'section_76_r01_c01'],
        ['section_90_r01_c01', 'section_31_r01_c01', 'section_17_r01_c01', 'section_38_r01_c01'],
        ['section_92_r01_c01', 'section_18_r01_c01'],
        ['section_19_r01_c01', 'section_34_r01_c01'],
        ['section_21_r01_c01', 'section_78_r01_c01'],
        ['section_22_r01_c01', 'section_26_r01_c01'],
        ['section_80_r01_c01', 'section_23_r01_c01', 'section_43_r01_c01'],
        ['section_24_r01_c01', 'section_40_r01_c01'],
        ['section_25_r01_c01', 'section_74_r01_c01'],
        ['section_28_r01_c01', 'section_73_r01_c01'],
        ['section_72_r01_c01', 'section_32_r01_c01', 'section_97_r01_c01'],
        ['section_2_r01_c01', 'section_36_r01_c01', 'section_102_r01_c01'],
        ['section_42_r01_c01', 'section_41_r01_c01'],
        ['section_45_r01_c01', 'section_105_r01_c01'],
        ['section_46_r01_c01', 'section_69_r01_c01'],
        ['section_47_r01_c01', 'section_96_r01_c01'],
        ['section_48_r01_c01', 'section_71_r01_c01'],
        ['section_49_r01_c01', 'section_75_r01_c01'],
        ['section_50_r01_c01', 'section_70_r01_c01'],
        ['section_55_r01_c01', 'section_63_r01_c01'],
        ['section_56_r01_c01', 'section_88_r01_c01'],
        ['section_35_r01_c01', 'section_6_r01_c01', 'section_57_r01_c01', 'section_61_r01_c01'],
        ['section_58_r01_c01', 'section_85_r01_c01'],
        ['section_51_r01_c01', 'section_60_r01_c01', 'section_100_r01_c01'],
        ['section_27_r01_c01', 'section_10_r01_c01', 'section_66_r01_c01'],
        ['section_67_r01_c01', 'section_86_r01_c01'],
        ['section_82_r01_c01', 'section_103_r01_c01'],
        ['section_87_r01_c01', 'section_91_r01_c01'],
        ['section_93_r01_c01', 'section_94_r01_c01'],
    ]
    return chains

def get_chain_head_tail(chains):
    """get head and tail of each chain"""
    head_tail = {}
    for i, chain in enumerate(chains):
        head_tail[i+1] = (chain[0], chain[-1])
    return head_tail

def is_chain_head_tail(section, head_tail):
    """check if section is the head or tail of some chain"""
    for chain_id, (head, tail) in head_tail.items():
        if section == head or section == tail:
            return True, chain_id
    return False, None

def load_top_two_pairs():
    """load data from top_two_pairs_for_each_section.txt"""
    pairs_data = {}
    
    with open('top_two_pairs_for_each_section.txt', 'r') as f:
        lines = f.readlines()
    
    for line in lines:
        if '->' in line and 'section_' in line:
            parts = line.strip().split('->')
            if len(parts) >= 3:
                section = parts[0].strip()
                pair1_part = parts[1].strip()
                pair2_part = parts[2].strip()
                
                # parse first pair
                pair1_match = pair1_part.split('(')[0].strip()
                score1 = float(pair1_part.split('score:')[1].split(')')[0].strip())
                
                # parse second pair
                pair2_match = pair2_part.split('(')[0].strip()
                score2 = float(pair2_part.split('score:')[1].split(')')[0].strip())
                
                pairs_data[section] = {
                    'pair1': (pair1_match, score1),
                    'pair2': (pair2_match, score2)
                }
    
    return pairs_data

def find_second_best_pairs_for_uncovered():
    """find second best pairs for uncovered sections"""
    # uncovered sections
    uncovered_sections = [
        'section_106_r01_c01',
        'section_20_r01_c01', 
        'section_30_r01_c01',
        'section_37_r01_c01',
        'section_44_r01_c01',
        'section_53_r01_c01',
        'section_62_r01_c01',
        'section_65_r01_c01',
        'section_68_r01_c01',
        'section_83_r01_c01',
        'section_84_r01_c01'
    ]
    
    chains = load_rigid_chains()
    head_tail = get_chain_head_tail(chains)
    pairs_data = load_top_two_pairs()
    
    print("find second best pairs for uncovered sections")
    print("=" * 60)
    
    results = []
    
    for section in uncovered_sections:
        if section in pairs_data:
            pair1, score1 = pairs_data[section]['pair1']
            pair2, score2 = pairs_data[section]['pair2']
            
            print(f"\n{section}:")
            print(f"  first best pair: {pair1} (score: {score1})")
            print(f"  second best pair: {pair2} (score: {score2})")
            
            # check if second best pair is connected to the head or tail of some chain
            is_head_tail, chain_id = is_chain_head_tail(pair2, head_tail)
            
            if is_head_tail:
                print(f"  ✓ second best pair is connected to the head or tail of chain {chain_id}")
                results.append({
                    'section': section,
                    'pair': pair2,
                    'score': score2,
                    'chain_id': chain_id,
                    'valid': True
                })
            else:
                print(f"  ✗ second best pair is not connected to any chain's head or tail")
                results.append({
                    'section': section,
                    'pair': pair2,
                    'score': score2,
                    'chain_id': None,
                    'valid': False
                })
        else:
            print(f"\n{section}: no pair data found")
            results.append({
                'section': section,
                'pair': None,
                'score': None,
                'chain_id': None,
                'valid': False
            })
    
    # statistics
    valid_pairs = [r for r in results if r['valid']]
    invalid_pairs = [r for r in results if not r['valid']]
    
    print(f"\nstatistics:")
    print(f"valid second best pairs: {len(valid_pairs)}")
    print(f"invalid second best pairs: {len(invalid_pairs)}")
    
    if valid_pairs:
        print(f"\nvalid connections:")
        for result in valid_pairs:
            print(f"  {result['section']} -> {result['pair']} (score: {result['score']}) -> chain {result['chain_id']}")
    
    if invalid_pairs:
        print(f"\ninvalid connections:")
        for result in invalid_pairs:
            if result['pair']:
                print(f"  {result['section']} -> {result['pair']} (score: {result['score']})") 
            else:
                print(f"  {result['section']} -> no pair data")
    
    return results

def main():
    """main function"""
    results = find_second_best_pairs_for_uncovered()
    
    # save results to file
    with open('second_best_pairs_analysis.txt', 'w') as f:
        f.write("second best pairs analysis for uncovered sections\n")
        f.write("=" * 60 + "\n\n")
        
        valid_pairs = [r for r in results if r['valid']]
        invalid_pairs = [r for r in results if not r['valid']]
        
        f.write(f"valid second best pairs: {len(valid_pairs)}\n")
        f.write(f"invalid second best pairs: {len(invalid_pairs)}\n\n")
        
        if valid_pairs:
            f.write("valid connections:\n")
            for result in valid_pairs:
                f.write(f"  {result['section']} -> {result['pair']} (score: {result['score']}) -> chain {result['chain_id']}\n")
            f.write("\n")
        
        if invalid_pairs:
            f.write("invalid connections:\n")
            for result in invalid_pairs:
                if result['pair']:
                    f.write(f"  {result['section']} -> {result['pair']} (score: {result['score']})\n")
                else:
                    f.write(f"  {result['section']} -> no pair data\n")
    
    print(f"\nresults saved to second_best_pairs_analysis.txt")

if __name__ == "__main__":
    main() 
