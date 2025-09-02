#!/usr/bin/env python3
"""
find_second_best_pairs.py - 为未覆盖的sections找到第二好的pair
"""

def load_rigid_chains():
    """加载40个刚性链"""
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
    """获取每个链的头尾节点"""
    head_tail = {}
    for i, chain in enumerate(chains):
        head_tail[i+1] = (chain[0], chain[-1])
    return head_tail

def is_chain_head_tail(section, head_tail):
    """检查section是否是某个链的头或尾"""
    for chain_id, (head, tail) in head_tail.items():
        if section == head or section == tail:
            return True, chain_id
    return False, None

def load_top_two_pairs():
    """加载top_two_pairs_for_each_section.txt的数据"""
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
                
                # 解析第一个pair
                pair1_match = pair1_part.split('(')[0].strip()
                score1 = float(pair1_part.split('score:')[1].split(')')[0].strip())
                
                # 解析第二个pair
                pair2_match = pair2_part.split('(')[0].strip()
                score2 = float(pair2_part.split('score:')[1].split(')')[0].strip())
                
                pairs_data[section] = {
                    'pair1': (pair1_match, score1),
                    'pair2': (pair2_match, score2)
                }
    
    return pairs_data

def find_second_best_pairs_for_uncovered():
    """为未覆盖的sections找到第二好的pair"""
    # 未覆盖的sections
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
    
    print("为未覆盖的sections寻找第二好的pair")
    print("=" * 60)
    
    results = []
    
    for section in uncovered_sections:
        if section in pairs_data:
            pair1, score1 = pairs_data[section]['pair1']
            pair2, score2 = pairs_data[section]['pair2']
            
            print(f"\n{section}:")
            print(f"  第一好pair: {pair1} (score: {score1})")
            print(f"  第二好pair: {pair2} (score: {score2})")
            
            # 检查第二好pair是否连接到链的头尾
            is_head_tail, chain_id = is_chain_head_tail(pair2, head_tail)
            
            if is_head_tail:
                print(f"  ✓ 第二好pair连接到链{chain_id}的头尾")
                results.append({
                    'section': section,
                    'pair': pair2,
                    'score': score2,
                    'chain_id': chain_id,
                    'valid': True
                })
            else:
                print(f"  ✗ 第二好pair不连接到任何链的头尾")
                results.append({
                    'section': section,
                    'pair': pair2,
                    'score': score2,
                    'chain_id': None,
                    'valid': False
                })
        else:
            print(f"\n{section}: 没有找到配对数据")
            results.append({
                'section': section,
                'pair': None,
                'score': None,
                'chain_id': None,
                'valid': False
            })
    
    # 统计结果
    valid_pairs = [r for r in results if r['valid']]
    invalid_pairs = [r for r in results if not r['valid']]
    
    print(f"\n统计结果:")
    print(f"有效的第二好pair: {len(valid_pairs)}")
    print(f"无效的第二好pair: {len(invalid_pairs)}")
    
    if valid_pairs:
        print(f"\n有效的连接:")
        for result in valid_pairs:
            print(f"  {result['section']} -> {result['pair']} (score: {result['score']}) -> 链{result['chain_id']}")
    
    if invalid_pairs:
        print(f"\n无效的连接:")
        for result in invalid_pairs:
            if result['pair']:
                print(f"  {result['section']} -> {result['pair']} (score: {result['score']})")
            else:
                print(f"  {result['section']} -> 无配对数据")
    
    return results

def main():
    """主函数"""
    results = find_second_best_pairs_for_uncovered()
    
    # 保存结果到文件
    with open('second_best_pairs_analysis.txt', 'w') as f:
        f.write("未覆盖sections的第二好pair分析\n")
        f.write("=" * 60 + "\n\n")
        
        valid_pairs = [r for r in results if r['valid']]
        invalid_pairs = [r for r in results if not r['valid']]
        
        f.write(f"有效的第二好pair: {len(valid_pairs)}\n")
        f.write(f"无效的第二好pair: {len(invalid_pairs)}\n\n")
        
        if valid_pairs:
            f.write("有效的连接:\n")
            for result in valid_pairs:
                f.write(f"  {result['section']} -> {result['pair']} (score: {result['score']}) -> 链{result['chain_id']}\n")
            f.write("\n")
        
        if invalid_pairs:
            f.write("无效的连接:\n")
            for result in invalid_pairs:
                if result['pair']:
                    f.write(f"  {result['section']} -> {result['pair']} (score: {result['score']})\n")
                else:
                    f.write(f"  {result['section']} -> 无配对数据\n")
    
    print(f"\n结果已保存到 second_best_pairs_analysis.txt")

if __name__ == "__main__":
    main() 