#!/usr/bin/env python3
"""
find_best_pairs.py - 找出每个section的最佳配对（最高score）
"""

import pandas as pd
import numpy as np

def load_csv(csv_file):
    """加载CSV文件"""
    df = pd.read_csv(csv_file)
    print(f"加载CSV文件: {csv_file}")
    print(f"总行数: {len(df)}")
    return df

def get_all_sections(df):
    """获取所有唯一的sections，按数字顺序排序"""
    all_sections = set(df['fixed'].unique()) | set(df['moving'].unique())
    
    # 按section数字排序
    def extract_number(section_name):
        try:
            # 提取section_后面的数字
            return int(section_name.split('_')[1])
        except:
            return 999  # 如果无法解析，放在最后
    
    sorted_sections = sorted(all_sections, key=extract_number)
    print(f"总共有 {len(sorted_sections)} 个唯一sections")
    return sorted_sections

def find_top_two_pairs_for_section(section, df):
    """找出指定section的前两个最佳配对"""
    # 查找包含该section的所有行
    mask_fixed = df['fixed'] == section
    mask_moving = df['moving'] == section
    
    # 合并所有相关行
    relevant_rows = df[mask_fixed | mask_moving].copy()
    
    if len(relevant_rows) == 0:
        return []
    
    # 收集所有配对和分数
    pairs = []
    for _, row in relevant_rows.iterrows():
        if row['fixed'] == section:
            other_section = row['moving']
            score = row['score']
            direction = "fixed->moving"
        else:
            other_section = row['fixed']
            score = row['score']
            direction = "moving->fixed"
        
        pairs.append({
            'other_section': other_section,
            'score': score,
            'direction': direction
        })
    
    # 按分数排序，取前两个
    pairs.sort(key=lambda x: x['score'], reverse=True)
    return pairs[:2]

def main():
    # 加载数据
    df = load_csv('new_pairwise_filtered.csv')
    
    # 获取所有sections
    all_sections = get_all_sections(df)
    
    # 找出每个section的前两个最佳配对
    results = []
    
    print(f"\n开始分析每个section的前两个最佳配对...")
    
    for section in all_sections:
        top_pairs = find_top_two_pairs_for_section(section, df)
        
        results.append({
            'section': section,
            'pairs': top_pairs
        })
        
        if top_pairs:
            print(f"{section:20s}:")
            for i, pair in enumerate(top_pairs, 1):
                print(f"  {i}. -> {pair['other_section']:20s} (score: {pair['score']:.4f}) [{pair['direction']}]")
        else:
            print(f"{section:20s}: 无配对")
    
    # 保存结果到文件
    output_file = 'top_two_pairs_for_each_section.txt'
    with open(output_file, 'w') as f:
        f.write("Section前两个最佳配对分析结果\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"总sections数: {len(all_sections)}\n")
        f.write(f"分析时间: {pd.Timestamp.now()}\n\n")
        
        f.write("格式: Section -> 配对1 (分数) [方向] -> 配对2 (分数) [方向]\n")
        f.write("-" * 60 + "\n")
        
        for result in results:
            section = result['section']
            pairs = result['pairs']
            
            if len(pairs) >= 2:
                f.write(f"{section:20s} -> {pairs[0]['other_section']:20s} (score: {pairs[0]['score']:.4f}) [{pairs[0]['direction']}] -> {pairs[1]['other_section']:20s} (score: {pairs[1]['score']:.4f}) [{pairs[1]['direction']}]\n")
            elif len(pairs) == 1:
                f.write(f"{section:20s} -> {pairs[0]['other_section']:20s} (score: {pairs[0]['score']:.4f}) [{pairs[0]['direction']}] -> N/A\n")
            else:
                f.write(f"{section:20s} -> N/A -> N/A\n")
        
        # 添加统计信息
        f.write("\n" + "=" * 60 + "\n")
        f.write("统计信息:\n")
        
        all_scores = []
        for result in results:
            for pair in result['pairs']:
                all_scores.append(pair['score'])
        
        if all_scores:
            f.write(f"平均分数: {np.mean(all_scores):.4f}\n")
            f.write(f"最高分数: {np.max(all_scores):.4f}\n")
            f.write(f"最低分数: {np.min(all_scores):.4f}\n")
            f.write(f"分数标准差: {np.std(all_scores):.4f}\n")
        
        sections_with_pairs = sum(1 for r in results if len(r['pairs']) > 0)
        f.write(f"有配对的sections: {sections_with_pairs}/{len(all_sections)}\n")
    
    print(f"\n结果已保存到: {output_file}")
    
    # 显示一些统计信息
    all_scores = []
    for result in results:
        for pair in result['pairs']:
            all_scores.append(pair['score'])
    
    if all_scores:
        print(f"\n统计信息:")
        print(f"平均分数: {np.mean(all_scores):.4f}")
        print(f"最高分数: {np.max(all_scores):.4f}")
        print(f"最低分数: {np.min(all_scores):.4f}")
        print(f"分数标准差: {np.std(all_scores):.4f}")
    
    sections_with_pairs = sum(1 for r in results if len(r['pairs']) > 0)
    print(f"有配对的sections: {sections_with_pairs}/{len(all_sections)}")

if __name__ == "__main__":
    main() 
"""
find_best_pairs.py - 找出每个section的最佳配对（最高score）
"""

import pandas as pd
import numpy as np

def load_csv(csv_file):
    """加载CSV文件"""
    df = pd.read_csv(csv_file)
    print(f"加载CSV文件: {csv_file}")
    print(f"总行数: {len(df)}")
    return df

def get_all_sections(df):
    """获取所有唯一的sections，按数字顺序排序"""
    all_sections = set(df['fixed'].unique()) | set(df['moving'].unique())
    
    # 按section数字排序
    def extract_number(section_name):
        try:
            # 提取section_后面的数字
            return int(section_name.split('_')[1])
        except:
            return 999  # 如果无法解析，放在最后
    
    sorted_sections = sorted(all_sections, key=extract_number)
    print(f"总共有 {len(sorted_sections)} 个唯一sections")
    return sorted_sections

def find_top_two_pairs_for_section(section, df):
    """找出指定section的前两个最佳配对"""
    # 查找包含该section的所有行
    mask_fixed = df['fixed'] == section
    mask_moving = df['moving'] == section
    
    # 合并所有相关行
    relevant_rows = df[mask_fixed | mask_moving].copy()
    
    if len(relevant_rows) == 0:
        return []
    
    # 收集所有配对和分数
    pairs = []
    for _, row in relevant_rows.iterrows():
        if row['fixed'] == section:
            other_section = row['moving']
            score = row['score']
            direction = "fixed->moving"
        else:
            other_section = row['fixed']
            score = row['score']
            direction = "moving->fixed"
        
        pairs.append({
            'other_section': other_section,
            'score': score,
            'direction': direction
        })
    
    # 按分数排序，取前两个
    pairs.sort(key=lambda x: x['score'], reverse=True)
    return pairs[:2]

def main():
    # 加载数据
    df = load_csv('new_pairwise_filtered.csv')
    
    # 获取所有sections
    all_sections = get_all_sections(df)
    
    # 找出每个section的前两个最佳配对
    results = []
    
    print(f"\n开始分析每个section的前两个最佳配对...")
    
    for section in all_sections:
        top_pairs = find_top_two_pairs_for_section(section, df)
        
        results.append({
            'section': section,
            'pairs': top_pairs
        })
        
        if top_pairs:
            print(f"{section:20s}:")
            for i, pair in enumerate(top_pairs, 1):
                print(f"  {i}. -> {pair['other_section']:20s} (score: {pair['score']:.4f}) [{pair['direction']}]")
        else:
            print(f"{section:20s}: 无配对")
    
    # 保存结果到文件
    output_file = 'top_two_pairs_for_each_section.txt'
    with open(output_file, 'w') as f:
        f.write("Section前两个最佳配对分析结果\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"总sections数: {len(all_sections)}\n")
        f.write(f"分析时间: {pd.Timestamp.now()}\n\n")
        
        f.write("格式: Section -> 配对1 (分数) [方向] -> 配对2 (分数) [方向]\n")
        f.write("-" * 60 + "\n")
        
        for result in results:
            section = result['section']
            pairs = result['pairs']
            
            if len(pairs) >= 2:
                f.write(f"{section:20s} -> {pairs[0]['other_section']:20s} (score: {pairs[0]['score']:.4f}) [{pairs[0]['direction']}] -> {pairs[1]['other_section']:20s} (score: {pairs[1]['score']:.4f}) [{pairs[1]['direction']}]\n")
            elif len(pairs) == 1:
                f.write(f"{section:20s} -> {pairs[0]['other_section']:20s} (score: {pairs[0]['score']:.4f}) [{pairs[0]['direction']}] -> N/A\n")
            else:
                f.write(f"{section:20s} -> N/A -> N/A\n")
        
        # 添加统计信息
        f.write("\n" + "=" * 60 + "\n")
        f.write("统计信息:\n")
        
        all_scores = []
        for result in results:
            for pair in result['pairs']:
                all_scores.append(pair['score'])
        
        if all_scores:
            f.write(f"平均分数: {np.mean(all_scores):.4f}\n")
            f.write(f"最高分数: {np.max(all_scores):.4f}\n")
            f.write(f"最低分数: {np.min(all_scores):.4f}\n")
            f.write(f"分数标准差: {np.std(all_scores):.4f}\n")
        
        sections_with_pairs = sum(1 for r in results if len(r['pairs']) > 0)
        f.write(f"有配对的sections: {sections_with_pairs}/{len(all_sections)}\n")
    
    print(f"\n结果已保存到: {output_file}")
    
    # 显示一些统计信息
    all_scores = []
    for result in results:
        for pair in result['pairs']:
            all_scores.append(pair['score'])
    
    if all_scores:
        print(f"\n统计信息:")
        print(f"平均分数: {np.mean(all_scores):.4f}")
        print(f"最高分数: {np.max(all_scores):.4f}")
        print(f"最低分数: {np.min(all_scores):.4f}")
        print(f"分数标准差: {np.std(all_scores):.4f}")
    
    sections_with_pairs = sum(1 for r in results if len(r['pairs']) > 0)
    print(f"有配对的sections: {sections_with_pairs}/{len(all_sections)}")

if __name__ == "__main__":
    main() 
 
 
 
 
 
 