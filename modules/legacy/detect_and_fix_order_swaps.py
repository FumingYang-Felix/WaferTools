#!/usr/bin/env python3
"""
detect_and_fix_order_swaps.py - 检测和修复order中的swaps

用法:
    python detect_and_fix_order_swaps.py --order improved_order_no_duplicates.txt --csv pair_ssim_ss10_scale07.csv --output cleaned_order.txt
"""

import argparse
import pandas as pd
import numpy as np

def load_order(order_file):
    """加载order文件"""
    with open(order_file, 'r') as f:
        content = f.read().strip()
        # 处理空格分隔的格式
        order = content.split()
    return order

def load_csv(csv_file):
    """加载CSV文件"""
    df = pd.read_csv(csv_file)
    return df

def calculate_order_score(order, df):
    """计算order的总分数"""
    total_score = 0
    count = 0
    
    for i in range(len(order) - 1):
        section1 = order[i]
        section2 = order[i + 1]
        
        # 查找这对section的分数
        mask = ((df['fixed'] == section1) & (df['moving'] == section2)) | \
               ((df['fixed'] == section2) & (df['moving'] == section1))
        
        if mask.any():
            score = df[mask]['ssim'].iloc[0]
            total_score += score
            count += 1
    
    return total_score / count if count > 0 else 0

def find_best_swaps(order, df, max_swaps=10):
    """寻找最佳的swaps来改善order"""
    best_order = order.copy()
    best_score = calculate_order_score(order, df)
    
    print(f"初始order分数: {best_score:.4f}")
    
    for swap_count in range(max_swaps):
        improved = False
        
        # 尝试所有可能的相邻swap
        for i in range(len(best_order) - 1):
            # 创建新的order，交换相邻的两个section
            new_order = best_order.copy()
            new_order[i], new_order[i + 1] = new_order[i + 1], new_order[i]
            
            # 计算新order的分数
            new_score = calculate_order_score(new_order, df)
            
            # 如果分数提高了，接受这个swap
            if new_score > best_score:
                best_order = new_order
                best_score = new_score
                improved = True
                print(f"Swap {swap_count + 1}: 交换 {best_order[i+1]} 和 {best_order[i]}, 新分数: {new_score:.4f}")
                break
        
        # 如果没有改进，停止
        if not improved:
            print(f"没有更多改进，在第 {swap_count} 次swap后停止")
            break
    
    return best_order, best_score

def main():
    parser = argparse.ArgumentParser(description='检测和修复order中的swaps')
    parser.add_argument('--order', required=True, help='输入order文件')
    parser.add_argument('--csv', required=True, help='输入CSV文件')
    parser.add_argument('--output', required=True, help='输出order文件')
    parser.add_argument('--max_swaps', type=int, default=10, help='最大swap次数')
    
    args = parser.parse_args()
    
    # 加载数据
    print(f"加载order文件: {args.order}")
    order = load_order(args.order)
    print(f"Order长度: {len(order)}")
    
    print(f"加载CSV文件: {args.csv}")
    df = load_csv(args.csv)
    print(f"CSV行数: {len(df)}")
    
    # 寻找最佳swaps
    print("\n开始寻找最佳swaps...")
    best_order, best_score = find_best_swaps(order, df, args.max_swaps)
    
    # 保存结果
    print(f"\n保存结果到: {args.output}")
    with open(args.output, 'w') as f:
        for section in best_order:
            f.write(section + '\n')
    
    print(f"最终order分数: {best_score:.4f}")
    print("完成!")

if __name__ == "__main__":
    main() 
"""
detect_and_fix_order_swaps.py - 检测和修复order中的swaps

用法:
    python detect_and_fix_order_swaps.py --order improved_order_no_duplicates.txt --csv pair_ssim_ss10_scale07.csv --output cleaned_order.txt
"""

import argparse
import pandas as pd
import numpy as np

def load_order(order_file):
    """加载order文件"""
    with open(order_file, 'r') as f:
        content = f.read().strip()
        # 处理空格分隔的格式
        order = content.split()
    return order

def load_csv(csv_file):
    """加载CSV文件"""
    df = pd.read_csv(csv_file)
    return df

def calculate_order_score(order, df):
    """计算order的总分数"""
    total_score = 0
    count = 0
    
    for i in range(len(order) - 1):
        section1 = order[i]
        section2 = order[i + 1]
        
        # 查找这对section的分数
        mask = ((df['fixed'] == section1) & (df['moving'] == section2)) | \
               ((df['fixed'] == section2) & (df['moving'] == section1))
        
        if mask.any():
            score = df[mask]['ssim'].iloc[0]
            total_score += score
            count += 1
    
    return total_score / count if count > 0 else 0

def find_best_swaps(order, df, max_swaps=10):
    """寻找最佳的swaps来改善order"""
    best_order = order.copy()
    best_score = calculate_order_score(order, df)
    
    print(f"初始order分数: {best_score:.4f}")
    
    for swap_count in range(max_swaps):
        improved = False
        
        # 尝试所有可能的相邻swap
        for i in range(len(best_order) - 1):
            # 创建新的order，交换相邻的两个section
            new_order = best_order.copy()
            new_order[i], new_order[i + 1] = new_order[i + 1], new_order[i]
            
            # 计算新order的分数
            new_score = calculate_order_score(new_order, df)
            
            # 如果分数提高了，接受这个swap
            if new_score > best_score:
                best_order = new_order
                best_score = new_score
                improved = True
                print(f"Swap {swap_count + 1}: 交换 {best_order[i+1]} 和 {best_order[i]}, 新分数: {new_score:.4f}")
                break
        
        # 如果没有改进，停止
        if not improved:
            print(f"没有更多改进，在第 {swap_count} 次swap后停止")
            break
    
    return best_order, best_score

def main():
    parser = argparse.ArgumentParser(description='检测和修复order中的swaps')
    parser.add_argument('--order', required=True, help='输入order文件')
    parser.add_argument('--csv', required=True, help='输入CSV文件')
    parser.add_argument('--output', required=True, help='输出order文件')
    parser.add_argument('--max_swaps', type=int, default=10, help='最大swap次数')
    
    args = parser.parse_args()
    
    # 加载数据
    print(f"加载order文件: {args.order}")
    order = load_order(args.order)
    print(f"Order长度: {len(order)}")
    
    print(f"加载CSV文件: {args.csv}")
    df = load_csv(args.csv)
    print(f"CSV行数: {len(df)}")
    
    # 寻找最佳swaps
    print("\n开始寻找最佳swaps...")
    best_order, best_score = find_best_swaps(order, df, args.max_swaps)
    
    # 保存结果
    print(f"\n保存结果到: {args.output}")
    with open(args.output, 'w') as f:
        for section in best_order:
            f.write(section + '\n')
    
    print(f"最终order分数: {best_score:.4f}")
    print("完成!")

if __name__ == "__main__":
    main() 
 
 
 
 
 
 