#!/usr/bin/env python3
import pandas as pd
import numpy as np
import argparse

def greedy_chain_order(df):
    # 获取所有section
    all_sections = sorted(set(df['fixed'].unique()) | set(df['moving'].unique()))
    used = set()
    chains = []
    
    # 按ssim降序排列所有pair
    pairs = df.sort_values('ssim', ascending=False)[['fixed', 'moving', 'ssim']].values.tolist()
    
    # 逐步构建链
    for fixed, moving, ssim in pairs:
        if fixed not in used and moving not in used:
            chains.append([fixed, moving])
            used.add(fixed)
            used.add(moving)
        elif fixed in used and moving not in used:
            # 尝试接到链尾
            for chain in chains:
                if chain[-1] == fixed:
                    chain.append(moving)
                    used.add(moving)
                    break
        elif moving in used and fixed not in used:
            for chain in chains:
                if chain[0] == moving:
                    chain.insert(0, fixed)
                    used.add(fixed)
                    break
    # 合并所有链
    while len(chains) > 1:
        merged = False
        for i in range(len(chains)):
            for j in range(len(chains)):
                if i != j and chains[i][-1] == chains[j][0]:
                    chains[i].extend(chains[j][1:])
                    chains.pop(j)
                    merged = True
                    break
            if merged:
                break
        if not merged:
            break
    # 补充未覆盖的section
    final_chain = chains[0] if chains else []
    for s in all_sections:
        if s not in final_chain:
            final_chain.append(s)
    return final_chain

def main():
    parser = argparse.ArgumentParser(description="Greedy chain ordering")
    parser.add_argument("csv_file", help="Input CSV file")
    parser.add_argument("--output", help="Output file")
    parser.add_argument("--target", help="Target order file to compare")
    args = parser.parse_args()
    df = pd.read_csv(args.csv_file)
    order = greedy_chain_order(df)
    output_file = args.output or "greedy_chain_order.txt"
    with open(output_file, 'w') as f:
        f.write(' '.join(order))
    print(f"Saved order to {output_file}")
    # 对比目标
    if args.target:
        with open(args.target, 'r') as f:
            target = f.read().strip().split()
        if order == target:
            print("✓ Success! Generated order matches target exactly!")
        else:
            print("✗ Generated order does not match target")
            for i, (a, b) in enumerate(zip(order, target)):
                if a != b:
                    print(f"First difference at position {i+1}: {a} vs {b}")
                    break
if __name__ == "__main__":
    main() 
import pandas as pd
import numpy as np
import argparse

def greedy_chain_order(df):
    # 获取所有section
    all_sections = sorted(set(df['fixed'].unique()) | set(df['moving'].unique()))
    used = set()
    chains = []
    
    # 按ssim降序排列所有pair
    pairs = df.sort_values('ssim', ascending=False)[['fixed', 'moving', 'ssim']].values.tolist()
    
    # 逐步构建链
    for fixed, moving, ssim in pairs:
        if fixed not in used and moving not in used:
            chains.append([fixed, moving])
            used.add(fixed)
            used.add(moving)
        elif fixed in used and moving not in used:
            # 尝试接到链尾
            for chain in chains:
                if chain[-1] == fixed:
                    chain.append(moving)
                    used.add(moving)
                    break
        elif moving in used and fixed not in used:
            for chain in chains:
                if chain[0] == moving:
                    chain.insert(0, fixed)
                    used.add(fixed)
                    break
    # 合并所有链
    while len(chains) > 1:
        merged = False
        for i in range(len(chains)):
            for j in range(len(chains)):
                if i != j and chains[i][-1] == chains[j][0]:
                    chains[i].extend(chains[j][1:])
                    chains.pop(j)
                    merged = True
                    break
            if merged:
                break
        if not merged:
            break
    # 补充未覆盖的section
    final_chain = chains[0] if chains else []
    for s in all_sections:
        if s not in final_chain:
            final_chain.append(s)
    return final_chain

def main():
    parser = argparse.ArgumentParser(description="Greedy chain ordering")
    parser.add_argument("csv_file", help="Input CSV file")
    parser.add_argument("--output", help="Output file")
    parser.add_argument("--target", help="Target order file to compare")
    args = parser.parse_args()
    df = pd.read_csv(args.csv_file)
    order = greedy_chain_order(df)
    output_file = args.output or "greedy_chain_order.txt"
    with open(output_file, 'w') as f:
        f.write(' '.join(order))
    print(f"Saved order to {output_file}")
    # 对比目标
    if args.target:
        with open(args.target, 'r') as f:
            target = f.read().strip().split()
        if order == target:
            print("✓ Success! Generated order matches target exactly!")
        else:
            print("✗ Generated order does not match target")
            for i, (a, b) in enumerate(zip(order, target)):
                if a != b:
                    print(f"First difference at position {i+1}: {a} vs {b}")
                    break
if __name__ == "__main__":
    main() 
 
 
 
 
 
 