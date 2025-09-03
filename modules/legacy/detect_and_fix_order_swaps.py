#!/usr/bin/env python3
"""
detect_and_fix_order_swaps.py - detect and fix swaps in order

usage:
    python detect_and_fix_order_swaps.py --order improved_order_no_duplicates.txt --csv pair_ssim_ss10_scale07.csv --output cleaned_order.txt
"""

import argparse
import pandas as pd
import numpy as np

def load_order(order_file):
    """load order file"""
    with open(order_file, 'r') as f:
        content = f.read().strip()
        # process space-separated format
        order = content.split()
    return order

def load_csv(csv_file):
    """load CSV file"""
    df = pd.read_csv(csv_file)
    return df

def calculate_order_score(order, df):
    """calculate total score of order"""
    total_score = 0
    count = 0
    
    for i in range(len(order) - 1):
        section1 = order[i]
        section2 = order[i + 1]
        
        # find score of this pair of sections
        mask = ((df['fixed'] == section1) & (df['moving'] == section2)) | \
               ((df['fixed'] == section2) & (df['moving'] == section1))
        
        if mask.any():
            score = df[mask]['ssim'].iloc[0]
            total_score += score
            count += 1
    
    return total_score / count if count > 0 else 0

def find_best_swaps(order, df, max_swaps=10):
    """find best swaps to improve order"""
    best_order = order.copy()
    best_score = calculate_order_score(order, df)
    
    print(f"initial order score: {best_score:.4f}")
    
    for swap_count in range(max_swaps):
        improved = False
        
        # try all possible adjacent swaps
        for i in range(len(best_order) - 1):
            # create new order, swap adjacent two sections
            new_order = best_order.copy()
            new_order[i], new_order[i + 1] = new_order[i + 1], new_order[i]
            
            # calculate score of new order
            new_score = calculate_order_score(new_order, df)
            
            # if score is improved, accept this swap
            if new_score > best_score:
                best_order = new_order
                best_score = new_score
                improved = True
                print(f"Swap {swap_count + 1}: swap {best_order[i+1]} and {best_order[i]}, new score: {new_score:.4f}")
                break
        
        # if no improvement, stop
        if not improved:
            print(f"no more improvement, stop at {swap_count} swaps")
            break
    
    return best_order, best_score

def main():
    parser = argparse.ArgumentParser(description='detect and fix swaps in order')
    parser.add_argument('--order', required=True, help='input order file')
    parser.add_argument('--csv', required=True, help='input CSV file')
    parser.add_argument('--output', required=True, help='output order file')
    parser.add_argument('--max_swaps', type=int, default=10, help='maximum number of swaps')
    
    args = parser.parse_args()
    
    # load data
    print(f"load order file: {args.order}")
    order = load_order(args.order)
    print(f"order length: {len(order)}")
    
    print(f"load CSV file: {args.csv}")
    df = load_csv(args.csv)
    print(f"CSV rows: {len(df)}")
    
    # find best swaps
    print("\nfind best swaps...")
    best_order, best_score = find_best_swaps(order, df, args.max_swaps)
    
    # save results
    print(f"\nsave results to: {args.output}")
    with open(args.output, 'w') as f:
        for section in best_order:
            f.write(section + '\n')
    
    print(f"final order score: {best_score:.4f}")
    print("done!")

if __name__ == "__main__":
    main() 
"""
detect_and_fix_order_swaps.py - detect and fix swaps in order

usage:
    python detect_and_fix_order_swaps.py --order improved_order_no_duplicates.txt --csv pair_ssim_ss10_scale07.csv --output cleaned_order.txt
"""

import argparse
import pandas as pd
import numpy as np

def load_order(order_file):
    """load order file"""
    with open(order_file, 'r') as f:
        content = f.read().strip()
        # process space-separated format
        order = content.split()
    return order

def load_csv(csv_file):
    """load CSV file"""
    df = pd.read_csv(csv_file)
    return df

def calculate_order_score(order, df):
    """calculate total score of order"""
    total_score = 0
    count = 0
    
    for i in range(len(order) - 1):
        section1 = order[i]
        section2 = order[i + 1]
        
        # find score of this pair of sections
        mask = ((df['fixed'] == section1) & (df['moving'] == section2)) | \
               ((df['fixed'] == section2) & (df['moving'] == section1))
        
        if mask.any():
            score = df[mask]['ssim'].iloc[0]
            total_score += score
            count += 1
    
    return total_score / count if count > 0 else 0

def find_best_swaps(order, df, max_swaps=10):
    """find best swaps to improve order"""
    best_order = order.copy()
    best_score = calculate_order_score(order, df)
    
    print(f"initial order score: {best_score:.4f}")
    
    for swap_count in range(max_swaps):
        improved = False
        
        # try all possible adjacent swaps
        for i in range(len(best_order) - 1):
            # create new order, swap adjacent two sections
            new_order = best_order.copy()
            new_order[i], new_order[i + 1] = new_order[i + 1], new_order[i]
            
            # calculate score of new order
            new_score = calculate_order_score(new_order, df)
            
            # if score is improved, accept this swap
            if new_score > best_score:
                best_order = new_order
                best_score = new_score
                improved = True
                print(f"Swap {swap_count + 1}: swap {best_order[i+1]} and {best_order[i]}, new score: {new_score:.4f}")
                break
        
        # if no improvement, stop
        if not improved:
            print(f"no more improvement, stop at {swap_count} swaps")
            break
    
    return best_order, best_score

def main():
    parser = argparse.ArgumentParser(description='detect and fix swaps in order')
    parser.add_argument('--order', required=True, help='input order file')
    parser.add_argument('--csv', required=True, help='input CSV file')
    parser.add_argument('--output', required=True, help='output order file')
    parser.add_argument('--max_swaps', type=int, default=10, help='maximum number of swaps')
    
    args = parser.parse_args()
    
    # load data
    print(f"load order file: {args.order}")
    order = load_order(args.order)
    print(f"order length: {len(order)}")
    
    print(f"load CSV file: {args.csv}")
    df = load_csv(args.csv)
    print(f"CSV rows: {len(df)}")
    
    # find best swaps
    print("\nfind best swaps...")
    best_order, best_score = find_best_swaps(order, df, args.max_swaps)
    
    # save results
    print(f"\nsave results to: {args.output}")
    with open(args.output, 'w') as f:
        for section in best_order:
            f.write(section + '\n')
    
    print(f"final order score: {best_score:.4f}")
    print("done!")

if __name__ == "__main__":
    main() 
 
 
 
 
 
 
