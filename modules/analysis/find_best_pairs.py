#!/usr/bin/env python3
"""
find_best_pairs.py - find best pairs for each section
"""

import pandas as pd
import numpy as np

def load_csv(csv_file):
    """load CSV file"""
    df = pd.read_csv(csv_file)
    print(f"load CSV file: {csv_file}")
    print(f"total rows: {len(df)}")
    return df

def get_all_sections(df):
    """get all unique sections, sorted by number"""
    all_sections = set(df['fixed'].unique()) | set(df['moving'].unique())
    
    # sort by section number
    def extract_number(section_name):
        try:
            # extract number after section_
            return int(section_name.split('_')[1])
        except:
            return 999  # if cannot parse, put at the end
    
    sorted_sections = sorted(all_sections, key=extract_number)
    print(f"total unique sections: {len(sorted_sections)}")
    return sorted_sections

def find_top_two_pairs_for_section(section, df):
    """find top two best pairs for specified section"""
    # find all rows containing the specified section
    mask_fixed = df['fixed'] == section
    mask_moving = df['moving'] == section
    
    # merge all relevant rows
    relevant_rows = df[mask_fixed | mask_moving].copy()
    
    if len(relevant_rows) == 0:
        return []
    
    # collect all pairs and scores
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
    
    # sort by score, take top two
    pairs.sort(key=lambda x: x['score'], reverse=True)
    return pairs[:2]

def main():
    # load data
    df = load_csv('new_pairwise_filtered.csv')
    
    # get all sections
    all_sections = get_all_sections(df)
    
    # find top two best pairs for each section
    results = []
    
    print(f"\nstart analyzing top two best pairs for each section...")
    
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
            print(f"{section:20s}: no pairs")
    
    # save results to file
    output_file = 'top_two_pairs_for_each_section.txt'
    with open(output_file, 'w') as f:
        f.write("Top two best pairs for each section\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"total sections: {len(all_sections)}\n")
        f.write(f"analysis time: {pd.Timestamp.now()}\n\n")
        
        f.write("format: Section -> pair1 (score) [direction] -> pair2 (score) [direction]\n")
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
        
        # add statistics
        f.write("\n" + "=" * 60 + "\n")
        f.write("statistics:\n")
        
        all_scores = []
        for result in results:
            for pair in result['pairs']:
                all_scores.append(pair['score'])
        
        if all_scores:
            f.write(f"average score: {np.mean(all_scores):.4f}\n")
            f.write(f"highest score: {np.max(all_scores):.4f}\n")
            f.write(f"lowest score: {np.min(all_scores):.4f}\n")
            f.write(f"score standard deviation: {np.std(all_scores):.4f}\n")
        
        sections_with_pairs = sum(1 for r in results if len(r['pairs']) > 0)
        f.write(f"sections with pairs: {sections_with_pairs}/{len(all_sections)}\n")
    
    print(f"\nresults saved to: {output_file}")
    
    # show some statistics
    all_scores = []
    for result in results:
        for pair in result['pairs']:
            all_scores.append(pair['score'])
    
    if all_scores:
        print(f"\nstatistics:")
        print(f"average score: {np.mean(all_scores):.4f}")
        print(f"highest score: {np.max(all_scores):.4f}")
        print(f"lowest score: {np.min(all_scores):.4f}")
        print(f"score standard deviation: {np.std(all_scores):.4f}")
    
    sections_with_pairs = sum(1 for r in results if len(r['pairs']) > 0)
    print(f"sections with pairs: {sections_with_pairs}/{len(all_sections)}")

if __name__ == "__main__":
    main() 
"""
find_best_pairs.py - find best pairs for each section
"""

import pandas as pd
import numpy as np

def load_csv(csv_file):
    """load CSV file"""
    df = pd.read_csv(csv_file)
    print(f"load CSV file: {csv_file}")
    print(f"total rows: {len(df)}")
    return df

def get_all_sections(df):
    """get all unique sections, sorted by number"""
    all_sections = set(df['fixed'].unique()) | set(df['moving'].unique())
    
    # sort by section number
    def extract_number(section_name):
        try:
            # extract number after section_
            return int(section_name.split('_')[1])
        except:
            return 999  # if cannot parse, put at the end
    
    sorted_sections = sorted(all_sections, key=extract_number)
    print(f"total unique sections: {len(sorted_sections)}")
    return sorted_sections

def find_top_two_pairs_for_section(section, df):
    """find top two best pairs for specified section"""
    # find all rows containing the specified section
    mask_fixed = df['fixed'] == section
    mask_moving = df['moving'] == section
    
    # merge all relevant rows
    relevant_rows = df[mask_fixed | mask_moving].copy()
    
    if len(relevant_rows) == 0:
        return []
    
    # collect all pairs and scores
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
    
    # sort by score, take top two
    pairs.sort(key=lambda x: x['score'], reverse=True)
    return pairs[:2]

def main():
    # load data
    df = load_csv('new_pairwise_filtered.csv')
    
    # get all sections
    all_sections = get_all_sections(df)
    
    # find top two best pairs for each section
    results = []
    
    print(f"\nstart analyzing top two best pairs for each section...")
    
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
            print(f"{section:20s}: no pairs")
    
    # save results to file
    output_file = 'top_two_pairs_for_each_section.txt'
    with open(output_file, 'w') as f:
        f.write("Top two best pairs for each section\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"total sections: {len(all_sections)}\n")
        f.write(f"analysis time: {pd.Timestamp.now()}\n\n")
        
        f.write("format: Section -> pair1 (score) [direction] -> pair2 (score) [direction]\n")
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
        
        # add statistics
        f.write("\n" + "=" * 60 + "\n")
        f.write("statistics:\n")
        
        all_scores = []
        for result in results:
            for pair in result['pairs']:
                all_scores.append(pair['score'])
        
        if all_scores:
            f.write(f"average score: {np.mean(all_scores):.4f}\n")
            f.write(f"highest score: {np.max(all_scores):.4f}\n")
            f.write(f"lowest score: {np.min(all_scores):.4f}\n")
            f.write(f"score standard deviation: {np.std(all_scores):.4f}\n")
        
        sections_with_pairs = sum(1 for r in results if len(r['pairs']) > 0)
        f.write(f"sections with pairs: {sections_with_pairs}/{len(all_sections)}\n")
    
    print(f"\nresults saved to: {output_file}")
    
    # show some statistics
    all_scores = []
    for result in results:
        for pair in result['pairs']:
            all_scores.append(pair['score'])
    
    if all_scores:
        print(f"\nstatistics:")
        print(f"average score: {np.mean(all_scores):.4f}")
        print(f"highest score: {np.max(all_scores):.4f}")
        print(f"lowest score: {np.min(all_scores):.4f}")
        print(f"score standard deviation: {np.std(all_scores):.4f}")
    
    sections_with_pairs = sum(1 for r in results if len(r['pairs']) > 0)
    print(f"sections with pairs: {sections_with_pairs}/{len(all_sections)}")

if __name__ == "__main__":
    main() 
 
 
 
 
 
 
