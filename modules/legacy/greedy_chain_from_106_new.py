import pandas as pd

# read CSV
csv_path = 'new_pairwise_filtered.csv'
df = pd.read_csv(csv_path)

print(f"read CSV file: {csv_path}")
print(f"CSV file rows: {len(df)}")

# get all unique section names
sections = set(df['fixed']).union(set(df['moving']))
print(f"unique section count: {len(sections)}")

# initialize
start_section = 'section_106_r01_c01'
used_sections = set([start_section])
order = [start_section]

current_section = start_section

print(f"start greedy chain from {start_section}...")

while len(used_sections) < len(sections):
    # find all pairs containing current_section, and the other section is not used
    mask = ((df['fixed'] == current_section) & (~df['moving'].isin(used_sections))) | \
           ((df['moving'] == current_section) & (~df['fixed'].isin(used_sections)))
    candidates = df[mask]
    if candidates.empty:
        print(f"No more candidates from {current_section}, stopping early.")
        break
    # select the pair with the highest ssim
    best_row = candidates.loc[candidates['ssim'].idxmax()]
    # find the next section name
    if best_row['fixed'] == current_section:
        next_section = best_row['moving']
    else:
        next_section = best_row['fixed']
    order.append(next_section)
    used_sections.add(next_section)
    current_section = next_section
    print(f"  {current_section} -> {next_section} (ssim: {best_row['ssim']:.3f})")

# output order
output_file = 'greedy_chain_from_106_new_order.txt'
with open(output_file, 'w') as f:
    for s in order:
        f.write(s + '\n')

print(f"\nresults:")
print(f"total section count: {len(order)}")
print(f"covered section ratio: {len(order)/len(sections)*100:.1f}%")
print(f"order saved to: {output_file}")

# show unused sections
unused_sections = sections - used_sections
if unused_sections:
    print(f"\nunused sections ({len(unused_sections)} sections):")
    for s in sorted(unused_sections):
        print(f"  {s}")
else:
    print(f"\nall sections are included in the chain!") 

# read CSV
csv_path = 'new_pairwise_filtered.csv'
df = pd.read_csv(csv_path)

print(f"read CSV file: {csv_path}")
print(f"CSV file rows: {len(df)}")

# get all unique section names
sections = set(df['fixed']).union(set(df['moving']))
print(f"unique section count: {len(sections)}")

# initialize
start_section = 'section_106_r01_c01'
used_sections = set([start_section])
order = [start_section]

current_section = start_section

print(f"start greedy chain from {start_section}...")

while len(used_sections) < len(sections):
    # find all pairs containing current_section, and the other section is not used
    mask = ((df['fixed'] == current_section) & (~df['moving'].isin(used_sections))) | \
           ((df['moving'] == current_section) & (~df['fixed'].isin(used_sections)))
    candidates = df[mask]
    if candidates.empty:
        print(f"No more candidates from {current_section}, stopping early.")
        break
    # select the pair with the highest ssim
    best_row = candidates.loc[candidates['ssim'].idxmax()]
    # find the next section name
    if best_row['fixed'] == current_section:
        next_section = best_row['moving']
    else:
        next_section = best_row['fixed']
    order.append(next_section)
    used_sections.add(next_section)
    current_section = next_section
    print(f"  {current_section} -> {next_section} (ssim: {best_row['ssim']:.3f})")

# output order
output_file = 'greedy_chain_from_106_new_order.txt'
with open(output_file, 'w') as f:
    for s in order:
        f.write(s + '\n')

print(f"\nresults:")
print(f"total section count: {len(order)}")
print(f"covered section ratio: {len(order)/len(sections)*100:.1f}%")
print(f"order saved to: {output_file}")

# show unused sections
unused_sections = sections - used_sections
if unused_sections:
    print(f"\nunused sections ({len(unused_sections)} sections):")
    for s in sorted(unused_sections):
        print(f"  {s}")
else:
    print(f"\nall sections are included in the chain!") 
 
 
 
 
 
 
