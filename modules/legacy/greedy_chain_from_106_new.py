import pandas as pd

# 读取CSV
csv_path = 'new_pairwise_filtered.csv'
df = pd.read_csv(csv_path)

print(f"读取CSV文件: {csv_path}")
print(f"CSV文件行数: {len(df)}")

# 获取所有唯一的section名
sections = set(df['fixed']).union(set(df['moving']))
print(f"唯一section数量: {len(sections)}")

# 初始化
start_section = 'section_106_r01_c01'
used_sections = set([start_section])
order = [start_section]

current_section = start_section

print(f"从 {start_section} 开始贪心链式排序...")

while len(used_sections) < len(sections):
    # 找到所有包含current_section的pair，且另一个section未被用过
    mask = ((df['fixed'] == current_section) & (~df['moving'].isin(used_sections))) | \
           ((df['moving'] == current_section) & (~df['fixed'].isin(used_sections)))
    candidates = df[mask]
    if candidates.empty:
        print(f"No more candidates from {current_section}, stopping early.")
        break
    # 选最大ssim的pair
    best_row = candidates.loc[candidates['ssim'].idxmax()]
    # 找到下一个section名
    if best_row['fixed'] == current_section:
        next_section = best_row['moving']
    else:
        next_section = best_row['fixed']
    order.append(next_section)
    used_sections.add(next_section)
    current_section = next_section
    print(f"  {current_section} -> {next_section} (ssim: {best_row['ssim']:.3f})")

# 输出顺序
output_file = 'greedy_chain_from_106_new_order.txt'
with open(output_file, 'w') as f:
    for s in order:
        f.write(s + '\n')

print(f"\n结果:")
print(f"总section数量: {len(order)}")
print(f"覆盖的section比例: {len(order)/len(sections)*100:.1f}%")
print(f"顺序已保存到: {output_file}")

# 显示未包含的section
unused_sections = sections - used_sections
if unused_sections:
    print(f"\n未包含的section ({len(unused_sections)}个):")
    for s in sorted(unused_sections):
        print(f"  {s}")
else:
    print(f"\n所有section都已包含在链中!") 

# 读取CSV
csv_path = 'new_pairwise_filtered.csv'
df = pd.read_csv(csv_path)

print(f"读取CSV文件: {csv_path}")
print(f"CSV文件行数: {len(df)}")

# 获取所有唯一的section名
sections = set(df['fixed']).union(set(df['moving']))
print(f"唯一section数量: {len(sections)}")

# 初始化
start_section = 'section_106_r01_c01'
used_sections = set([start_section])
order = [start_section]

current_section = start_section

print(f"从 {start_section} 开始贪心链式排序...")

while len(used_sections) < len(sections):
    # 找到所有包含current_section的pair，且另一个section未被用过
    mask = ((df['fixed'] == current_section) & (~df['moving'].isin(used_sections))) | \
           ((df['moving'] == current_section) & (~df['fixed'].isin(used_sections)))
    candidates = df[mask]
    if candidates.empty:
        print(f"No more candidates from {current_section}, stopping early.")
        break
    # 选最大ssim的pair
    best_row = candidates.loc[candidates['ssim'].idxmax()]
    # 找到下一个section名
    if best_row['fixed'] == current_section:
        next_section = best_row['moving']
    else:
        next_section = best_row['fixed']
    order.append(next_section)
    used_sections.add(next_section)
    current_section = next_section
    print(f"  {current_section} -> {next_section} (ssim: {best_row['ssim']:.3f})")

# 输出顺序
output_file = 'greedy_chain_from_106_new_order.txt'
with open(output_file, 'w') as f:
    for s in order:
        f.write(s + '\n')

print(f"\n结果:")
print(f"总section数量: {len(order)}")
print(f"覆盖的section比例: {len(order)/len(sections)*100:.1f}%")
print(f"顺序已保存到: {output_file}")

# 显示未包含的section
unused_sections = sections - used_sections
if unused_sections:
    print(f"\n未包含的section ({len(unused_sections)}个):")
    for s in sorted(unused_sections):
        print(f"  {s}")
else:
    print(f"\n所有section都已包含在链中!") 
 
 
 
 
 
 