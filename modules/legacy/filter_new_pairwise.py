import pandas as pd

# 读取CSV文件
print("正在读取 new_pairwise.csv...")
df = pd.read_csv('new_pairwise.csv')

print(f"原始数据行数: {len(df)}")

# 检查列名
print("列名:", df.columns.tolist())

# 过滤条件1: 去掉dx是0的所有row
# 注意：dx列可能为空，我们需要处理这种情况
if 'dx_px' in df.columns:
    # 去掉dx为0或空的行
    df_filtered = df[df['dx_px'].notna() & (df['dx_px'] != 0)]
    print(f"过滤dx=0后行数: {len(df_filtered)}")
else:
    print("警告: 未找到dx_px列")
    df_filtered = df

# 过滤条件2: 去掉scale<0.7的所有row
if 'scale' in df.columns:
    # 去掉scale小于0.7的行
    df_filtered = df_filtered[df_filtered['scale'] >= 0.7]
    print(f"过滤scale<0.7后行数: {len(df_filtered)}")
else:
    print("警告: 未找到scale列")

# 过滤条件3: 去掉包含指定section的行
sections_to_remove = ['section_54_r01_c01', 'section_79_r01_c01', 'section_33_r01_c01', 
                     'section_95_r01_c01', 'section_98_r01_c01', 'section_81_r01_c01']

# 显示过滤前的section统计
print(f"\n过滤前包含指定section的行数:")
for section in sections_to_remove:
    fixed_count = len(df_filtered[df_filtered['fixed'] == section])
    moving_count = len(df_filtered[df_filtered['moving'] == section])
    if fixed_count > 0 or moving_count > 0:
        print(f"  {section}: fixed={fixed_count}, moving={moving_count}")

# 创建掩码，标记需要保留的行
mask = ~(df_filtered['fixed'].isin(sections_to_remove) | df_filtered['moving'].isin(sections_to_remove))
df_filtered = df_filtered[mask]
print(f"过滤指定section后行数: {len(df_filtered)}")

# 保存过滤后的数据
output_filename = 'new_pairwise_filtered.csv'
df_filtered.to_csv(output_filename, index=False)
print(f"过滤后的数据已保存到: {output_filename}")

# 显示一些统计信息
print("\n统计信息:")
print(f"最终保留的行数: {len(df_filtered)}")
print(f"过滤掉的行数: {len(df) - len(df_filtered)}")

# 显示scale列的统计信息
if 'scale' in df_filtered.columns:
    print(f"\nScale列统计:")
    print(f"最小值: {df_filtered['scale'].min()}")
    print(f"最大值: {df_filtered['scale'].max()}")
    print(f"平均值: {df_filtered['scale'].mean():.3f}")

# 显示前几行数据
print(f"\n过滤后的前5行数据:")
print(df_filtered.head()) 

# 读取CSV文件
print("正在读取 new_pairwise.csv...")
df = pd.read_csv('new_pairwise.csv')

print(f"原始数据行数: {len(df)}")

# 检查列名
print("列名:", df.columns.tolist())

# 过滤条件1: 去掉dx是0的所有row
# 注意：dx列可能为空，我们需要处理这种情况
if 'dx_px' in df.columns:
    # 去掉dx为0或空的行
    df_filtered = df[df['dx_px'].notna() & (df['dx_px'] != 0)]
    print(f"过滤dx=0后行数: {len(df_filtered)}")
else:
    print("警告: 未找到dx_px列")
    df_filtered = df

# 过滤条件2: 去掉scale<0.7的所有row
if 'scale' in df.columns:
    # 去掉scale小于0.7的行
    df_filtered = df_filtered[df_filtered['scale'] >= 0.7]
    print(f"过滤scale<0.7后行数: {len(df_filtered)}")
else:
    print("警告: 未找到scale列")

# 过滤条件3: 去掉包含指定section的行
sections_to_remove = ['section_54_r01_c01', 'section_79_r01_c01', 'section_33_r01_c01', 
                     'section_95_r01_c01', 'section_98_r01_c01', 'section_81_r01_c01']

# 显示过滤前的section统计
print(f"\n过滤前包含指定section的行数:")
for section in sections_to_remove:
    fixed_count = len(df_filtered[df_filtered['fixed'] == section])
    moving_count = len(df_filtered[df_filtered['moving'] == section])
    if fixed_count > 0 or moving_count > 0:
        print(f"  {section}: fixed={fixed_count}, moving={moving_count}")

# 创建掩码，标记需要保留的行
mask = ~(df_filtered['fixed'].isin(sections_to_remove) | df_filtered['moving'].isin(sections_to_remove))
df_filtered = df_filtered[mask]
print(f"过滤指定section后行数: {len(df_filtered)}")

# 保存过滤后的数据
output_filename = 'new_pairwise_filtered.csv'
df_filtered.to_csv(output_filename, index=False)
print(f"过滤后的数据已保存到: {output_filename}")

# 显示一些统计信息
print("\n统计信息:")
print(f"最终保留的行数: {len(df_filtered)}")
print(f"过滤掉的行数: {len(df) - len(df_filtered)}")

# 显示scale列的统计信息
if 'scale' in df_filtered.columns:
    print(f"\nScale列统计:")
    print(f"最小值: {df_filtered['scale'].min()}")
    print(f"最大值: {df_filtered['scale'].max()}")
    print(f"平均值: {df_filtered['scale'].mean():.3f}")

# 显示前几行数据
print(f"\n过滤后的前5行数据:")
print(df_filtered.head()) 
 
 
 
 
 
 