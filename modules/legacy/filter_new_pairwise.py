import pandas as pd

# read CSV file
print("reading new_pairwise.csv...")
df = pd.read_csv('new_pairwise.csv')

print(f"original data rows: {len(df)}")

# check column names
print("column names:", df.columns.tolist())

# filter condition 1: remove rows where dx is 0
# note: dx column may be empty, we need to handle this case
if 'dx_px' in df.columns:
    # remove rows where dx is 0 or empty
    df_filtered = df[df['dx_px'].notna() & (df['dx_px'] != 0)]
    print(f"filtered dx=0 rows: {len(df_filtered)}")
else:
    print("warning: dx_px column not found")
    df_filtered = df

# filter condition 2: remove rows where scale < 0.7
if 'scale' in df.columns:
    # remove rows where scale < 0.7
    df_filtered = df_filtered[df_filtered['scale'] >= 0.7]
    print(f"filtered scale<0.7 rows: {len(df_filtered)}")
else:
    print("warning: scale column not found")

# filter condition 3: remove rows containing specified sections
sections_to_remove = ['section_54_r01_c01', 'section_79_r01_c01', 'section_33_r01_c01', 
                     'section_95_r01_c01', 'section_98_r01_c01', 'section_81_r01_c01']

# show section statistics before filtering
print(f"\nsections before filtering:")
for section in sections_to_remove:
    fixed_count = len(df_filtered[df_filtered['fixed'] == section])
    moving_count = len(df_filtered[df_filtered['moving'] == section])
    if fixed_count > 0 or moving_count > 0:
        print(f"  {section}: fixed={fixed_count}, moving={moving_count}")

# create mask, mark rows to keep
mask = ~(df_filtered['fixed'].isin(sections_to_remove) | df_filtered['moving'].isin(sections_to_remove))
df_filtered = df_filtered[mask]
print(f"filtered rows after removing specified sections: {len(df_filtered)}")

# save filtered data
output_filename = 'new_pairwise_filtered.csv'
df_filtered.to_csv(output_filename, index=False)
print(f"filtered data saved to: {output_filename}")

# show some statistics
print("\nstatistics:")
print(f"final retained rows: {len(df_filtered)}")
print(f"filtered rows: {len(df) - len(df_filtered)}")

# show scale column statistics
if 'scale' in df_filtered.columns:
    print(f"\nscale column statistics:")
    print(f"minimum: {df_filtered['scale'].min()}")
    print(f"maximum: {df_filtered['scale'].max()}")
    print(f"average: {df_filtered['scale'].mean():.3f}")

# show first few rows of filtered data
print(f"\nfirst few rows of filtered data:")
print(df_filtered.head()) 

# read CSV file
print("reading new_pairwise.csv...")
df = pd.read_csv('new_pairwise.csv')

print(f"original data rows: {len(df)}")

# check column names
print("column names:", df.columns.tolist())

# filter condition 1: remove rows where dx is 0
# note: dx column may be empty, we need to handle this case
if 'dx_px' in df.columns:
    # remove rows where dx is 0 or empty
    df_filtered = df[df['dx_px'].notna() & (df['dx_px'] != 0)]
    print(f"filtered dx=0 rows: {len(df_filtered)}")
else:
    print("warning: dx_px column not found")
    df_filtered = df

# filter condition 2: remove rows where scale < 0.7
if 'scale' in df.columns:
    # remove rows where scale < 0.7
    df_filtered = df_filtered[df_filtered['scale'] >= 0.7]
    print(f"filtered scale<0.7 rows: {len(df_filtered)}")
else:
    print("warning: scale column not found")

# filter condition 3: remove rows containing specified sections
sections_to_remove = ['section_54_r01_c01', 'section_79_r01_c01', 'section_33_r01_c01', 
                     'section_95_r01_c01', 'section_98_r01_c01', 'section_81_r01_c01']

# show section statistics before filtering
print(f"\nsections before filtering:")
for section in sections_to_remove:
    fixed_count = len(df_filtered[df_filtered['fixed'] == section])
    moving_count = len(df_filtered[df_filtered['moving'] == section])
    if fixed_count > 0 or moving_count > 0:
        print(f"  {section}: fixed={fixed_count}, moving={moving_count}")

# create mask, mark rows to keep
mask = ~(df_filtered['fixed'].isin(sections_to_remove) | df_filtered['moving'].isin(sections_to_remove))
df_filtered = df_filtered[mask]
print(f"filtered rows after removing specified sections: {len(df_filtered)}")

# save filtered data
output_filename = 'new_pairwise_filtered.csv'
df_filtered.to_csv(output_filename, index=False)
print(f"filtered data saved to: {output_filename}")

# show some statistics
print("\nstatistics:")
print(f"final retained rows: {len(df_filtered)}")
print(f"filtered rows: {len(df) - len(df_filtered)}")

# show scale column statistics
if 'scale' in df_filtered.columns:
    print(f"\nscale column statistics:")
    print(f"minimum: {df_filtered['scale'].min()}")
    print(f"maximum: {df_filtered['scale'].max()}")
    print(f"average: {df_filtered['scale'].mean():.3f}")

# show first few rows of filtered data
print(f"\nfirst few rows of filtered data:")
print(df_filtered.head()) 
 
 
 
 
 
 
