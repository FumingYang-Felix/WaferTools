# 指定的56个section
mst_sections = [
    'section_106_r01_c01', 'section_105_r01_c01', 'section_45_r01_c01', 'section_60_r01_c01',
    'section_100_r01_c01', 'section_87_r01_c01', 'section_91_r01_c01', 'section_53_r01_c01',
    'section_69_r01_c01', 'section_46_r01_c01', 'section_104_r01_c01', 'section_7_r01_c01',
    'section_11_r01_c01', 'section_67_r01_c01', 'section_86_r01_c01', 'section_85_r01_c01',
    'section_55_r01_c01', 'section_63_r01_c01', 'section_1_r01_c01', 'section_92_r01_c01',
    'section_6_r01_c01', 'section_57_r01_c01', 'section_89_r01_c01', 'section_12_r01_c01',
    'section_88_r01_c01', 'section_16_r01_c01', 'section_29_r01_c01', 'section_30_r01_c01',
    'section_74_r01_c01', 'section_25_r01_c01', 'section_9_r01_c01', 'section_43_r01_c01',
    'section_65_r01_c01', 'section_97_r01_c01', 'section_62_r01_c01', 'section_102_r01_c01',
    'section_36_r01_c01', 'section_71_r01_c01', 'section_48_r01_c01', 'section_66_r01_c01',
    'section_4_r01_c01', 'section_22_r01_c01', 'section_26_r01_c01', 'section_93_r01_c01',
    'section_96_r01_c01', 'section_78_r01_c01', 'section_101_r01_c01', 'section_34_r01_c01',
    'section_15_r01_c01', 'section_39_r01_c01', 'section_31_r01_c01', 'section_20_r01_c01',
    'section_77_r01_c01', 'section_49_r01_c01', 'section_75_r01_c01', 'section_70_r01_c01'
]

with open('correct_order.txt') as f:
    order = f.read().strip().split()

filtered = [s for s in order if s in mst_sections]

with open('correct_order_mst_only.txt', 'w') as f:
    f.write(' '.join(filtered))

print(f"保留了{len(filtered)}个section，已写入correct_order_mst_only.txt") 
mst_sections = [
    'section_106_r01_c01', 'section_105_r01_c01', 'section_45_r01_c01', 'section_60_r01_c01',
    'section_100_r01_c01', 'section_87_r01_c01', 'section_91_r01_c01', 'section_53_r01_c01',
    'section_69_r01_c01', 'section_46_r01_c01', 'section_104_r01_c01', 'section_7_r01_c01',
    'section_11_r01_c01', 'section_67_r01_c01', 'section_86_r01_c01', 'section_85_r01_c01',
    'section_55_r01_c01', 'section_63_r01_c01', 'section_1_r01_c01', 'section_92_r01_c01',
    'section_6_r01_c01', 'section_57_r01_c01', 'section_89_r01_c01', 'section_12_r01_c01',
    'section_88_r01_c01', 'section_16_r01_c01', 'section_29_r01_c01', 'section_30_r01_c01',
    'section_74_r01_c01', 'section_25_r01_c01', 'section_9_r01_c01', 'section_43_r01_c01',
    'section_65_r01_c01', 'section_97_r01_c01', 'section_62_r01_c01', 'section_102_r01_c01',
    'section_36_r01_c01', 'section_71_r01_c01', 'section_48_r01_c01', 'section_66_r01_c01',
    'section_4_r01_c01', 'section_22_r01_c01', 'section_26_r01_c01', 'section_93_r01_c01',
    'section_96_r01_c01', 'section_78_r01_c01', 'section_101_r01_c01', 'section_34_r01_c01',
    'section_15_r01_c01', 'section_39_r01_c01', 'section_31_r01_c01', 'section_20_r01_c01',
    'section_77_r01_c01', 'section_49_r01_c01', 'section_75_r01_c01', 'section_70_r01_c01'
]

with open('correct_order.txt') as f:
    order = f.read().strip().split()

filtered = [s for s in order if s in mst_sections]

with open('correct_order_mst_only.txt', 'w') as f:
    f.write(' '.join(filtered))

print(f"保留了{len(filtered)}个section，已写入correct_order_mst_only.txt") 
 
 
 
 
 
 