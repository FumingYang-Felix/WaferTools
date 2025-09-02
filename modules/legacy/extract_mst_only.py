#!/usr/bin/env python3

def extract_mst_only():
    """
    从mst_complete_fixed.txt中提取MST裸结果中的section
    """
    # 读取MST裸结果中的section
    with open('mst_raw_no_duplicates.txt', 'r') as f:
        mst_raw_line = f.read().strip()
    mst_raw_sections = set(mst_raw_line.split())
    
    # 读取修正后的完整顺序
    with open('mst_complete_fixed.txt', 'r') as f:
        fixed_line = f.read().strip()
    fixed_sections = fixed_line.split()
    
    # 提取MST裸结果中的section，保持它们在修正后顺序中的位置
    mst_only_sections = []
    for section in fixed_sections:
        if section in mst_raw_sections:
            mst_only_sections.append(section)
    
    # 保存到新文件
    with open('mst_complete_fixed_mst_only.txt', 'w') as f:
        f.write(' '.join(mst_only_sections))
    
    print(f"提取了 {len(mst_only_sections)} 个section")
    print("保存到 mst_complete_fixed_mst_only.txt")
    
    # 显示提取的section
    print("\n提取的section:")
    for i, section in enumerate(mst_only_sections, 1):
        print(f"{i:2d}. {section}")

if __name__ == "__main__":
    extract_mst_only() 

def extract_mst_only():
    """
    从mst_complete_fixed.txt中提取MST裸结果中的section
    """
    # 读取MST裸结果中的section
    with open('mst_raw_no_duplicates.txt', 'r') as f:
        mst_raw_line = f.read().strip()
    mst_raw_sections = set(mst_raw_line.split())
    
    # 读取修正后的完整顺序
    with open('mst_complete_fixed.txt', 'r') as f:
        fixed_line = f.read().strip()
    fixed_sections = fixed_line.split()
    
    # 提取MST裸结果中的section，保持它们在修正后顺序中的位置
    mst_only_sections = []
    for section in fixed_sections:
        if section in mst_raw_sections:
            mst_only_sections.append(section)
    
    # 保存到新文件
    with open('mst_complete_fixed_mst_only.txt', 'w') as f:
        f.write(' '.join(mst_only_sections))
    
    print(f"提取了 {len(mst_only_sections)} 个section")
    print("保存到 mst_complete_fixed_mst_only.txt")
    
    # 显示提取的section
    print("\n提取的section:")
    for i, section in enumerate(mst_only_sections, 1):
        print(f"{i:2d}. {section}")

if __name__ == "__main__":
    extract_mst_only() 
 
 
 
 
 
 