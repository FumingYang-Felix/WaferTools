#!/usr/bin/env python3

def verify_mst_order():
    """
    验证MST裸结果和修正后顺序的一致性
    """
    # 读取MST裸结果
    with open('mst_raw_no_duplicates.txt', 'r') as f:
        mst_raw_line = f.read().strip()
    mst_raw_sections = mst_raw_line.split()
    
    # 读取修正后的完整顺序
    with open('mst_complete_fixed.txt', 'r') as f:
        fixed_line = f.read().strip()
    fixed_sections = fixed_line.split()
    
    print(f"MST裸结果长度: {len(mst_raw_sections)}")
    print(f"修正后完整顺序长度: {len(fixed_sections)}")
    
    # 从修正后的顺序中提取MST裸结果中的section，保持顺序
    extracted_sections = []
    for section in fixed_sections:
        if section in mst_raw_sections:
            extracted_sections.append(section)
    
    print(f"提取的section数量: {len(extracted_sections)}")
    
    # 比较两个顺序
    if extracted_sections == mst_raw_sections:
        print("✓ MST裸结果和修正后顺序中的MST部分完全一致！")
        print("✓ 主树没有问题，问题出现在插入过程中")
    else:
        print("✗ 发现不一致！")
        print("MST裸结果:", mst_raw_sections)
        print("提取结果:", extracted_sections)
        
        # 找出第一个不一致的位置
        for i, (orig, extr) in enumerate(zip(mst_raw_sections, extracted_sections)):
            if orig != extr:
                print(f"第一个不一致位置 {i+1}: {orig} vs {extr}")
                break
    
    # 显示插入的section
    inserted_sections = [s for s in fixed_sections if s not in mst_raw_sections]
    print(f"\n插入的section数量: {len(inserted_sections)}")
    print("插入的section:", inserted_sections)

if __name__ == "__main__":
    verify_mst_order() 

def verify_mst_order():
    """
    验证MST裸结果和修正后顺序的一致性
    """
    # 读取MST裸结果
    with open('mst_raw_no_duplicates.txt', 'r') as f:
        mst_raw_line = f.read().strip()
    mst_raw_sections = mst_raw_line.split()
    
    # 读取修正后的完整顺序
    with open('mst_complete_fixed.txt', 'r') as f:
        fixed_line = f.read().strip()
    fixed_sections = fixed_line.split()
    
    print(f"MST裸结果长度: {len(mst_raw_sections)}")
    print(f"修正后完整顺序长度: {len(fixed_sections)}")
    
    # 从修正后的顺序中提取MST裸结果中的section，保持顺序
    extracted_sections = []
    for section in fixed_sections:
        if section in mst_raw_sections:
            extracted_sections.append(section)
    
    print(f"提取的section数量: {len(extracted_sections)}")
    
    # 比较两个顺序
    if extracted_sections == mst_raw_sections:
        print("✓ MST裸结果和修正后顺序中的MST部分完全一致！")
        print("✓ 主树没有问题，问题出现在插入过程中")
    else:
        print("✗ 发现不一致！")
        print("MST裸结果:", mst_raw_sections)
        print("提取结果:", extracted_sections)
        
        # 找出第一个不一致的位置
        for i, (orig, extr) in enumerate(zip(mst_raw_sections, extracted_sections)):
            if orig != extr:
                print(f"第一个不一致位置 {i+1}: {orig} vs {extr}")
                break
    
    # 显示插入的section
    inserted_sections = [s for s in fixed_sections if s not in mst_raw_sections]
    print(f"\n插入的section数量: {len(inserted_sections)}")
    print("插入的section:", inserted_sections)

if __name__ == "__main__":
    verify_mst_order() 
 
 
 
 
 
 