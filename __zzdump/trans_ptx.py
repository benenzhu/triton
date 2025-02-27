import re
import sys
import os

def process_ptx_with_source(in_ptx):
    # 读取 PTX 文件
    with open(in_ptx, 'r') as f:
        ptx_lines = f.readlines()
    
    # 生成输出文件名
    out_ptx = os.path.splitext(in_ptx)[0] + "_anote.ptx"
    
    # 从PTX文件中提取源文件路径
    source_path = None
    for line in ptx_lines:
        file_match = re.match(r'\s*\.file\s+(\d+)\s+"([^"]+)"', line)
        if file_match:
            file_idx = int(file_match.group(1))
            if file_idx == 1:  # 通常第一个文件是主源文件
                source_path = file_match.group(2)
                break
    
    # 如果在PTX中找不到源文件路径，尝试从debug_info部分提取
    if not source_path:
        for i, line in enumerate(ptx_lines):
            if ".section	.debug_info" in line:
                # 在debug_info部分查找源文件路径
                for j in range(i, min(i+50, len(ptx_lines))):
                    path_match = re.search(r'/[^\s]+\.py', ptx_lines[j])
                    if path_match:
                        source_path = path_match.group(0)
                        break
                if source_path:
                    break
    
    # 如果仍然找不到源文件，尝试从文件名推断
    if not source_path:
        print("Warning: Could not find source file path in PTX. Using a default path.")
        # 尝试从PTX文件名推断源文件
        base_name = os.path.basename(in_ptx).replace("triton_kernel_", "").replace(".ptx", ".py")
        possible_paths = [
            f"/A/code/triton/__zzdump/{base_name}",
            f"./{base_name}"
        ]
        for path in possible_paths:
            if os.path.exists(path):
                source_path = path
                break
    
    if not source_path or not os.path.exists(source_path):
        print(f"Warning: Source file {source_path} not found. Proceeding without source annotations.")
        source_code = []
    else:
        # 读取原始源代码
        with open(source_path, 'r') as f:
            source_code = f.readlines()
    
    # 处理每一行 PTX
    output = []
    for line in ptx_lines:
        
        # 检测 .loc 1 31 0 或 .loc 1 31 5 这样的指令 (包含列号)
        loc_match = re.match(r'\s*\.loc\s+(\d+)\s+(\d+)\s+(\d+)', line)
        if loc_match and source_code:
            file_idx = int(loc_match.group(1))
            line_no = int(loc_match.group(2))
            col_no = int(loc_match.group(3))
            
            # 这里假设 file_idx=1 对应我们的 Python 文件
            if file_idx == 1 and 0 <= line_no - 1 < len(source_code):
                source_line = source_code[line_no - 1].rstrip()
                
                # 添加列位置标记
                if col_no >= 0 and col_no < len(source_line):
                    # 创建一个指向列位置的标记
                    col_marker = ' ' * col_no + '^'
                    output.append(f"// Source {line_no:3d}: {source_line}\n")
                    output.append(f"//        {col_no :3d}: {col_marker}\n")
                else:
                    output.append(f"// Source Line {line_no:3d}: {source_line}\n")
                    output.append(f"// Column Wrong here..\n")
        else: 
            # 保留原始行
            output.append(line)
            
    
    # 写入新 PTX
    with open(out_ptx, 'w') as f:
        f.writelines(output)
    
    return out_ptx

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python trans.py input.ptx [output.ptx] [source.py]")
        print("  - If output.ptx is not provided, will save to input_anote.ptx")
        print("  - If source.py is not provided, will try to extract from PTX")
        sys.exit(1)
    
    in_ptx = sys.argv[1]
    out_ptx = process_ptx_with_source(in_ptx)
    print(f"Annotated PTX saved to {out_ptx}")
