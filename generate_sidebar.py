import os

# --- 配置区 ---
EXCLUDE_DIRS = ['.git', '.github', 'docs', 'assets', '.vscode']  # 忽略这些文件夹
EXCLUDE_FILES = ['README.md', 'SUMMARY.md']                      # 侧边栏通常不重复列出首页
SIDEBAR_PATH = 'docs/_sidebar.md'                                # 输出路径
# --------------

def generate_sidebar():
    sidebar_content = [
        "* [🏠 首页](README.md)\n",
        "* **资料库目录**\n"
    ]

    # 获取当前根目录下所有一级目录
    for item in sorted(os.listdir('.')):
        item_path = os.path.join('.', item)
        
        # 如果是符合条件的文件夹
        if os.path.isdir(item_path) and item not in EXCLUDE_DIRS:
            sidebar_content.append(f"  * **{item}**\n")
            
            # 扫描二级目录下的 .md 文件
            for root, dirs, files in os.walk(item_path):
                # 计算缩进层级
                rel_path = os.path.relpath(root, '.')
                level = rel_path.count(os.sep)
                indent = "    " * (level + 1)
                
                # 添加子文件夹名（如果有）
                if root != item_path:
                    folder_name = os.path.basename(root)
                    sidebar_content.append(f"{indent}* **{folder_name}**\n")

                # 添加 .md 文件链接
                for file in sorted(files):
                    if file.endswith('.md') and file not in EXCLUDE_FILES:
                        file_name = file.replace('.md', '')
                        # 转换路径分隔符为网页通用的 /
                        full_path = os.path.join(root, file).replace('\\', '/')
                        sidebar_content.append(f"{indent}  * [{file_name}]({full_path})\n")
    
    # 写入文件
    with open(SIDEBAR_PATH, 'w', encoding='utf-8') as f:
        f.writelines(sidebar_content)
    
    print(f"✅ 侧边栏已成功更新至: {SIDEBAR_PATH}")

if __name__ == "__main__":
    generate_sidebar()