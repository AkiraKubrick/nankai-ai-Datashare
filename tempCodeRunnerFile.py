import os
import urllib.parse

# --- 配置区 ---
# 排除不需要扫描的文件夹
EXCLUDE_DIRS = ['.git', '.github', 'docs', 'assets', '.vscode', 'node_modules']
# 排除不需要在侧边栏显示的文件
EXCLUDE_FILES = ['README.md', 'SUMMARY.md', '_sidebar.md', '_navbar.md', 'generate_sidebar.py', '.nojekyll']
# 允许显示的文件类型
ALLOWED_EXTENSIONS = ['.md', '.pdf', '.zip', '.rar', '.7z', '.docx', '.doc', '.pptx', '.ppt', '.xlsx', '.jpg', '.png']

SIDEBAR_PATH = '_sidebar.md'
# --------------

def generate_sidebar():
    # 首页链接也必须是相对路径
    sidebar_content = ["* [🏠 首页](README.md)\n\n"]

    # topdown=True 允许我们过滤 dirs
    for root, dirs, files in os.walk('.', topdown=True):
        # 排除隐藏目录和特定目录
        dirs[:] = sorted([d for d in dirs if d not in EXCLUDE_DIRS and not d.startswith('.')])
        
        rel_path = os.path.relpath(root, '.')
        if rel_path == '.':
            continue

        # 计算缩进层级
        level = rel_path.count(os.sep)
        indent = "  " * level
        folder_name = os.path.basename(root)

        # 添加文件夹标题（加粗）
        sidebar_content.append(f"{indent}* **{folder_name}**\n")

        # 遍历当前目录下的文件
        for file in sorted(files):
            if file in EXCLUDE_FILES or file.startswith('.'):
                continue
            
            ext = os.path.splitext(file)[1].lower()
            if ext in ALLOWED_EXTENSIONS:
                # 【关键修正】直接构造相对路径，不拼接任何网址前缀
                raw_path = os.path.join(rel_path, file).replace('\\', '/')
                # 对中文和空格进行编码，确保链接在浏览器中有效
                url_path = urllib.parse.quote(raw_path)
                
                # 生成 Markdown 链接：* [文件名](路径/文件名)
                sidebar_content.append(f"{indent}  * [{file}]({url_path})\n")

    # 写入文件
    with open(SIDEBAR_PATH, 'w', encoding='utf-8') as f:
        f.writelines(sidebar_content)
    
    print(f"✅ 侧边栏已成功更新！当前模式：站内相对路径（无 GitHub 绝对地址）。")

if __name__ == "__main__":
    generate_sidebar()