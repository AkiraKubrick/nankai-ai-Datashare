import os
import urllib.parse

# --- 配置区 ---
EXCLUDE_DIRS = ['.git', '.github', 'docs', 'assets', '.vscode']
EXCLUDE_FILES = ['README.md', 'SUMMARY.md', '_sidebar.md', '_navbar.md', 'generate_sidebar.py', '.nojekyll']

# 定义允许出现在侧边栏的文件类型及对应的图标
FILE_TYPES = {
    '.md': '📝',
    '.pdf': '📕',
    '.zip': '📦',
    '.rar': '📦',
    '.7z': '📦',
    '.docx': '📄',
    '.doc': '📄',
    '.pptx': '📊',
    '.ppt': '📊',
    '.xlsx': '📈',
    '.jpg': '🖼️',
    '.png': '🖼️'
}

SIDEBAR_PATH = '_sidebar.md'
# --------------

def generate_sidebar():
    sidebar_content = [
        "* [🏠 首页](README.md)\n\n"
    ]

    for root, dirs, files in sorted(os.walk('.')):
        # 过滤掉隐藏文件夹和排除名单
        dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS and not d.startswith('.')]
        
        rel_path = os.path.relpath(root, '.')
        if rel_path == '.':
            continue

        level = rel_path.count(os.sep)
        indent = "  " * level
        folder_name = os.path.basename(root)

        # 添加文件夹标题
        sidebar_content.append(f"{indent}* **{folder_name}**\n")

        # 遍历所有文件
        for file in sorted(files):
            if file in EXCLUDE_FILES or file.startswith('.'):
                continue
            
            ext = os.path.splitext(file)[1].lower()
            
            # 如果文件在我们的白名单内
            if ext in FILE_TYPES:
                emoji = FILE_TYPES[ext]
                display_name = file
                
                # 构造路径并处理 URL 编码（防止文件名中有空格导致链接失效）
                raw_path = os.path.join(rel_path, file).replace('\\', '/')
                url_path = urllib.parse.quote(raw_path)
                
                # 生成 Markdown 链接
                sidebar_content.append(f"{indent}  * {emoji} [{display_name}]({url_path})\n")

    with open(SIDEBAR_PATH, 'w', encoding='utf-8') as f:
        f.writelines(sidebar_content)
    
    print(f"✅ 多格式导航已修复！已支持 {len(FILE_TYPES)} 种文件格式。")

if __name__ == "__main__":
    generate_sidebar()