#!/bin/bash

# 删除可执行文件的Bash脚本（改进版，不会删除自己）
# 作者：AI助手
# 版本：1.1

# 设置颜色代码
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 获取脚本自身的完整路径
SCRIPT_PATH="$(realpath "$0")"
# 获取脚本所在的目录
SCRIPT_DIR="$(dirname "$SCRIPT_PATH")"
# 计算目标清理目录（脚本所在目录的上一级）
TARGET_DIR="$(realpath "$SCRIPT_DIR/..")"

# 显示脚本标题
echo -e "${YELLOW}=== 可执行文件清理工具 ===${NC}"
echo ""

# 安全警告
echo -e "${RED}警告：此脚本将删除目标目录 (${TARGET_DIR}) 及其子目录中的所有可执行文件！${NC}"
echo -e "${YELLOW}注意：脚本自身 (${SCRIPT_PATH}) 不会被删除。${NC}"
echo ""

# 显示目标目录
echo -e "目标清理目录: ${GREEN}${TARGET_DIR}${NC}"
echo ""

# 第一步：查找并显示目标目录中的所有可执行文件（排除自己）
echo -e "${YELLOW}正在查找目标目录 (${TARGET_DIR}) 中的可执行文件...${NC}"
executable_files=$(find "$TARGET_DIR" -type f -executable ! -path "$SCRIPT_PATH")
count=$(echo "$executable_files" | grep -c '[^[:space:]]') # 更准确的行数统计

if [ "$count" -eq 0 ]; then
    echo -e "${GREEN}未找到任何可执行文件。${NC}"
    exit 0
fi

echo ""
echo -e "找到 ${RED}${count}${NC} 个可执行文件："
echo "$executable_files"
echo ""

# 第二步：确认删除
read -p "您确定要删除这些文件吗？(y/n): " confirm
if [[ "$confirm" != "y" && "$confirm" != "Y" ]]; then
    echo -e "${GREEN}操作已取消。${NC}"
    exit 0
fi

# 第三步：执行删除（排除自己）
echo ""
echo -e "${YELLOW}正在删除文件...${NC}"

# 使用while循环处理文件名中的特殊字符
find "$TARGET_DIR" -type f -executable ! -path "$SCRIPT_PATH" -print0 | while IFS= read -r -d '' file; do
    echo "删除: $file"
    rm -f "$file"
done

# 检查删除结果
remaining=$(find "$TARGET_DIR" -type f -executable ! -path "$SCRIPT_PATH" | wc -l)
echo ""
if [ "$remaining" -eq 0 ]; then
    echo -e "${GREEN}所有可执行文件已成功删除。${NC}"
else
    echo -e "${RED}警告：仍有 ${remaining} 个可执行文件未被删除。${NC}"
    echo "可能是由于权限不足或文件被锁定。"
fi

exit 0
