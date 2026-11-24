#!/usr/bin/env sh
# Linux 与 macOS 环境下的一键安装脚本
# 用法: curl -fsSL https://raw.githubusercontent.com/.../install.sh | bash

set -e

INSTALL_DIR="${HOME}/.shed"
BIN_DIR="${INSTALL_DIR}/bin"

echo "=== 安装 shed ==="

# 1. 检查基础工具
if ! command -v node >/dev/null 2>&1; then
    echo "提示: 未检测到 Node.js，shed 交互界面需要 Node.js (>=18)。"
    echo "请先安装 Node.js: https://nodejs.org"
fi

mkdir -p "${BIN_DIR}"

# 2. 生成本地全局可执行文件软链接或启动包装脚本
WRAPPER="${BIN_DIR}/shed"
cat <<'EOF' >"${WRAPPER}"
#!/usr/bin/env sh
DIR="$(cd "$(dirname "$0")/.." && pwd)"

# 优先探测系统 npx / node 全局包或本地环境
if command -v shed >/dev/null 2>&1 && [ "$(command -v shed)" != "$0" ]; then
    exec "$(command -v shed)" "$@"
fi

if [ -f "${DIR}/shed-tui/bin/shed.js" ]; then
    exec node "${DIR}/shed-tui/bin/shed.js" "$@"
fi

exec npx my-pi-agent "$@"
EOF

chmod +x "${WRAPPER}"

# 3. 环境变量 PATH 配置检查
SHELL_NAME="$(basename "${SHELL:-sh}")"
RC_FILE=""

case "${SHELL_NAME}" in
zsh)
    RC_FILE="${HOME}/.zshrc"
    ;;
bash)
    if [ -f "${HOME}/.bash_profile" ]; then
        RC_FILE="${HOME}/.bash_profile"
    else
        RC_FILE="${HOME}/.bashrc"
    fi
    ;;
*)
    RC_FILE="${HOME}/.profile"
    ;;
esac

if [ -n "${RC_FILE}" ] && [ -f "${RC_FILE}" ]; then
    if ! grep -q "\.shed/bin" "${RC_FILE}"; then
        echo "" >>"${RC_FILE}"
        echo '# shed PATH' >>"${RC_FILE}"
        echo 'export PATH="${HOME}/.shed/bin:${PATH}"' >>"${RC_FILE}"
        echo "✓ 已将 ${BIN_DIR} 添加到 ${RC_FILE}"
    fi
fi

echo "✓ shed 安装就绪: ${WRAPPER}"
echo "提示: 重启终端或执行 'source ${RC_FILE}' 后即可直接敲 'shed' 体验！"
