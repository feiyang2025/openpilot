#!/usr/bin/env bash

export API_HOST=https://api.konik.ai
export ATHENA_HOST=wss://athena.konik.ai

# 必须先切到脚本所在目录: 下面 exec 的是相对路径, 从别处调用会报
# "行 5: ./launch_chffrplus.sh: 没有那个文件或目录" 且看起来像"没反应"。
cd "$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
exec ./launch_chffrplus.sh
