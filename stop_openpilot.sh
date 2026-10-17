#!/usr/bin/env bash
# 干净退出 openpilot(本目录)。
# 用法: ./stop_openpilot.sh
#
# 1) 先 SIGINT manager(等价于前台 Ctrl+C, 同进程组内的东西会一起退)
# 2) 再清 Ctrl+C 够不着的: nohup 起的 fake panda、keep_alive 守护(fake_touch/magic/aid/webui)
#    和 manager 退出时如果没带走的孩子
# 3) 兜底 SIGTERM → SIGKILL, 全程只碰"属于本目录"的进程:
#    - 命令行含本目录路径
#    - 或 cwd 在本目录、且自己是 python / 本启动器的 bash 子 shell
#    交互式 shell(/bin/bash)与 sudo tw_camera_cfg 这类外部进程不在清理范围。
set -u
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

# 调用者自己的进程链(脚本本身、发起 shell、上层会话)永远不进清理名单:
# bash 工具/终端的命令行里常带着本仓库路径, 会误伤。
EXCLUDE="$$"; p=$$
while [ -n "$p" ] && [ "$p" -gt 1 ]; do
  p=$(awk '/^PPid:/{print $2}' "/proc/$p/status" 2>/dev/null)
  [ -z "$p" ] && break
  EXCLUDE="$EXCLUDE $p"
done

gather() {
  # 打印"属于本树"的 pid 清单
  local pid cmdline cwd exe out=""
  for pid in $(pgrep -f "launch_chffrplus|launch_openpilot|tools/sim/fake_panda|manager[.]py|openpilot[.][a-z]|openpilot/" 2>/dev/null; pgrep -x ui 2>/dev/null); do
    case " $EXCLUDE " in *" $pid "*) continue ;; esac
    [ -r "/proc/$pid/cwd" ] || continue
    cmdline=$(tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null || echo "")
    cwd=$(readlink "/proc/$pid/cwd" 2>/dev/null || echo "")
    case "$cmdline" in
      *"$DIR"*) out="$out $pid"; continue ;;
    esac
    # cmdline 被 setproctitle 改写(只剩 "ui" / "openpilot.system.xxx") → 看 cwd + exe
    case "$cwd" in
      "$DIR"|"$DIR"/*)
        exe=$(readlink "/proc/$pid/exe" 2>/dev/null || echo "")
        case "$exe" in
          *python*) out="$out $pid" ;;
          *bash*)
            case "$cmdline" in
              *launch_chffrplus*|*launch_openpilot*) out="$out $pid" ;;
            esac ;;
        esac ;;
    esac
  done
  echo $out
}

INT_SIGHTS=0
for pid in $(gather); do
  [ "$pid" = "$$" ] && continue
  cmdline=$(tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null | head -c 120)
  case "$cmdline" in *manager.py*|*"manager"*) kill -INT "$pid" 2>/dev/null && echo "[stop] SIGINT -> manager ($pid $cmdline)";; esac
done

sleep 2

L=$(gather)
for i in 1 2 3 4 5 6; do
  [ -z "$L" ] && break
  sleep 1
  L=$(gather)
done

if [ -n "$L" ]; then
  echo "[stop] 再给一轮 SIGTERM: $L"
  for pid in $L; do kill -TERM "$pid" 2>/dev/null; done
  sleep 3
  L=$(gather)
fi

if [ -n "$L" ]; then
  echo "[stop] 顽固进程, SIGKILL: $L"
  for pid in $L; do kill -9 "$pid" 2>/dev/null; done
fi

echo "[stop] openpilot 已退出"
