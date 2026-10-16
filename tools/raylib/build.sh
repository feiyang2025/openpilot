#!/usr/bin/env bash
set -e

# 本机直连 github.com 是间歇性的(git ls-remote 能通, clone 却 135s 超时), 而本地
# 代理能整段稳定 clone。RAYLIB_GIT_PROXY=http://127.0.0.1:7897 ./build.sh 即可。
# raygui.h 优先取 $DIR/raygui.h —— 那是本脚本对同一锁定 commit 的预取件
# (与代理重新下载的文件 sha256 一致: 29a9c7bf8cbea93a), 因为
# raw.githubusercontent.com 从本机直连是 SSL 超时、连代理也时好时坏。

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && pwd )"
cd $DIR

RAYLIB_PLATFORM="PLATFORM_DESKTOP"

ARCHNAME=$(uname -m)
if [ -f /TICI ]; then
  ARCHNAME="larch64"
  RAYLIB_PLATFORM="PLATFORM_COMMA"
fi

if [[ "$OSTYPE" == "darwin"* ]]; then
  ARCHNAME="Darwin"
fi

INSTALL_DIR="$DIR/$ARCHNAME"
rm -rf $INSTALL_DIR
mkdir -p $INSTALL_DIR

INSTALL_H_DIR="$DIR/include"
rm -rf $INSTALL_H_DIR
mkdir -p $INSTALL_H_DIR

if [ ! -d raylib_repo ]; then
  git clone -b master --no-tags https://github.com/commaai/raylib.git raylib_repo
fi

cd raylib_repo

COMMIT="66030a7de62c9e1ee8ab30a1d657a740333bb4f2"
git fetch origin $COMMIT
git reset --hard $COMMIT
git clean -xdff .

cd src

make -j$(nproc) PLATFORM=$RAYLIB_PLATFORM RAYLIB_RELEASE_PATH=$INSTALL_DIR
cp raylib.h raymath.h rlgl.h $INSTALL_H_DIR/
echo "raylib development files installed/updated in $INSTALL_H_DIR"

# this commit needs to be in line with raylib
set -x
RAYGUI_COMMIT="76b36b597edb70ffaf96f046076adc20d67e7827"
# raw.githubusercontent.com is not reachable from this host (SSL timeout), while
# github.com itself is. Prefer a raygui.h that is already here - it is the file
# this script produced for the same pinned commit - and only fall back to a
# download, going through the local proxy when one is set.
RAYGUI_URL="https://raw.githubusercontent.com/raysan5/raygui/$RAYGUI_COMMIT/src/raygui.h"
if [ -s "$DIR/raygui.h" ]; then
  cp "$DIR/raygui.h" $INSTALL_H_DIR/raygui.h
  echo "raygui.h taken from $DIR/raygui.h (pre-fetched for $RAYGUI_COMMIT)"
elif curl -fsSLo $INSTALL_H_DIR/raygui.h "$RAYGUI_URL"; then
  echo "raygui.h downloaded"
elif [ -n "${HTTPS_PROXY:-}" ] || curl -fsSLo $INSTALL_H_DIR/raygui.h --proxy http://127.0.0.1:7897 "$RAYGUI_URL"; then
  echo "raygui.h downloaded via proxy"
else
  echo "ERROR: cannot obtain raygui.h ($RAYGUI_COMMIT)" >&2
  exit 1
fi

# 原脚本把绑定构建门控在 /TICI 下: AGNOS 上 pyray 是预置的, 不必重编。
# 本 fork 的 UI 用了 raygui (GuiTextAlignment / gui_set_font / gui_set_style),
# 而 .venv 里预装的 pyray 不带 raygui, 于是 ui 起来就 AttributeError 崩、
# 被 manager 无限重启。所以这里默认就编绑定, 用 SKIP_BINDINGS=1 可退回旧行为。
if [ "${SKIP_BINDINGS:-0}" != "1" ]; then

  # Building the python bindings
  cd $DIR

  if [ ! -d raylib_python_repo ]; then
    git clone -b master --no-tags https://github.com/commaai/raylib-python-cffi.git raylib_python_repo
  fi

  cd raylib_python_repo

  BINDINGS_COMMIT="ef8141c7979d5fa630ef4108605fc221f07d8cb7"
  git fetch origin $BINDINGS_COMMIT
  git reset --hard $BINDINGS_COMMIT
  git clean -xdff .

  RAYLIB_PLATFORM=$RAYLIB_PLATFORM RAYLIB_INCLUDE_PATH=$INSTALL_H_DIR RAYLIB_LIB_PATH=$INSTALL_DIR python setup.py bdist_wheel
  cd $DIR

  rm -rf wheel
  mkdir wheel
  cp raylib_python_repo/dist/*.whl wheel/

fi
