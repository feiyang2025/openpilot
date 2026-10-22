#!/usr/bin/env python3
"""Install the pinned neural signal models for the opt-in shadow observer.

The observer validates both ONNX files against fixed SHA-256 digests before it
runs them. This script copies a matching pair into <shadow>/neural/ and can
enable the CUDA backend; it never downloads anything.

  python tools/signal_analysis/install_neural_models.py --check
  python tools/signal_analysis/install_neural_models.py \
      --source /path/to/models --enable --backend cuda
"""
import argparse
import hashlib
import os
from pathlib import Path
import shutil
import sys

DETECTOR_SHA = '1ad633066a1195006f4709f8fa07800dd65a74a814b3efb4c99bcc5a1a7962f6'
CLASSIFIER_SHA = '455b71b3b20d3a96aa0e49f32714ba50421f668a2f9b9907c30b1346ac8a3703'
NAMES = {'detector.onnx': DETECTOR_SHA, 'classifier.onnx': CLASSIFIER_SHA}
BACKENDS = ('auto', 'cuda', 'cpu')


def default_directory() -> Path:
  env = os.environ.get('SIGNAL_SHADOW_DIR')
  if env:
    return Path(env)
  legacy = Path('/data/signal-color-shadow')
  if os.access(legacy.parent, os.W_OK):
    return legacy
  return Path.home() / '.comma' / 'signal-color-shadow'


def sha256(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open('rb') as f:
    for block in iter(lambda: f.read(1024 * 1024), b''):
      digest.update(block)
  return digest.hexdigest()


def check(directory: Path) -> int:
  neural = directory / 'neural'
  ok = True
  for name, expected in NAMES.items():
    path = neural / name
    if not path.is_file():
      print(f'MISSING {path}')
      ok = False
      continue
    actual = sha256(path)
    print(f'{"OK     " if actual == expected else "BAD    "} {path} {actual[:16]}')
    ok &= actual == expected
  print(f'{"enabled" if (directory / "neural_enabled").is_file() else "disabled"} in {directory}')
  return 0 if ok else 1


def install(source: Path, directory: Path, backend: str, enable: bool) -> int:
  neural = directory / 'neural'
  neural.mkdir(parents=True, exist_ok=True)
  for name, expected in NAMES.items():
    src = source / name
    if not src.is_file():
      print(f'not found: {src}', file=sys.stderr)
      return 1
    actual = sha256(src)
    if actual != expected:
      print(f'{src}: sha256 {actual} does not match the pinned {expected}', file=sys.stderr)
      return 1
    shutil.copy2(src, neural / name)
    print(f'installed {neural / name}')
  (directory / 'neural_backend').write_text(backend + '\n')
  if enable:
    (directory / 'neural_enabled').write_text('1\n')
  print(f'backend={backend} enabled={enable} directory={directory}')
  return 0


if __name__ == '__main__':
  p = argparse.ArgumentParser()
  p.add_argument('--directory', type=Path, default=default_directory())
  p.add_argument('--source', type=Path, help='directory holding detector.onnx/classifier.onnx')
  p.add_argument('--backend', choices=BACKENDS, default='cuda')
  p.add_argument('--enable', action='store_true')
  p.add_argument('--disable', action='store_true')
  p.add_argument('--check', action='store_true')
  args = p.parse_args()

  if args.disable:
    (args.directory / 'neural_enabled').unlink(missing_ok=True)
    print(f'disabled in {args.directory}')
  if args.check:
    raise SystemExit(check(args.directory))
  if args.source is None:
    p.error('--source is required unless --check/--disable is used')
  raise SystemExit(install(args.source, args.directory, args.backend, args.enable))
