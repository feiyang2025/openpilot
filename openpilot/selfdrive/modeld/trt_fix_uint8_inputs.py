#!/usr/bin/env python3
"""Rewrite UINT8 image inputs so TensorRT can parse the graph.

TensorRT 10.16 refuses to import a graph in which a UINT8 tensor feeds certain
ops. The big model this fork runs opens with

    Concat(img, big_img) -> cat          # both UINT8
    Cast(cat) -> _to_copy                # to = FLOAT16
    Sub(_to_copy, mean) / Div(.., std)

and the Concat is exactly one of those ops:

    [E] importerUtils.cpp:46 In function checkNotInvalidType:
    [8] Assertion failed: status.is_success(): Found unsupported input type of UINT8
    [E] Failed to parse onnx file

This is not a general "make any uint8 model work" pass. It rewrites the image
inputs to FLOAT16 and drops the UINT8->FLOAT16 Cast that immediately followed,
rewiring its consumers onto the concat result. Because the Cast was the only
place the 0..255 integers became floats, the arithmetic downstream is unchanged
- this is a dtype-plumbing edit, not a numeric one.

Anything unexpected is left alone and reported, so a model this does not
understand still fails loudly in trtexec rather than being silently altered.

Usage:
  python trt_fix_uint8_inputs.py IN.onnx OUT.onnx
"""

import shutil
import sys
from pathlib import Path

import onnx
from onnx import TensorProto as TP

FLOAT16 = TP.FLOAT16


def _uint8_input_names(graph) -> list[str]:
  return [i.name for i in graph.input if i.type.tensor_type.elem_type == TP.UINT8]


def _set_elem_type(value_info, elem_type: int) -> None:
  value_info.type.tensor_type.elem_type = elem_type


def _cast_to_f16(node) -> bool:
  return node.op_type == "Cast" and any(a.name == "to" and a.i == FLOAT16 for a in node.attribute)


def rewrite(src: Path, dst: Path) -> int:
  model = onnx.load(str(src))
  graph = model.graph

  uint8_inputs = _uint8_input_names(graph)
  if not uint8_inputs:
    print(f"{src.name}: no UINT8 graph inputs, copying unchanged")
    shutil.copyfile(src, dst)
    return 0

  declared = {vi.name: vi for vi in list(graph.value_info) + list(graph.input)}

  # Walk forward from the UINT8 inputs through ops that preserve dtype, so we know
  # exactly which tensors the Cast we are about to remove was feeding.
  producer = {o: n for n in graph.node for o in n.output}
  uint8_tensors = set(uint8_inputs)
  changed = True
  while changed:
    changed = False
    for node in graph.node:
      if node.op_type in {"Cast", "Identity"}:
        continue
      if any(i in uint8_tensors for i in node.input) and all(i in uint8_tensors for i in node.input if i):
        for o in node.output:
          if o not in uint8_tensors:
            uint8_tensors.add(o)
            changed = True

  removed = []
  for node in list(graph.node):
    # Only drop a Cast when its source really is one of the UINT8 tensors, and its
    # result is float16 - i.e. this is the img normalisation cast, not some cast
    # deeper in the network that happens to sit on the same value.
    if _cast_to_f16(node) and node.input and node.input[0] in uint8_tensors:
      removed.append(node)

  if not removed:
    print(f"{src.name}: UINT8 inputs found ({', '.join(uint8_inputs)}) but no "
          "UINT8->FLOAT16 Cast to remove; leaving the graph alone", file=sys.stderr)
    return 1

  retarget = {}
  for node in removed:
    for o in node.output:
      retarget[o] = node.input[0]

  for node in graph.node:
    for i, name in enumerate(node.input):
      if name in retarget:
        node.input[i] = retarget[name]
  for vi in list(graph.value_info) + list(graph.input) + list(graph.output):
    for i, name in enumerate(getattr(vi, "input", [])):
      if name in retarget:
        vi.input[i] = retarget[name]

  keep = [n for n in graph.node if n not in removed]
  del graph.node[:]
  graph.node.extend(keep)

  for name in uint8_inputs:
    _set_elem_type(declared[name], FLOAT16)
  for name in uint8_tensors:
    vi = declared.get(name)
    if vi is not None and vi.type.tensor_type.elem_type == TP.UINT8:
      _set_elem_type(vi, FLOAT16)

  dst.parent.mkdir(parents=True, exist_ok=True)
  onnx.save(model, str(dst))

  still_uint8 = [n for n in uint8_tensors if n not in retarget and n not in uint8_inputs]
  print(f"{src.name}: rewrote {len(uint8_inputs)} UINT8 input(s) "
        f"({', '.join(uint8_inputs)}) to FLOAT16, dropped {len(removed)} Cast(s) -> {dst.name}")
  if still_uint8:
    print(f"  note: {len(still_uint8)} intermediate UINT8 tensor(s) kept as-is: "
          f"{', '.join(sorted(still_uint8)[:5])}", file=sys.stderr)
  return 0


def main(argv: list[str]) -> int:
  if len(argv) != 3:
    print(__doc__.strip().splitlines()[-1], file=sys.stderr)
    return 2
  src, dst = Path(argv[1]), Path(argv[2])
  if not src.is_file():
    print(f"no such onnx: {src}", file=sys.stderr)
    return 2
  return rewrite(src, dst)


if __name__ == "__main__":
  raise SystemExit(main(sys.argv))