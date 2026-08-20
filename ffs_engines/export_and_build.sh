#!/bin/bash
# One-time export of Fast-FoundationStereo to ONNX + TensorRT engines.
# Runs in the TEMPORARY conda env `ffs_export` (python 3.12, torch 2.6 cu124,
# tensorrt 11.2.1.2). The live code never uses this env.
#
# Two shapes, both from checkpoint 23-36-37 (the most accurate documented one),
# 8 refinement iterations, max_disp 192:
#   736x1280  = HD720 full resolution (720 stretched to 736)   -> best precision
#   384x640   = HD720 half resolution (360 stretched to 384)   -> fast fallback
set -ex
PY=/home/yizhouch/miniforge3/envs/ffs_export/bin/python
FFS=/home/yizhouch/move_some_robots/Fast-FoundationStereo
OUT=/home/yizhouch/move_some_robots/ffs_engines
CKPT=$FFS/weights/weights/23-36-37/model_best_bp2_serialize.pth

cd $FFS

$PY scripts/make_single_onnx.py --model_dir $CKPT --save_path $OUT \
    --height 736 --width 1280 --valid_iters 8 --max_disp 192 \
    --onnx_name ffs_23-36-37_it8_736x1280

$PY scripts/make_single_onnx.py --model_dir $CKPT --save_path $OUT \
    --height 384 --width 640 --valid_iters 8 --max_disp 192 \
    --onnx_name ffs_23-36-37_it8_384x640

$PY $OUT/build_engine.py $OUT/ffs_23-36-37_it8_736x1280.onnx $OUT/ffs_23-36-37_it8_736x1280.engine
$PY $OUT/build_engine.py $OUT/ffs_23-36-37_it8_384x640.onnx $OUT/ffs_23-36-37_it8_384x640.engine

echo "EXPORT_AND_BUILD_DONE"
