"""Build a TensorRT engine from a Fast-FoundationStereo single ONNX file.

The pip tensorrt wheel ships no trtexec binary, so this is the equivalent of:
    trtexec --onnx=<in.onnx> --saveEngine=<out.engine> --fp16

An engine is specific to ONE GPU model and ONE TensorRT version. This one is
built on the RTX A6000 with the TensorRT version printed below. The consumer
(realtime/ffs_trt.py in the humble pixi env) must run the SAME TensorRT
version. Rebuild after any GPU or TensorRT change.

Usage:
    python build_engine.py <in.onnx> <out.engine>
"""
import sys

import tensorrt as trt


def main(onnx_path, engine_path):
    print(f"TensorRT {trt.__version__}")
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(0)
    parser = trt.OnnxParser(network, logger)
    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print(parser.get_error(i))
            sys.exit(1)
    config = builder.create_builder_config()
    config.set_flag(trt.BuilderFlag.FP16)
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 8 << 30)
    blob = builder.build_serialized_network(network, config)
    if blob is None:
        print("engine build FAILED")
        sys.exit(1)
    with open(engine_path, "wb") as f:
        f.write(blob)
    print(f"engine -> {engine_path}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
