
import argparse
import __main__

from pruned_modules import C2fPruningFriendly

__main__.C2fPruningFriendly = C2fPruningFriendly

from ultralytics import YOLO


parser = argparse.ArgumentParser()

parser.add_argument(
    "--model",
    default="yolo11s_p20_best.pt"
)

parser.add_argument(
    "--precision",
    choices=["fp16", "int8"],
    required=True
)

parser.add_argument(
    "--data",
    default=None,
    help="Dataset YAML required for INT8 calibration"
)

args = parser.parse_args()

model = YOLO(args.model)

params = sum(
    p.numel()
    for p in model.model.parameters()
)

print("Parameters:", f"{params:,}")

if params != 6867792:
    raise RuntimeError(
        "Wrong P20 architecture"
    )

if args.precision == "fp16":

    print("Exporting TensorRT FP16...")

    result = model.export(
        format="engine",
        imgsz=640,
        batch=1,
        device=0,
        quantize=16
    )

else:

    if args.data is None:
        raise ValueError(
            "--data YAML is required for INT8"
        )

    print("Exporting TensorRT INT8...")

    result = model.export(
        format="engine",
        imgsz=640,
        batch=1,
        device=0,
        quantize=8,
        data=args.data,
        fraction=1.0
    )

print("Exported:")
print(result)
