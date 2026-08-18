
import argparse
import __main__

from pruned_modules import C2fPruningFriendly

# Required because checkpoint was serialized as
# __main__.C2fPruningFriendly
__main__.C2fPruningFriendly = C2fPruningFriendly

from ultralytics import YOLO


def parse_source(value):
    try:
        return int(value)
    except ValueError:
        return value


parser = argparse.ArgumentParser()

parser.add_argument(
    "--model",
    default="yolo11s_p20_best.pt"
)

parser.add_argument(
    "--source",
    default="0"
)

parser.add_argument(
    "--conf",
    type=float,
    default=0.15
)

parser.add_argument(
    "--imgsz",
    type=int,
    default=640
)

parser.add_argument(
    "--device",
    default="0"
)

args = parser.parse_args()

source = parse_source(args.source)

print("Loading P20 model...")

model = YOLO(args.model)

params = sum(
    p.numel()
    for p in model.model.parameters()
)

print("Parameters:", f"{params:,}")

if params != 6867792:
    raise RuntimeError(
        f"Unexpected model: {params:,} parameters"
    )

print("Model loaded successfully.")

model.predict(
    source=source,
    conf=args.conf,
    imgsz=args.imgsz,
    device=args.device,
    show=True,
    stream=False,
    save=False
)
