"""Build-time rectangular YOLOv4-tiny graph smoke test for the fork."""
from pathlib import Path

import torch

from tool.darknet2pytorch import Darknet


def main() -> None:
    cfg = Path("cfg/yolov4-tiny.cfg")
    model = Darknet(str(cfg), width=224, height=160)
    with torch.inference_mode():
        heads = model(torch.zeros(1, 3, 160, 224))
    expected = [(1, 255, 5, 7), (1, 255, 10, 14)]
    actual = [tuple(head.shape) for head in heads]
    if actual != expected:
        raise AssertionError(f"Expected {expected}, got {actual}")
    print(f"Tianxiaomo YOLOv4-tiny smoke test passed: {actual}")


if __name__ == "__main__":
    main()
