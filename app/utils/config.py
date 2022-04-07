from typing import Any

import torch


def load_yolov5_model() -> Any:
    return torch.hub.load('ultralytics/yolov5', 'yolov5s')
