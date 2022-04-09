from typing import Any

import torch

config = {
    "yolo_path": r"C:\Users\Viktor\Desktop\git\object-detection-playground\yolov5",
    "model_path": r"C:\Users\Viktor\Desktop\git\object-detection-playground\model\best.pt",
    "device": torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
}


def load_yolov5_model() -> Any:
    model = torch.hub.load(config.get("yolo_path"),
                           "custom",
                           path=config.get("model_path"),
                           force_reload=True,
                           device=config.get("device"),
                           source="local",
                           )
    return model
