import numpy as np
from ultralytics import YOLO
from mmocr.apis import MMOCRInferencer
from PIL import Image


class YOLOBoxClass:
    def __init__(self, weight):
        self.model = YOLO(weight)

    def forward(self, image):
        result = self.model(image, show_labels=False, show_conf=False, save=True)
        return result
