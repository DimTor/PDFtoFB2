import numpy as np
from ultralytics import YOLO
from mmocr.apis import MMOCRInferencer
from PIL import Image


class YOLOClass:
    def __init__(self, weight):
        self.model = YOLO(weight)

    def forward(self, image):
        result = self.model(image)
        return result

