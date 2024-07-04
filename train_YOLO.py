import numpy as np
from ultralytics import YOLO
from mmocr.apis import MMOCRInferencer
from PIL import Image
import pytesseract

model = YOLO("yolov8n.pt")
model.train(data='my_dt2.yaml', epochs=100, overlap_mask=False)  # train the model