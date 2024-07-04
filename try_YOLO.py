import numpy as np
from ultralytics import YOLO
from mmocr.apis import MMOCRInferencer
from PIL import Image
import pytesseract

# Load a model

image = '/home/tor/PycharmProjects/Samsung/datasets/yolo/images/val/f0a90210-24.jpg'
image2 = '/home/tor/PycharmProjects/Samsung/my_data/27.jpg'
image3 = 'my_data/inf/yy0.jpg'
im = Image.open(image3)
model = YOLO('runs/detect/train14/weights/best.pt')  # load a pretrained YOLOv8n detection model
result = model(image3, save=True)
"""for res in result:
  #  res.show()
    boxes = res.boxes.cpu()  # Boxes object for bounding box outputs
    class_name = boxes.cls.numpy()
    for n, cls in enumerate(class_name):
        if cls != 1:
            x, y, w, h = boxes.xywh.numpy()[n]
            im_crop = im.crop((x - w/2, y - h / 2, x + w / 2, y + h / 2))
            if n == 3:
                im_crop.save('out6.jpg')"""