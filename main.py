import numpy as np
from ultralytics import YOLO
from mmocr.apis import MMOCRInferencer
from PIL import Image
import pytesseract

# Load a model

image = 'my_data_comp/111.jpg'
im = Image.open(image)
model = YOLO('runs/detect/train4/weights/best.pt')  # load a pretrained YOLOv8n detection model
#model.train(data='my_dt.yaml', epochs=100)  # train the model
result = model(image)  # predict on an image
ocr = MMOCRInferencer(det='FCEnet')
# Perform inference

for res in result:
    res.show()
    boxes = res.boxes.cpu()  # Boxes object for bounding box outputs
    class_name = boxes.cls.numpy()
    for n, cls in enumerate(class_name):
        if cls != 1:
            x, y, w, h = boxes.xywh.numpy()[n]
            im_crop = im.crop((x - w/2, y - h / 2, x + w / 2, y + h / 2))
            im_crop.show()
            im_crop.save('out.jpg')
            r = pytesseract.image_to_string('out.jpg', lang='rus+eng')
            print(r)