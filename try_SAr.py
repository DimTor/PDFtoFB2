from mmocr.apis import MMOCRInferencer, TextRecInferencer
import pytesseract

image = 'out8.png'

image2 = '/home/tor/PycharmProjects/Samsung/Sar/recognition_generator/out/textrecog_imgs/test/img_18.jpg'
config = '/home/tor/PycharmProjects/Samsung/Sar/sar_config/configs/textrecog/sar/sar_icdar.py'
weight = '/home/tor/PycharmProjects/Samsung/Sar/sar_config/work_dir/epoch_5.pth'
r = pytesseract.image_to_string(image, lang='rus')
print(r)
model = TextRecInferencer(weights=weight, model=config)
r = model(image, show=True)