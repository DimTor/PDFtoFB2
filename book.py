import base64

image = open('datasets/coco8/images/val/000000000036.jpg', 'rb').read() #open binary file in read mode

image_64_encode = base64.encodebytes(image)
f = open("jpg1_b64.txt", "wb")
f.write(image_64_encode)
f.close()