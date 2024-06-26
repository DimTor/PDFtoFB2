from mmocr.apis import TextRecInferencer


class SarClass:
    def __init__(self, config, weights):
        self.model = TextRecInferencer(model=config, weights=weights)\


    def forward(self, image):
        rec = self.model(image, show=False)
        return rec['predictions'][0]
