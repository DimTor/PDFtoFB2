from PyPDF2 import PdfReader, PdfWriter
from pdfminer.high_level import extract_pages
from PIL import Image
import pdf2image
from io import BytesIO
import os
import numpy as np


class Loader:
    def __init__(self, name):
        self.reader = PdfReader(name)
        self.start_page = 0
        self.extract = list(extract_pages(name))

    def return_page_jpg(self, n_page):
        page = self.reader.pages[n_page]
        BytesIO(page.images[0].data)
        im = Image.open(BytesIO(page.images[0].data))
        im_ar = np.asarray(im)
        return im_ar

    def return_page(self, n_page):
        page = self.reader.pages[n_page]
        return page

    def forward(self):
        page = self.reader.pages[self.start_page]
        BytesIO(page.images[0].data)
        im = Image.open(BytesIO(page.images[0].data))
        im_ar = np.asarray(im)
        self.start_page += 1
        return im_ar

    def to_png(self, n):
        page = self.reader.pages[n]
        writer = PdfWriter()
        writer.add_page(page)
        with open('output.pdf', 'wb') as outfile:
            writer.write(outfile)
        c = pdf2image.convert_from_path('output.pdf')
        os.remove('output.pdf')
        return np.asarray(c[0])

    def forward_jpg(self):
        page = self.reader.pages[self.start_page]
        writer = PdfWriter()
        writer.add_page(page)
        with open('output.pdf', 'wb') as outfile:
            writer.write(outfile)
        c = pdf2image.convert_from_path('output.pdf')
        os.remove('output.pdf')
        self.start_page += 1
        Image.fromarray(np.asarray(c[0])).save(f'my_data_comp.jpg')
        return 'my_data_comp.jpg'

    def forward_jpg_n(self, n):
        page = self.reader.pages[n]
        writer = PdfWriter()
        writer.add_page(page)
        with open('output.pdf', 'wb') as outfile:
            writer.write(outfile)
        c = pdf2image.convert_from_path('output.pdf')
        os.remove('output.pdf')
        Image.fromarray(np.asarray(c[0])).save(f'my_data_comp.jpg')
        return 'my_data_comp.jpg'

    def len_pages(self):
        return len(self.reader.pages)

    def is_scan(self, n):
        if len(self.return_page(n).images):
            return True
        return False

    def extract_page(self, n):
        return self.extract[n]





"""s = Loader('pdf_files/2346_removed(1).pdf')
print(s.len_pages())
for i in range(1):
    Image.fromarray(s.to_png(i)).save(f'my_data/inf/yy{i}.jpg')"""
