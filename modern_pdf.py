# To read the PDF
import PyPDF2
# To analyze the PDF layout and extract text
from pdfminer.high_level import extract_pages, extract_text
from pdfminer.layout import LTTextContainer, LTChar, LTRect, LTFigure
# To extract the images from the PDFs
from PIL import Image, ImageDraw
from pdf2image import convert_from_path
import pdfplumber
# To perform OCR to extract text from images
import pytesseract
import pdf2image
import numpy as np
import random
# To remove the additional created files
import os
from models.Yolo_main_class import YOLOClass
import config
from dataloader import Loader
from create_fb2 import FB2


# Create function to extract text

class ModernPdf:
    def __init__(self, path, fb2_obj):
        self.YOLOMain = YOLOClass(config.yolo_weight_mod)
        self.current_page = 0
        # Create a pdf file object
        self.pdfFileObj = open(path, 'rb')
        # Create a pdf reader object
        self.images = {}
        self.pdfReaded = PyPDF2.PdfReader(self.pdfFileObj)
        # Create the dictionary to extract text from each image
        self.text_per_page = []
        self.pdf = pdfplumber.open(path)
        self.page_text = []
        self.w, self.h = 0, 0
        self.w_cof, self.h_cof = 0, 0
        self.text_from_images = {}
        self.lines = []
        self.FB2 = fb2_obj

    def to_jpg(self, n):
        page = self.pdfReaded.pages[n]
        writer = PyPDF2.PdfWriter()
        writer.add_page(page)
        with open('output.pdf', 'wb') as outfile:
            writer.write(outfile)
        c = pdf2image.convert_from_path('output.pdf')
        Image.fromarray(np.asarray(c[0])).save('pdf_img.jpg')
        os.remove('output.pdf')
        return 'pdf_img.jpg'

    def find_images(self, page_num):
        page = self.to_jpg(page_num)
        blocks = self.YOLOMain.forward(page)
        im = Image.open(page)
        w_im, h_im = im.size
        w_pdf, h_pdf = self.w, self.h
        self.w_cof, self.h_cof = w_pdf / w_im, h_pdf / h_im
        for res in blocks:
            boxes = res.boxes.cpu()  # Boxes object for bounding box outputs
            class_name = boxes.cls.numpy()
            for n, cls in enumerate(class_name):
                if cls == 1:
                    x, y, w, h = boxes.xywh.numpy()[n]
                    self.images[f'image_{n}'] = [x * self.w_cof, y * self.h_cof, w * self.w_cof, h * self.h_cof]
                    im_crop = im.crop((x - w / 2, y - h / 2, x + w / 2, y + h / 2))
                    im_crop.save(f'image_{n}.jpg')

    def del_repeat_imgs(self, page_elements):
        for i, component in enumerate(page_elements):
            # Extract the element of the page layout
            element = component[1]
            if isinstance(element, LTFigure):
                x0, y0up, x1, y1up = element.bbox
                x_center, y_center = x0 + (x1 - x0) / 2, self.h - (y0up + (y1up - y0up) / 2)
                new_images = {}
                for im in self.images.keys():
                    x, y, w, h = self.images[im]
                    if np.abs(x_center - x) < w / 2 and np.abs(y_center - y) / 2 < h/2:
                        pass
                    else:
                        new_images[im] = self.images[im]
                self.images = new_images

    def del_repeat_tables(self, tables):
        for table in tables:
            x0, y0up, x1, y1up = table.bbox
            x_center, y_center = x0 + (x1 - x0) / 2, self.h - (y0up + (y1up - y0up) / 2)
            new_images = {}
            for im in self.images.keys():
                x, y, w, h = self.images[im]
                if np.abs(x_center - x) < w / 2 and np.abs(y_center - y) / 2 < h / 2:
                    pass
                else:
                    new_images[im] = self.images[im]

    def text_overlap_imgs(self, element):
        x0, y0up, x1, y1up = element.bbox
        x_center, y_center = x0 + (x1 - x0) / 2, self.h - (y0up + (y1up - y0up) / 2)
        new_images = [x0 / self.w_cof, (self.h - y1up) / self.h_cof, x1 / self.w_cof, (self.h - y0up) / self.h_cof]
        self.lines.append(new_images)
        for im in self.images.keys():
            x, y, w, h = self.images[im]
            if np.abs(x_center - x) < w / 2 and np.abs(y_center - y) < h / 2:
                self.page_text.append(f'[{im}.jpg]')
                return False
        return True

    def forward(self, page, pagenum):
        pageObj = self.pdfReaded.pages[pagenum]
        self.w, self.h = float(pageObj.mediabox.width), float(pageObj.mediabox.height)
        self.page_text = []
        self.images = {}
        self.text_from_images = {}
        self.page_content = []
        self.text_per_page = [self.page_text, self.text_from_images, self.page_content]
        # Find all the elements
        page_elements = [(element.y1, element) for element in page._objs]
        # Sort all the element as they appear in the page
        page_elements.sort(key=lambda a: a[0], reverse=True)
        exist_table = 0
        # Find the examined page
        page_tables = self.pdf.pages[pagenum]
        # Find the number of tables in the page
        tables = page_tables.find_tables()
        self.find_images(pagenum)
        self.del_repeat_imgs(page_elements)
        # Initialize the number of the examined tables
        table_in_page = -1
        # Open the pdf file
        if len(tables) != 0:
            table_in_page = 0
            exist_table = 1
            self.del_repeat_tables(tables)
        # Find the elements that composed a page
        for i, component in enumerate(page_elements):
            # Extract the element of the page layout
            element = component[1]
            if exist_table and is_element_inside_any_table(element, page, tables) and self.text_overlap_imgs(element):
                table_found = find_table_for_element(element, page, tables)
                if table_found == table_in_page and table_found != None:
                    crop_table(tables[table_found], pageObj, table_in_page)
                    self.page_text.append(f'[table_{table_in_page}.jpg]')
                    continue

            # Check if the element is text element
            if isinstance(element, LTTextContainer):
                if self.text_overlap_imgs(element):
                    # Use the function to extract the text and format for each text element
                    line_text = text_extraction(element)
                    # Append the text of each line to the page text
                    self.page_text.append(line_text)

            # Check the elements for images
            if isinstance(element, LTFigure):
                # Crop the image from PDF
                crop_image(element, pageObj, f'image_from_miner_{i}.jpg')
                # Convert the croped pdf to image
                # Extract the text from image
                # Add a placeholder in the text and format lists
                self.page_text.append(f'[image_from_miner_{i}.jpg]')
                # Update the flag for image detection

        # Add the list of list as value of the page key
        self.text_per_page = [self.page_text, self.text_from_images]
        self.del_repeats()

    def del_repeats(self):
        slac_mas = []
        next_join = False
        text_last = []
        for z, text in enumerate(self.text_per_page[0]):
            if ('image' in text or 'table' in text) and '.jpg' in text and '[' in text and ']' in text:
                if text not in slac_mas:
                    new_name = f'{random.randint(1, 1000000000000)}{text[1:-1]}'
                    os.rename(text[1:-1], new_name)
                    slac_mas.append(text)
                    self.FB2.image(new_name)
                    os.remove(new_name)
            else:
                if '-' in text:
                    if text.replace(' ', '')[-1] == '-':
                        if z != len(self.text_per_page[0]) - 1:
                            text_last.append(''.join(text.split('-')[:-1]))
                            continue
                        else:
                            text_last.append(text)
                    else:
                        text_last.append(text)
                elif text == ' ':
                    pass
                elif all(f in '0123456789' for f in text.replace(' ', '')):
                    pass
                elif 'Глава' in text and (z == len(self.text_per_page[0]) - 1 or z == 0 or z == 1):
                    pass
                else:
                    text_last.append(text)
                text_last = ''.join(text_last)
                text_last = text_last.replace(' - ', '#@!')
                text_last = text_last.replace('- ', '')
                text_last = text_last.replace('#@!', ' - ')
                text_last = text_last.replace('<', '')
                text_last = text_last.replace('>', '')
                text_last = text_last.replace('&', '')
                text_last = text_last.replace('\'', '')
                text_last = text_last.replace('\"', '')
                self.FB2.text(text_last)
                text_last = []

       # show_bbox(self.lines)


def text_extraction(element):
    # Extracting the text from the in line text element
    line_text = element.get_text()

    return ' '.join(line_text.split('\n'))


def crop_image(element, pageObj, name):
    # Get the coordinates to crop the image from PDF
    [image_left, image_top, image_right, image_bottom] = [element.x0, element.y0, element.x1, element.y1]
    # Crop the page using coordinates (left, bottom, right, top)
    pageObj.mediabox.lower_left = (image_left, image_bottom)
    pageObj.mediabox.upper_right = (image_right, image_top)
    # Save the cropped page to a new PDF
    cropped_pdf_writer = PyPDF2.PdfWriter()
    cropped_pdf_writer.add_page(pageObj)
    # Save the cropped PDF to a new file
    with open('cropped_image.pdf', 'wb') as cropped_pdf_file:
        cropped_pdf_writer.write(cropped_pdf_file)
    c = pdf2image.convert_from_path('cropped_image.pdf')
    Image.fromarray(np.asarray(c[0])).save(name)
# Create a function to convert the PDF to images


def crop_table(element, pageObj, tab):
    # Get the coordinates to crop the image from PDF
    image_left, image_top, image_right, image_bottom = element.bbox
    # Crop the page using coordinates (left, bottom, right, top)
    pageObj.mediabox.lower_left = (image_left, image_bottom)
    pageObj.mediabox.upper_right = (image_right, image_top - 30)
    # Save the cropped page to a new PDF
    cropped_pdf_writer = PyPDF2.PdfWriter()
    cropped_pdf_writer.add_page(pageObj)
    # Save the cropped PDF to a new file
    with open('cropped_table.pdf', 'wb') as cropped_pdf_file:
        cropped_pdf_writer.write(cropped_pdf_file)
    c = pdf2image.convert_from_path('cropped_table.pdf')
    Image.fromarray(np.asarray(c[0])).save(f'table_{tab}.jpg')

def convert_to_images(input_file,):
    images = convert_from_path(input_file)
    image = images[0]
    output_file = 'PDF_image.jpg'
    image.save(output_file, 'JPG')


def extract_table(pdf_path, page_num, table_num):
    # Open the pdf file
    pdf = pdfplumber.open(pdf_path)
    # Find the examined page
    table_page = pdf.pages[page_num]
    # Extract the appropriate table
    table = table_page.extract_tables()[table_num]

    return table


# Convert table into appropriate fromat
def table_converter(table):
    table_string = ''
    # Iterate through each row of the table
    for row_num in range(len(table)):
        row = table[row_num]
        # Remove the line breaker from the wrapted texts
        cleaned_row = [
            item.replace('\n', ' ') if item is not None and '\n' in item else 'None' if item is None else item for item
            in row]
        # Convert the table into a string
        table_string += ('|' + '|'.join(cleaned_row) + '|' + '\n')
    # Removing the last line break
    table_string = table_string[:-1]
    return table_string


# Create a function to check if the element is in any tables present in the page
def is_element_inside_any_table(element, page, tables):
    x0, y0up, x1, y1up = element.bbox
    # Change the cordinates because the pdfminer counts from the botton to top of the page
    y0 = page.bbox[3] - y1up
    y1 = page.bbox[3] - y0up
    for table in tables:
        tx0, ty0, tx1, ty1 = table.bbox
        if tx0 <= x0 <= x1 <= tx1 and ty0 <= y0 <= y1 <= ty1:
            return True
    return False


# Function to find the table for a given element
def find_table_for_element(element, page, tables):
    x0, y0up, x1, y1up = element.bbox
    # Change the cordinates because the pdfminer counts from the botton to top of the page
    y0 = page.bbox[3] - y1up
    y1 = page.bbox[3] - y0up
    for i, table in enumerate(tables):
        tx0, ty0, tx1, ty1 = table.bbox
        if tx0 <= x0 <= x1 <= tx1 and ty0 <= y0 <= y1 <= ty1:
            return i  # Return the index of the table
    return None

def box(mass):
    lines = []
    for i in mass:
        x1, y1, x2, y2 = i
        lines.append([(x1, y1), (x2, y1), (x2, y2), (x1, y2), (x1, y1)])
    return lines


def show_bbox(mass):
    image = Image.open('pdf_img.jpg')
    draw = ImageDraw.Draw(image)
    lines = box(mass)
    for z in lines:
        draw.line(
            xy=z, fill='red', width=2)
    image.show()



"""# Find the PDF path
pdf_path = 'pdf_files/2346.pdf'

# Create a pdf file object
pdfFileObj = open(pdf_path, 'rb')
# Create a pdf reader object
pdfReaded = PyPDF2.PdfReader(pdfFileObj)

# We extract the pages from the PDF

# Close the pdf file object
pdfFileObj.close()"""

# Delete the additional files created if image is detected
"""if image_flag:
    os.remove('cropped_image.pdf')
    os.remove('PDF_image.png')"""
"""s = FB2()
z = ModernPdf(pdf_path, s)
load = Loader(pdf_path)
for pagenum, page in enumerate(extract_pages(pdf_path)):
    if pagenum == 12:
        z.forward(load.extract_page(pagenum), pagenum)
        break"""
# Display the content of the page