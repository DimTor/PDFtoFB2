import os
from typing import List, Tuple

from ..data_generator import FakeTextDataGenerator
from ..utils import load_dict, load_fonts



class GeneratorFromStrings:
    """Generator that uses a given list of strings"""

    def __init__(
        self,
        strings: List[str],
        count: int = -1,
        fonts: List[str] = [],
        language: str = "en",
        size: int = 32,
        skewing_angle: int = 0,
        random_skew: bool = False,
        blur: int = 0,
        random_blur: bool = False,
        background_type: int = 0,
        distorsion_type: int = 0,
        distorsion_orientation: int = 0,
        is_handwritten: bool = False,
        width: int = -1,
        alignment: int = 1,
        text_color: str = "#282828",
        orientation: int = 0,
        space_width: float = 1.0,
        character_spacing: int = 0,
        margins: Tuple[int, int, int, int] = (5, 5, 5, 5),
        fit: bool = False,
        output_mask: bool = False,
        word_split: bool = False,
        image_dir: str = os.path.join(
            "..", os.path.split(os.path.realpath(__file__))[0], "images"
        ),
        stroke_width: int = 0,
        stroke_fill: str = "#282828",
        image_mode: str = "RGB",
        output_bboxes: int = 0,
        rtl: bool = False,
    ):
        self.count = count
        self.strings = strings
        self.fonts = fonts
        if len(fonts) == 0:
            self.fonts = load_fonts(language)
        self.rtl = rtl
        self.orig_strings = []
        self.language = language
        self.size = size
        self.skewing_angle = skewing_angle
        self.random_skew = random_skew
        self.blur = blur
        self.random_blur = random_blur
        self.background_type = background_type
        self.distorsion_type = distorsion_type
        self.distorsion_orientation = distorsion_orientation
        self.is_handwritten = is_handwritten
        self.width = width
        self.alignment = alignment
        self.text_color = text_color
        self.orientation = orientation
        self.space_width = space_width
        self.character_spacing = character_spacing
        self.margins = margins
        self.fit = fit
        self.output_mask = output_mask
        self.word_split = word_split
        self.image_dir = image_dir
        self.output_bboxes = output_bboxes
        self.generated_count = 0
        self.stroke_width = stroke_width
        self.stroke_fill = stroke_fill
        self.image_mode = image_mode

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def next(self):
        if self.generated_count == self.count:
            raise StopIteration
        self.generated_count += 1
        return (
            FakeTextDataGenerator.generate(
                self.generated_count,
                self.strings[(self.generated_count - 1) % len(self.strings)],
                self.fonts[(self.generated_count - 1) % len(self.fonts)],
                None,
                self.size,
                None,
                self.skewing_angle,
                self.random_skew,
                self.blur,
                self.random_blur,
                self.background_type,
                self.distorsion_type,
                self.distorsion_orientation,
                self.is_handwritten,
                0,
                self.width,
                self.alignment,
                self.text_color,
                self.orientation,
                self.space_width,
                self.character_spacing,
                self.margins,
                self.fit,
                self.output_mask,
                self.word_split,
                self.image_dir,
                self.stroke_width,
                self.stroke_fill,
                self.image_mode,
                self.output_bboxes,
            ),
            self.orig_strings[(self.generated_count - 1) % len(self.orig_strings)]
            if self.rtl
            else self.strings[(self.generated_count - 1) % len(self.strings)],
        )


if __name__ == "__main__":
    from recog_generator.generators.from_wikipedia import GeneratorFromWikipedia

    s = GeneratorFromWikipedia("test")
    next(s)
