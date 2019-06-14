""" Extract face thumbnails from images. """

import face_recognition
import argparse
import sys
import os.path as path

from PIL import Image
from os.path import exists

models = set(['cnn', 'hog'])

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('source', type=str,
                        help="Source image.")
    parser.add_argument('target', type=str,
                        help="Target image.")
    parser.add_argument('--width', type=int, default=400,
                        help="Output width.")
    parser.add_argument('--height', type=int, default=500,
                        help="Output height.")
    parser.add_argument('-p', '--padding', type=float, default=0.5,
                        help="Padding around face. Specified as fraction of wider dimension.")
    parser.add_argument('-m', '--allow_multiple', default=False, action='store_true',
                        help="Process multiple faces if found instead of throwing an error.")
    parser.add_argument('--model', type=str, default='hog',
                        help="Detection model to use. Options are 'hog' (fast) and 'cnn' (slow). See face_detection documentation for more details.")
    args = parser.parse_args()

    return args

class rect(object):
    def __init__(self, top, right, bottom, left):
        self.left = left
        self.right = right
        self.top = top
        self.bottom = bottom

        self.width = right - left
        self.height = bottom - top

        self.aspect = self.width / self.height

    def __str__(self):
        return "left: {}, right: {}, bottom: {}, top: {}".format(
            self.left, self.right, self.bottom, self.top
            )

    def rescale_to_dims(self, width, height):
        slack_width = width - self.width
        slack_height = height - self.height

        left = int(self.left - 0.5 * slack_width)
        right = left + width

        top = int(self.top - 0.5 * slack_height)
        bottom = top + height

        return rect(top, right, bottom, left)

    def expand_to_aspect(self, aspect):
        if self.aspect < aspect:
            width = int(self.height * aspect)
            height = self.height
        else:
            width = self.width
            height = int(self.width / aspect)

        return self.rescale_to_dims(width, height)

    def pad_ratio(self, padding_ratio):
        if self.aspect > 1:
            height = int(self.height * (1.0 + padding_ratio))
            width = self.width + (height - self.height)
        else:
            width = int(self.width * (1.0 + padding_ratio))
            height = self.height + (width - self.width)

        return self.rescale_to_dims(width, height)

def run():
    args = get_args()

    width = args.width
    height = args.height
    padding = args.padding
    aspect = width / height

    model = args.model

    if model not in models:
        raise ValueError("Unknown model %s. Options are %s." % (model, str(models)))

    if not exists(args.source):
        raise ValueError("Source %s not found." % args.source)

    target_base, target_ext = path.splitext(args.target)

    image = face_recognition.load_image_file(args.source)

    face_locations = face_recognition.face_locations(image, model=args.model)

    if len(face_locations) == 0:
        raise ValueError("No faces found.")

    postfix = None
    if len(face_locations) > 1:
        if not args.allow_multiple:
            raise ValueError("Multiple faces (%d) found. " % len(face_locations) +
                             "To generate multiple thumbnails, specify --allow_multiple .")
        postfix = 0

    for face_location in face_locations:
        frect = rect(*face_location)

        erect = frect.expand_to_aspect(aspect)
        prect = erect.pad_ratio(padding)

        face_image = image[prect.top:prect.bottom,
                           prect.left:prect.right]
        pil_image = Image.fromarray(face_image).resize((width, height))

        output_file = args.target
        if postfix is not None:
            output_file = target_base + "." + str(postfix) + target_ext
            postfix += 1

        pil_image.save(output_file)
        print("face at {} written to {}".format(face_location, output_file))

try:
    run()
except ValueError as err:
    print(err, file=sys.stderr)
    exit(1)
