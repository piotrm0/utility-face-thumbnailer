""" Extract face thumbnails from images. """

import face_recognition
import argparse
import sys

from PIL import Image
from os.path import exists

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

    if not exists(args.source):
        raise ValueError("Source %s not found." % args.source)

    image = face_recognition.load_image_file(args.source)

    face_locations = face_recognition.face_locations(image)

    if len(face_locations) == 0:
        raise ValueError("No faces found.")

    postfix = None
    if len(face_locations) > 1:
        postfix = 0

    for face_location in face_locations:
        frect = rect(*face_location)

        erect = frect.expand_to_aspect(aspect)
        prect = erect.pad_ratio(padding)
        
        #print(frect)
        #print(erect)
        #print(prect)

        face_image = image[prect.top:prect.bottom,
                           prect.left:prect.right]
        pil_image = Image.fromarray(face_image).resize((width, height))

        output_file = args.target
        if postfix is not None:
            output_file += "." + str(postfix)
            postfix += 1

        pil_image.save(output_file)
        print("wrote {}".format(output_file))

try:
    run()
except ValueError as err:
    print(err, file=sys.stderr)
    exit(1)
