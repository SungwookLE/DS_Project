import cv2
from viz.config import get_parse_args
import random
from tensorflow.keras.models import load_model
import numpy as np
import sys
import os

class ImageReader(object):
    def __init__(self, file_names):
        self.file_names = file_names
        self.max_idx = len(file_names)

    def __iter__(self):
        self.idx = 0
        return self

    def __next__(self):
        if self.idx == self.max_idx:
            raise StopIteration
        img = cv2.imread(self.file_names[self.idx], cv2.IMREAD_COLOR)
        if img.size == 0:
            raise IOError('Image {} cannot be read'.format(self.file_names[self.idx]))
        self.idx = self.idx + 1
        return img


class VideoReader(object):
    def __init__(self, file_name):
        self.file_name = file_name
        try:  # OpenCV needs int to read from webcam
            self.file_name = int(file_name)
        except ValueError:
            pass

    def __iter__(self):
        self.cap = cv2.VideoCapture(self.file_name)
        width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        print("Video Shape is ({0},{1}), FPS is {2}.".format(width, height, fps))

        if not self.cap.isOpened():
            raise IOError('Video {} cannot be opened'.format(self.file_name))
        return self

    def __next__(self):
        was_read, img = self.cap.read()
        if not was_read:
            raise StopIteration
        return img
def demo_init(args):
    net = args.model_name

    frame_provider = ImageReader(args.images)
    if args.video != '':
        frame_provider = VideoReader(args.video)
        image_flag = False
    else:
        image_flag = True


    return net, frame_provider

def run_demo(net, frame_provider):
    this_dict = {'c6': 0, 'c5': 1, 'c7': 2, 'c1': 3, 'c0': 4} ######################
    this_dict2 = {'c5': 'close', 'c0': 'center', 'c6': 'far', 'c1': 'phone', 'c7': 'behind'}

    model_load = load_model("./ckpt/"+net)
    for img in frame_provider:
        img = cv2.resize(img, dsize=(64,64), interpolation=cv2.INTER_AREA)
        orig_img = img.copy()
        pred= model_load.predict(img.reshape(1, img.shape[0], img.shape[1], img.shape[2]))
        
        print(np.argmax(pred))
        for key, val in this_dict.items():
            if val == (np.argmax(pred)):
                img = cv2.resize(img, dsize=(320, 320), interpolation=cv2.INTER_AREA)
                cv2.putText(img, this_dict2[key], (100,100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,128,0), 2)

        cv2.imshow("OOP Classifier", img)
        key = cv2.waitKey(1)

        if key == 27:  # esc
            break
        elif key == 112:  # 'p'
            if delay == 1:
                delay = 0
            else:
                delay = 1

        # python run_demo.py --model_name 'model_oop_cnn' --images ../Data/safety_class_dataset/20211025_*_unbelt_nomask_jieun/Color/*.jpg
        # python run_demo.py --model_name 'model_oop_cnn' --video 0


    return