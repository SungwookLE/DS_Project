import cv2
from viz.config import get_parse_args
import random
from tensorflow.keras.models import load_model
import numpy as np
import sys
import os
from tensorflow import keras
import matplotlib.cm as cm
import tensorflow as tf



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


def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
        # First, we create a model that maps the input image to the activations
        # of the last conv layer as well as the output predictions
        grad_model = tf.keras.models.Model(
            [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
        )

        # Then, we compute the gradient of the top predicted class for our input image
        # with respect to the activations of the last conv layer
        with tf.GradientTape() as tape:
            last_conv_layer_output, preds = grad_model(img_array)
            if pred_index is None:
                pred_index = tf.argmax(preds[0])
            class_channel = preds[:, pred_index]

        # This is the gradient of the output neuron (top predicted or chosen)
        # with regard to the output feature map of the last conv layer
        grads = tape.gradient(class_channel, last_conv_layer_output)

        # This is a vector where each entry is the mean intensity of the gradient
        # over a specific feature map channel
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

        # We multiply each channel in the feature map array
        # by "how important this channel is" with regard to the top predicted class
        # then sum all the channels to obtain the heatmap class activation
        last_conv_layer_output = last_conv_layer_output[0]
        heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)

        # For visualization purpose, we will also normalize the heatmap between 0 & 1
        heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
        return heatmap.numpy()

def display_gradcam(img, heatmap, alpha=0.4):
        # Load the original image
    
        img = keras.preprocessing.image.img_to_array(img)
        # Rescale heatmap to a range 0-255
        heatmap = np.uint8(255 * heatmap)

        # Use jet colormap to colorize heatmap
        jet = cm.get_cmap("jet")

        # Use RGB values of the colormap
        jet_colors = jet(np.arange(256))[:, :3]
        jet_heatmap = jet_colors[heatmap]

        # Create an image with RGB colorized heatmap
        jet_heatmap = keras.preprocessing.image.array_to_img(jet_heatmap)
        jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
        jet_heatmap = keras.preprocessing.image.img_to_array(jet_heatmap)

        # Superimpose the heatmap on original image
        superimposed_img = jet_heatmap * alpha + img
        superimposed_img = keras.preprocessing.image.array_to_img(superimposed_img)
        superimposed_img = np.array(superimposed_img)
        #superimposed_img = cv2.cvtColor(superimposed_img, cv2.COLOR_RGB2BGR)

        return superimposed_img


def explainable(model, img, last_conv_layer_name, alpha =0.4):
    array = keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(array, axis=0)

    # Generate class activation heatmap
    heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer_name)
    superimposed_img = display_gradcam(img*255.0, heatmap, alpha=alpha)

    return superimposed_img


def run_demo(net, frame_provider):
    this_dict = {'c6': 0, 'c5': 1, 'c7': 2, 'c1': 3, 'c0': 4} ######################
    this_dict2 = {'c5': 'close', 'c0': 'center', 'c6': 'far', 'c1': 'phone', 'c7': 'behind'}

    model_load = load_model("./ckpt/"+net)
    for img in frame_provider:
        img = cv2.resize(img, dsize=(64,64), interpolation=cv2.INTER_AREA)
        orig_img = img.copy()
        img = img/255.0
        pred= model_load.predict(img.reshape(1, img.shape[0], img.shape[1], img.shape[2]))
        
        print("Pred label: ", (np.argmax(pred)))

        for key, val in this_dict.items():
            if val == (np.argmax(pred)):
                img1 = cv2.resize(orig_img, dsize=(320, 320), interpolation=cv2.INTER_AREA)
                cv2.putText(img1, this_dict2[key], (100,100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,128,0), 2)
        
        img2 = explainable(model_load, img, "3rd_maxpool", alpha=0.4)
        img2 = cv2.resize(img2, dsize=(320, 320), interpolation=cv2.INTER_AREA)

        img_list = [img1, img2]
        img_v = cv2.hconcat(img_list)
        cv2.imshow("OOP Classifier", img_v)
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