import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import dlib
import cv2
import argparse as ap
from skimage.transform import resize



import keras

from keras_retinanet import models
from keras_retinanet.utils.image import read_image_bgr, preprocess_image, resize_image
from keras_retinanet.utils.visualization import draw_box, draw_caption
from keras_retinanet.utils.colors import label_color

import matplotlib.pyplot as plt
import cv2
import os
import numpy as np
import time
import serial
import time
import tensorflow as tf

def get_session():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    return tf.Session(config=config)

keras.backend.tensorflow_backend.set_session(get_session())

def detection(image):
    draw = image.copy()
    draw = cv2.cvtColor(draw, cv2.COLOR_BGR2RGB)

    # preprocess image for network
    image = preprocess_image(image)
    image, scale = resize_image(image)

    # process image
    start = time.time()
    boxes, scores, labels = model.predict_on_batch(np.expand_dims(image, axis=0))
    print("processing time: ", time.time() - start)

    # correct for image scale
    boxes /= scale
    b= ()
    # visualize detections
    for box, score, label in zip(boxes[0], scores[0], labels[0]):
        # scores are sorted so we can break
        if score < 0.5:
            break

        color = label_color(label)

        b = box.astype(int)
        return b



#
# def classifier(img, crop ):
#     image_array = np.asarray(img)
#     image_array = image_array / 255.
#     image_array = resize(image_array, (120, 360), mode='constant', anti_aliasing=True)
#     X = np.empty((1, 120, 360, 3))
#     X[0,] = image_array
#     return myclassifier.predict(X)[0][0] > .1
#

def run(source=0, dispLoc=False):
    # Create the VideoCapture object
    cam = cv2.VideoCapture(source)

    # If Camera Device is not opened, exit the program
    if not cam.isOpened():
        print("Video device or file couldn't be opened")
        exit()

    print("Press key `p` to pause the video to start tracking")
    while True:
        # Retrieve an image and Display it.
        retval, img = cam.read()
        if not retval:
            print ("Cannot capture frame device")
            exit()
        if(cv2.waitKey(10)==ord('p')):
            break
        cv2.namedWindow("Image", cv2.WINDOW_FULLSCREEN)
        cv2.imshow("Image", img)
    cv2.destroyWindow("Image")
    retval, img = cam.read()

    cv2.namedWindow("Image", cv2.WINDOW_FULLSCREEN)
    cv2.imshow("Image", img)
    timer = 0
    temp = 0

    tracker = dlib.correlation_tracker()
    points =detection(img)
    tracker.start_track(img, dlib.rectangle(*points))
    g=0
    while True:
        # Read frame from device or file
        retval, img = cam.read()
        if not retval:
            print("Cannot capture frame device | CODE TERMINATING :(")
            exit()
        g+= 1
        # Update the tracker
        #

        tracker.update(img)
        rect = tracker.get_position()
        # if(not classifier(img) and g > 150):
        #     g=0
        #     points = detection(img, rect)
        #     if len(points) < 4:
        #         points = (0, 0, 2, 2)
        #     print(points)
        #     tracker.start_track(img, dlib.rectangle(*points))


        pt1 = (int(rect.left()), int(rect.top()))
        pt2 = (int(rect.right()), int(rect.bottom()))
        if timer > 5:
            print(pt1[0])	
            if(pt1[0] > 250):
                print('send 1')
                myser.write("1\r".encode())

            elif (pt1[0] < 100):
                print('send 2')
                myser.write("2\r".encode())
            timer = 0

        timer +=1



        cv2.rectangle(img, pt1, pt2, (255, 255, 255), 3)
        cv2.namedWindow("Image", cv2.WINDOW_FULLSCREEN)
        cv2.imshow("Image", img)
        # Continue until the user presses ESC key
        if cv2.waitKey(1) == 27:
            break

    # Relase the VideoCapture object
    cam.release()


if __name__ == "__main__":
    # Parse command line arguments
    ####detection
    model = models.load_model('./resnet50_coco_best_v2.1.0.h5', backbone_name='resnet50')
    labels_to_names = {0: 'Person'}
#    myclassifier = keras.models.load_model('/home/msadegh/workspace/embedded_project/tracker/system/mymodel')

    myser = serial.Serial('/dev/ttyACM0', 9600)
    ####detection
    parser = ap.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('-d', "--deviceID", help="Device ID")
    group.add_argument('-v', "--videoFile", help="Path to Video File")
    parser.add_argument('-l', "--dispLoc", dest="dispLoc", action="store_true")
    args = vars(parser.parse_args())

    # Get the source of video
    if args["videoFile"]:
        source = args["videoFile"]
    else:
        source = int(args["deviceID"])
    run(source, args["dispLoc"])
