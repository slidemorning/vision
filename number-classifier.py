import os
import sys
import cv2
import numpy as np
import tensorflow as tf

oldx = oldy = -1

def onMouse(event, x, y, flags, param):
    
    global oldx, oldy
    
    if event == cv2.EVENT_LBUTTONDOWN:
        oldx, oldy = x, y
    
    elif event == cv2.EVENT_MOUSEMOVE:
        
        if flags & cv2.EVENT_FLAG_LBUTTON:
            cv2.line(param, (oldx, oldy), (x, y), (255, 255), 5, cv2.LINE_AA)
            cv2.imshow('board', param)
            oldx, oldy = x, y

def draw_number():

    board = np.zeros((200, 200))

    cv2.imshow('board', board)
    cv2.setMouseCallback('board', onMouse, param=board)
    print('If you finish draw number, press enter key')
    if cv2.waitKey(0) == 13:
        # save image
        
        cv2.destroyAllWindows()

        return board

def load_model(path):

    model = tf.keras.models.load_model(path)

    model.summary()

    return model

def preprocess(image):

    image = cv2.resize(image, (28, 28))

    image = image[tf.newaxis, ..., tf.newaxis]

    image = np.array(image)

    image = image / 255.

    return image



if __name__ == '__main__':

    print(tf.__version__)

    model = load_model('./model/mnist-classifier.h5')

    while True:

        img = draw_number()

        input_img = cv2.copyTo(img, None)

        img = preprocess(img)

        pred = model.predict(img)
        
        for n, idx in enumerate(range(10)):
            msg = '{} : {}'.format(n, np.round(pred[:, idx]*100, 4))
            print(msg)

        cv2.imshow('input_img', input_img)

        if cv2.waitKey(0) == 27:
            break
        elif cv2.waitKey(0) == 13:
            oldx = oldy = -1
            cv2.destroyAllWindows()
    
    cv2.destroyAllWindows()
    