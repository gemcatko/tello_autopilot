import logging

import numpy as np

from dev_env_vars import *
from datetime import datetime
import cv2


def navigate_o(objects,to_which_object,to_which_object_2):
    """

    :param objects:
        self.id = id
        self.category = category
        self.score = score
        self.bounds = bounds
        self.is_big = False
        self.is_detected_by_detector = True
        self.ignore = False
        self.is_picture_saved = False
        self.distance = distance
    :return:
    """
    navigate_frame = np.zeros((200, 200, 3), np.uint8)
    for id in objects:
        try:
            if to_which_object == objects[id].category and objects[id].is_detected_by_detector==True :
                if to_close(objects[id].distance):
                    cv2.line(navigate_frame, (100, 100), (100,200 ), green, 5)
                if not to_close(objects[id].distance):
                    cur_time = str(datetime.now().time()) +"   " + str(objects[id].id) + " GO"
                    cv2.line(navigate_frame, (100, 100), (100,0 ), green, 5)
                    detection = [objects[id].distance,objects[id].category,objects[id].score,objects[id].bounds]
                    rotate_to_target(navigate_frame,detection)
                    #tello(rotate_to_target(detection))
                    #logging.info(cur_time)

            if to_which_object_2 == objects[id].category:
                if to_close(objects[id].distance):
                    cv2.line(navigate_frame, (100, 100), (100,200 ), green, 5)

                if not to_close(objects[id].distance):
                    cur_time = str(datetime.now().time()) +"   " +  str(objects[id].id) + " GO"
                    #logging.info(cur_time)
                    cv2.line(navigate_frame, (100, 100), (100, 0), green, 5)
            cv2.imshow("navigate_frame",navigate_frame)
        except Exception as e:
            print(e)


def to_close(distance):
    if distance < min_distance:
        cur_time = str(datetime.now().time()) + " BACKWARD"
        #logging.info(cur_time)
        return True

def rotate_to_target (navigate_frame,detection):

    x, y, w, h = detection[3][0], \
                         detection[3][1], \
                         detection[3][2], \
                         detection[3][3]
    middleX= Xresolution / 2
    middleY= Yresolution / 2
    darknetvscameraresolutionx = (Xresolution / network_width)
    darknetvscameraresolutiony = (Yresolution / network_heigth)
    #x = x * darknetvscameraresolutionx
    #y = y * darknetvscameraresolutiony
    #w = w * darknetvscameraresolutionx
    #h = h * darknetvscameraresolutiony
    try:
        if x < middleX:
            logging.info("Rotate to left")
            #calculate how much to rotate for now 1=Left with max power,
            cv2.line(navigate_frame, (0, 100), (100, 100), green, 5)
            return 1
        if x > middleX:
            logging.info("Rotate to right")
            cv2.line(navigate_frame, (100, 100), (200, 100), green, 5)
            return -1
    except Exception as e:
        print(e)



