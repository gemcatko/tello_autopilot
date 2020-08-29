import cv2
import numpy as np
from dev_env_vars import *


def map_visualization(objects):
    """

    :type objects:
    is visualizing objects you need to feed it with  objects. Check class YObject: for details.
    """
    zoom = 2  # how many times smoler
    map_X = int(Xresolution / zoom)
    map_Y = int(Xresolution / zoom)
    middle = int(map_X / 2)
    map_frame = np.zeros((map_X, map_Y, 3), np.uint8)
    cv2.circle(map_frame, (middle, middle), 5, yellow, 5)   #helicopter positions
    for id in objects:

        x = int(objects[id].bounds[0] / zoom)  # x
        y = int((objects[id].distance) / zoom) + middle  # dis
        # radius = int(objects[id].bounds[3]/zoom)       #w
        radius = 5
        #print("X,y", x, y, objects[id].id)

        if objects[id].is_detected_by_detector == True:
            cv2.circle(map_frame, (x, y), radius, green, 5)
        else:
            cv2.circle(map_frame, (x, y), radius, brown, 5)
            #print(objects[id].id,"red object")
    cv2.imshow('map1', map_frame)
    # image = cv2.circle(image, center_coordinates, radius, color, thickness)
    return map_frame
