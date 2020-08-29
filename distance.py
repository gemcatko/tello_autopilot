

from dev_env_vars import *
import logging
from datetime import datetime
import cv2
KNOWN_DISTANCE = 24.0 #
KNOWN_WIDTH = 11.0
def know_distance(widthPx):
    # initialize the known distance from the camera to the object, which
    # in this case is 24 inches
    #KNOWN_DISTANCE = 24.0
    # initialize the known object width, which in this case, the piece of
    # paper is 12 inches wide
    # KNOWN_WIDTH = 11.0
    # load the furst image that contains an object that is KNOWN TO BE 2 feet
    # from our camera, then find the paper marker in the image, and initialize
    # the focal length
    focalLength = (widthPx * KNOWN_DISTANCE) / KNOWN_WIDTH
    return focalLength

def distance_to_camera(KNOWN_WIDTH, focalLength, perWidth):
	# compute and return the distance from the maker to the camera
	return (KNOWN_WIDTH * focalLength) / perWidth


def update_for_dist(id,cat, score, bounds,results):
    distance_results = []
    try:
        for cat, score, bounds in results:
            x, y, w, h = bounds
            # loop over the tracked objects from Centroid
            distance_result = distance_result, cat, score, bounds
            distance_results.append(distance_result)
        return distance_results
    except:
        return distance_results


def update_resutls_for_distance(detections):

    distance_results = []
    defined_categories = ["person", "cell phone"]
    for id,cat, score, bounds in detections:
        # print(cat)
        if "person" == bytes.decode(cat):
            dist_t_camera_p = distance_to_camera(7, know_distance(Xresolution), bounds[2])
            distance_result = dist_t_camera_p, id, cat, score, bounds

        if "cell phone" == bytes.decode(cat):
            dist_t_camera_c = distance_to_camera(5, know_distance(Xresolution), bounds[2])
            distance_result = dist_t_camera_c, id ,cat, score, bounds

        if not bytes.decode(cat) in defined_categories:
            #logging.info("Not defined cat using default fordistance calculation ")
            dist_t_camera_default = distance_to_camera(6, know_distance(Xresolution), bounds[2])
            distance_result = dist_t_camera_default, id, cat, score, bounds

        distance_results.append(distance_result)
        # print(distance_results)
    return  distance_results


def draw_distance_next_to_bb_box(img,distance_results):
    #print("Distance",distance_results)
    for detection in distance_results:
        x, y, w, h = detection[4][0], \
                     detection[4][1], \
                     detection[4][2], \
                     detection[4][3]
        darknetvscameraresolutionx = (Xresolution / network_width)
        darknetvscameraresolutiony = (Yresolution / network_heigth)
        x = x * darknetvscameraresolutionx
        y = y * darknetvscameraresolutiony
        w = w * darknetvscameraresolutionx
        h = h * darknetvscameraresolutiony
        cv2.putText(img, "DIST " + str(int(detection[0] )), (int(x), int(y)), cv2.FONT_HERSHEY_COMPLEX,
                    font_size,
                    azzure)
    return img

