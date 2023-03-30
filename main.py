import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math
import time

# We need to initiate video cam to capture images
image_capture = cv2.VideoCapture(0)

#adding classifier
classify_images = Classifier("applied_models/keras_model.h5", "applied_models/labels.txt")

# we will locate hands by hand detector and specify max no. of hands to be 1 to make it simple
locate_hands = HandDetector(maxHands=1)

buffer = 20
dimention_image = 250
#
collected_images = "collected_Images/numeric_3"
count = 0

schema = ["A", "B", "C", "1", "2", "3"]

while True:
    success, img = image_capture.read()
    img_out = img.copy()
    objects, img = locate_hands.findHands(img) # this will detect hand in the capture image.
#now we want to crop images to be of same size therefore we will create a bounding box
    if objects:
        hand = objects[0]
        x, y, w, h = hand["bbox"]
        img_clrd = np.ones((dimention_image, dimention_image, 3), np.uint8) * 255
        #now we need to specify width and hight to crop our image
        crop_image = img[y - buffer:y + h + buffer, x - buffer:x + w + buffer]
        crop_image_Shape = crop_image.shape

        #now we need to keep the aspect ratio of the image in check,
        # if width is less we need to increase and if height is less
        # we need to increrase so that the image stays square.
        hwRatio = h/w
        if hwRatio > 1:#if hieght is greater then we need to increase width therefor
            const_val = dimention_image/h
            new_width = math.ceil(const_val*w)
            resized_image = cv2.resize(crop_image,(new_width,dimention_image))
            resized_image_shape = crop_image.shape
            padded_width = math.ceil((dimention_image-new_width)/2)
            img_clrd[:, padded_width : new_width+padded_width] = resized_image
            #with this our image will change its width but the height will be same as dimention_image
            cls_output, index = classify_images.getPrediction(img_clrd, draw = False)
            print(cls_output, index)

        else:
            const_val = dimention_image / w
            new_height = math.ceil(const_val * h)
            resized_image = cv2.resize(crop_image, (new_height, dimention_image))
            resized_image_shape = crop_image.shape
            padded_height = math.ceil((dimention_image - new_height) / 2)
            img_clrd[:, padded_height: new_height + padded_height] = resized_image
            # with this our image will change its height but the width will be same as dimention_image
            cls_output, index = classify_images.getPrediction(img_clrd, draw = False)
            print(cls_output, index)


        cv2.putText(img_out, schema[index],(x,y+10),cv2.FONT_HERSHEY_COMPLEX,1,(150,21,255),1)



        cv2.imshow("ImageCrop", crop_image)
        cv2.imshow("Img_Clrd", img_clrd)


    cv2.imshow("Image", img_out)
    cv2.waitKey(1)
