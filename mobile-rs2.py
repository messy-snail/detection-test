# -*- coding: utf-8 -*-
import numpy as np
import cv2
import jetson.inference
import jetson.utils

import sys
import copy

import realsense as rs

FONT = cv2.FONT_HERSHEY_SIMPLEX
LINE = cv2.LINE_AA
    

if __name__=='__main__':
    cam = rs.realsense()
    cam.open()
    
    net = jetson.inference.detectNet("ssd-mobilenet-v2", threshold=0.7)
    net.SetOverlayAlpha(60);

    while (True):
        if cam.run() ==False:
            continue
        
        image = copy.deepcopy(cam.color_image)
        depth = copy.deepcopy(cam.depth_image)
        h,w,_ = image.shape
        rgba_image = cv2.cvtColor(image, cv2.COLOR_RGB2RGBA).astype(np.float32)
        # move the image to CUDA:
        cuda_image = jetson.utils.cudaFromNumpy(rgba_image)
        detections = net.Detect(cuda_image, w, h)
        count = len(detections)
        # print(f"{count}개 오브젝트를 찾았습니다")

        # print("detected {:d} objects in image".format(len(detections)))
        numpyImg = jetson.utils.cudaToNumpy(cuda_image, w, h, 4)
        # now back to unit8
        result = numpyImg.astype(np.uint8)
        result = cv2.cvtColor(result, cv2.COLOR_RGBA2BGR)
        for i in range(0, count):
            id_img = result[int(detections[i].ROI[1]):int(detections[i].ROI[3]), int(detections[i].ROI[0]):int(detections[i].ROI[2]), :]
            if id_img.shape[0] !=0 and id_img.shape[1] !=0:
                cv2.putText(id_img, f'{i}', (11, 30), FONT, 0.5, (32, 32, 32), 4, LINE)
                cv2.putText(id_img, f'{i}', (10, 20), FONT, 0.5, (240, 240, 240), 1, LINE)   
        
        cv2.putText(result, "Total: " + str(count), (11, 20), FONT, 0.5, (32, 32, 32), 4, LINE)
        cv2.putText(result, "Total: " + str(count), (10, 20), FONT, 0.5, (240, 240, 240), 1, LINE)
        # # show frames
        cv2.imshow('result', result)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
