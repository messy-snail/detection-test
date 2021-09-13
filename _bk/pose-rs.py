# -*- coding: utf-8 -*-
import jetson.inference
import jetson.utils
import realsense
import numpy as np
import cv2
import copy



FONT = cv2.FONT_HERSHEY_SIMPLEX
LINE = cv2.LINE_AA

# # parse the command line
# parser = argparse.ArgumentParser(description="Run pose estimation DNN on a video/image stream.", 
#                                  formatter_class=argparse.RawTextHelpFormatter, epilog=jetson.inference.poseNet.Usage() +
#                                  jetson.utils.videoSource.Usage() + jetson.utils.videoOutput.Usage() + jetson.utils.logUsage())


# parser.add_argument("--network", type=str, default="resnet18-body", help="pre-trained model to load (see below for options)")
# parser.add_argument("--overlay", type=str, default="links,keypoints", help="pose overlay flags (e.g. --overlay=links,keypoints)\nvalid combinations are:  'links', 'keypoints', 'boxes', 'none'")
# parser.add_argument("--threshold", type=float, default=0.15, help="minimum detection threshold to use") 



# load the pose estimation model



# if __name__=='__main__':
    
    # cam.open()

    # net = jetson.inference.poseNet('resnet18-body', 0.15)
    

    # while (True):
    #     if cam.run() ==False:
    #         continue
      
    #     image = copy.deepcopy(cam.color_image)
    #     depth = copy.deepcopy(cam.depth_image)
    #     h,w,_ = image.shape
    #     rgba_image = cv2.cvtColor(image, cv2.COLOR_RGB2RGBA).astype(np.float32)
    #     # move the image to CUDA:
    #     cuda_image = jetson.utils.cudaFromNumpy(rgba_image)
    #     poses = net.Process(cuda_image, overlay="links,keypoints")

    #     count = len(poses)
        
    #     print(f"{count}개 오브젝트를 찾았습니다.")

    #     for pose in poses:
    #         print(pose)
    #         print(pose.Keypoints)
    #         print('Links', pose.Links)

    #     # print out performance info
    #     net.PrintProfilerTimes()

    #     numpyImg = jetson.utils.cudaToNumpy(cuda_image, w, h, 4)
    #     # now back to unit8
    #     result = numpyImg.astype(np.uint8)
    #     # for detection in detections:
    #     #     if detection.ClassID==1:
    #     #         print(detection.ROI)
    #     #         image = cv2.rectangle(result, (int(detection.ROI[0]), int(detection.ROI[1])),
    #     #         (int(detection.ROI[2]), int(detection.ROI[3])),(0,0,255), 1, cv2.LINE_AA
    #     #         )
    #     #         center_pts = (int((detection.ROI[0]+detection.ROI[2])/2), int((detection.ROI[1]+detection.ROI[3])/2))
    #     #         image = cv2.circle(image, (center_pts[0], center_pts[1]), 3, (0,0,255), -1, cv2.LINE_AA
    #     #         )

    #     #         cv2.putText(image, f'Depth: {depth[center_pts]*cam.scale_factor:.3f}m' , (11, 40), FONT, 0.5, (32, 32, 32), 4, LINE)
    #     #         cv2.putText(image, f"Depth: {depth[center_pts]*cam.scale_factor:.3f}m" , (10, 40), FONT, 0.5, (240, 240, 240), 1, LINE)
        
    #     cv2.putText(image, "Total: " + str(count), (11, 20), FONT, 0.5, (32, 32, 32), 4, LINE)
    #     cv2.putText(image, "Total: " + str(count), (10, 20), FONT, 0.5, (240, 240, 240), 1, LINE)
    #     # # show frames
    #     cv2.imshow('result', image)
    #     if cv2.waitKey(1) & 0xFF == ord('q'):
    #         break
    



