import jetson.inference
import jetson.utils

import numpy as np
import copy
import cv2

cap = cv2.VideoCapture(3)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# load mono depth network
net = jetson.inference.depthNet()

# depthNet re-uses the same memory for the depth field,
# so you only need to do this once (not every frame)
depth_field = net.GetDepthField()

# cudaToNumpy() will map the depth field cudaImage to numpy
# this mapping is persistent, so you only need to do it once
depth_numpy = jetson.utils.cudaToNumpy(depth_field)

print(f"depth field resolution is {depth_field.width}x{depth_field.height}, format={depth_field.format}")

while True:
    ret, frame = cap.read()
    if ret ==False:
        continue

    image = copy.deepcopy(frame)
    h,w,_ = image.shape
    rgba_image = cv2.cvtColor(image, cv2.COLOR_RGB2RGBA).astype(np.float32)
    # move the image to CUDA:
    cuda_image = jetson.utils.cudaFromNumpy(rgba_image)
    net.Process(cuda_image)
    jetson.utils.cudaDeviceSynchronize() # wait for GPU to finish processing, so we can use the results on CPU
	
    # find the min/max values with numpy
    min_depth = np.amin(depth_numpy)
    max_depth = np.amax(depth_numpy)