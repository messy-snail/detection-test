import jetson.inference
import jetson.utils

import cv2
import pyrealsense2 as rs2

net = jetson.inference.detectNet("ssd-mobilenet-v2", threshold=0.5)
# camera = jetson.utils.videoSource("/dev/video2")      # '/dev/video0' for V4L2
camera = cv2.VideoCapture(2)
# camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
# camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
display = jetson.utils.videoOutput("display://0") # 'my_video.mp4' for file

while display.IsStreaming():
	# img = camera.Capture()
	_, bgr = camera.read()
	bgr = jetson.utils.cudaFromNumpy(bgr, isBGR=True)

	rgb = jetson.utils.cudaAllocMapped(width=bgr.width,
							    height=bgr.height,
							    format='rgb8')

	jetson.utils.cudaConvertColor(bgr, rgb)
	
	# detections = net.Detect(img)
	detections = net.Detect(rgb)
	# print(detections)
	
	# display.Render(img)
	display.Render(rgb)
	display.SetStatus("Object Detection | Network {:.0f} FPS".format(net.GetNetworkFPS()))
