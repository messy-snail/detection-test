# -*- coding: utf-8 -*-
import pyrealsense2.pyrealsense2 as rs
import numpy as np

class realsense:
    def __init__(self):
        self.pipeline = None
        self.config = None
        self.align = None

        self.depth_intrinsic = None
        self.scale_factor = None

        self.depth_image = None
        self.color_image = None

    def open(self):
        # Create a pipeline
        self.pipeline = rs.pipeline()
        self.config = rs.config()

        # Get device product line for setting a supporting resolution
        pipeline_wrapper = rs.pipeline_wrapper(self.pipeline)
        pipeline_profile = self.config.resolve(pipeline_wrapper)
        device = pipeline_profile.get_device()
        device_product_line = str(device.get_info(rs.camera_info.product_line))
        print(device_product_line)
        found_rgb = False   
        for s in device.sensors:
            if s.get_info(rs.camera_info.name) == 'RGB Camera':
                found_rgb = True
                break
        if not found_rgb:
            print("The demo requires Depth camera with Color sensor")
            exit(0)

        self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

        if device_product_line == 'L500':
            self.config.enable_stream(rs.stream.color, 960, 540, rs.format.rgb8, 30)
        else:
            self.config.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, 30)
            # config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
        profile = self.pipeline.start(self.config)

        depth_sensor = profile.get_device().first_depth_sensor()
        self.scale_factor = depth_sensor.get_depth_scale()
        print("Depth Scale is: " , self.scale_factor)

        clipping_distance_in_meters = 5 #1 meter
        clipping_distance = clipping_distance_in_meters / self.scale_factor

        align_to = rs.stream.color
        # align_to = rs.stream.depth
        self.align = rs.align(align_to)

    def run(self):
        frames = self.pipeline.wait_for_frames()
        aligned_frames = self.align.process(frames)

        
        # Get aligned frames
        aligned_depth_frame = aligned_frames.get_depth_frame() # aligned_depth_frame is a 640x480 depth image
        # color_frame = frames.get_color_frame()
        color_frame = aligned_frames.get_color_frame()
        self.depth_intrinsic = aligned_frames.profile.as_video_stream_profile().intrinsics

        # Validate that both frames are valid
        if not aligned_depth_frame or not color_frame:
            return False

        self.depth_image = np.asanyarray(aligned_depth_frame.get_data())
        self.color_image = np.asanyarray(color_frame.get_data())
        return True
        