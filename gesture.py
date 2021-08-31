import numpy as np
import json
import math

class gesture():
    def __init__(self):
        self.angle_list = []
        with open('human_pose.json') as f:
            data = json.load(f)
        self.keypoints_name_list = data['keypoints']

        self.neck = 0

        self.left_shoulder = 0
        self.left_elbow = 0
        self.left_wrist = 0

        self.right_shoulder = 0
        self.right_elbow = 0
        self.right_wrist = 0

        self.left_angle = 0
        self.right_angle = 0

   
    def compute_angle(self, neck, shoulder, elbow):
        vec1 = np.array([shoulder.x-neck.x, shoulder.y-neck.y])
        vec2 = np.array([shoulder.x-elbow.x, shoulder.y-elbow.y])
        vec1_nrom = np.linalg.norm(vec1)
        vec2_nrom = np.linalg.norm(vec2)
        # a.b/a*b
        #|a||b|cosq = a.b
        cos_val = np.dot(vec1, vec2)/(vec1_nrom*vec2_nrom)
        
        deg = math.degrees(math.acos(cos_val))
        return deg
        

    def run(self, poses):
        
        for pose in poses:
            neck_idx = pose.FindKeypoint(self.keypoints_name_list.index('neck'))

            left_shoulder_idx = pose.FindKeypoint(self.keypoints_name_list.index('left_shoulder'))
            left_elbow_idx = pose.FindKeypoint(self.keypoints_name_list.index('left_elbow'))
            left_wrist_idx = pose.FindKeypoint(self.keypoints_name_list.index('left_wrist'))

            right_shoulder_idx = pose.FindKeypoint(self.keypoints_name_list.index('right_shoulder'))
            right_elbow_idx = pose.FindKeypoint(self.keypoints_name_list.index('right_elbow'))
            right_wrist_idx = pose.FindKeypoint(self.keypoints_name_list.index('right_wrist'))

            left_flag = False
            right_flag = False
            if left_shoulder_idx > 0 and left_elbow_idx>0 and neck_idx>0:
                left_flag = True
            if right_shoulder_idx > 0 and right_elbow_idx>0 and neck_idx>0:
                right_flag = True

            if left_flag ==False and right_flag==False:
                continue
        
            
            if left_flag:
                neck = pose.Keypoints[neck_idx]    
                left_shoulder = pose.Keypoints[left_shoulder_idx]      
                left_elbow = pose.Keypoints[left_elbow_idx]
                
                self.left_angle =self.compute_angle(neck, left_shoulder, left_elbow)
                
                
            if right_flag:
                neck = pose.Keypoints[neck_idx]          
                right_shoulder = pose.Keypoints[right_shoulder_idx]  
                right_elbow = pose.Keypoints[right_elbow_idx]
                
                self.right_angle = self.compute_angle(neck, right_shoulder, right_elbow)

        angle = f'left: {self.left_angle:.2f}, right: {self.right_angle:.2f}'

        return angle