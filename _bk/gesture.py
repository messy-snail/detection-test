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

        self.left_angle = [-1, -1]
        self.right_angle = [-1, -1]

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
        neck = None
        num_obj = len(poses)
        print(f'obj: {num_obj}')
        for pose in poses:
            neck_idx = pose.FindKeypoint(self.keypoints_name_list.index('neck'))

            left_shoulder_idx = pose.FindKeypoint(self.keypoints_name_list.index('left_shoulder'))
            left_elbow_idx = pose.FindKeypoint(self.keypoints_name_list.index('left_elbow'))
            left_wrist_idx = pose.FindKeypoint(self.keypoints_name_list.index('left_wrist'))

            right_shoulder_idx = pose.FindKeypoint(self.keypoints_name_list.index('right_shoulder'))
            right_elbow_idx = pose.FindKeypoint(self.keypoints_name_list.index('right_elbow'))
            right_wrist_idx = pose.FindKeypoint(self.keypoints_name_list.index('right_wrist'))

            left_flags = [False, False]
            right_flags = [False, False]
            if left_shoulder_idx > 0 and left_elbow_idx>0 and neck_idx>0:
                left_flags[0] = True
            if right_shoulder_idx > 0 and right_elbow_idx>0 and neck_idx>0:
                right_flags[0] = True
            if left_shoulder_idx > 0 and left_elbow_idx>0 and left_wrist_idx>0:
                left_flags[1] = True
            if right_shoulder_idx > 0 and right_elbow_idx>0 and right_wrist_idx>0:
                right_flags[1] = True                

            if left_flags ==[False, False] and right_flags==[False, False]:
                continue

            self.left_angle = [-1, -1]
            self.right_angle = [-1, -1]
            if left_flags[0]:
                neck = pose.Keypoints[neck_idx]    
                left_shoulder = pose.Keypoints[left_shoulder_idx]      
                left_elbow = pose.Keypoints[left_elbow_idx]
                
                self.left_angle[0] =self.compute_angle(neck, left_shoulder, left_elbow)
  
            if right_flags[0]:
                neck = pose.Keypoints[neck_idx]          
                right_shoulder = pose.Keypoints[right_shoulder_idx]  
                right_elbow = pose.Keypoints[right_elbow_idx]
                
                self.right_angle[0] = self.compute_angle(neck, right_shoulder, right_elbow)

            if left_flags[1]:
                left_shoulder = pose.Keypoints[left_shoulder_idx]      
                left_elbow = pose.Keypoints[left_elbow_idx]
                left_wrist = pose.Keypoints[left_wrist_idx]  
                
                self.left_angle[1] =self.compute_angle(left_shoulder, left_elbow, left_wrist)
  
            if right_flags[1]:
                right_shoulder = pose.Keypoints[right_shoulder_idx]      
                right_elbow = pose.Keypoints[right_elbow_idx]
                right_wrist = pose.Keypoints[right_wrist_idx]  
                
                self.right_angle[1] =self.compute_angle(right_shoulder, right_elbow, right_wrist)             

        angle = f'left0: {self.left_angle[0]:.2f}, right0: {self.right_angle[0]:.2f}, left1: {self.left_angle[1]:.2f}, right1: {self.right_angle[1]:.2f}'
        state = ['N/A', 'N/A']
        
        if self.left_angle[0]>150 and self.right_angle[0]>150:
            state[0] = 'POSE1 [BOTH]'
        else:
            if self.left_angle[0]>150:
                state[0] = 'POSE1 [LEFT]'
            if self.right_angle[0]>150:
                state[0] = 'POSE1 [RIGHT]'

        if self.left_angle[1]<80 and self.left_angle[1]>0 and self.right_angle[1]<80 and self.right_angle[1]>0:
            state[1] = 'POSE2 [BOTH]'
        else:
            if self.left_angle[1]<70 and self.left_angle[1]>0:
                state[1] = 'POSE2 [LEFT]'
            if self.right_angle[1]<70 and self.right_angle[1]>0:
                state[1] = 'POSE2 [RIGHT]'                

        if neck is None:
            return angle, f'{state[0]}, {state[1]}', None
        else:
            return angle, f'{state[0]}, {state[1]}', (int(neck.y), int(neck.x))
        