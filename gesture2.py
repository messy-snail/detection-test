import numpy as np
import json
import math

class gesture():
    def __init__(self):
        self.angle_list = []
        with open('human_pose.json') as f:
            data = json.load(f)
        self.keypoints_name_list = data['keypoints']
        
        ## 키포인트 있는지 없는지 체크하기 위한 인덱스
        self.neck_idx = -1

        self.left_shoulder_idx =-1
        self.left_elbow_idx = -1
        self.left_wrist_idx = -1

        self.right_shoulder_idx =-1
        self.right_elbow_idx = -1
        self.right_wrist_idx = -1

        ## 실제 포인트에 대한 위치 정보
        self.neck_position = None

        self.left_shoulder_position = None
        self.left_elbow_position = None
        self.left_wrist_position = None

        self.right_shoulder_position = None
        self.right_elbow_position = None
        self.right_wrist_position = None

    def compute_angle(self, neck_position, shoulder_position, elbow_position):
        vec1 = np.array([shoulder_position[0]-neck_position[0], shoulder_position[1]-neck_position[1]])
        vec2 = np.array([shoulder_position[0]-elbow_position[0], shoulder_position[1]-elbow_position[1]])
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
        # print(f'obj: {num_obj}')

        total_state=[]
        for pose in poses:

            self.neck_idx = pose.FindKeypoint(self.keypoints_name_list.index('neck'))

            self.left_shoulder_idx = pose.FindKeypoint(self.keypoints_name_list.index('left_shoulder'))
            self.left_elbow_idx = pose.FindKeypoint(self.keypoints_name_list.index('left_elbow'))
            self.left_wrist_idx = pose.FindKeypoint(self.keypoints_name_list.index('left_wrist'))

            self.right_shoulder_idx = pose.FindKeypoint(self.keypoints_name_list.index('right_shoulder'))
            self.right_elbow_idx = pose.FindKeypoint(self.keypoints_name_list.index('right_elbow'))
            self.right_wrist_idx = pose.FindKeypoint(self.keypoints_name_list.index('right_wrist'))

            left_flags, right_flags = self.lr_checker()                

            if left_flags ==[False, False] and right_flags==[False, False]:
                continue

            left_angle, right_angle = self.find_lr_angles(pose, left_flags, right_flags)             
            
            state = self.find_pose(left_angle, right_angle)     
            total_state.append(state)

        # angle = f'left0: {self.left_angle[0]:.2f}, right0: {self.right_angle[0]:.2f}, left1: {self.left_angle[1]:.2f}, right1: {self.right_angle[1]:.2f}'
            
        print(total_state)
        return total_state

    def find_pose(self, left_angle, right_angle):
        state = ['N/A', 'N/A']
        
        if left_angle[0]>150 and right_angle[0]>150:
            state[0] = 'POSE1 [BOTH]'
        else:
            if left_angle[0]>150:
                state[0] = 'POSE1 [LEFT]'
            if right_angle[0]>150:
                state[0] = 'POSE1 [RIGHT]'

        if left_angle[1]<80 and left_angle[1]>0 and right_angle[1]<80 and right_angle[1]>0:
            state[1] = 'POSE2 [BOTH]'
        else:
            if left_angle[1]<80 and left_angle[1]>0:
                state[1] = 'POSE2 [LEFT]'
            if right_angle[1]<80 and right_angle[1]>0:
                state[1] = 'POSE2 [RIGHT]'
        return state

    def lr_checker(self):
        left_flags = [False, False]
        right_flags = [False, False]
        
        # 왼쪽 숄더 각도 판단이 가능할 때
        if self.left_shoulder_idx > 0 and self.left_elbow_idx>0 and self.neck_idx>0:
            left_flags[0] = True
        # 오른쪽 숄더 각도 판단이 가능할 때
        if self.right_shoulder_idx > 0 and self.right_elbow_idx>0 and self.neck_idx>0:
            right_flags[0] = True
        # 왼쪽 엘보우 각도 판단이 가능할 때
        if self.left_shoulder_idx > 0 and self.left_elbow_idx>0 and self.left_wrist_idx>0:
            left_flags[1] = True
        # 오른쪽 엘보우 각도 판단이 가능할 때            
        if self.right_shoulder_idx > 0 and self.right_elbow_idx>0 and self.right_wrist_idx>0:
            right_flags[1] = True

        return left_flags,right_flags

    def find_lr_angles(self, pose, left_flags, right_flags):

        left_angle = [-1, -1]
        right_angle = [-1, -1]

        if left_flags[0]:
            neck_key_pts = pose.Keypoints[self.neck_idx]    
            left_shoulder_key_pts = pose.Keypoints[self.left_shoulder_idx]      
            left_elbow_key_pts = pose.Keypoints[self.left_elbow_idx]

            self.neck_position = (neck_key_pts.x, neck_key_pts.y)
            self.left_shoulder_position = (left_shoulder_key_pts.x, left_shoulder_key_pts.y)
            self.left_elbow_position = (left_elbow_key_pts.x, left_elbow_key_pts.y)
                
            left_angle[0] =self.compute_angle(self.neck_position, 
                                              self.left_shoulder_position, 
                                              self.left_elbow_position)
        if right_flags[0]:
            neck_key_pts = pose.Keypoints[self.neck_idx]          
            right_shoulder_key_pts = pose.Keypoints[self.right_shoulder_idx]  
            right_elbow_key_pts = pose.Keypoints[self.right_elbow_idx]

            self.neck_position = (neck_key_pts.x, neck_key_pts.y)
            self.right_shoulder_position = (right_shoulder_key_pts.x, right_shoulder_key_pts.y)
            self.right_elbow_position = (right_elbow_key_pts.x, right_elbow_key_pts.y)

            right_angle[0] = self.compute_angle(self.neck_position, 
                                                self.right_shoulder_position, 
                                                self.right_elbow_position)


        if left_flags[1]:
            left_shoulder_key_pts = pose.Keypoints[self.left_shoulder_idx]      
            left_elbow_key_pts = pose.Keypoints[self.left_elbow_idx]
            left_wrist_key_pts = pose.Keypoints[self.left_wrist_idx]  

            self.left_shoulder_position = (left_shoulder_key_pts.x, left_shoulder_key_pts.y)
            self.left_elbow_position = (left_elbow_key_pts.x, left_elbow_key_pts.y)
            self.left_wrist_position = (left_wrist_key_pts.x, left_wrist_key_pts.y)
                
            left_angle[1] =self.compute_angle(self.left_shoulder_position, 
                                                   self.left_elbow_position, 
                                                   self.left_wrist_position)
  
        if right_flags[1]:
            right_shoulder_key_pts = pose.Keypoints[self.right_shoulder_idx]      
            right_elbow_key_pts = pose.Keypoints[self.right_elbow_idx]
            right_wrist_key_pts = pose.Keypoints[self.right_wrist_idx]  

            self.right_shoulder_position = (right_shoulder_key_pts.x, right_shoulder_key_pts.y)
            self.right_elbow_position = (right_elbow_key_pts.x, right_elbow_key_pts.y)
            self.right_wrist_position = (right_wrist_key_pts.x, right_wrist_key_pts.y)

            right_angle[1] =self.compute_angle(self.right_shoulder_position, 
                                                    self.right_elbow_position, 
                                                    self.right_wrist_position)

        return left_angle, right_angle                                                    

        