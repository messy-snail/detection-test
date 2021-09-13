import numpy as np
import json
import math
from collections import defaultdict
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

        self.total_state_dict = defaultdict(str)
        self.registration_id = -1
        

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
        
    def run(self, poses, depth, scale):

        for pose in poses:

            self.neck_idx = pose.FindKeypoint(self.keypoints_name_list.index('neck'))

            self.left_shoulder_idx = pose.FindKeypoint(self.keypoints_name_list.index('left_shoulder'))
            self.left_elbow_idx = pose.FindKeypoint(self.keypoints_name_list.index('left_elbow'))
            self.left_wrist_idx = pose.FindKeypoint(self.keypoints_name_list.index('left_wrist'))

            self.right_shoulder_idx = pose.FindKeypoint(self.keypoints_name_list.index('right_shoulder'))
            self.right_elbow_idx = pose.FindKeypoint(self.keypoints_name_list.index('right_elbow'))
            self.right_wrist_idx = pose.FindKeypoint(self.keypoints_name_list.index('right_wrist'))

            left_angle, right_angle = self.find_lr_angles(pose)      
            # left_angle, right_angle = self.find_lr_angles(pose, left_flags, right_flags)             
            
            state = self.find_pose(left_angle, right_angle)     
            print(f'left sholder: {left_angle[0]}, left elbow: {left_angle[1]}')
            print(f'right sholder: {right_angle[0]}, right elbow: {right_angle[1]}')

            
            if self.total_state_dict[pose.ID]=='reg [RIGHT]':
                if self.total_state_dict[pose.ID]!='N/A':
                    self.total_state_dict[pose.ID] = state
                if self.total_state_dict[pose.ID]=='reg [LEFT]' or self.total_state_dict[pose.ID]=='reg [BOTH]':
                    self.registration_id = pose.ID
            else:
                self.total_state_dict[pose.ID] = state                


        # print(self.total_state_dict)
        

    def find_pose(self, left_angle, right_angle):
        state = 'N/A'
        
        registration_condition_both = left_angle[1]<80 and left_angle[1]>15 and right_angle[1]<80 and right_angle[1]>15
        registration_condition_left = left_angle[1]<80 and left_angle[1]>15
        registration_condition_right = right_angle[1]<80 and right_angle[1]>15

        if self.registration_id==-1:
            if registration_condition_both:
                state = 'reg [BOTH]'
            else:
                if registration_condition_left:
                    state = 'reg [LEFT]'
                if registration_condition_right:
                    state = 'reg [RIGHT]'
        else:
            go_condition_right = right_angle[1]<90 and right_angle[1]>15 and \
                abs(self.right_wrist_position[0] - self.right_elbow_position[0])< 20 and self.right_wrist_position[1] < self.right_elbow_position[1]
  
            go_condition_left = left_angle[1]<90 and left_angle[1]>15 and \
                abs(self.left_wrist_position[0] - self.left_elbow_position[0])< 20 and self.left_wrist_position[1] < self.left_elbow_position[1]

            # stop_condition = right_angle[1]<50 and right_angle[1]>15 and left_angle[1]<50 and left_angle[1]>15 and \
            #     abs(self.right_elbow_position[1] - self.right_wrist_position[1])<20 and abs(self.left_elbow_position[1] - self.left_wrist_position[1])<20 and\
            #         self.right_elbow_position[0]< self.right_wrist_position[0] and self.left_elbow_position[0] > self.left_wrist_position[0]

            # right_condition = right_angle[1]<50 and right_angle[1]>15 and \
            #     abs(self.right_elbow_position[1] - self.right_wrist_position[1])<30 and\
            #         self.right_elbow_position[0]< self.right_wrist_position[0] 

            # left_condition = left_angle[1]<50 and left_angle[1]>15 and \
            #     abs(self.left_elbow_position[1] - self.left_wrist_position[1])<30 and \
            #         self.left_elbow_position[0] > self.left_wrist_position[0]
            right_condition = right_angle[1]<50 and right_angle[1]>15 and \
                self.right_shoulder_position[1] < self.right_wrist_position[1]and\
                    self.right_elbow_position[0]< self.right_wrist_position[0] 

            left_condition = left_angle[1]<50 and left_angle[1]>15 and \
                self.left_shoulder_position[1] < self.left_wrist_position[1]and\
                    self.left_elbow_position[0] > self.left_wrist_position[0]            

            if go_condition_right ^ go_condition_left:
                state = 'GO'
            if left_condition and right_condition:
                state = 'STOP'
            else:
                if left_condition:
                    state = 'LEFT'
                if right_condition:
                    state = 'RIGHT'



                

                        
        return state

  

    def find_lr_angles(self, pose):

        self.compute_position(pose)                        

        left_angles, right_angles = self.compute_lr_angles()                                                

        return left_angles, right_angles 

    def compute_lr_angles(self):
        left_angles = [-1, -1]
        right_angles = [-1, -1]

        left_angle0_condition = self.neck_idx>0 and self.left_shoulder_idx>0 and self.left_elbow_idx>0
        rightt_angle0_condition = self.neck_idx>0 and self.right_shoulder_idx>0 and self.right_elbow_idx>0

        left_angle1_condition = self.left_shoulder_idx>0 and self.left_elbow_idx>0 and self.left_wrist_idx>0
        rightt_angle1_condition = self.right_shoulder_idx>0 and self.right_elbow_idx>0 and self.right_wrist_idx>0

        if left_angle0_condition:
            left_angles[0] =self.compute_angle(self.neck_position, 
                                              self.left_shoulder_position, 
                                              self.left_elbow_position)      

        if rightt_angle0_condition:
            right_angles[0] = self.compute_angle(self.neck_position, 
                                                self.right_shoulder_position, 
                                                self.right_elbow_position)
        if left_angle1_condition:
            left_angles[1] =self.compute_angle(self.left_shoulder_position, 
                                              self.left_elbow_position, 
                                              self.left_wrist_position)      
                                              
        if rightt_angle1_condition:
            right_angles[1] = self.compute_angle(self.right_shoulder_position, 
                                                self.right_elbow_position, 
                                                self.right_wrist_position)
                                                
        return left_angles,right_angles

    def compute_position(self, pose):
        ###### NECK
        if self.neck_idx>0:
            neck_key_pts = pose.Keypoints[self.neck_idx]
            self.neck_position = (neck_key_pts.x, neck_key_pts.y)
        else:
            self.neck_position = None

        ###### LEFT SOULDER, ELBOW, WRIST
        if self.left_shoulder_idx>0:
            left_shoulder_key_pts = pose.Keypoints[self.left_shoulder_idx]   
            self.left_shoulder_position = (left_shoulder_key_pts.x, left_shoulder_key_pts.y)
        else:
            self.left_shoulder_position = None         

        if self.left_elbow_idx>0:
            left_elbow_key_pts = pose.Keypoints[self.left_elbow_idx]
            self.left_elbow_position = (left_elbow_key_pts.x, left_elbow_key_pts.y)
        else:
            self.left_elbow_position = None         

        if self.left_wrist_idx>0:
            left_wrist_key_pts = pose.Keypoints[self.left_wrist_idx]
            self.left_wrist_position = (left_wrist_key_pts.x, left_wrist_key_pts.y)
        else:
            self.left_wrist_position = None    

     
        ###### RIGHT SOULDER, ELBOW, WRIST
        if self.right_shoulder_idx>0:
            right_shoulder_key_pts = pose.Keypoints[self.right_shoulder_idx]  
            self.right_shoulder_position = (right_shoulder_key_pts.x, right_shoulder_key_pts.y)
        else:
            self.right_shoulder_position = None

        if self.right_elbow_idx>0:
            right_elbow_key_pts = pose.Keypoints[self.right_elbow_idx]            
            self.right_elbow_position = (right_elbow_key_pts.x, right_elbow_key_pts.y)    
        else:
            self.right_elbow_position = None
               
        if self.right_wrist_idx>0:
            right_wrist_key_pts = pose.Keypoints[self.right_wrist_idx]            
            self.right_wrist_position = (right_wrist_key_pts.x, right_wrist_key_pts.y)    
        else:
            self.right_wrist_position = None

        