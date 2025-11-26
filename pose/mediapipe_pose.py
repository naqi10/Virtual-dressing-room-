import mediapipe as mp
import numpy as np
from PIL import Image
import cv2
import torch
import torch.nn.functional as F


class PoseGenerator:
    """
    Creates 18-channel pose heatmaps identical to VITON-HD dataset format.
    """

    def __init__(self,
                 target_size=(512, 384),
                 sigma=6):
        self.mp_pose = mp.solutions.pose.Pose(
            static_image_mode=True,
            model_complexity=2,
            enable_segmentation=False,
            min_detection_confidence=0.5
        )
        self.H, self.W = target_size
        self.sigma = sigma

        # Mapping 33 MediaPipe landmarks → 18 OpenPose/VITON landmarks
        # OpenPose 18-keypoint format:
        # 0=nose, 1=neck, 2=right_shoulder, 3=right_elbow, 4=right_wrist,
        # 5=left_shoulder, 6=left_elbow, 7=left_wrist, 8=right_hip, 9=left_hip,
        # 10=right_knee, 11=left_knee, 12=right_ankle, 13=left_ankle,
        # 14=right_eye, 15=left_eye, 16=right_ear, 17=left_ear
        # MediaPipe → OpenPose mapping
        self.viton_keypoint_ids = {
            0: 0,    # nose → 0
            11: 5,   # left_shoulder → 5
            12: 2,   # right_shoulder → 2
            13: 6,   # left_elbow → 6
            14: 3,   # right_elbow → 3
            15: 7,   # left_wrist → 7
            16: 4,   # right_wrist → 4
            23: 9,   # left_hip → 9
            24: 8,   # right_hip → 8
            25: 11,  # left_knee → 11
            26: 10,  # right_knee → 10
            27: 13,  # left_ankle → 13
            28: 12,  # right_ankle → 12
        }
        
        # For neck (keypoint 1), we'll estimate it from shoulders
        self.neck_landmarks = [11, 12]  # left_shoulder, right_shoulder in MediaPipe

    def gaussian(self, x, y, x0, y0, sigma):
        return np.exp(-((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))

    def generate_heatmap(self, x, y):
        """
        Generates a single 2D Gaussian heatmap channel.
        """
        xx, yy = np.meshgrid(np.arange(self.W), np.arange(self.H))
        return self.gaussian(xx, yy, x, y, self.sigma)

    def create_pose_map(self, img_path):
        """
        Returns a (18, 512, 384) pose heatmap tensor.
        """
        img = Image.open(img_path).convert("RGB")
        img_resized = img.resize((self.W, self.H))

        img_np = np.array(img_resized)
        results = self.mp_pose.process(img_np)

        pose_map = np.zeros((18, self.H, self.W), dtype=np.float32)

        if not results.pose_landmarks:
            print("[POSE] WARNING: No landmarks detected.")
            return torch.from_numpy(pose_map)

        # Extract 33 MediaPipe keypoints
        landmarks = results.pose_landmarks.landmark

        for mp_id, viton_id in self.viton_keypoint_ids.items():
            lm = landmarks[mp_id]
            x = int(lm.x * self.W)
            y = int(lm.y * self.H)

            if 0 <= x < self.W and 0 <= y < self.H:
                pose_map[viton_id] = self.generate_heatmap(x, y)

        # Normalize to [0,1]
        pose_map /= pose_map.max() + 1e-8

        return torch.from_numpy(pose_map)


# Helper function used by new1.py
def run_pose(image_path):
    """
    Generates the 18-channel pose tensor for the uploaded person image.
    Returns: (pose_heatmap_tensor, pose_keypoints_array)
        - pose_heatmap_tensor: (1,18,H,W) tensor for visualization
        - pose_keypoints_array: (18, 2) numpy array with keypoint coordinates [x, y]
    """
    generator = PoseGenerator()
    pose_map = generator.create_pose_map(image_path)
    
    # Also extract keypoint coordinates for get_parse_agnostic and get_img_agnostic
    # Use FULL image resolution for keypoints (not resized), as they'll be scaled later
    img = Image.open(image_path).convert("RGB")
    img_full_np = np.array(img)
    results = generator.mp_pose.process(img_full_np)
    
    # Get original image dimensions
    orig_h, orig_w = img_full_np.shape[:2]
    
    pose_keypoints = np.zeros((18, 2), dtype=np.float32)
    
    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark
        
        # Extract mapped keypoints in ORIGINAL image coordinates
        for mp_id, viton_id in generator.viton_keypoint_ids.items():
            lm = landmarks[mp_id]
            if lm.visibility > 0.5:  # Only use visible keypoints
                pose_keypoints[viton_id, 0] = lm.x * orig_w
                pose_keypoints[viton_id, 1] = lm.y * orig_h
        
        # Estimate neck (keypoint 1) as midpoint between shoulders
        if landmarks[11].visibility > 0.5 and landmarks[12].visibility > 0.5:
            left_shoulder = landmarks[11]
            right_shoulder = landmarks[12]
            pose_keypoints[1, 0] = ((left_shoulder.x + right_shoulder.x) / 2) * orig_w
            pose_keypoints[1, 1] = ((left_shoulder.y + right_shoulder.y) / 2) * orig_h
    
    return pose_map.unsqueeze(0), pose_keypoints  # (1,18,H,W), (18,2) in original image coords
