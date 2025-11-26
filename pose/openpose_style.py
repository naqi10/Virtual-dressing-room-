"""
OpenPose-style pose generator that matches VITON-HD dataset format.
Outputs rendered RGB pose images and OpenPose JSON-style keypoints.
"""
import mediapipe as mp
import numpy as np
from PIL import Image, ImageDraw
import cv2
import torch


class OpenPoseStyleGenerator:
    """
    Creates OpenPose-style pose visualization matching VITON-HD dataset format.
    Uses MediaPipe internally but converts to OpenPose 18-keypoint format.
    """

    def __init__(self):
        self.mp_pose = mp.solutions.pose.Pose(
            static_image_mode=True,
            model_complexity=2,
            enable_segmentation=False,
            min_detection_confidence=0.5
        )

        # Mapping 33 MediaPipe landmarks → 18 OpenPose/VITON landmarks
        # OpenPose 18-keypoint format (COCO format):
        # 0=nose, 1=neck, 2=right_shoulder, 3=right_elbow, 4=right_wrist,
        # 5=left_shoulder, 6=left_elbow, 7=left_wrist, 8=right_hip, 9=left_hip,
        # 10=right_knee, 11=left_knee, 12=right_ankle, 13=left_ankle,
        # 14=right_eye, 15=left_eye, 16=right_ear, 17=left_ear
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
        
        # MediaPipe eye/ear landmarks (approximate mapping)
        # OpenPose doesn't have direct eye/ear in MediaPipe, so we'll estimate
        self.eye_ear_landmarks = {
            2: 16,   # right_ear (approximate from right eye outer)
            5: 17,   # left_ear (approximate from left eye outer)
        }

    def extract_keypoints(self, img_np):
        """
        Extract OpenPose-style 18 keypoints from MediaPipe results.
        Returns: (18, 2) numpy array with [x, y] coordinates, or zeros if not detected.
        """
        results = self.mp_pose.process(img_np)
        h, w = img_np.shape[:2]
        
        pose_keypoints = np.zeros((18, 2), dtype=np.float32)
        
        if not results.pose_landmarks:
            return pose_keypoints
        
        landmarks = results.pose_landmarks.landmark
        
        # Extract mapped keypoints
        for mp_id, viton_id in self.viton_keypoint_ids.items():
            lm = landmarks[mp_id]
            if lm.visibility > 0.3:  # Lower threshold for better detection
                pose_keypoints[viton_id, 0] = lm.x * w
                pose_keypoints[viton_id, 1] = lm.y * h
        
        # Estimate neck (keypoint 1) as midpoint between shoulders
        if landmarks[11].visibility > 0.3 and landmarks[12].visibility > 0.3:
            left_shoulder = landmarks[11]
            right_shoulder = landmarks[12]
            pose_keypoints[1, 0] = ((left_shoulder.x + right_shoulder.x) / 2) * w
            pose_keypoints[1, 1] = ((left_shoulder.y + right_shoulder.y) / 2) * h
        
        # Estimate eyes and ears (approximate)
        # Right eye (14) - use right eye outer corner
        if landmarks[2].visibility > 0.3:  # right eye outer
            pose_keypoints[14, 0] = landmarks[2].x * w
            pose_keypoints[14, 1] = landmarks[2].y * h
        
        # Left eye (15) - use left eye outer corner
        if landmarks[5].visibility > 0.3:  # left eye outer
            pose_keypoints[15, 0] = landmarks[5].x * w
            pose_keypoints[15, 1] = landmarks[5].y * h
        
        # Right ear (16) - approximate from right ear tip
        if len(landmarks) > 8 and landmarks[8].visibility > 0.3:  # right ear tip
            pose_keypoints[16, 0] = landmarks[8].x * w
            pose_keypoints[16, 1] = landmarks[8].y * h
        
        # Left ear (17) - approximate from left ear tip
        if len(landmarks) > 7 and landmarks[7].visibility > 0.3:  # left ear tip
            pose_keypoints[17, 0] = landmarks[7].x * w
            pose_keypoints[17, 1] = landmarks[7].y * h
        
        return pose_keypoints

    def render_pose(self, img_pil, pose_keypoints, target_size=(768, 1024)):
        """
        Render OpenPose-style pose visualization matching VITON-HD format.
        Returns: PIL Image (RGB) with pose skeleton drawn.
        """
        # Resize image to target size
        img_resized = img_pil.resize(target_size, Image.BICUBIC)
        w, h = target_size
        
        # Scale keypoints to target size
        orig_w, orig_h = img_pil.size
        scale_x = w / float(orig_w)
        scale_y = h / float(orig_h)
        
        scaled_keypoints = pose_keypoints.copy()
        scaled_keypoints[:, 0] *= scale_x
        scaled_keypoints[:, 1] *= scale_y
        
        # Create black background
        pose_img = Image.new('RGB', (w, h), color=(0, 0, 0))
        draw = ImageDraw.Draw(pose_img)
        
        # Define skeleton connections (OpenPose format)
        # Each tuple: (start_keypoint_idx, end_keypoint_idx, color)
        skeleton = [
            # Head
            (0, 1, (255, 0, 0)),   # nose to neck
            (0, 14, (255, 0, 0)),  # nose to right_eye
            (0, 15, (255, 0, 0)),  # nose to left_eye
            (14, 16, (255, 0, 0)), # right_eye to right_ear
            (15, 17, (255, 0, 0)), # left_eye to left_ear
            # Torso
            (1, 2, (0, 255, 0)),   # neck to right_shoulder
            (1, 5, (0, 255, 0)),   # neck to left_shoulder
            (2, 8, (0, 255, 0)),   # right_shoulder to right_hip
            (5, 9, (0, 255, 0)),   # left_shoulder to left_hip
            (8, 9, (0, 255, 0)),   # right_hip to left_hip
            # Right arm
            (2, 3, (0, 0, 255)),   # right_shoulder to right_elbow
            (3, 4, (0, 0, 255)),   # right_elbow to right_wrist
            # Left arm
            (5, 6, (0, 0, 255)),   # left_shoulder to left_elbow
            (6, 7, (0, 0, 255)),   # left_elbow to left_wrist
            # Right leg
            (8, 10, (255, 255, 0)), # right_hip to right_knee
            (10, 12, (255, 255, 0)), # right_knee to right_ankle
            # Left leg
            (9, 11, (255, 255, 0)), # left_hip to left_knee
            (11, 13, (255, 255, 0)), # left_knee to left_ankle
        ]
        
        # Draw skeleton connections
        for start_idx, end_idx, color in skeleton:
            start_pt = tuple(scaled_keypoints[start_idx].astype(int))
            end_pt = tuple(scaled_keypoints[end_idx].astype(int))
            
            # Only draw if both keypoints are valid (non-zero)
            if (scaled_keypoints[start_idx, 0] > 0 or scaled_keypoints[start_idx, 1] > 0) and \
               (scaled_keypoints[end_idx, 0] > 0 or scaled_keypoints[end_idx, 1] > 0):
                draw.line([start_pt, end_pt], fill=color, width=3)
        
        # Draw keypoints as circles
        for i, (x, y) in enumerate(scaled_keypoints):
            if x > 0 or y > 0:  # Valid keypoint
                radius = 4
                draw.ellipse([x-radius, y-radius, x+radius, y+radius], 
                           fill=(255, 255, 255), outline=(255, 255, 255))
        
        return pose_img

    def process(self, image_path, target_size=(768, 1024)):
        """
        Process image and generate OpenPose-style pose visualization.
        Returns:
            pose_rgb: PIL Image (RGB) with pose skeleton
            pose_keypoints: (18, 2) numpy array with keypoint coordinates [x, y] in ORIGINAL image coords
        """
        img_pil = Image.open(image_path).convert("RGB")
        img_np = np.array(img_pil)
        
        # Extract keypoints in original image coordinates
        pose_keypoints = self.extract_keypoints(img_np)
        
        # Render pose visualization at target size
        pose_rgb = self.render_pose(img_pil, pose_keypoints, target_size)
        
        return pose_rgb, pose_keypoints


# Helper function used by new1.py
def run_openpose_style(image_path, target_size=(768, 1024)):
    """
    Generate OpenPose-style pose visualization matching VITON-HD format.
    Returns:
        pose_rgb: PIL Image (RGB) with pose skeleton, resized to target_size
        pose_keypoints: (18, 2) numpy array with keypoint coordinates [x, y] in ORIGINAL image coords
    """
    generator = OpenPoseStyleGenerator()
    pose_rgb, pose_keypoints = generator.process(image_path, target_size)
    return pose_rgb, pose_keypoints

