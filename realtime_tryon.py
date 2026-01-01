"""
MediaPipe-based Real-Time Virtual Try-On Pipeline
Integrates with Flask backend for web-based real-time camera overlay
"""

import cv2
import mediapipe as mp
import numpy as np
import os
from collections import deque
from datetime import datetime
import time


class RealtimeTryOn:
    def __init__(self, shirt_folder=None, process_width=640, smooth_frames=15):
        """
        Initialize real-time try-on system.
        
        Args:
            shirt_folder: Path to folder containing shirt PNGs
            process_width: Width to resize frames for processing
            smooth_frames: Number of frames for moving average smoothing
        """
        # Default shirt folder
        if shirt_folder is None:
            shirt_folder = os.path.join("static", "shirts")
        
        self.shirt_folder = shirt_folder
        self.process_width = process_width
        self.smooth_frames = smooth_frames
        
        # Create shirt folder if it doesn't exist
        os.makedirs(self.shirt_folder, exist_ok=True)
        
        # Load shirts
        self.shirt_files = [f for f in os.listdir(self.shirt_folder) 
                           if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        if not self.shirt_files:
            print(f"[WARNING] No shirts found in {self.shirt_folder}")
            print(f"[INFO] Please add shirt images (PNG/JPG) to {self.shirt_folder}")
            self.shirts = []
            self.thumbs = []
            self.shirt_names = []
        else:
            self.shirts = [self._load_rgba(os.path.join(self.shirt_folder, f)) 
                          for f in self.shirt_files]
            thumb_size = (100, 100)
            self.thumbs = [cv2.resize(s[:, :, :3], thumb_size, 
                                     interpolation=cv2.INTER_AREA) for s in self.shirts]
            self.shirt_names = [os.path.splitext(f)[0] for f in self.shirt_files]
        
        # MediaPipe setup
        self.mp_pose = mp.solutions.pose
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7,
            model_complexity=1,
            smooth_landmarks=True
        )
        
        self.hands = self.mp_hands.Hands(
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7,
            static_image_mode=False,
            max_num_hands=2
        )
        
        # State variables
        self.shirt_index = 0
        self.target_shirt_index = 0
        self.shoulder_history = deque(maxlen=smooth_frames)
        self.size_history = deque(maxlen=smooth_frames)
        self.pose_history = deque(maxlen=smooth_frames)
        self.rotation_history = deque(maxlen=smooth_frames)
        self.last_transition_time = time.time()
        self.gallery_offset = 0.0
        self.target_gallery_offset = 0.0
        
        # Smoothed values
        self.avg_neck = None
        self.avg_shoulder_dist = None
        self.avg_size = None
        self.avg_center = None
        self.avg_rotation = None
        
        # Settings
        self.thumb_size = (100, 100)
        self.gallery_height = 140
        self.show_landmarks = False
        self.shirt_transition_duration = 0.2
        self.gallery_transition_duration = 0.3
    
    def _load_rgba(self, path):
        """Load image with alpha channel."""
        im = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if im is None:
            raise RuntimeError(f"Failed to read {path}")
        if im.shape[2] == 3:
            b, g, r = cv2.split(im)
            alpha = np.full(b.shape, 255, dtype=b.dtype)
            im = cv2.merge([b, g, r, alpha])
        return im
    
    def overlay_image_alpha(self, bg, fg, x, y):
        """Overlay RGBA image onto BGR background with alpha blending."""
        if fg is None:
            return bg
        bg_h, bg_w = bg.shape[:2]
        fg_h, fg_w = fg.shape[:2]
        
        if x >= bg_w or y >= bg_h or x + fg_w <= 0 or y + fg_h <= 0:
            return bg
        
        x1 = max(0, x)
        y1 = max(0, y)
        x2 = min(bg_w, x + fg_w)
        y2 = min(bg_h, y + fg_h)
        
        fg_x1 = x1 - x
        fg_y1 = y1 - y
        fg_x2 = fg_x1 + (x2 - x1)
        fg_y2 = fg_y1 + (y2 - y1)
        
        fg_roi = fg[fg_y1:fg_y2, fg_x1:fg_x2]
        if fg_roi.shape[2] == 3:
            alpha = np.full(fg_roi.shape[:2] + (1,), 255, dtype=np.uint8)
            fg_roi = np.dstack((fg_roi, alpha))
        
        fg_rgb = fg_roi[:, :, :3].astype(float)
        alpha = (fg_roi[:, :, 3:] / 255.0).astype(float)
        bg_roi = bg[y1:y2, x1:x2].astype(float)
        
        comp = (1.0 - alpha) * bg_roi + alpha * fg_rgb
        bg[y1:y2, x1:x2] = comp.astype(np.uint8)
        return bg
    
    def build_arm_mask(self, image_shape, landmarks, include_full_sleeves=True):
        """Create mask for arms so shirt appears behind them."""
        ih, iw = image_shape[:2]
        mask = np.zeros((ih, iw), dtype=np.uint8)
        if landmarks is None:
            return mask
        
        def to_px(landmark):
            return int(landmark.x * iw), int(landmark.y * ih)
        
        try:
            ls = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value]
            le = landmarks[self.mp_pose.PoseLandmark.LEFT_ELBOW.value]
            lw = landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST.value]
            rs = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
            re = landmarks[self.mp_pose.PoseLandmark.RIGHT_ELBOW.value]
            rw = landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST.value]
            lh = landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value]
            rh = landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value]
            
            ls_px = to_px(ls)
            le_px = to_px(le)
            lw_px = to_px(lw)
            rs_px = to_px(rs)
            re_px = to_px(re)
            rw_px = to_px(rw)
            lh_px = to_px(lh)
            rh_px = to_px(rh)
            
            left_upper_arm_width = int(np.hypot(le_px[0] - ls_px[0], le_px[1] - ls_px[1]) * 0.18)
            right_upper_arm_width = int(np.hypot(re_px[0] - rs_px[0], re_px[1] - rs_px[1]) * 0.18)
            
            left_lower_arm_width = max(3, int(left_upper_arm_width * 0.8))
            right_lower_arm_width = max(3, int(right_upper_arm_width * 0.8))
            
            cv2.line(mask, ls_px, le_px, 255, max(4, left_upper_arm_width))
            cv2.line(mask, le_px, lw_px, 255, max(3, left_lower_arm_width))
            cv2.line(mask, rs_px, re_px, 255, max(4, right_upper_arm_width))
            cv2.line(mask, re_px, rw_px, 255, max(3, right_lower_arm_width))
            
            cv2.circle(mask, ls_px, max(6, left_upper_arm_width), 255, -1)
            cv2.circle(mask, le_px, max(10, left_upper_arm_width + 3), 255, -1)
            cv2.circle(mask, lw_px, max(7, left_lower_arm_width + 2), 255, -1)
            cv2.circle(mask, rs_px, max(6, right_upper_arm_width), 255, -1)
            cv2.circle(mask, re_px, max(10, right_upper_arm_width + 3), 255, -1)
            cv2.circle(mask, rw_px, max(7, right_lower_arm_width + 2), 255, -1)
            
            if include_full_sleeves:
                lw_extended = (lw_px[0], lw_px[1] + int(left_lower_arm_width * 0.5))
                rw_extended = (rw_px[0], rw_px[1] + int(right_lower_arm_width * 0.5))
                cv2.circle(mask, lw_extended, max(8, left_lower_arm_width + 3), 255, -1)
                cv2.circle(mask, rw_extended, max(8, right_lower_arm_width + 3), 255, -1)
        except Exception:
            pass
        
        mask = cv2.GaussianBlur(mask, (21, 21), 0)
        return mask
    
    def get_midpoint(self, a, b):
        return ((a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0)
    
    def process_frame(self, frame):
        """
        Process a single frame and return the result with shirt overlay.
        
        Args:
            frame: BGR frame from camera
            
        Returns:
            display_frame: Frame with shirt overlay
            info: Dictionary with processing info
        """
        if len(self.shirts) == 0:
            cv2.putText(frame, "No shirts available. Add shirts to static/shirts/", 
                       (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            return frame, {"shirt_count": 0}
        
        frame = cv2.flip(frame, 1)  # Mirror for UX
        display_frame = frame.copy()
        
        # Resize for processing
        orig_h, orig_w = frame.shape[:2]
        proc_w = self.process_width
        proc_h = int(orig_h * (proc_w / orig_w))
        proc_frame = cv2.resize(frame, (proc_w, proc_h))
        proc_rgb = cv2.cvtColor(proc_frame, cv2.COLOR_BGR2RGB)
        
        # Process pose and hands
        pose_res = self.pose.process(proc_rgb)
        hands_res = self.hands.process(proc_rgb)
        
        pose_landmarks = None
        if pose_res.pose_landmarks:
            pose_landmarks = pose_res.pose_landmarks.landmark
        
        # Process shirt placement if pose detected
        if pose_landmarks:
            ih_proc, iw_proc = proc_frame.shape[:2]
            
            ls = pose_landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value]
            rs = pose_landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
            lh = pose_landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value]
            rh = pose_landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value]
            le = pose_landmarks[self.mp_pose.PoseLandmark.LEFT_ELBOW.value]
            re = pose_landmarks[self.mp_pose.PoseLandmark.RIGHT_ELBOW.value]
            
            ls_px = (ls.x * iw_proc, ls.y * ih_proc)
            rs_px = (rs.x * iw_proc, rs.y * ih_proc)
            lh_px = (lh.x * iw_proc, lh.y * ih_proc)
            rh_px = (rh.x * iw_proc, rh.y * ih_proc)
            le_px = (le.x * iw_proc, le.y * ih_proc)
            re_px = (re.x * iw_proc, re.y * ih_proc)
            
            neck_px = self.get_midpoint(ls_px, rs_px)
            hip_center_px = self.get_midpoint(lh_px, rh_px)
            chest_center_px = self.get_midpoint(neck_px, hip_center_px)
            
            shoulder_dist = np.hypot(rs_px[0] - ls_px[0], rs_px[1] - ls_px[1])
            torso_length = np.hypot(hip_center_px[0] - neck_px[0], hip_center_px[1] - neck_px[1])
            chest_width_est = max(shoulder_dist, np.hypot(re_px[0] - le_px[0], re_px[1] - le_px[1]) * 0.7)
            
            body_ratio = torso_length / max(shoulder_dist, 1)
            
            if body_ratio > 1.8:
                width_factor = 1.6
                height_factor = 1.8
            elif body_ratio < 1.0:
                width_factor = 1.4
                height_factor = 1.6
            else:
                width_factor = 1.5
                height_factor = 1.7
            
            shirt_w = int(chest_width_est * width_factor)
            shirt_h = int(torso_length * height_factor)
            rotation_angle = np.degrees(np.arctan2(rs_px[1] - ls_px[1], rs_px[0] - ls_px[0]))
            
            if shirt_w > 10 and shirt_h > 10:
                scale_x = frame.shape[1] / iw_proc
                scale_y = frame.shape[0] / ih_proc
                neck_px_full = (int(neck_px[0] * scale_x), int(neck_px[1] * scale_y))
                chest_center_full = (int(chest_center_px[0] * scale_x), int(chest_center_px[1] * scale_y))
                shoulder_dist_full = int(shoulder_dist * ((scale_x + scale_y)/2.0))
                shirt_w_full = max(20, int(shirt_w * ((scale_x + scale_y)/2.0)))
                shirt_h_full = max(20, int(shirt_h * ((scale_x + scale_y)/2.0)))
                
                use_new_value = True
                if len(self.shoulder_history) > 3:
                    last_neck = self.shoulder_history[-1][0]
                    neck_diff = np.hypot(neck_px_full[0] - last_neck[0], neck_px_full[1] - last_neck[1])
                    max_allowed_movement = frame.shape[1] * 0.2
                    if neck_diff > max_allowed_movement:
                        use_new_value = False
                
                if use_new_value:
                    self.shoulder_history.append((neck_px_full, shoulder_dist_full))
                    self.size_history.append((shirt_w_full, shirt_h_full))
                    self.pose_history.append(chest_center_full)
                    self.rotation_history.append(rotation_angle)
                
                # Exponential moving average
                alpha = 0.15
                if self.avg_neck is None or len(self.shoulder_history) == 1:
                    self.avg_neck = np.array(neck_px_full, dtype=float)
                    self.avg_shoulder_dist = float(shoulder_dist_full)
                    self.avg_size = np.array((shirt_w_full, shirt_h_full), dtype=float)
                    self.avg_center = np.array(chest_center_full, dtype=float)
                    self.avg_rotation = float(rotation_angle)
                else:
                    new_neck = np.array(neck_px_full, dtype=float)
                    new_size = np.array((shirt_w_full, shirt_h_full), dtype=float)
                    new_center = np.array(chest_center_full, dtype=float)
                    new_rotation = float(rotation_angle)
                    
                    self.avg_neck = self.avg_neck * (1 - alpha) + new_neck * alpha
                    self.avg_shoulder_dist = self.avg_shoulder_dist * (1 - alpha) + shoulder_dist_full * alpha
                    self.avg_size = self.avg_size * (1 - alpha) + new_size * alpha
                    self.avg_center = self.avg_center * (1 - alpha) + new_center * alpha
                    self.avg_rotation = self.avg_rotation * (1 - alpha) + new_rotation * alpha
                
                self.avg_neck = self.avg_neck.astype(int)
                self.avg_shoulder_dist = int(self.avg_shoulder_dist)
                self.avg_size = self.avg_size.astype(int)
                self.avg_center = self.avg_center.astype(int)
                self.avg_rotation = float(self.avg_rotation)
                
                final_w, final_h = int(self.avg_size[0]), int(self.avg_size[1])
                x = int(self.avg_center[0] - final_w * 0.5)
                y = int(self.avg_neck[1] - final_h * 0.15)
                
                shirt_img = self.shirts[self.shirt_index]
                
                current_time = time.time()
                transition_progress = min(1.0, (current_time - self.last_transition_time) / self.shirt_transition_duration)
                
                if self.shirt_index != self.target_shirt_index:
                    alpha_blend = transition_progress
                    if transition_progress >= 1.0:
                        self.shirt_index = self.target_shirt_index
                        alpha_blend = 1.0
                else:
                    alpha_blend = 1.0
                
                shirt_resized = cv2.resize(shirt_img, (max(2, final_w), max(2, final_h)), 
                                          interpolation=cv2.INTER_LANCZOS4)
                
                if alpha_blend < 1.0 and self.shirt_index != self.target_shirt_index:
                    target_shirt = self.shirts[self.target_shirt_index]
                    target_resized = cv2.resize(target_shirt, (max(2, final_w), max(2, final_h)), 
                                               interpolation=cv2.INTER_LANCZOS4)
                    
                    if shirt_resized.shape[2] == 4 and target_resized.shape[2] == 4:
                        blend_alpha = shirt_resized[:, :, 3:4] / 255.0
                        target_alpha = target_resized[:, :, 3:4] / 255.0
                        combined_alpha = np.maximum(blend_alpha * (1 - alpha_blend), target_alpha * alpha_blend)
                        shirt_rgb = shirt_resized[:, :, :3] * (1 - alpha_blend) + target_resized[:, :, :3] * alpha_blend
                        shirt_resized = np.dstack([shirt_rgb.astype(np.uint8), 
                                                  (combined_alpha * 255).astype(np.uint8)])
                
                landmarks_full = []
                for lm in pose_landmarks:
                    x_norm_full = (lm.x * iw_proc) / frame.shape[1]
                    y_norm_full = (lm.y * ih_proc) / frame.shape[0]
                    class L:
                        pass
                    l = L()
                    l.x = x_norm_full
                    l.y = y_norm_full
                    landmarks_full.append(l)
                
                arm_mask = self.build_arm_mask(frame.shape, landmarks_full, include_full_sleeves=True)
                inv_mask = cv2.bitwise_not(arm_mask)
                
                temp = display_frame.copy()
                temp = self.overlay_image_alpha(temp, shirt_resized, x, y)
                
                fg_part = cv2.bitwise_and(display_frame, display_frame, mask=arm_mask)
                bg_part = cv2.bitwise_and(temp, temp, mask=inv_mask)
                display_frame = cv2.add(fg_part, bg_part)
                
                current_shirt_name = self.shirt_names[self.target_shirt_index] if self.target_shirt_index < len(self.shirt_names) else f"Shirt {self.target_shirt_index + 1}"
                cv2.putText(display_frame, f"Shirt: {current_shirt_name}", (20, 40),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.0, (230, 230, 230), 2, cv2.LINE_AA)
                
                if self.show_landmarks and pose_res and pose_res.pose_landmarks:
                    self.mp_drawing.draw_landmarks(display_frame, pose_res.pose_landmarks, 
                                                   self.mp_pose.POSE_CONNECTIONS)
        
        # Update gallery offset
        if abs(self.gallery_offset - self.target_gallery_offset) > 0.5:
            diff = self.target_gallery_offset - self.gallery_offset
            self.gallery_offset += diff * 0.15
        else:
            self.gallery_offset = self.target_gallery_offset
        
        info = {
            "shirt_count": len(self.shirts),
            "current_shirt": self.target_shirt_index,
            "shirt_name": self.shirt_names[self.target_shirt_index] if self.target_shirt_index < len(self.shirt_names) else "Unknown"
        }
        
        return display_frame, info
    
    def set_shirt_index(self, index):
        """Set target shirt index with smooth transition."""
        if 0 <= index < len(self.shirts):
            self.target_shirt_index = index
            self.last_transition_time = time.time()
    
    def next_shirt(self):
        """Switch to next shirt."""
        if len(self.shirts) > 0:
            self.set_shirt_index((self.target_shirt_index + 1) % len(self.shirts))
    
    def prev_shirt(self):
        """Switch to previous shirt."""
        if len(self.shirts) > 0:
            self.set_shirt_index((self.target_shirt_index - 1) % len(self.shirts))
    
    def cleanup(self):
        """Clean up resources."""
        if hasattr(self, 'pose'):
            self.pose.close()
        if hasattr(self, 'hands'):
            self.hands.close()

