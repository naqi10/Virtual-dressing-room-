import cv2
import numpy as np
from PIL import Image
import torch


class ClothMaskGenerator:
    """
    Generates:
        1. Cloth binary mask  (1 channel)
        2. Cloth edge map     (1 channel)
    Output resolution = (768, 1024) exactly like VITON-HD dataset.
    """

    def __init__(self, target_size=(768, 1024)):
        self.W, self.H = target_size  # width, height

    def resize(self, img):
        return cv2.resize(img, (self.W, self.H), interpolation=cv2.INTER_LINEAR)

    def create_mask(self, cloth_img_pil):
        """
        Auto-separates cloth from background by color + morphology.
        Handles both light and dark clothes (including black).
        Returns binary mask (0 or 1).
        """
        cloth_np = np.array(cloth_img_pil.resize((self.W, self.H)))
        img = cloth_np.copy()

        # Method 1: Remove white/light backgrounds using HSV
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        
        # Detect white/light backgrounds (high value, low saturation)
        # White background: V > 200, S < 30
        white_mask = (hsv[:, :, 2] > 200) & (hsv[:, :, 1] < 30)
        
        # Detect gray backgrounds (medium value, very low saturation)
        gray_mask = (hsv[:, :, 2] > 100) & (hsv[:, :, 2] < 200) & (hsv[:, :, 1] < 20)
        
        # Combine: cloth is everything EXCEPT white/gray background
        background_mask = white_mask | gray_mask
        cloth_mask = (~background_mask).astype(np.uint8) * 255
        
        # Method 2: If cloth mask is too small (<5%), try edge-based detection
        cloth_pixels = np.sum(cloth_mask > 0)
        total_pixels = cloth_mask.size
        coverage = cloth_pixels / total_pixels
        
        if coverage < 0.05:  # Less than 5% coverage - try alternative method
            # Use OTSU thresholding for dark clothes (like black)
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            # Invert for black clothes (OTSU works better on inverted)
            gray_inv = 255 - gray
            _, thresh = cv2.threshold(gray_inv, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            # Invert back
            cloth_mask = 255 - thresh
        
        # Morphology clean-up to remove noise
        kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        cloth_mask = cv2.morphologyEx(cloth_mask, cv2.MORPH_OPEN, kernel_open)
        cloth_mask = cv2.morphologyEx(cloth_mask, cv2.MORPH_CLOSE, kernel_close)
        
        # Final check: if still too small, use entire image (assume no background)
        cloth_pixels_final = np.sum(cloth_mask > 0)
        coverage_final = cloth_pixels_final / total_pixels
        
        if coverage_final < 0.05:  # Still too small - assume full image is cloth
            print(f"[WARNING] Cloth mask coverage {coverage_final*100:.1f}% too small. Using full image as cloth mask.")
            cloth_mask = np.ones((self.H, self.W), dtype=np.uint8) * 255

        # Normalize mask to 0/1
        mask = (cloth_mask > 128).astype(np.float32)

        return mask  # (H, W)

    def create_edge(self, cloth_np):
        """
        Runs Canny edge detection on cloth.
        Returns normalized edge map (0-1).
        """
        gray = cv2.cvtColor(cloth_np, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 20, 80)
        edges = edges.astype(np.float32) / 255.0
        return edges  # (H, W)

    def process(self, cloth_path):
        """
        Full cloth preprocessing:
            - Loads cloth
            - Creates cloth mask
            - Creates cloth edge
        Returns:
            cloth_tensor (1,3,H,W)
            mask_tensor  (1,1,H,W)
            edge_tensor  (1,1,H,W)
        """
        cloth_pil = Image.open(cloth_path).convert("RGB")
        cloth_resized = cloth_pil.resize((self.W, self.H))
        cloth_np = np.array(cloth_resized)

        mask = self.create_mask(cloth_resized)      # (H,W)
        edge = self.create_edge(cloth_np)           # (H,W)

        # Convert to tensors
        cloth_tensor = torch.tensor(cloth_np.transpose(2, 0, 1)).float().unsqueeze(0) / 255.0
        mask_tensor = torch.tensor(mask).float().unsqueeze(0).unsqueeze(0)
        edge_tensor = torch.tensor(edge).float().unsqueeze(0).unsqueeze(0)

        return cloth_tensor, mask_tensor, edge_tensor


# Helper function used inside new1.py
def run_cloth_preprocess(cloth_path):
    """
    Creates cloth_tensor, cloth_mask_tensor, cloth_edge_tensor
    """
    generator = ClothMaskGenerator()
    return generator.process(cloth_path)
