


import os
import copy
import uuid
import types
import traceback

import torch
import torch.nn.functional as F
from PIL import Image, ImageOps, ImageFilter
from flask import Flask, request, jsonify, render_template_string
import numpy as np
from torchvision import transforms as T
from werkzeug.utils import secure_filename

from networks import SegGenerator, GMM, ALIASGenerator
from datasets import VITONDataset

# -------------------------------------------------------------------------
# Checkpoint helpers
# -------------------------------------------------------------------------
try:
    from utils import load_checkpoint, gen_noise
except ImportError:
    def load_checkpoint(model, path, map_location="cpu"):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        state = torch.load(path, map_location=map_location)
        if isinstance(state, dict):
            if "state_dict" in state:
                state = state["state_dict"]
            elif "model" in state:
                state = state["model"]
        model.load_state_dict(state, strict=False)
        return model

    def gen_noise(shape):
        return torch.randn(shape)


# -------------------------------------------------------------------------
# Imports for SCHP, pose, cloth-mask from ROOT-LEVEL files
# -------------------------------------------------------------------------
try:
    # schp_parser.py in parsing/ directory
    # run_schp_parsing(image_path, checkpoint=...) -> (prob_map, argmax_mask)
    from parsing.schp_parser import run_schp_parsing
except Exception as e:
    # Capture the actual error for debugging
    import traceback
    error_msg = f"Failed to import run_schp_parsing: {type(e).__name__}: {str(e)}\n{traceback.format_exc()}"
    print(f"[WARNING] {error_msg}")
    def run_schp_parsing(image_path, checkpoint=None):
        raise RuntimeError(
            f"run_schp_parsing not found. Ensure parsing/schp_parser.py exists "
            f"and defines run_schp_parsing(image_path, checkpoint=...).\n"
            f"Original error: {error_msg}"
        )

try:
    # openpose_style.py - OpenPose-style pose matching VITON-HD format
    # run_openpose_style(image_path, target_size) -> (pose_rgb_pil, pose_keypoints_array)
    from pose.openpose_style import run_openpose_style
except ImportError:
    try:
        # Fallback to mediapipe_pose.py
        from pose.mediapipe_pose import run_pose as run_pose_legacy
        def run_openpose_style(image_path, target_size=(768, 1024)):
            # Legacy fallback - convert MediaPipe output to OpenPose format
            pose_heatmap, pose_keypoints = run_pose_legacy(image_path)
            # Create a simple pose visualization from heatmap
            from PIL import Image
            import numpy as np
            pose_np = pose_heatmap.squeeze().cpu().numpy() if isinstance(pose_heatmap, torch.Tensor) else pose_heatmap
            if pose_np.ndim == 3:
                vis = pose_np.max(axis=0)
                vis = (vis - vis.min()) / (vis.max() - vis.min() + 1e-8)
                pose_rgb = Image.fromarray((vis * 255.0).astype(np.uint8), mode="L").convert("RGB")
                pose_rgb = pose_rgb.resize(target_size, Image.BICUBIC)
            else:
                pose_rgb = Image.new("RGB", target_size, color=(0, 0, 0))
            return pose_rgb, pose_keypoints
    except ImportError:
        def run_openpose_style(image_path, target_size=(768, 1024)):
            raise RuntimeError(
                "run_openpose_style not found. Ensure pose/openpose_style.py exists "
                "and defines run_openpose_style(image_path, target_size)."
            )

try:
    # cloth_mask.py in project root
    # run_cloth_preprocess(cloth_path) -> (cloth_tensor, mask_tensor, edge_tensor)
    from cloth.cloth_mask import run_cloth_preprocess
except ImportError:
    def run_cloth_preprocess(cloth_path):
        raise RuntimeError(
            "run_cloth_preprocess not found. Ensure cloth_mask.py exists in project root "
            "and defines run_cloth_preprocess(cloth_path)."
        )


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_tensor_type("torch.FloatTensor")

app = Flask(__name__, static_folder="static")
os.makedirs("static", exist_ok=True)
RESULT_DIR = os.path.join("static", "results")
os.makedirs(RESULT_DIR, exist_ok=True)
UPLOAD_DIR = os.path.join("static", "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)


def to_url(path):
    return "/" + path.replace("\\", "/")


def make_base_opt():
    opt = types.SimpleNamespace()
    opt.load_height = 1024
    opt.load_width = 768
    opt.semantic_nc = 13
    opt.grid_size = 5
    opt.norm_G = "spectralaliasinstance"
    opt.ngf = 64
    opt.num_upsampling_layers = "most"
    opt.init_type = "xavier"
    opt.init_variance = 0.02
    opt.dataset_dir = "./datasets/zalando-hd-resized"
    opt.dataset_mode = "test"
    opt.dataset_list = "test_pairs.txt"
    opt.batch_size = 1
    opt.workers = 0
    opt.shuffle = False
    opt.checkpoint_dir = "./checkpoints/"
    opt.seg_checkpoint = "seg_final.pth"
    opt.gmm_checkpoint = "gmm_final.pth"
    opt.alias_checkpoint = "alias_final.pth"
    opt.save_dir = RESULT_DIR
    return opt


BASE_OPT = make_base_opt()
MODEL_OPT = copy.deepcopy(BASE_OPT)
DATASET_OPT = copy.deepcopy(BASE_OPT)


def load_models(opt):
    print("[info] Initializing VITON-HD models...")
    seg = SegGenerator(opt, input_nc=opt.semantic_nc + 8, output_nc=opt.semantic_nc).to(device).eval()
    gmm = GMM(opt, inputA_nc=7, inputB_nc=3).to(device).eval()

    alias_opt = copy.deepcopy(opt)
    alias_opt.semantic_nc = 7
    alias = ALIASGenerator(alias_opt, input_nc=9).to(device).eval()

    for model, ckpt_name, label in [
        (seg, opt.seg_checkpoint, "SegGenerator"),
        (gmm, opt.gmm_checkpoint, "GMM"),
        (alias, opt.alias_checkpoint, "ALIASGenerator"),
    ]:
        ckpt_path = os.path.join(opt.checkpoint_dir, ckpt_name)
        try:
            load_checkpoint(model, ckpt_path, map_location=device)
            print(f"[info] Loaded {label} checkpoint from {ckpt_path}")
        except Exception as exc:
            print(f"[warn] Could not load {label} checkpoint ({ckpt_path}): {exc}")

    return seg, gmm, alias


SEG_MODEL, GMM_MODEL, ALIAS_MODEL = load_models(MODEL_OPT)


def load_dataset(opt):
    dataset_root = os.path.join(opt.dataset_dir, opt.dataset_mode)
    if not os.path.isdir(dataset_root):
        raise FileNotFoundError(f"Dataset directory not found: {dataset_root}")
    dataset = VITONDataset(opt)
    print(f"[info] Loaded dataset with {len(dataset)} pairs from {dataset_root}")
    return dataset


try:
    DATASET = load_dataset(DATASET_OPT)
    PAIR_LIST = list(zip(DATASET.img_names, DATASET.c_names["unpaired"]))
except Exception as dataset_exc:
    print(f"[warn] Failed to load preprocessed dataset: {dataset_exc}")
    DATASET = None
    PAIR_LIST = []

TOTAL_PAIRS = len(PAIR_LIST)
PREVIEW_COUNT = min(100, TOTAL_PAIRS)
PAIR_OPTIONS_PREVIEW = list(enumerate(PAIR_LIST[:PREVIEW_COUNT]))


def gaussian_blur_tensor(tensor, kernel_size=15, sigma=3.0):
    if kernel_size <= 1 or tensor.shape[1] == 0:
        return tensor

    coords = torch.arange(kernel_size, dtype=torch.float32, device=tensor.device) - (kernel_size - 1) / 2.0
    try:
        grid_y, grid_x = torch.meshgrid(coords, coords, indexing="ij")
    except TypeError:
        grid_y, grid_x = torch.meshgrid(coords, coords)

    kernel = torch.exp(-(grid_x ** 2 + grid_y ** 2) / (2 * sigma ** 2))
    kernel = kernel / kernel.sum()
    kernel = kernel.view(1, 1, kernel_size, kernel_size)
    kernel = kernel.repeat(tensor.shape[1], 1, 1, 1)

    return F.conv2d(tensor, kernel, padding=kernel_size // 2, groups=tensor.shape[1])


def tensor_to_pil(tensor, is_mask=False):
    if tensor.dim() == 4:
        tensor = tensor[0]
    tensor = tensor.detach().cpu()

    if is_mask or tensor.size(0) == 1:
        array = tensor.squeeze(0)
        if array.min() < 0 or array.max() > 1:
            array = (array.clamp(-1, 1) + 1) * 0.5
        array = array.clamp(0, 1).mul(255).byte().numpy()
        return Image.fromarray(array, mode="L")

    # For RGB images: ensure proper shape and normalization
    if tensor.size(0) == 3:
        # Check if tensor is already in [0, 1] range
        if tensor.min() >= 0 and tensor.max() <= 1:
            # Already normalized, just scale to [0, 255]
            array = tensor.mul(255).permute(1, 2, 0).byte().numpy()
        else:
            # Assume [-1, 1] range, normalize to [0, 1] then scale
            array = tensor.clamp(-1, 1)
            array = (array + 1) * 0.5
            array = array.mul(255).permute(1, 2, 0).byte().numpy()
        return Image.fromarray(array, mode="RGB")
    
    # Fallback for other cases
    array = tensor.clamp(-1, 1)
    array = (array + 1) * 0.5
    if array.dim() == 3 and array.size(0) == 3:
        array = array.mul(255).permute(1, 2, 0).byte().numpy()
        return Image.fromarray(array, mode="RGB")
    else:
        array = array.mul(255).byte().numpy()
        return Image.fromarray(array)


def save_tensor_image(tensor, path, is_mask=False):
    img = tensor_to_pil(tensor, is_mask=is_mask)
    img.save(path, quality=95)


# -------------------------------------------------------------------------
# VITON-HD style center crop (full body)
# -------------------------------------------------------------------------
def compute_viton_crop_box(img_w, img_h, pose_keypoints, target_size=(768, 1024)):
    """
    Compute crop box for VITON-HD style center crop.
    Returns: (left, top, right, bottom) crop coordinates
    """
    target_w, target_h = target_size
    
    # Calculate bounding box from pose keypoints
    valid_keypoints = pose_keypoints[(pose_keypoints[:, 0] > 0) | (pose_keypoints[:, 1] > 0)]
    
    if len(valid_keypoints) == 0:
        # No valid keypoints - use full image with center crop
        print("[WARNING] No valid pose keypoints, using full image center crop")
        # Center crop maintaining aspect ratio
        aspect_ratio = target_w / target_h
        img_aspect = img_w / img_h
        
        if img_aspect > aspect_ratio:
            # Image is wider - crop width
            new_w = int(img_h * aspect_ratio)
            left = (img_w - new_w) // 2
            right = left + new_w
            top, bottom = 0, img_h
        else:
            # Image is taller - crop height
            new_h = int(img_w / aspect_ratio)
            top = (img_h - new_h) // 2
            bottom = top + new_h
            left, right = 0, img_w
    else:
        # Use keypoints to determine bounding box
        min_x = max(0, int(valid_keypoints[:, 0].min() - 50))
        max_x = min(img_w, int(valid_keypoints[:, 0].max() + 50))
        min_y = max(0, int(valid_keypoints[:, 1].min() - 100))  # More padding at top for head
        max_y = min(img_h, int(valid_keypoints[:, 1].max() + 100))  # More padding at bottom for feet
        
        # Calculate crop dimensions maintaining aspect ratio
        crop_w = max_x - min_x
        crop_h = max_y - min_y
        aspect_ratio = target_w / target_h
        crop_aspect = crop_w / crop_h
        
        if crop_aspect > aspect_ratio:
            # Crop is wider - adjust height
            new_h = int(crop_w / aspect_ratio)
            center_y = (min_y + max_y) // 2
            min_y = max(0, center_y - new_h // 2)
            max_y = min(img_h, min_y + new_h)
        else:
            # Crop is taller - adjust width
            new_w = int(crop_h * aspect_ratio)
            center_x = (min_x + max_x) // 2
            min_x = max(0, center_x - new_w // 2)
            max_x = min(img_w, min_x + new_w)
        
        left, top, right, bottom = min_x, min_y, max_x, max_y
    
    return left, top, right, bottom


def viton_center_crop(person_pil, pose_keypoints, target_size=(768, 1024)):
    """
    Center crop person image to full body, matching VITON-HD preprocessing.
    Uses pose keypoints to determine bounding box, then center crops with padding.
    """
    orig_w, orig_h = person_pil.size
    target_w, target_h = target_size
    
    # Compute crop box
    left, top, right, bottom = compute_viton_crop_box(orig_w, orig_h, pose_keypoints, target_size)
    
    # Crop and resize
    cropped = person_pil.crop((left, top, right, bottom))
    resized = cropped.resize(target_size, Image.BICUBIC)
    
    # Scale keypoints to match resized image
    crop_w = right - left
    crop_h = bottom - top
    scale_x = target_w / float(crop_w)
    scale_y = target_h / float(crop_h)
    
    scaled_keypoints = pose_keypoints.copy()
    scaled_keypoints[:, 0] = (scaled_keypoints[:, 0] - left) * scale_x
    scaled_keypoints[:, 1] = (scaled_keypoints[:, 1] - top) * scale_y
    
    return resized, scaled_keypoints, (left, top, right, bottom)


# -------------------------------------------------------------------------
# Custom sample preprocessing: use SCHP, pose, cloth mask to mimic VITONDataset
# -------------------------------------------------------------------------
def preprocess_custom_viton(person_pil, cloth_pil, parse_array, pose_rgb_pil, pose_keypoints, cloth_mask_np):
    """
    Prepare a single custom pair (person, cloth) into the exact dict format
    the app expects, mimicking VITONDataset output so it can flow through
    Seg → GMM → ALIAS.
    Inputs:
        person_pil    : PIL Image (RGB) - ORIGINAL size
        cloth_pil      : PIL Image (RGB) - ORIGINAL size
        parse_array    : HxW numpy array (SCHP argmax labels 0–19) - ORIGINAL size
        pose_rgb_pil   : PIL Image (RGB) with OpenPose-style skeleton, already resized to target
        pose_keypoints : (18, 2) numpy array with keypoint coordinates [x, y] in ORIGINAL image coords
        cloth_mask_np : (H,W) numpy mask in [0,1] - ORIGINAL size
    Returns:
        dict with keys: img_name, c_name, img, img_agnostic, parse_agnostic, pose,
                        cloth, cloth_mask
    """
    from datasets import VITONDataset
    
    load_h, load_w = DATASET_OPT.load_height, DATASET_OPT.load_width
    resize_img = T.Resize((load_h, load_w), interpolation=Image.BICUBIC)
    resize_mask_nearest = T.Resize((load_h, load_w), interpolation=Image.NEAREST)
    to_tensor_norm = T.Compose([T.ToTensor(), T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    # STEP 1: VITON-HD style center crop (full body) for person image
    print(f"[DEBUG] Applying VITON-HD center crop to person image")
    person_resized, pose_keypoints_scaled, crop_box = viton_center_crop(
        person_pil.convert("RGB"), 
        pose_keypoints, 
        target_size=(load_w, load_h)
    )
    left, top, right, bottom = crop_box
    print(f"[DEBUG] Crop box: ({left}, {top}, {right}, {bottom})")
    
    # STEP 2: Resize cloth image to target size
    cloth_resized = resize_img(cloth_pil.convert("RGB"))
    
    # STEP 3: Crop and resize parse array to match cropped/resized person image
    # Parse array from SCHP is at (512, 384) - need to resize to original image size first
    orig_w, orig_h = person_pil.size
    parse_pil_orig = Image.fromarray(parse_array.astype(np.uint8))
    # SCHP outputs at (512, 384), resize back to original image size
    parse_pil_orig = parse_pil_orig.resize((orig_w, orig_h), Image.NEAREST)
    # Use the SAME crop box as person image
    parse_cropped = parse_pil_orig.crop((left, top, right, bottom))
    parse_pil = resize_mask_nearest(parse_cropped)
    parse_array_resized = np.array(parse_pil, dtype=np.int64)
    print(f"[DEBUG] Parse array: SCHP output (512,384) -> resized to ({orig_w}, {orig_h}) -> cropped to ({right-left}, {bottom-top}) -> resized to ({load_w}, {load_h})")
    
    # Use dataset methods to create parse_agnostic and img_agnostic
    # Create a dummy dataset instance to access the methods
    dummy_opt = types.SimpleNamespace()
    dummy_opt.load_height = load_h
    dummy_opt.load_width = load_w
    dataset_helper = VITONDataset.__new__(VITONDataset)
    dataset_helper.load_height = load_h
    dataset_helper.load_width = load_w
    
    # Get parse_agnostic using dataset method
    parse_pil_for_agnostic = Image.fromarray(parse_array_resized.astype(np.uint8))
    
    # DEBUG: Check what SCHP actually outputs BEFORE masking
    parse_before_mask = np.array(parse_pil_for_agnostic)
    unique_before = np.unique(parse_before_mask)
    print(f"[DEBUG] SCHP parse BEFORE masking - unique values: {unique_before}")
    print(f"[DEBUG] Value counts BEFORE: {[(val, np.sum(parse_before_mask == val)) for val in unique_before[:15]]}")
    
    # Check if SCHP output is valid - should have multiple body parts (not just 1-2 labels)
    # Also check for meaningful body parts (excluding background 0 and 19)
    non_bg_labels = unique_before[(unique_before != 0) & (unique_before != 19)]
    
    # Check for complete body structure - need multiple categories for good try-on
    has_face = any(label in unique_before for label in [4, 13])  # face
    has_torso = any(label in unique_before for label in [5, 6, 7, 12])  # upper/lower clothing, torso
    has_arms = any(label in unique_before for label in [14, 15])  # left/right arm
    has_legs = any(label in unique_before for label in [16, 17])  # left/right leg
    has_hair = any(label in unique_before for label in [1, 2, 3])  # hair
    
    # Count how many body part categories we have (face, torso, arms, legs, hair)
    category_count = sum([has_face, has_torso, has_arms, has_legs, has_hair])
    
    if len(unique_before) <= 2:
        print(f"[WARNING] SCHP parse only contains {len(unique_before)} label(s): {unique_before}. This indicates SCHP may have failed or is not working correctly!")
        print(f"[WARNING] Expected multiple body part labels (0-19), but got only: {unique_before.tolist()}")
        print(f"[WARNING] Continuing with processing, but results may be poor. Consider checking:")
        print(f"[WARNING]   1. SCHP checkpoint file exists and is valid")
        print(f"[WARNING]   2. Input image quality and format")
        print(f"[WARNING]   3. SCHP model is loaded correctly")
    elif len(non_bg_labels) < 5 or (not has_face and not has_torso) or category_count < 3:
        # Need at least 5 non-background labels OR (face AND torso) OR at least 3 body part categories for good results
        # For try-on, we especially need face and torso regions
        missing_parts = []
        if not has_face: missing_parts.append("face")
        if not has_torso: missing_parts.append("torso")
        if not has_arms: missing_parts.append("arms")
        if not has_legs: missing_parts.append("legs")
        if not has_hair: missing_parts.append("hair")
        
        print(f"[WARNING] SCHP parse has insufficient body structure! Detected {len(non_bg_labels)} non-background labels: {non_bg_labels.tolist()}")
        print(f"[WARNING] Body part categories detected: {category_count}/5 (face={has_face}, torso={has_torso}, arms={has_arms}, legs={has_legs}, hair={has_hair})")
        print(f"[WARNING] Missing: {', '.join(missing_parts) if missing_parts else 'none'}")
        print(f"[WARNING] SCHP may have failed partially. The synthesis step will attempt to fill in missing parts, but quality may be reduced.")
        print(f"[WARNING] Consider checking SCHP model and input image quality.")
    
    # Check if we have enough keypoints for proper masking
    # Critical keypoints: 1 (neck), 2 (right_shoulder), 5 (left_shoulder), 8 (right_hip), 9 (left_hip)
    critical_kpts = [1, 2, 5, 8, 9]
    missing_critical = [i for i in critical_kpts if (pose_keypoints_scaled[i, 0] == 0 and pose_keypoints_scaled[i, 1] == 0)]
    
    # Estimate missing keypoints
    if 1 in missing_critical and 2 not in missing_critical and 5 not in missing_critical:
        # Estimate neck from shoulders
        pose_keypoints_scaled[1, 0] = (pose_keypoints_scaled[2, 0] + pose_keypoints_scaled[5, 0]) / 2
        pose_keypoints_scaled[1, 1] = (pose_keypoints_scaled[2, 1] + pose_keypoints_scaled[5, 1]) / 2
        print(f"[INFO] Estimated neck keypoint from shoulders")
    
    # The dataset code uses pose_data[12] for hip, but OpenPose 12 is right_ankle
    # If 12 is missing but 8 (right_hip) exists, use 8's position for 12
    if (pose_keypoints_scaled[12, 0] == 0 and pose_keypoints_scaled[12, 1] == 0) and \
       (pose_keypoints_scaled[8, 0] > 0 or pose_keypoints_scaled[8, 1] > 0):
        pose_keypoints_scaled[12, 0] = pose_keypoints_scaled[8, 0]
        pose_keypoints_scaled[12, 1] = pose_keypoints_scaled[8, 1]
        print(f"[INFO] Using right_hip (8) position for keypoint 12")
    
    if missing_critical:
        print(f"[WARNING] Missing critical pose keypoints: {missing_critical}. Results may be distorted.")
    
    # CRITICAL FIX 1: Check for SCHP misclassification - if any single label is >50% of image, it's wrong
    total_pixels = load_h * load_w
    unique_labels, label_counts = np.unique(parse_array_resized, return_counts=True)
    label_percentages = {label: (count / total_pixels * 100) for label, count in zip(unique_labels, label_counts) if label != 0}
    
    # Find labels that are too dominant (misclassification)
    dominant_labels = {label: pct for label, pct in label_percentages.items() if pct > 50}
    
    if dominant_labels:
        print(f"[CRITICAL] SCHP MISCLASSIFICATION DETECTED! Labels >50% of image: {dominant_labels}")
        print(f"[CRITICAL] This indicates SCHP parsing failed. Redistributing misclassified pixels...")
        
        for misclassified_label, pct in dominant_labels.items():
            print(f"[CRITICAL] Label {misclassified_label} is {pct:.1f}% of image - redistributing...")
            
            # Get mask of misclassified pixels
            misclassified_mask = (parse_array_resized == misclassified_label)
            total_misclassified = np.sum(misclassified_mask)
            
            # CRITICAL: Never redistribute back to the same misclassified label!
            # Determine safe redistribution targets based on misclassified label
            if misclassified_label == 13:  # Face misclassified
                # If face is misclassified, redistribute head region to hair instead
                head_target = 2  # Hair
                upper_target = 5  # Upper-clothes
                lower_target = 9  # Pants
            elif misclassified_label == 2:  # Hair misclassified
                # If hair is misclassified, redistribute head region to face (but only small area)
                head_target = 13  # Face (small area only)
                upper_target = 5  # Upper-clothes
                lower_target = 9  # Pants
            elif misclassified_label in [5, 6, 7]:  # Upper clothing misclassified
                # If upper clothing is misclassified, use pose to determine proper regions
                head_target = 13  # Face
                upper_target = 5  # Upper-clothes (will be redistributed based on pose)
                lower_target = 9  # Pants
            else:
                # Default: redistribute based on position
                head_target = 2  # Hair (safer than face)
                upper_target = 5  # Upper-clothes
                lower_target = 9  # Pants
            
            # Use pose to determine proper body regions with better fallbacks
            neck_y = pose_keypoints_scaled[1, 1] if pose_keypoints_scaled[1, 1] > 0 else load_h * 0.15
            hip_y = (pose_keypoints_scaled[8, 1] + pose_keypoints_scaled[9, 1]) / 2 if (pose_keypoints_scaled[8, 1] > 0 and pose_keypoints_scaled[9, 1] > 0) else load_h * 0.6
            
            h_coords = np.arange(load_h, dtype=np.float32)
            w_coords = np.arange(load_w, dtype=np.float32)
            h_grid, w_grid = np.meshgrid(h_coords, w_coords, indexing='ij')
            
            # Redistribute based on position:
            # - Above neck: hair (2) or face (13) - but NOT if that's the misclassified label
            # - Neck to hip: upper clothing (5)
            # - Below hip: pants (9) or legs (16, 17)
            
            # Upper body region (neck to hip) - most likely for misclassified pixels
            upper_mask = (h_grid >= neck_y) & (h_grid < hip_y) & misclassified_mask
            if np.sum(upper_mask) > 0:
                parse_array_resized[upper_mask] = upper_target
                print(f"[INFO] Redistributed {np.sum(upper_mask)} pixels to label {upper_target}")
            
            # Lower body region (below hip)
            lower_mask = (h_grid >= hip_y) & misclassified_mask
            if np.sum(lower_mask) > 0:
                parse_array_resized[lower_mask] = lower_target
                print(f"[INFO] Redistributed {np.sum(lower_mask)} pixels to label {lower_target}")
            
            # Head region (above neck) - be careful not to redistribute to misclassified label
            head_mask = (h_grid < neck_y) & misclassified_mask
            if np.sum(head_mask) > 0:
                if misclassified_label == 13:  # Face is misclassified, so use hair for all head region
                    parse_array_resized[head_mask] = 2  # Hair
                    print(f"[INFO] Redistributed {np.sum(head_mask)} pixels to Hair (label 2) - avoiding misclassified Face")
                elif misclassified_label == 2:  # Hair is misclassified, use face for lower head, upper-clothes for upper head
                    # Split head region: upper part goes to upper-clothes (might be misclassified hair on shoulders)
                    # Lower part (face area) goes to face
                    nose_y = pose_keypoints_scaled[0, 1] if pose_keypoints_scaled[0, 1] > 0 else neck_y - 50
                    face_mask = (h_grid >= nose_y - 30) & (h_grid < neck_y) & head_mask
                    hair_upper_mask = (h_grid < nose_y - 30) & head_mask
                    
                    if np.sum(face_mask) > 0:
                        parse_array_resized[face_mask] = 13  # Face
                        print(f"[INFO] Redistributed {np.sum(face_mask)} pixels to Face (label 13)")
                    if np.sum(hair_upper_mask) > 0:
                        # Upper head region might actually be shoulders/upper body
                        parse_array_resized[hair_upper_mask] = 5  # Upper-clothes
                        print(f"[INFO] Redistributed {np.sum(hair_upper_mask)} pixels to Upper-clothes (label 5) - upper head region")
                else:
                    # Normal case: split between hair and face
                    nose_y = pose_keypoints_scaled[0, 1] if pose_keypoints_scaled[0, 1] > 0 else neck_y - 50
                    face_mask = (h_grid >= nose_y - 50) & (h_grid < neck_y) & head_mask
                    hair_mask = (h_grid < nose_y - 50) & head_mask
                    
                    if np.sum(face_mask) > 0 and misclassified_label != 13:
                        parse_array_resized[face_mask] = 13  # Face
                        print(f"[INFO] Redistributed {np.sum(face_mask)} pixels to Face (label 13)")
                    elif np.sum(face_mask) > 0:
                        # Face is misclassified, so use hair instead
                        parse_array_resized[face_mask] = 2  # Hair
                        print(f"[INFO] Redistributed {np.sum(face_mask)} pixels to Hair (label 2) - avoiding misclassified Face")
                    
                    if np.sum(hair_mask) > 0 and misclassified_label != 2:
                        parse_array_resized[hair_mask] = 2  # Hair
                        print(f"[INFO] Redistributed {np.sum(hair_mask)} pixels to Hair (label 2)")
                    elif np.sum(hair_mask) > 0:
                        # Hair is misclassified, so use upper-clothes (might be shoulders)
                        parse_array_resized[hair_mask] = 5  # Upper-clothes
                        print(f"[INFO] Redistributed {np.sum(hair_mask)} pixels to Upper-clothes (label 5) - avoiding misclassified Hair")
            
            # Ensure ALL misclassified pixels are redistributed (fallback for any remaining)
            remaining_mask = (parse_array_resized == misclassified_label)
            remaining_count = np.sum(remaining_mask)
            if remaining_count > 0:
                print(f"[WARNING] {remaining_count} pixels still have misclassified label {misclassified_label} - redistributing based on position...")
                # Redistribute remaining pixels based on their position
                remaining_upper = (h_grid >= neck_y) & (h_grid < hip_y) & remaining_mask
                remaining_lower = (h_grid >= hip_y) & remaining_mask
                remaining_head = (h_grid < neck_y) & remaining_mask
                
                if np.sum(remaining_upper) > 0:
                    parse_array_resized[remaining_upper] = 5  # Upper-clothes
                    print(f"[INFO] Redistributed remaining {np.sum(remaining_upper)} upper pixels to Upper-clothes (label 5)")
                if np.sum(remaining_lower) > 0:
                    parse_array_resized[remaining_lower] = 9  # Pants
                    print(f"[INFO] Redistributed remaining {np.sum(remaining_lower)} lower pixels to Pants (label 9)")
                if np.sum(remaining_head) > 0:
                    # For head, avoid the misclassified label
                    if misclassified_label == 13:
                        head_label = 2  # Hair
                        parse_array_resized[remaining_head] = head_label
                    elif misclassified_label == 2:
                        head_label = 5  # Upper-clothes (might be shoulders)
                        parse_array_resized[remaining_head] = head_label
                    else:
                        head_label = 2  # Hair (safe default)
                        parse_array_resized[remaining_head] = head_label
                    print(f"[INFO] Redistributed remaining {np.sum(remaining_head)} head pixels to label {head_label}")
        
        parse_pil_for_agnostic = Image.fromarray(parse_array_resized.astype(np.uint8))
        print(f"[INFO] SCHP misclassification fixed. New unique labels: {np.unique(parse_array_resized)}")
    
    # Also check for label 3 (Glove) misclassification
    glove_pixels = np.sum(parse_array_resized == 3)
    glove_percent = glove_pixels / total_pixels * 100
    
    if glove_percent > 20:  # More than 20% classified as Glove = misclassification
        print(f"[CRITICAL] Label 3 (Glove) is {glove_percent:.1f}% of image - converting to Upper-clothes...")
        parse_array_resized[parse_array_resized == 3] = 5
        parse_pil_for_agnostic = Image.fromarray(parse_array_resized.astype(np.uint8))
        print(f"[INFO] Converted {glove_pixels} Glove pixels to Upper-clothes (label 5)")
    
    # CRITICAL FIX 2: Ensure original parse has upper clothing labels (5,6,7) BEFORE get_parse_agnostic
    # The dataset's original parse images ALREADY have these labels (preprocessed)
    has_upper_in_original = np.any((parse_array_resized == 5) | (parse_array_resized == 6) | (parse_array_resized == 7))
    
    if not has_upper_in_original:
        print(f"[CRITICAL] Original parse missing upper clothing labels (5,6,7). Adding from pose...")
        # Use pose to add upper clothing region to original parse
        neck_y = pose_keypoints_scaled[1, 1] if pose_keypoints_scaled[1, 1] > 0 else load_h * 0.15
        hip_y = (pose_keypoints_scaled[8, 1] + pose_keypoints_scaled[9, 1]) / 2 if (pose_keypoints_scaled[8, 1] > 0 and pose_keypoints_scaled[9, 1] > 0) else load_h * 0.6
        
        # Create upper body mask
        h_coords = np.arange(load_h, dtype=np.float32)
        w_coords = np.arange(load_w, dtype=np.float32)
        h_grid, w_grid = np.meshgrid(h_coords, w_coords, indexing='ij')
        
        # Upper body region: between neck and hips
        upper_mask = (h_grid >= neck_y) & (h_grid < hip_y)
        
        # Exclude areas already assigned to other body parts
        exclude_mask = (parse_array_resized == 2) | (parse_array_resized == 4) | (parse_array_resized == 13) | \
                      (parse_array_resized == 9) | (parse_array_resized == 12) | \
                      (parse_array_resized == 16) | (parse_array_resized == 17) | \
                      (parse_array_resized == 18) | (parse_array_resized == 19)
        
        # Add upper clothing label (5 = Upper-clothes) where not excluded
        parse_array_resized[upper_mask & ~exclude_mask] = 5
        # Update parse_pil_for_agnostic too
        parse_pil_for_agnostic = Image.fromarray(parse_array_resized.astype(np.uint8))
        print(f"[INFO] Added upper clothing label (5) to {np.sum(upper_mask & ~exclude_mask)} pixels in original parse")
    
    # CRITICAL FIX 3: Ensure face is present in original parse
    has_face_in_original = np.any((parse_array_resized == 4) | (parse_array_resized == 13))
    if not has_face_in_original and pose_keypoints_scaled[0, 1] > 0:  # nose keypoint exists
        print(f"[CRITICAL] Face missing in original parse. Adding from pose...")
        from PIL import ImageDraw
        mask_img = Image.new('L', (load_w, load_h), 0)
        draw = ImageDraw.Draw(mask_img)
        nose = tuple(pose_keypoints_scaled[0].astype(int))
        face_size = 95
        draw.ellipse([nose[0]-face_size, nose[1]-face_size*1.5, 
                     nose[0]+face_size, nose[1]+face_size*1.1], fill=255)
        mask = np.array(mask_img) > 128
        
        # CRITICAL: Overwrite hair/background in face region - don't just check for 0
        # Face should be above hair, so overwrite hair (1, 2) and background (0) in face region
        face_mask = mask & ((parse_array_resized == 0) | (parse_array_resized == 1) | (parse_array_resized == 2))
        parse_array_resized[face_mask] = 13  # face
        parse_pil_for_agnostic = Image.fromarray(parse_array_resized.astype(np.uint8))
        face_pixels_added = np.sum(face_mask)
        print(f"[INFO] Added face label (13) to {face_pixels_added} pixels in original parse")
        
        if face_pixels_added == 0:
            print(f"[WARNING] Face mask found 0 pixels! Nose position: {nose}, Face size: {face_size}")
            print(f"[WARNING] Parse array shape: {parse_array_resized.shape}, Unique values in mask region: {np.unique(parse_array_resized[mask])}")
    
    # CRITICAL: Save ORIGINAL parse (before get_parse_agnostic) for building parse_agnostic_map
    # We need this because after masking, upper clothing labels (5,6,7) become 0
    # But we need them in the map to ensure proper structure for SEG model
    parse_original_for_map = parse_array_resized.copy()  # Original with upper clothing labels 5,6,7
    print(f"[DEBUG] Saved original parse for map building. Has upper clothing: {np.any((parse_original_for_map == 5) | (parse_original_for_map == 6) | (parse_original_for_map == 7))}")
    
    # Build parse_agnostic EXACTLY like the dataset does (lines 184-204 in datasets.py):
    # 1. Dataset uses MASKED parse_agnostic to build parse_agnostic_map
    # 2. But original parse should have upper clothing labels for proper structure
    # 3. The SEG model will generate channel 3 from the input structure
    
    # Apply get_parse_agnostic to mask arms and upper clothing (for img_agnostic)
    parse_agnostic_pil = dataset_helper.get_parse_agnostic(parse_pil_for_agnostic, pose_keypoints_scaled)
    parse_agnostic_np = np.array(parse_agnostic_pil, dtype=np.int64)
    unique_after = np.unique(parse_agnostic_np)
    print(f"[DEBUG] get_parse_agnostic result. Unique values: {unique_after}")
    print(f"[DEBUG] Value counts: {[(val, np.sum(parse_agnostic_np == val)) for val in unique_after[:10]]}")
    
    # CRITICAL: For parse_agnostic_map, we use MASKED parse_agnostic (like dataset does)
    # But we ensured original parse has upper clothing labels before masking
    # The SEG model will generate channel 3 from the input structure
    
    # CRITICAL FIX: If SCHP failed (missing critical body parts), use pose to fill them in
    # We MUST have body structure for try-on to work - SEG model can't work without it
    non_zero_labels = unique_after[unique_after != 0]
    has_torso_like = any(l in unique_after for l in [5, 6, 7, 9, 12])  # upper/lower clothing
    has_legs = any(l in unique_after for l in [16, 17])  # legs
    has_right_arm = 15 in unique_after
    has_left_arm = 14 in unique_after
    
    # If missing critical body parts, synthesize from pose
    if len(non_zero_labels) < 5 or (not has_torso_like and not has_legs):
        print(f"[CRITICAL] SCHP missing body structure! Synthesizing missing parts from pose keypoints...")
        print(f"[CRITICAL] Currently have labels: {unique_after.tolist()}")
        
        from PIL import ImageDraw
        
        def is_valid_kpt(kpt_idx):
            return (pose_keypoints_scaled[kpt_idx, 0] > 0 and pose_keypoints_scaled[kpt_idx, 1] > 0 and
                    pose_keypoints_scaled[kpt_idx, 0] < load_w and pose_keypoints_scaled[kpt_idx, 1] < load_h)
        
        # Build enhanced parse with pose-synthesized body parts
        enhanced_parse = parse_agnostic_np.copy()
        
        # SYNTHESIZE IN ORDER: face first, then hair, then body parts
        # This ensures proper layering (face should be below hair)
        
        # Synthesize face FIRST if missing (critical for try-on)
        # Check if face is missing in MASKED parse
        has_face_in_masked = 4 in unique_after or 13 in unique_after
        if not has_face_in_masked and is_valid_kpt(0):  # nose
            mask_img = Image.new('L', (load_w, load_h), 0)
            draw = ImageDraw.Draw(mask_img)
            nose = tuple(pose_keypoints_scaled[0].astype(int))
            face_size = 95
            draw.ellipse([nose[0]-face_size, nose[1]-face_size*1.5, 
                         nose[0]+face_size, nose[1]+face_size*1.1], fill=255)
            mask = np.array(mask_img) > 128
            # CRITICAL: Overwrite background/hair in face region - face should be visible
            face_mask = mask & ((enhanced_parse == 0) | (enhanced_parse == 1) | (enhanced_parse == 2))
            enhanced_parse[face_mask] = 13  # face
            face_pixels = np.sum(face_mask)
            print(f"[INFO] Synthesized face from pose in MASKED parse - {face_pixels} pixels")
        
        # Synthesize hair if missing - above face (after face so they layer properly)
        has_hair_in_original = 1 in parse_array_resized or 2 in parse_array_resized
        if not has_hair_in_original and is_valid_kpt(0):  # nose
            mask_img = Image.new('L', (load_w, load_h), 0)
            draw = ImageDraw.Draw(mask_img)
            nose = tuple(pose_keypoints_scaled[0].astype(int))
            hair_y_top = max(0, nose[1] - 160)
            hair_y_bottom = nose[1] - 65
            hair_width = 150
            draw.ellipse([nose[0]-hair_width, hair_y_top, nose[0]+hair_width, hair_y_bottom], fill=255)
            mask = np.array(mask_img) > 128
            # Don't overwrite face - add hair above face
            enhanced_parse[mask & (enhanced_parse != 13) & (enhanced_parse == 0)] = 2  # hair
            hair_pixels = np.sum(mask & (enhanced_parse != 13))
            print(f"[INFO] Synthesized hair from pose - {hair_pixels} pixels")
        
        # Synthesize right arm if missing
        if not has_right_arm and is_valid_kpt(2) and is_valid_kpt(3) and is_valid_kpt(4):
            mask_img = Image.new('L', (load_w, load_h), 0)
            draw = ImageDraw.Draw(mask_img)
            shoulder = tuple(pose_keypoints_scaled[2].astype(int))
            elbow = tuple(pose_keypoints_scaled[3].astype(int))
            wrist = tuple(pose_keypoints_scaled[4].astype(int))
            width = 45
            draw.line([shoulder, elbow], fill=255, width=width)
            draw.line([elbow, wrist], fill=255, width=width)
            draw.ellipse([shoulder[0]-width//2, shoulder[1]-width//2, shoulder[0]+width//2, shoulder[1]+width//2], fill=255)
            mask = np.array(mask_img) > 128
            enhanced_parse[mask & (enhanced_parse == 0)] = 15  # right_arm
            print(f"[INFO] Synthesized right_arm from pose")
        
        # Synthesize left arm if missing (even if SCHP detected it, it might be masked)
        if not has_left_arm and is_valid_kpt(5) and is_valid_kpt(6) and is_valid_kpt(7):
            mask_img = Image.new('L', (load_w, load_h), 0)
            draw = ImageDraw.Draw(mask_img)
            shoulder = tuple(pose_keypoints_scaled[5].astype(int))
            elbow = tuple(pose_keypoints_scaled[6].astype(int))
            wrist = tuple(pose_keypoints_scaled[7].astype(int))
            width = 45
            draw.line([shoulder, elbow], fill=255, width=width)
            draw.line([elbow, wrist], fill=255, width=width)
            draw.ellipse([shoulder[0]-width//2, shoulder[1]-width//2, shoulder[0]+width//2, shoulder[1]+width//2], fill=255)
            mask = np.array(mask_img) > 128
            enhanced_parse[mask & (enhanced_parse == 0)] = 14  # left_arm
            print(f"[INFO] Synthesized left_arm from pose")
        
        # Synthesize legs if missing
        if not has_legs:
            # Right leg
            if is_valid_kpt(8) and is_valid_kpt(10) and is_valid_kpt(12):
                mask_img = Image.new('L', (load_w, load_h), 0)
                draw = ImageDraw.Draw(mask_img)
                hip = tuple(pose_keypoints_scaled[8].astype(int))
                knee = tuple(pose_keypoints_scaled[10].astype(int))
                ankle = tuple(pose_keypoints_scaled[12].astype(int))
                width = 55
                draw.line([hip, knee], fill=255, width=width)
                draw.line([knee, ankle], fill=255, width=width)
                draw.ellipse([ankle[0]-width//2, ankle[1]-width//2, ankle[0]+width//2, ankle[1]+width//2], fill=255)
                mask = np.array(mask_img) > 128
                leg_mask = mask & (enhanced_parse == 0)
                enhanced_parse[leg_mask] = 17  # right_leg
                # Shoe at bottom of leg - create another mask for shoe region
                shoe_mask_img = Image.new('L', (load_w, load_h), 0)
                shoe_draw = ImageDraw.Draw(shoe_mask_img)
                shoe_draw.ellipse([ankle[0]-width//2-10, ankle[1]-width//2-10, 
                                  ankle[0]+width//2+10, ankle[1]+width//2+10], fill=255)
                shoe_mask = np.array(shoe_mask_img) > 128
                shoe_region = shoe_mask & mask & (enhanced_parse == 17)
                enhanced_parse[shoe_region] = 19  # right_shoe
                print(f"[INFO] Synthesized right_leg and shoe from pose")
            
            # Left leg
            if is_valid_kpt(9) and is_valid_kpt(11) and is_valid_kpt(13):
                mask_img = Image.new('L', (load_w, load_h), 0)
                draw = ImageDraw.Draw(mask_img)
                hip = tuple(pose_keypoints_scaled[9].astype(int))
                knee = tuple(pose_keypoints_scaled[11].astype(int))
                ankle = tuple(pose_keypoints_scaled[13].astype(int))
                width = 55
                draw.line([hip, knee], fill=255, width=width)
                draw.line([knee, ankle], fill=255, width=width)
                draw.ellipse([ankle[0]-width//2, ankle[1]-width//2, ankle[0]+width//2, ankle[1]+width//2], fill=255)
                mask = np.array(mask_img) > 128
                leg_mask = mask & (enhanced_parse == 0)
                enhanced_parse[leg_mask] = 16  # left_leg
                # Shoe at bottom of leg - create another mask for shoe region
                shoe_mask_img = Image.new('L', (load_w, load_h), 0)
                shoe_draw = ImageDraw.Draw(shoe_mask_img)
                shoe_draw.ellipse([ankle[0]-width//2-10, ankle[1]-width//2-10, 
                                  ankle[0]+width//2+10, ankle[1]+width//2+10], fill=255)
                shoe_mask = np.array(shoe_mask_img) > 128
                shoe_region = shoe_mask & mask & (enhanced_parse == 16)
                enhanced_parse[shoe_region] = 18  # left_shoe
                print(f"[INFO] Synthesized left_leg and shoe from pose")
        
        # Synthesize lower body (pants/skirt) if missing - between hips
        if not has_torso_like and (is_valid_kpt(8) or is_valid_kpt(9)):
            mask_img = Image.new('L', (load_w, load_h), 0)
            draw = ImageDraw.Draw(mask_img)
            hips = []
            if is_valid_kpt(8):
                hips.append(pose_keypoints_scaled[8])
            if is_valid_kpt(9):
                hips.append(pose_keypoints_scaled[9])
            if len(hips) >= 1:
                hip_y = int(np.mean([h[1] for h in hips]))
                hip_x = int(np.mean([h[0] for h in hips]))
                # Estimate knee position
                if is_valid_kpt(10) and is_valid_kpt(11):
                    knee_y = int(np.mean([pose_keypoints_scaled[10, 1], pose_keypoints_scaled[11, 1]]))
                else:
                    knee_y = hip_y + 180
                
                width = 220
                draw.rectangle([hip_x-width//2, hip_y, hip_x+width//2, knee_y], fill=255)
                draw.ellipse([hip_x-width//2, hip_y-25, hip_x+width//2, hip_y+25], fill=255)
                mask = np.array(mask_img) > 128
                # Only add where not already leg and not background (but not 0 means keep existing)
                not_leg = (enhanced_parse != 16) & (enhanced_parse != 17) & (enhanced_parse != 18) & (enhanced_parse != 19)
                enhanced_parse[mask & (enhanced_parse == 0) & not_leg] = 9  # pants
                print(f"[INFO] Synthesized lower_body from pose")
        
        parse_agnostic_np = enhanced_parse
        print(f"[INFO] Enhanced parse now has labels: {np.unique(parse_agnostic_np).tolist()}")
    
    # CRITICAL: Ensure face is in the final parse_agnostic_np
    # If face is still missing, add it aggressively
    has_face_final = np.any((parse_agnostic_np == 4) | (parse_agnostic_np == 13))
    if not has_face_final and pose_keypoints_scaled[0, 1] > 0:
        print(f"[CRITICAL] Face still missing in final parse! Adding aggressively...")
        from PIL import ImageDraw
        mask_img = Image.new('L', (load_w, load_h), 0)
        draw = ImageDraw.Draw(mask_img)
        nose = tuple(pose_keypoints_scaled[0].astype(int))
        face_size = 95
        draw.ellipse([nose[0]-face_size, nose[1]-face_size*1.5, 
                     nose[0]+face_size, nose[1]+face_size*1.1], fill=255)
        mask = np.array(mask_img) > 128
        # AGGRESSIVE: Overwrite everything in face region except critical body parts
        # Keep: legs (16,17), shoes (18,19), pants (9,12) - everything else can be overwritten
        face_mask = mask & ((parse_agnostic_np != 16) & (parse_agnostic_np != 17) & 
                           (parse_agnostic_np != 18) & (parse_agnostic_np != 19) &
                           (parse_agnostic_np != 9) & (parse_agnostic_np != 12))
        parse_agnostic_np[face_mask] = 13  # face
        face_pixels = np.sum(face_mask)
        print(f"[INFO] Aggressively added face label (13) to {face_pixels} pixels in final parse")
    
    # Note: Even if SCHP parse was incomplete, we've now filled in missing body parts from pose
    
    try:
        img_agnostic_pil = dataset_helper.get_img_agnostic(person_resized, parse_pil_for_agnostic, pose_keypoints_scaled)
        print(f"[DEBUG] get_img_agnostic succeeded")
    except Exception as e:
        print(f"[ERROR] get_img_agnostic failed: {e}")
        import traceback
        traceback.print_exc()
        img_agnostic_pil = person_resized.copy()
    
    # CRITICAL: Ensure upper clothing is COMPLETELY removed using parse
    # The pose-based masking might miss some areas, so use parse to be thorough
    parse_array_np = np.array(parse_pil_for_agnostic)
    img_np = np.array(img_agnostic_pil)
    
    # Identify upper clothing regions from parse
    # Upper clothing labels: 5 (Upper-clothes), 6 (Dress), 7 (Coat)
    upper_clothing_mask = ((parse_array_np == 5) | (parse_array_np == 6) | (parse_array_np == 7))
    
    # Also mask arms (14, 15) and neck (10) - these should be grayed out
    arms_neck_mask = ((parse_array_np == 14) | (parse_array_np == 15) | (parse_array_np == 10))
    
    # Combine masks
    gray_out_mask = upper_clothing_mask | arms_neck_mask
    
    # Gray out upper clothing and arms completely
    img_np[gray_out_mask] = [128, 128, 128]  # Gray color (RGB)
    
    # Also use pose-based region to ensure complete coverage
    # If pose keypoints are available, gray out the torso region more aggressively
    if pose_keypoints_scaled[1, 1] > 0:  # neck keypoint exists
        neck_y = int(pose_keypoints_scaled[1, 1])
        hip_y = int((pose_keypoints_scaled[8, 1] + pose_keypoints_scaled[9, 1]) / 2) if (pose_keypoints_scaled[8, 1] > 0 and pose_keypoints_scaled[9, 1] > 0) else int(load_h * 0.6)
        
        # Gray out entire torso region (neck to hip)
        h_coords = np.arange(load_h)
        torso_mask = (h_coords >= neck_y) & (h_coords < hip_y)
        torso_mask_2d = torso_mask[:, np.newaxis]  # (H, 1)
        
        # Only gray out if not already head or lower body
        head_mask = (parse_array_np == 4) | (parse_array_np == 13)
        lower_mask = (parse_array_np == 9) | (parse_array_np == 12) | \
                     (parse_array_np == 16) | (parse_array_np == 17) | \
                     (parse_array_np == 18) | (parse_array_np == 19)
        
        # Gray out torso region excluding head and lower body
        torso_gray_mask = torso_mask_2d & (~head_mask) & (~lower_mask)
        img_np[torso_gray_mask] = [128, 128, 128]
        
        print(f"[INFO] Grayed out {np.sum(gray_out_mask)} pixels from parse + {np.sum(torso_gray_mask)} pixels from pose-based torso region")
    else:
        print(f"[INFO] Grayed out {np.sum(gray_out_mask)} pixels from parse (upper clothing + arms + neck)")
    
    img_agnostic_pil = Image.fromarray(img_np)
    print(f"[DEBUG] img_agnostic: Upper clothing completely removed using parse + pose")
    
    # Convert to tensors
    person = to_tensor_norm(person_resized)  # (3,H,W)
    cloth = to_tensor_norm(cloth_resized)    # (3,H,W)
    img_agnostic = to_tensor_norm(img_agnostic_pil)  # (3,H,W)
    
    # Build parse_agnostic semantic map EXACTLY like the dataset (lines 203-208 in datasets.py)
    # parse_agnostic_np is already the result from get_parse_agnostic - use it directly
    
    # CRITICAL FIX: Save ORIGINAL parse (before get_parse_agnostic) for building parse_agnostic_map
    # The dataset uses masked parse_agnostic, but we need to ensure channel 3 (upper) has structure
    # So we'll use ORIGINAL parse (with upper clothing) to build the map, then mask it properly
    parse_original_for_map = parse_array_resized.copy()  # Original with upper clothing labels 5,6,7
    
    # Build parse_agnostic semantic map EXACTLY like the dataset (lines 184-204 in datasets.py)
    # BUT: We'll use ORIGINAL parse to ensure channel 3 has proper values, then apply masking logic
    parse_for_map_np = np.clip(parse_original_for_map, 0, 19)
    parse_for_map_tensor = torch.tensor(parse_for_map_np[None], dtype=torch.long)  # (1,H,W)
    
    print(f"[DEBUG] Building parse_agnostic map from ORIGINAL parse (before masking)")
    print(f"[DEBUG] Original parse unique values: {np.unique(parse_for_map_np)}")
    print(f"[DEBUG] Has upper clothing (5,6,7): {np.any((parse_for_map_np == 5) | (parse_for_map_np == 6) | (parse_for_map_np == 7))}")
    
    # Build one-hot map over 20 classes (0-19) from ORIGINAL parse
    parse_agnostic_map = torch.zeros(20, load_h, load_w, dtype=torch.float32)
    parse_agnostic_map.scatter_(0, parse_for_map_tensor, 1.0)
    
    # CRITICAL: DON'T mask upper clothing in the map - we need it for channel 3!
    # The dataset uses masked parse_agnostic, but channel 3 (upper) still needs structure
    # We'll mask arms and neck, but KEEP upper clothing (5,6,7) for proper channel 3 mapping
    parse_agnostic_map[14] = 0  # Left-arm (masked)
    parse_agnostic_map[15] = 0  # Right-arm (masked)
    parse_agnostic_map[10] = 0  # Neck (masked)
    # KEEP: parse_agnostic_map[5,6,7] for channel 3 (upper) mapping
    
    print(f"[DEBUG] Applied partial masking to parse_agnostic_map (arms + neck set to 0, upper clothing KEPT for channel 3)")
    
    # Map 20 classes to 13 semantic classes exactly like dataset
    # LIP 20 classes: 0=Background, 1=Hat, 2=Hair, 3=Glove, 4=Sunglasses, 5=Upper-clothes, 
    # 6=Dress, 7=Coat, 8=Socks, 9=Pants, 10=Jumpsuits, 11=Scarf, 12=Skirt, 13=Face,
    # 14=Left-arm, 15=Right-arm, 16=Left-leg, 17=Right-leg, 18=Left-shoe, 19=Right-shoe
    labels = {
        0: ['background', [0, 10]],      # background + Jumpsuits (treated as background)
        1: ['hair', [1, 2]],              # Hat + Hair
        2: ['face', [4, 13]],             # Sunglasses + Face
        3: ['upper', [5, 6, 7]],          # Upper-clothes + Dress + Coat (masked out in agnostic)
        4: ['bottom', [9, 12]],           # Pants + Skirt
        5: ['left_arm', [14]],            # Left-arm (masked out in agnostic)
        6: ['right_arm', [15]],          # Right-arm (masked out in agnostic)
        7: ['left_leg', [16]],           # Left-leg
        8: ['right_leg', [17]],          # Right-leg
        9: ['left_shoe', [18]],           # Left-shoe
        10: ['right_shoe', [19]],         # Right-shoe
        11: ['socks', [8]],               # Socks
        12: ['noise', [3, 11]]            # Glove + Scarf (treated as noise)
    }
    
    # CRITICAL FIX: If SCHP detected mostly "Scarf" (label 11) or other noise labels,
    # it means SCHP failed. We need to redistribute those pixels to proper body parts.
    # Check if label 11 (Scarf) or label 3 (Glove) has too many pixels (>30% of image)
    total_pixels = load_h * load_w
    scarf_pixels = parse_agnostic_map[11].sum().item() if 11 < parse_agnostic_map.size(0) else 0
    glove_pixels = parse_agnostic_map[3].sum().item() if 3 < parse_agnostic_map.size(0) else 0
    noise_pixels = scarf_pixels + glove_pixels
    
    if noise_pixels > total_pixels * 0.3:  # More than 30% classified as noise
        print(f"[CRITICAL] SCHP misclassified {noise_pixels/total_pixels*100:.1f}% as noise (Scarf/Glove)")
        print(f"[CRITICAL] This indicates SCHP parsing failed. Redistributing noise pixels to body parts...")
        
        # Get noise mask (Scarf + Glove) as boolean tensor
        scarf_mask = parse_agnostic_map[11] > 0.5 if 11 < parse_agnostic_map.size(0) else torch.zeros(load_h, load_w, dtype=torch.bool)
        glove_mask = parse_agnostic_map[3] > 0.5 if 3 < parse_agnostic_map.size(0) else torch.zeros(load_h, load_w, dtype=torch.bool)
        noise_mask = scarf_mask | glove_mask
        
        # Use pose keypoints to determine where noise pixels should go
        # If we have valid pose keypoints, redistribute based on position
        valid_kpts = pose_keypoints_scaled[(pose_keypoints_scaled[:, 0] > 0) | (pose_keypoints_scaled[:, 1] > 0)]
        
        if len(valid_kpts) > 5:  # Enough keypoints to determine body regions
            # Create body region masks based on pose
            h_coords, w_coords = torch.meshgrid(
                torch.arange(load_h, dtype=torch.float32),
                torch.arange(load_w, dtype=torch.float32),
                indexing='ij'
            )
            
            # Estimate body regions from keypoints
            neck_y = pose_keypoints_scaled[1, 1] if pose_keypoints_scaled[1, 1] > 0 else load_h * 0.2
            hip_y = (pose_keypoints_scaled[8, 1] + pose_keypoints_scaled[9, 1]) / 2 if (pose_keypoints_scaled[8, 1] > 0 and pose_keypoints_scaled[9, 1] > 0) else load_h * 0.6
            
            # Redistribute noise pixels:
            # - Above neck: hair/face region
            # - Neck to hip: upper body (torso)
            # - Below hip: lower body (legs)
            
            # Upper body (torso) - most likely for misclassified pixels
            upper_mask = ((h_coords >= neck_y) & (h_coords < hip_y) & noise_mask).float()
            if upper_mask.sum() > 0:
                # Map to upper clothing (label 5, 6, or 7) - but these get masked, so use background
                # Actually, keep as background since upper gets masked anyway
                parse_agnostic_map[0] = torch.clamp(parse_agnostic_map[0] + upper_mask, 0, 1)
                parse_agnostic_map[11] = parse_agnostic_map[11] * (1 - upper_mask)
                parse_agnostic_map[3] = parse_agnostic_map[3] * (1 - upper_mask)
                print(f"[INFO] Redistributed {upper_mask.sum().item():.0f} noise pixels to background (upper body region)")
            
            # Lower body (legs)
            lower_mask = ((h_coords >= hip_y) & noise_mask).float()
            if lower_mask.sum() > 0:
                # Map to pants (label 9)
                parse_agnostic_map[9] = torch.clamp(parse_agnostic_map[9] + lower_mask, 0, 1)
                parse_agnostic_map[11] = parse_agnostic_map[11] * (1 - lower_mask)
                parse_agnostic_map[3] = parse_agnostic_map[3] * (1 - lower_mask)
                print(f"[INFO] Redistributed {lower_mask.sum().item():.0f} noise pixels to pants (lower body region)")
            
            # Head region
            head_mask = ((h_coords < neck_y) & noise_mask).float()
            if head_mask.sum() > 0:
                # Map to hair (label 2) or face (label 13)
                parse_agnostic_map[2] = torch.clamp(parse_agnostic_map[2] + head_mask, 0, 1)  # Hair
                parse_agnostic_map[11] = parse_agnostic_map[11] * (1 - head_mask)
                parse_agnostic_map[3] = parse_agnostic_map[3] * (1 - head_mask)
                print(f"[INFO] Redistributed {head_mask.sum().item():.0f} noise pixels to hair (head region)")
        else:
            # Not enough keypoints - just convert noise to background
            noise_mask_float = noise_mask.float()
            parse_agnostic_map[0] = torch.clamp(parse_agnostic_map[0] + noise_mask_float, 0, 1)
            parse_agnostic_map[11] = parse_agnostic_map[11] * (1 - noise_mask_float)
            parse_agnostic_map[3] = parse_agnostic_map[3] * (1 - noise_mask_float)
            print(f"[INFO] Redistributed {noise_mask_float.sum().item():.0f} noise pixels to background (no pose info)")
    
    new_parse_agnostic_map = torch.zeros(DATASET_OPT.semantic_nc, load_h, load_w, dtype=torch.float32)
    for i in range(len(labels)):
        for label in labels[i][1]:
            if label < 20:  # Safety check
                new_parse_agnostic_map[i] += parse_agnostic_map[label]
    
    # Debug: check what we actually have
    non_zero_channels = (new_parse_agnostic_map.sum(dim=(1, 2)) > 0).nonzero(as_tuple=True)[0].tolist()
    print(f"[DEBUG] Non-zero parse_agnostic channels: {non_zero_channels}")
    
    # CRITICAL: Ensure face channel (2) is populated
    face_channel_sum = new_parse_agnostic_map[2].sum().item()
    if face_channel_sum < total_pixels * 0.01:  # Less than 1% face
        print(f"[CRITICAL] Face channel (2) is too small: {face_channel_sum/total_pixels*100:.1f}%")
        print(f"[CRITICAL] Adding face to parse_agnostic_map from pose...")
        if pose_keypoints_scaled[0, 1] > 0:  # nose keypoint exists
            from PIL import ImageDraw
            mask_img = Image.new('L', (load_w, load_h), 0)
            draw = ImageDraw.Draw(mask_img)
            nose = tuple(pose_keypoints_scaled[0].astype(int))
            face_size = 95
            draw.ellipse([nose[0]-face_size, nose[1]-face_size*1.5, 
                         nose[0]+face_size, nose[1]+face_size*1.1], fill=255)
            mask = np.array(mask_img) > 128
            # Add face to channel 2 (face) - map from label 13 (Face) or 4 (Sunglasses)
            face_mask_tensor = torch.tensor(mask, dtype=torch.float32)
            new_parse_agnostic_map[2] = torch.clamp(new_parse_agnostic_map[2] + face_mask_tensor, 0, 1)
            # Remove from other channels where we added face
            new_parse_agnostic_map[0] = new_parse_agnostic_map[0] * (1 - face_mask_tensor * 0.8)  # Reduce background
            new_parse_agnostic_map[1] = new_parse_agnostic_map[1] * (1 - face_mask_tensor * 0.8)  # Reduce hair
            face_pixels_added = face_mask_tensor.sum().item()
            print(f"[INFO] Added face to channel 2: {face_pixels_added} pixels")
            print(f"[INFO] Face channel now: {new_parse_agnostic_map[2].sum().item()/total_pixels*100:.1f}%")
    
    total_pixels = load_h * load_w
    
    # CRITICAL FIX 1: Check if hair channel is too large (SCHP misclassification)
    # If hair > 50% of image, it's wrong - redistribute to proper body parts
    hair_channel_sum = new_parse_agnostic_map[1].sum().item()
    
    if hair_channel_sum > total_pixels * 0.5:  # More than 50% classified as hair
        print(f"[CRITICAL] Hair channel is too large: {hair_channel_sum/total_pixels*100:.1f}% - SCHP misclassified!")
        print(f"[CRITICAL] Redistributing hair pixels to proper body parts based on pose...")
        
        # Get hair mask
        hair_mask = new_parse_agnostic_map[1] > 0.5
        
        # Use pose keypoints to determine body regions
        neck_y = pose_keypoints_scaled[1, 1] if pose_keypoints_scaled[1, 1] > 0 else load_h * 0.15
        hip_y = (pose_keypoints_scaled[8, 1] + pose_keypoints_scaled[9, 1]) / 2 if (pose_keypoints_scaled[8, 1] > 0 and pose_keypoints_scaled[9, 1] > 0) else load_h * 0.6
        
        h_coords, w_coords = torch.meshgrid(
            torch.arange(load_h, dtype=torch.float32),
            torch.arange(load_w, dtype=torch.float32),
            indexing='ij'
        )
        
        # Redistribute hair pixels:
        # - Above neck: keep as hair (correct)
        # - Neck to hip: upper body (channel 3)
        # - Below hip: lower body (channel 4 - bottom/pants)
        
        # Upper body region (neck to hip) - redistribute to channel 3
        upper_from_hair = ((h_coords >= neck_y) & (h_coords < hip_y) & hair_mask).float()
        if upper_from_hair.sum() > 0:
            new_parse_agnostic_map[3] = torch.clamp(new_parse_agnostic_map[3] + upper_from_hair, 0, 1)
            new_parse_agnostic_map[1] = new_parse_agnostic_map[1] * (1 - upper_from_hair)
            print(f"[INFO] Redistributed {upper_from_hair.sum().item():.0f} hair pixels to upper body (channel 3)")
        
        # Lower body region (below hip) - redistribute to channel 4 (bottom/pants)
        lower_from_hair = ((h_coords >= hip_y) & hair_mask).float()
        if lower_from_hair.sum() > 0:
            new_parse_agnostic_map[4] = torch.clamp(new_parse_agnostic_map[4] + lower_from_hair, 0, 1)
            new_parse_agnostic_map[1] = new_parse_agnostic_map[1] * (1 - lower_from_hair)
            print(f"[INFO] Redistributed {lower_from_hair.sum().item():.0f} hair pixels to lower body (channel 4)")
        
        print(f"[INFO] Hair channel now: {new_parse_agnostic_map[1].sum().item()/total_pixels*100:.1f}%")
    
    # CRITICAL: Ensure channel 3 (upper) is properly populated for SEG model
    # SEG model generates parse_old from parse_agnostic, and parse_old[:, 3] → parse[:, 2] → GMM
    # If channel 3 is empty or too small, SEG won't generate it properly, and GMM will fail
    upper_channel_sum = new_parse_agnostic_map[3].sum().item()
    upper_percent = upper_channel_sum / total_pixels * 100
    
    print(f"[DEBUG] Channel 3 (upper) before final check: {upper_percent:.1f}%")
    
    if upper_percent < 20.0:  # Less than 20% - need to ensure proper coverage
        print(f"[CRITICAL] Channel 3 (upper) is {upper_percent:.1f}% - too small for SEG model!")
        print(f"[CRITICAL] SEG model needs proper channel 3 to generate parse_old[:, 3] for GMM")
        print(f"[CRITICAL] Ensuring channel 3 has at least 20% coverage...")
        
        # Use pose to create upper body region
        neck_y = pose_keypoints_scaled[1, 1] if pose_keypoints_scaled[1, 1] > 0 else load_h * 0.15
        hip_y = (pose_keypoints_scaled[8, 1] + pose_keypoints_scaled[9, 1]) / 2 if (pose_keypoints_scaled[8, 1] > 0 and pose_keypoints_scaled[9, 1] > 0) else load_h * 0.6
        
        h_coords, w_coords = torch.meshgrid(
            torch.arange(load_h, dtype=torch.float32),
            torch.arange(load_w, dtype=torch.float32),
            indexing='ij'
        )
        
        # Upper body region: between neck and hips
        if pose_keypoints_scaled[2, 0] > 0 and pose_keypoints_scaled[5, 0] > 0:
            shoulder_left = pose_keypoints_scaled[5, 0]
            shoulder_right = pose_keypoints_scaled[2, 0]
            center_x = (shoulder_left + shoulder_right) / 2
            width = abs(shoulder_right - shoulder_left) * 1.8
            upper_mask = ((h_coords >= neck_y) & (h_coords < hip_y) & 
                         (w_coords >= center_x - width/2) & (w_coords <= center_x + width/2)).float()
        else:
            upper_mask = ((h_coords >= neck_y) & (h_coords < hip_y)).float()
        
        # Don't overwrite face, but can add to background/sparse regions
        exclude_mask = (new_parse_agnostic_map[2] > 0.3)  # Face
        
        # Add to channel 3 where not excluded
        upper_mask = upper_mask * (~exclude_mask).float()
        new_parse_agnostic_map[3] = torch.clamp(new_parse_agnostic_map[3] + upper_mask, 0, 1)
        
        # Reduce background
        new_parse_agnostic_map[0] = new_parse_agnostic_map[0] * (1 - upper_mask * 0.7)
        
        final_upper_percent = new_parse_agnostic_map[3].sum().item() / total_pixels * 100
        print(f"[INFO] Channel 3 (upper) now: {final_upper_percent:.1f}% (target: ≥20%)")
    
    # Pose: use OpenPose-style rendered RGB image (already resized to target size)
    # pose_rgb_pil is already in target size from run_openpose_style
    if isinstance(pose_rgb_pil, Image.Image):
        # Ensure it's exactly the target size
        if pose_rgb_pil.size != (load_w, load_h):
            pose_rgb_pil = pose_rgb_pil.resize((load_w, load_h), Image.BICUBIC)
        pose = to_tensor_norm(pose_rgb_pil)
    else:
        # Fallback: create a blank pose image
        pose_img = Image.new("RGB", (load_w, load_h), color=(0, 0, 0))
        pose = to_tensor_norm(pose_img)

    # Cloth mask: ensure (1,H,W) float tensor in {0,1}
    cm_np = np.asarray(cloth_mask_np)
    cm_img = Image.fromarray((cm_np * 255.0).astype(np.uint8))
    cm_img = resize_mask_nearest(cm_img)
    cm_arr = (np.array(cm_img) >= 128).astype(np.float32)
    cloth_mask = torch.from_numpy(cm_arr).unsqueeze(0)  # (1,H,W)

    # Compose output sample exactly like dataset
    sample = {
        "img_name": "upload_person.jpg",
        "c_name": {"unpaired": "upload_cloth.jpg"},
        "img": person,
        "img_agnostic": img_agnostic,
        "parse_agnostic": new_parse_agnostic_map,
        "pose": pose,
        "cloth": {"unpaired": cloth},
        "cloth_mask": {"unpaired": cloth_mask},
    }
    return sample


# -------------------------------------------------------------------------
# Legacy overlay (kept for debugging; not used in main UI now)
# -------------------------------------------------------------------------
def generate_demo_overlay(person_file, cloth_file):
    canvas_w, canvas_h = DATASET_OPT.load_width, DATASET_OPT.load_height

    person_pil = Image.open(person_file).convert("RGB")
    cloth_pil = Image.open(cloth_file).convert("RGB")

    person_resized = ImageOps.fit(person_pil, (canvas_w, canvas_h), Image.BICUBIC)

    cloth_target_w = int(canvas_w * 0.6)
    cloth_ratio = cloth_pil.width / max(1, cloth_pil.height)
    cloth_target_h = int(cloth_target_w / max(0.6, cloth_ratio))
    cloth_resized = cloth_pil.resize((cloth_target_w, cloth_target_h), Image.LANCZOS)

    cloth_gray = cloth_resized.convert("L")
    mask = cloth_gray.point(lambda p: 255 if p > 20 else 0).convert("L")
    mask = mask.filter(ImageFilter.GaussianBlur(radius=3))

    x = (canvas_w - cloth_target_w) // 2
    y = int(canvas_h * 0.28)

    base = person_resized.convert("RGBA")
    cloth_rgba = cloth_resized.convert("RGBA")
    cloth_rgba.putalpha(mask)

    composite = base.copy()
    composite.paste(cloth_rgba, (x, y), cloth_rgba)
    result = Image.blend(base, composite, alpha=0.85).convert("RGB")

    result_path = os.path.join(RESULT_DIR, f"demo_{uuid.uuid4().hex}.jpg")
    result.save(result_path, quality=95)

    return {"result_url": to_url(result_path)}


# -------------------------------------------------------------------------
# ORIGINAL WORKING DATASET INFERENCE (unchanged)
# -------------------------------------------------------------------------
def run_dataset_inference(pair_index):
    if DATASET is None:
        raise RuntimeError("Preprocessed dataset is not available. Please check datasets/zalando-hd-resized.")
    if pair_index < 0 or pair_index >= len(DATASET):
        raise IndexError(f"Pair index {pair_index} is out of range (0 - {len(DATASET) - 1}).")

    sample = DATASET[pair_index]
    img_name, cloth_name = PAIR_LIST[pair_index]
    opt = DATASET_OPT

    img_agnostic = sample["img_agnostic"].unsqueeze(0).to(device)
    parse_agnostic = sample["parse_agnostic"].unsqueeze(0).to(device)
    pose = sample["pose"].unsqueeze(0).to(device)
    cloth = sample["cloth"]["unpaired"].unsqueeze(0).to(device)
    cloth_mask = sample["cloth_mask"]["unpaired"].unsqueeze(0).to(device)

    with torch.no_grad():
        parse_agnostic_down = F.interpolate(parse_agnostic, size=(256, 192), mode="bilinear", align_corners=False)
        pose_down = F.interpolate(pose, size=(256, 192), mode="bilinear", align_corners=False)
        c_masked_down = F.interpolate(cloth * cloth_mask, size=(256, 192), mode="bilinear", align_corners=False)
        cm_down = F.interpolate(cloth_mask, size=(256, 192), mode="bilinear", align_corners=False)

        noise = gen_noise(cm_down.size()).to(device, dtype=cm_down.dtype)

        seg_input = torch.cat((cm_down, c_masked_down, parse_agnostic_down, pose_down, noise), dim=1)
        parse_pred_down = SEG_MODEL(seg_input)

        parse_pred_up = F.interpolate(parse_pred_down, size=(opt.load_height, opt.load_width),
                                      mode="bilinear", align_corners=False)
        parse_pred_up = gaussian_blur_tensor(parse_pred_up)
        parse_pred = parse_pred_up.argmax(dim=1, keepdim=True)

        parse_old = torch.zeros(parse_pred.size(0), opt.semantic_nc, opt.load_height, opt.load_width, device=device)
        parse_old.scatter_(1, parse_pred, 1.0)

        label_map = {
            0: [0],
            1: [2, 4, 7, 8, 9, 10, 11],
            2: [3],
            3: [1],
            4: [5],
            5: [6],
            6: [12],
        }
        parse = torch.zeros(parse_pred.size(0), 7, opt.load_height, opt.load_width, device=device)
        for new_idx, old_ids in label_map.items():
            for old_id in old_ids:
                parse[:, new_idx] += parse_old[:, old_id]

        agnostic_gmm = F.interpolate(img_agnostic, size=(256, 192), mode="nearest")
        parse_cloth_gmm = F.interpolate(parse[:, 2:3], size=(256, 192), mode="nearest")
        pose_gmm = F.interpolate(pose, size=(256, 192), mode="nearest")
        c_gmm = F.interpolate(cloth, size=(256, 192), mode="nearest")
        gmm_input = torch.cat((parse_cloth_gmm, pose_gmm, agnostic_gmm), dim=1)

        _, warped_grid = GMM_MODEL(gmm_input, c_gmm)
        
        # CRITICAL: Check warped_grid validity - if invalid, cloth will be distorted
        grid_min = warped_grid.min().item()
        grid_max = warped_grid.max().item()
        grid_mean = warped_grid.mean().item()
        grid_std = warped_grid.std().item()
        
        print(f"[DEBUG] Warped grid stats: min={grid_min:.3f}, max={grid_max:.3f}, mean={grid_mean:.3f}, std={grid_std:.3f}")
        
        # Grid should be in range [-1, 1] for grid_sample - normalize instead of clamp
        if grid_min < -1.0 or grid_max > 1.0:
            print(f"[CRITICAL] Warped grid out of valid range [-1, 1]! Normalizing to prevent cloth distortion...")
            # Normalize to [-1, 1] range while preserving relative transformations
            grid_range = grid_max - grid_min
            if grid_range > 0:
                # Scale to fit in [-1, 1] while maintaining center
                warped_grid = (warped_grid - grid_mean) / (grid_range / 2.0) * 0.95  # 0.95 for safety margin
                # Ensure it's within bounds
                warped_grid = torch.clamp(warped_grid, -1.0, 1.0)
            else:
                # Fallback: just clamp if range is invalid
                warped_grid = torch.clamp(warped_grid, -1.0, 1.0)
            print(f"[INFO] Normalized warped grid to [{warped_grid.min().item():.3f}, {warped_grid.max().item():.3f}]")
        
        warped_c = F.grid_sample(cloth, warped_grid, padding_mode="border", align_corners=False)
        warped_cm = F.grid_sample(cloth_mask, warped_grid, padding_mode="border", align_corners=False)
        
        # Debug: Check warped cloth
        warped_c_sum = warped_c.sum().item()
        print(f"[DEBUG] Warped cloth sum: {warped_c_sum:.0f} (should be > 0)")

        misalign_mask = torch.clamp(parse[:, 2:3] - warped_cm, min=0.0)
        parse_div = torch.cat((parse, misalign_mask), dim=1)
        parse_div[:, 2:3] -= misalign_mask

        alias_input = torch.cat((img_agnostic, pose, warped_c), dim=1)
        output = ALIAS_MODEL(alias_input, parse, parse_div, misalign_mask)

    identifier = uuid.uuid4().hex
    result_path = os.path.join(RESULT_DIR, f"tryon_{identifier}.jpg")
    person_path = os.path.join(RESULT_DIR, f"person_{identifier}.jpg")
    cloth_path = os.path.join(RESULT_DIR, f"cloth_{identifier}.jpg")
    warped_path = os.path.join(RESULT_DIR, f"warped_{identifier}.jpg")

    save_tensor_image(output, result_path)
    save_tensor_image(sample["img"], person_path)
    save_tensor_image(sample["cloth"]["unpaired"], cloth_path)
    save_tensor_image(warped_c, warped_path)

    return {
        "pair_index": pair_index,
        "img_name": img_name,
        "cloth_name": cloth_name,
        "result_url": to_url(result_path),
        "person_url": to_url(person_path),
        "cloth_url": to_url(cloth_path),
        "warped_url": to_url(warped_path),
    }


# -------------------------------------------------------------------------
# Generic inference for a single sample dict (same pipeline)
# -------------------------------------------------------------------------
def run_sample_inference(sample, img_name="upload_person.jpg",
                         cloth_name="upload_cloth.jpg",
                         tag_prefix="custom",
                         use_seg=True):
    """
    Run Seg → GMM → ALIAS on a sample.
    Always use use_seg=True to use SegGenerator for proper clothing overlay.
    The SegGenerator refines the parse_agnostic map for better results.
    """
    opt = DATASET_OPT

    img_agnostic = sample["img_agnostic"].unsqueeze(0).to(device)
    parse_agnostic = sample["parse_agnostic"].unsqueeze(0).to(device)  # (1,13,H,W)
    pose = sample["pose"].unsqueeze(0).to(device)
    cloth = sample["cloth"]["unpaired"].unsqueeze(0).to(device)
    cloth_mask = sample["cloth_mask"]["unpaired"].unsqueeze(0).to(device)

    with torch.no_grad():

        if use_seg:
            # ---------------- SEG STAGE (original) ----------------
            parse_agnostic_down = F.interpolate(parse_agnostic, size=(256, 192),
                                                mode="bilinear", align_corners=False)
            pose_down = F.interpolate(pose, size=(256, 192),
                                      mode="bilinear", align_corners=False)
            c_masked_down = F.interpolate(cloth * cloth_mask, size=(256, 192),
                                          mode="bilinear", align_corners=False)
            cm_down = F.interpolate(cloth_mask, size=(256, 192),
                                    mode="bilinear", align_corners=False)

            noise = gen_noise(cm_down.size()).to(device, dtype=cm_down.dtype)

            seg_input = torch.cat(
                (cm_down, c_masked_down, parse_agnostic_down, pose_down, noise), dim=1
            )
            parse_pred_down = SEG_MODEL(seg_input)

            parse_pred_up = F.interpolate(
                parse_pred_down,
                size=(opt.load_height, opt.load_width),
                mode="bilinear",
                align_corners=False,
            )
            parse_pred_up = gaussian_blur_tensor(parse_pred_up)
            parse_pred = parse_pred_up.argmax(dim=1, keepdim=True)

            # parse_old: 13-channel semantic map
            parse_old = torch.zeros(
                parse_pred.size(0),
                opt.semantic_nc,
                opt.load_height,
                opt.load_width,
                device=device,
            )
            parse_old.scatter_(1, parse_pred, 1.0)

        else:
            # ---------------- NO-SEG MODE (use SCHP parse directly) ----------------
            # parse_agnostic already has shape (1,13,H,W), one-hot style.
            parse_old = parse_agnostic  # treat SCHP-derived map as the "old" 13 classes

        # ---- same label_map as before: 13 → 7 channels ----
        label_map = {
            0: [0],
            1: [2, 4, 7, 8, 9, 10, 11],
            2: [3],  # parse[:, 2] comes from parse_old[:, 3] (upper channel)
            3: [1],
            4: [5],
            5: [6],
            6: [12],
        }
        parse = torch.zeros(
            parse_old.size(0), 7, opt.load_height, opt.load_width, device=device
        )
        for new_idx, old_ids in label_map.items():
            for old_id in old_ids:
                parse[:, new_idx] += parse_old[:, old_id]
        
        # CRITICAL: Check parse[:, 2:3] (upper clothing for GMM) - if too small, fix it
        parse_cloth_sum = parse[:, 2:3].sum().item()
        total_pixels_parse = parse.size(2) * parse.size(3)
        parse_cloth_percent = parse_cloth_sum / total_pixels_parse * 100
        
        print(f"[DEBUG] parse[:, 2:3] (for GMM) = {parse_cloth_percent:.1f}% ({parse_cloth_sum:.0f} pixels)")
        print(f"[DEBUG] parse_old[:, 3] (upper) = {parse_old[:, 3].sum().item() / total_pixels_parse * 100:.1f}%")
        print(f"[DEBUG] parse_agnostic[:, 3] (upper) = {parse_agnostic[:, 3].sum().item() / total_pixels_parse * 100:.1f}%")
        
        if parse_cloth_percent < 20.0:  # Less than 20% - GMM will fail
            print(f"[CRITICAL] parse[:, 2:3] is {parse_cloth_percent:.1f}% - too small for GMM warping!")
            print(f"[CRITICAL] SEG model didn't generate parse_old[:, 3] properly. Using parse_agnostic[:, 3] directly...")
            
            # CRITICAL FIX: Use parse_agnostic[:, 3] directly to populate parse[:, 2]
            # This ensures GMM has proper upper body region for warping
            parse_agnostic_upper = parse_agnostic[:, 3:4].to(device)  # (1, 1, H, W)
            parse[:, 2:3] = torch.clamp(parse[:, 2:3] + parse_agnostic_upper, 0, 1)
            
            final_parse_cloth_percent = parse[:, 2:3].sum().item() / total_pixels_parse * 100
            print(f"[INFO] Fixed parse[:, 2:3] = {final_parse_cloth_percent:.1f}% (added from parse_agnostic[:, 3])")

        # ---------------- GMM STAGE ----------------
        # CRITICAL: Check parse[:, 2:3] before GMM - if empty, warping will fail
        parse_cloth_sum = parse[:, 2:3].sum().item()
        total_pixels_gmm = parse.size(2) * parse.size(3)
        parse_cloth_percent = parse_cloth_sum / total_pixels_gmm * 100
        
        if parse_cloth_percent < 10.0:  # Less than 10% - GMM will fail
            print(f"[CRITICAL] parse[:, 2:3] is {parse_cloth_percent:.1f}% - too small for GMM warping!")
            print(f"[CRITICAL] GMM warping will fail. Checking parse_old[:, 3]...")
            parse_old_upper_sum = parse_old[:, 3].sum().item() if parse_old.size(1) > 3 else 0
            parse_old_upper_percent = parse_old_upper_sum / total_pixels_gmm * 100
            print(f"[DEBUG] parse_old[:, 3] (upper) = {parse_old_upper_percent:.1f}%")
            print(f"[DEBUG] parse_agnostic[:, 3] (upper) = {parse_agnostic[:, 3].sum().item() / total_pixels_gmm * 100:.1f}%")
        
        img_agnostic_gmm = F.interpolate(img_agnostic, size=(256, 192), mode="nearest")
        parse_cloth_gmm = F.interpolate(parse[:, 2:3], size=(256, 192), mode="nearest")
        pose_gmm = F.interpolate(pose, size=(256, 192), mode="nearest")
        c_gmm = F.interpolate(cloth, size=(256, 192), mode="nearest")
        
        # Debug: Check GMM inputs
        parse_cloth_gmm_sum = parse_cloth_gmm.sum().item()
        parse_cloth_gmm_percent = parse_cloth_gmm_sum / (256 * 192) * 100
        print(f"[DEBUG] GMM inputs - parse_cloth: {parse_cloth_gmm_sum:.0f} pixels ({parse_cloth_gmm_percent:.1f}%), pose: {pose_gmm.shape}, cloth: {c_gmm.shape}")
        
        # CRITICAL: Ensure parse_cloth_gmm has enough coverage for GMM
        if parse_cloth_gmm_percent < 20.0:
            print(f"[CRITICAL] parse_cloth_gmm is {parse_cloth_gmm_percent:.1f}% - too small for GMM!")
            print(f"[CRITICAL] Using parse_agnostic[:, 3] directly at downsampled resolution...")
            parse_agnostic_upper_down = F.interpolate(parse_agnostic[:, 3:4].to(device), size=(256, 192), mode="nearest")
            parse_cloth_gmm = torch.clamp(parse_cloth_gmm + parse_agnostic_upper_down, 0, 1)
            print(f"[INFO] Fixed parse_cloth_gmm = {parse_cloth_gmm.sum().item() / (256 * 192) * 100:.1f}%")
        
        gmm_input = torch.cat((parse_cloth_gmm, pose_gmm, img_agnostic_gmm), dim=1)

        _, warped_grid = GMM_MODEL(gmm_input, c_gmm)
        
        # CRITICAL: Check and normalize warped_grid - MUST be in [-1, 1] for grid_sample
        grid_min = warped_grid.min().item()
        grid_max = warped_grid.max().item()
        grid_mean = warped_grid.mean().item()
        print(f"[DEBUG] Warped grid range: [{grid_min:.3f}, {grid_max:.3f}], mean: {grid_mean:.3f} (should be ~[-1, 1])")
        
        # Grid MUST be in range [-1, 1] for grid_sample - normalize instead of clamp to preserve transformations
        if grid_min < -1.0 or grid_max > 1.0:
            print(f"[CRITICAL] Warped grid out of valid range [-1, 1]! Normalizing to prevent cloth distortion...")
            # Normalize to [-1, 1] range while preserving relative transformations
            grid_range = grid_max - grid_min
            if grid_range > 0:
                # Scale to fit in [-1, 1] while maintaining center
                warped_grid = (warped_grid - grid_mean) / (grid_range / 2.0) * 0.95  # 0.95 for safety margin
                # Ensure it's within bounds
                warped_grid = torch.clamp(warped_grid, -1.0, 1.0)
            else:
                # Fallback: just clamp if range is invalid
                warped_grid = torch.clamp(warped_grid, -1.0, 1.0)
            print(f"[INFO] Normalized warped grid to [{warped_grid.min().item():.3f}, {warped_grid.max().item():.3f}]")
        
        warped_c = F.grid_sample(cloth, warped_grid, padding_mode="border", align_corners=False)
        warped_cm = F.grid_sample(cloth_mask, warped_grid, padding_mode="border", align_corners=False)

        # ---------------- ALIAS (try-on) STAGE ----------------
        misalign_mask = torch.clamp(parse[:, 2:3] - warped_cm, min=0.0)
        parse_div = torch.cat((parse, misalign_mask), dim=1)
        parse_div[:, 2:3] -= misalign_mask

        alias_input = torch.cat((img_agnostic, pose, warped_c), dim=1)
        output = ALIAS_MODEL(alias_input, parse, parse_div, misalign_mask)

    identifier = uuid.uuid4().hex
    result_path = os.path.join(RESULT_DIR, f"{tag_prefix}_tryon_{identifier}.jpg")
    person_path = os.path.join(RESULT_DIR, f"{tag_prefix}_person_{identifier}.jpg")
    cloth_path = os.path.join(RESULT_DIR, f"{tag_prefix}_cloth_{identifier}.jpg")
    warped_path = os.path.join(RESULT_DIR, f"{tag_prefix}_warped_{identifier}.jpg")

    save_tensor_image(output, result_path)
    save_tensor_image(sample["img"], person_path)
    save_tensor_image(sample["cloth"]["unpaired"], cloth_path)
    save_tensor_image(warped_c, warped_path)

    return {
        "pair_index": -1,
        "img_name": img_name,
        "cloth_name": cloth_name,
        "result_url": to_url(result_path),
        "person_url": to_url(person_path),
        "cloth_url": to_url(cloth_path),
        "warped_url": to_url(warped_path),
    }


# -------------------------------------------------------------------------
# FULL custom upload pipeline using SCHP + pose + cloth mask
# -------------------------------------------------------------------------
def run_custom_tryon_from_uploads(person_file, cloth_file):
    """
    Full custom-upload pipeline:
      1) save uploads to disk
      2) run SCHP parsing, pose estimation, cloth preprocessing
      3) build VITON-style sample with preprocess_custom_viton
      4) run Seg → GMM → ALIAS via run_sample_inference
    """
    uid = uuid.uuid4().hex

    # Save uploads
    person_filename = secure_filename(person_file.filename or f"person_{uid}.jpg")
    cloth_filename = secure_filename(cloth_file.filename or f"cloth_{uid}.jpg")

    person_path = os.path.join(UPLOAD_DIR, f"person_{uid}.jpg")
    cloth_path = os.path.join(UPLOAD_DIR, f"cloth_{uid}.jpg")
    person_file.save(person_path)
    cloth_file.save(cloth_path)

    # Load images as PIL
    person_pil = Image.open(person_path).convert("RGB")
    cloth_pil = Image.open(cloth_path).convert("RGB")
    
    # Verify image is suitable for SCHP
    person_w, person_h = person_pil.size
    print(f"[DEBUG] Person image size: {person_w}x{person_h}")
    if person_w < 256 or person_h < 256:
        print(f"[WARNING] Person image is very small ({person_w}x{person_h}) - SCHP may not work well!")
    if person_w > 2048 or person_h > 2048:
        print(f"[WARNING] Person image is very large ({person_w}x{person_h}) - may cause memory issues")

    # 1) Human parsing (SCHP) -> (prob_map, argmax_mask)
    # Try ATR checkpoint first (better generalization), fallback to LIP
    atr_checkpoint = "checkpoints/humanparsing/exp-schp-201908301523-atr.pth"
    lip_checkpoint = "checkpoints/humanparsing/exp-schp-201908261155-lip.pth"
    
    if os.path.exists(atr_checkpoint):
        print(f"[DEBUG] Using ATR checkpoint for better results")
        checkpoint_to_use = atr_checkpoint
    elif os.path.exists(lip_checkpoint):
        print(f"[DEBUG] ATR checkpoint not found, using LIP checkpoint")
        checkpoint_to_use = lip_checkpoint
    else:
        raise FileNotFoundError(
            f"Neither ATR nor LIP checkpoint found. "
            f"Please download exp-schp-201908301523-atr.pth or exp-schp-201908261155-lip.pth "
            f"to checkpoints/humanparsing/ folder"
        )
    
    print(f"[DEBUG] Running SCHP parsing on {person_path} with checkpoint: {checkpoint_to_use}")
    prob_map, argmax_mask = run_schp_parsing(person_path, checkpoint=checkpoint_to_use)
    parse_array = argmax_mask  # (H,W) uint8
    
    # Check SCHP probabilities - if argmax shows only one label, investigate the probability map
    if isinstance(prob_map, torch.Tensor):
        prob_map_np = prob_map.squeeze().cpu().numpy()  # (num_classes, H, W) or (1, num_classes, H, W)
        if prob_map_np.ndim == 3 and prob_map_np.shape[0] == 1:
            prob_map_np = prob_map_np[0]  # Remove batch dim: (num_classes, H, W)
        
        if prob_map_np.ndim == 3:
            # Check which classes have high probability (not just argmax)
            # Get mean probability per class across all pixels
            mean_probs = prob_map_np.mean(axis=(1, 2))  # (num_classes,) - mean prob per class
            top5_classes = np.argsort(mean_probs)[-5:][::-1]  # Top 5 classes by mean prob
            print(f"[DEBUG] SCHP probability map shape: {prob_map_np.shape}")
            print(f"[DEBUG] Top 5 classes by mean probability:")
            for i, cls in enumerate(top5_classes):
                print(f"[DEBUG]   Class {cls}: {mean_probs[cls]:.4f} mean prob")
            
            # Get actual number of classes from model output
            num_classes = prob_map_np.shape[0]
            print(f"[DEBUG] Model output has {num_classes} classes (ATR=18, LIP=20)")
            
            # Check if argmax is picking wrong class - look at pixels where face (13) has high prob
            # Face label exists in both ATR (label 11) and LIP (label 13)
            face_label = 11 if num_classes == 18 else 13  # ATR uses 11, LIP uses 13
            if face_label < num_classes:
                face_prob = prob_map_np[face_label, :, :]  # (H, W) - probability of face per pixel
                high_face_pixels = (face_prob > 0.5).sum()
                print(f"[DEBUG] Pixels with face prob > 0.5: {high_face_pixels} / {face_prob.size}")
            
            # Check what classes are actually present in probability map
            # Look at pixels where each class has > 0.3 probability
            active_classes = []
            for cls in range(num_classes):  # Use actual num_classes instead of hardcoded 20
                cls_prob = prob_map_np[cls, :, :]
                pixels_above_thresh = (cls_prob > 0.3).sum()
                if pixels_above_thresh > 100:  # More than 100 pixels
                    active_classes.append((cls, pixels_above_thresh, cls_prob.mean()))
            
            if active_classes:
                print(f"[DEBUG] Classes with >100 pixels above 0.3 probability threshold:")
                for cls, count, mean_prob in sorted(active_classes, key=lambda x: x[1], reverse=True):
                    print(f"[DEBUG]   Class {cls}: {count} pixels, {mean_prob:.4f} mean prob")
            
            # Get top-3 classes per pixel
            top3_indices = np.argsort(prob_map_np, axis=0)[-3:, :, :]  # (3, H, W)
            unique_top3 = np.unique(top3_indices)
            print(f"[DEBUG] Unique classes in top-3 predictions per pixel: {unique_top3.shape[0]} classes")
            print(f"[DEBUG] Top-3 class indices: {unique_top3.tolist()}")
    
    print(f"[DEBUG] SCHP parse array shape: {parse_array.shape}, unique values: {np.unique(parse_array)}")
    print(f"[DEBUG] SCHP value counts: {[(val, np.sum(parse_array == val)) for val in np.unique(parse_array)]}")
    
    # CRITICAL WARNING if SCHP failed badly
    unique_labels = np.unique(parse_array)
    if len(unique_labels) <= 2 or (len(unique_labels) == 1 and unique_labels[0] not in [0, 13]):
        print(f"[CRITICAL ERROR] SCHP FAILED - Only detected {unique_labels}. This WILL cause melted/distorted output!")
        print(f"[CRITICAL ERROR] The entire image is classified as: {unique_labels}")
        print(f"[CRITICAL ERROR] Without body structure (torso, arms, legs), the try-on model cannot work properly!")
        print(f"[CRITICAL ERROR] Check the SCHP visualization at: {os.path.join(RESULT_DIR, f'debug_schp_raw_{uid}.png')}")
        print(f"[CRITICAL ERROR] Possible causes: SCHP model broken, wrong checkpoint, bad input image quality")
    
    # Save raw SCHP output for debugging
    schp_debug_path = os.path.join(RESULT_DIR, f"debug_schp_raw_{uid}.png")
    # Create a color visualization of the parse (each label gets a different color)
    unique_labels = np.unique(parse_array)
    schp_vis = np.zeros((parse_array.shape[0], parse_array.shape[1], 3), dtype=np.uint8)
    # Simple color mapping: each label gets a distinct color
    # Using HSV-like colors for better distinction
    for label in unique_labels:
        mask = parse_array == label
        # Generate distinct colors for labels 0-19
        hue = (label * 180 // 20) % 180
        # Convert to RGB (simplified HSV to RGB)
        if label == 0:
            color = [0, 0, 0]  # Background = black
        else:
            # Create distinct colors
            r = ((label * 7) % 255)
            g = ((label * 11) % 255)
            b = ((label * 13) % 255)
            color = [r, g, b]
        schp_vis[mask] = color
    schp_vis_img = Image.fromarray(schp_vis)
    schp_vis_img.save(schp_debug_path)
    print(f"[DEBUG] Saved raw SCHP visualization: {schp_debug_path}")
    
    # Check if SCHP is working - if only one label, it's likely broken
    if len(unique_labels) <= 2:
        print(f"[WARNING] SCHP only detected {len(unique_labels)} label(s): {unique_labels}. This suggests SCHP may not be working correctly!")

    # 2) Pose estimation (OpenPose-style) -> (pose_rgb_pil, pose_keypoints)
    print(f"[DEBUG] Running OpenPose-style pose estimation on {person_path}")
    pose_rgb_pil, pose_keypoints = run_openpose_style(person_path, target_size=(768, 1024))
    # pose_rgb_pil: PIL Image (RGB) with OpenPose skeleton, resized to target
    # pose_keypoints: (18, 2) numpy array with coordinates in ORIGINAL image coords
    detected_keypoints = np.sum((pose_keypoints[:, 0] > 0) | (pose_keypoints[:, 1] > 0))
    print(f"[DEBUG] Detected {detected_keypoints}/18 pose keypoints")
    print(f"[DEBUG] Keypoint indices with values: {np.where((pose_keypoints[:, 0] > 0) | (pose_keypoints[:, 1] > 0))[0].tolist()}")

    # 3) Cloth preprocessing -> (cloth_tensor, mask_tensor, edge_tensor)
    print(f"[DEBUG] Running cloth preprocessing on {cloth_path}")
    _, mask_tensor, _ = run_cloth_preprocess(cloth_path)
    # mask_tensor: (1,1,H,W) float
    cloth_mask_np = mask_tensor.squeeze().cpu().numpy()  # (H,W)
    print(f"[DEBUG] Cloth mask shape: {cloth_mask_np.shape}, coverage: {np.sum(cloth_mask_np > 0.5) / cloth_mask_np.size * 100:.1f}%")

    # 4) Build sample using proper dataset methods
    print(f"[DEBUG] Building VITON sample...")
    sample = preprocess_custom_viton(person_pil, cloth_pil, parse_array, pose_rgb_pil, pose_keypoints, cloth_mask_np)
    print(f"[DEBUG] Sample built. Parse agnostic shape: {sample['parse_agnostic'].shape}")
    # parse_agnostic is (13, H, W) - sum over spatial dimensions (1, 2)
    parse_sums = sample['parse_agnostic'].sum(dim=(1, 2)).tolist()
    print(f"[DEBUG] Parse agnostic channel sums: {parse_sums}")
    print(f"[DEBUG] Img agnostic range: [{sample['img_agnostic'].min():.3f}, {sample['img_agnostic'].max():.3f}]")
    
    # Save debug images
    debug_id = uuid.uuid4().hex[:8]
    parse_debug_path = os.path.join(RESULT_DIR, f"debug_parse_{debug_id}.png")
    agnostic_debug_path = os.path.join(RESULT_DIR, f"debug_agnostic_{debug_id}.jpg")
    # parse_agnostic is (13, H, W), argmax over channel dim 0
    parse_vis = sample['parse_agnostic'].argmax(dim=0).cpu().numpy()
    parse_vis_img = Image.fromarray((parse_vis * 255 / (parse_vis.max() + 1e-8)).astype(np.uint8), mode='L')
    parse_vis_img.save(parse_debug_path)
    save_tensor_image(sample['img_agnostic'], agnostic_debug_path)
    print(f"[DEBUG] Saved debug images: {parse_debug_path}, {agnostic_debug_path}")
    
    # 5) Always use SegGenerator (use_seg=True) for proper clothing overlay
    result = run_sample_inference(sample,
                                  img_name=person_filename,
                                  cloth_name=cloth_filename,
                                  tag_prefix="custom",
                                  use_seg=True)

    return result


# -------------------------------------------------------------------------
# HTML templates
# -------------------------------------------------------------------------
HTML_HOME = """
<!doctype html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>VITON-HD Virtual Dressing Room</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            display: flex;
            align-items: center;
            justify-content: center;
            overflow: hidden;
        }
        .container {
            text-align: center;
            color: white;
            animation: fadeIn 1s ease-in;
        }
        h1 {
            font-size: 4rem;
            margin-bottom: 1rem;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
            animation: slideDown 0.8s ease-out;
        }
        p {
            font-size: 1.3rem;
            margin-bottom: 2rem;
            opacity: 0.95;
            animation: slideUp 0.8s ease-out 0.2s both;
        }
        .btn {
            display: inline-block;
            padding: 18px 45px;
            font-size: 1.2rem;
            background: rgba(255,255,255,0.2);
            color: white;
            text-decoration: none;
            border-radius: 50px;
            border: 2px solid rgba(255,255,255,0.3);
            transition: all 0.3s ease;
            backdrop-filter: blur(10px);
            animation: slideUp 0.8s ease-out 0.4s both;
        }
        .btn:hover {
            background: rgba(255,255,255,0.3);
            transform: translateY(-3px);
            box-shadow: 0 10px 25px rgba(0,0,0,0.2);
        }
        @keyframes fadeIn {
            from { opacity: 0; }
            to { opacity: 1; }
        }
        @keyframes slideDown {
            from { transform: translateY(-30px); opacity: 0; }
            to { transform: translateY(0); opacity: 1; }
        }
        @keyframes slideUp {
            from { transform: translateY(30px); opacity: 0; }
            to { transform: translateY(0); opacity: 1; }
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>✨ Virtual Dressing Room</h1>
        <p>Experience AI-Powered Virtual Try-On</p>
        <a href="/predict" class="btn">Enter Dressing Room</a>
    </div>
</body>
</html>
"""

HTML_PREDICT = """
<!doctype html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Select Outfit - Virtual Dressing Room</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            display: flex;
            align-items: center;
            justify-content: center;
            padding: 20px;
            overflow-x: hidden;
        }
        .main-container {
            width: 100%;
            max-width: 1200px;
            background: rgba(255, 255, 255, 0.95);
            border-radius: 30px;
            padding: 50px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            backdrop-filter: blur(10px);
            animation: slideUp 0.6s ease-out;
        }
        .header {
            text-align: center;
            margin-bottom: 40px;
        }
        .header h1 {
            font-size: 3rem;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            margin-bottom: 10px;
        }
        .header p {
            color: #666;
            font-size: 1.1rem;
        }
        .stats {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 15px;
            text-align: center;
            margin-bottom: 30px;
            font-size: 1.2rem;
            font-weight: 600;
        }
        .form-container {
            background: #f8f9fa;
            padding: 40px;
            border-radius: 20px;
            box-shadow: inset 0 2px 10px rgba(0,0,0,0.05);
        }
        .form-group {
            margin-bottom: 25px;
        }
        label {
            display: block;
            font-weight: 600;
            color: #333;
            margin-bottom: 12px;
            font-size: 1.1rem;
        }
        select, input[type="number"] {
            width: 100%;
            padding: 15px 20px;
            font-size: 1rem;
            border: 2px solid #e0e0e0;
            border-radius: 12px;
            background: white;
            transition: all 0.3s ease;
            font-family: inherit;
        }
        select:focus, input[type="number"]:focus {
            outline: none;
            border-color: #667eea;
            box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
        }
        select {
            cursor: pointer;
            appearance: none;
            background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='12' height='12' viewBox='0 0 12 12'%3E%3Cpath fill='%23667eea' d='M6 9L1 4h10z'/%3E%3C/svg%3E");
            background-repeat: no-repeat;
            background-position: right 20px center;
            padding-right: 50px;
        }
        .btn-submit {
            width: 100%;
            padding: 18px;
            font-size: 1.2rem;
            font-weight: 600;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            border-radius: 12px;
            cursor: pointer;
            transition: all 0.3s ease;
            margin-top: 10px;
            box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
        }
        .btn-submit:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(102, 126, 234, 0.5);
        }
        .btn-submit:active {
            transform: translateY(0);
        }
        .error-message {
            background: #fee;
            color: #c33;
            padding: 20px;
            border-radius: 12px;
            text-align: center;
            border: 2px solid #fcc;
        }
        .loading {
            display: none;
            text-align: center;
            padding: 20px;
        }
        .loading.active {
            display: block;
        }
        .spinner {
            border: 4px solid rgba(102, 126, 234, 0.2);
            border-top: 4px solid #667eea;
            border-radius: 50%;
            width: 40px;
            height: 40px;
            animation: spin 1s linear infinite;
            margin: 0 auto 15px;
        }
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        @keyframes slideUp {
            from {
                transform: translateY(30px);
                opacity: 0;
            }
            to {
                transform: translateY(0);
                opacity: 1;
            }
        }
        .divider {
            height: 1px;
            background: linear-gradient(90deg, transparent, #ddd, transparent);
            margin: 25px 0;
        }
    </style>
</head>
<body>
    <div class="main-container">
        <div class="header">
            <h1>✨ Select Your Outfit</h1>
            <p>Choose from our curated collection</p>
        </div>
        
        {% if total_pairs == 0 %}
        <div class="error-message">
            <strong>Dataset not found.</strong><br>
            Please place the preprocessed dataset under <code>datasets/zalando-hd-resized/</code>.
        </div>
        {% else %}
        <div class="stats">
            📦 Total Available Pairs: {{ total_pairs }}
        </div>
        
        <form method="POST" action="/predict" id="tryonForm" class="form-container">
            <input type="hidden" name="action" value="dataset">
            
            <div class="form-group">
                <label for="pair_index">🎯 Quick Select (First {{ preview_pairs|length }} pairs)</label>
                <select name="pair_index" id="pair_index">
                    <option value="">-- Select from list --</option>
                    {% for idx, pair in preview_pairs %}
                    <option value="{{ idx }}">{{ idx }} — {{ pair[0] }} ➜ {{ pair[1] }}</option>
                    {% endfor %}
                </select>
            </div>
            
            <div class="divider"></div>
            
            <div class="form-group">
                <label for="pair_index_manual">🔢 Or Enter Specific Index (0 – {{ total_pairs_minus_one }})</label>
                <input type="number" name="pair_index_manual" id="pair_index_manual" 
                       min="0" max="{{ total_pairs_minus_one }}" placeholder="Enter index (0-{{ total_pairs_minus_one }})">
            </div>
            
            <button type="submit" class="btn-submit">🚀 Generate Try-On</button>
            
            <div class="loading" id="loading">
                <div class="spinner"></div>
                <p>Generating your virtual try-on... Please wait</p>
            </div>
        </form>
        {% endif %}
    </div>
    
    <script>
        document.getElementById('tryonForm').addEventListener('submit', function(e) {
            // Validate that at least one field is filled
            var selectValue = document.getElementById('pair_index').value;
            var manualValue = document.getElementById('pair_index_manual').value;
            
            if (!selectValue && !manualValue) {
                e.preventDefault();
                alert('Please select from the dropdown OR enter an index number.');
                return false;
            }
            
            // Clear the other field before submission to avoid confusion
            if (manualValue) {
                document.getElementById('pair_index').value = '';
            } else if (selectValue) {
                document.getElementById('pair_index_manual').value = '';
            }
            
            document.getElementById('loading').classList.add('active');
            document.querySelector('.btn-submit').disabled = true;
        });
        
        // Clear the other field when one is used
        document.getElementById('pair_index_manual').addEventListener('input', function(e) {
            if (e.target.value !== '') {
                document.getElementById('pair_index').value = '';
            }
        });
        
        document.getElementById('pair_index').addEventListener('change', function(e) {
            if (e.target.value !== '') {
                document.getElementById('pair_index_manual').value = '';
            }
        });
    </script>
</body>
</html>
"""

HTML_RESULT = """
<!doctype html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Try-On Result - Virtual Dressing Room</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 30px 20px;
            overflow-x: hidden;
        }
        .container {
            max-width: 1400px;
            margin: 0 auto;
            background: rgba(255, 255, 255, 0.98);
            border-radius: 30px;
            padding: 50px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            animation: fadeIn 0.6s ease-out;
        }
        .header {
            text-align: center;
            margin-bottom: 40px;
            padding-bottom: 30px;
            border-bottom: 2px solid #f0f0f0;
        }
        .header h1 {
            font-size: 2.5rem;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            margin-bottom: 15px;
        }
        .header .info {
            color: #666;
            font-size: 1.1rem;
        }
        .result-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 30px;
            margin-bottom: 40px;
        }
        .result-card {
            background: #f8f9fa;
            border-radius: 20px;
            padding: 25px;
            text-align: center;
            transition: all 0.3s ease;
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        }
        .result-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 8px 25px rgba(0,0,0,0.15);
        }
        .result-card.featured {
            grid-column: 1 / -1;
            background: linear-gradient(135deg, #667eea15 0%, #764ba215 100%);
            border: 2px solid #667eea;
        }
        .result-card h3 {
            color: #333;
            margin-bottom: 20px;
            font-size: 1.3rem;
        }
        .result-card.featured h3 {
            color: #667eea;
            font-size: 1.5rem;
        }
        .result-card img {
            width: 100%;
            max-width: 100%;
            height: auto;
            border-radius: 15px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.2);
            transition: transform 0.3s ease;
        }
        .result-card:hover img {
            transform: scale(1.02);
        }
        .btn-container {
            text-align: center;
            margin-top: 40px;
        }
        .btn {
            display: inline-block;
            padding: 18px 45px;
            font-size: 1.1rem;
            font-weight: 600;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            text-decoration: none;
            border-radius: 50px;
            transition: all 0.3s ease;
            box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
        }
        .btn:hover {
            transform: translateY(-3px);
            box-shadow: 0 6px 20px rgba(102, 126, 234, 0.5);
        }
        @keyframes fadeIn {
            from {
                opacity: 0;
                transform: translateY(20px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }
        @media (max-width: 768px) {
            .result-grid {
                grid-template-columns: 1fr;
            }
            .result-card.featured {
                grid-column: 1;
            }
            .container {
                padding: 30px 20px;
            }
            .header h1 {
                font-size: 2rem;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>✨ Try-On Result</h1>
            <div class="info">
                {% if pair_index >= 0 %}
                Pair #{{ pair_index }}: <strong>{{ img_name }}</strong> ➜ <strong>{{ cloth_name }}</strong>
                {% else %}
                Custom: <strong>{{ img_name }}</strong> ➜ <strong>{{ cloth_name }}</strong>
                {% endif %}
            </div>
        </div>
        
        <div class="result-grid">
            <div class="result-card featured">
                <h3>🎯 Generated Try-On</h3>
                <img src="{{ result_url }}" alt="Generated Try-On Result">
            </div>
            
            <div class="result-card">
                <h3>👤 Original Person</h3>
                <img src="{{ person_url }}" alt="Original Person">
            </div>
            
            <div class="result-card">
                <h3>👕 Original Cloth</h3>
                <img src="{{ cloth_url }}" alt="Original Cloth">
            </div>
            
            <div class="result-card">
                <h3>🔄 Warped Cloth</h3>
                <img src="{{ warped_url }}" alt="Warped Cloth">
            </div>
        </div>
        
        <div class="btn-container">
            <a href="/predict" class="btn">🔄 Try Another Outfit</a>
        </div>
    </div>
</body>
</html>
"""

HTML_OVERLAY_RESULT = """
<!doctype html>
<title>Demo Overlay</title>
<h1>Demo Overlay Result</h1>
<img src="{{ result_url }}" width="512">
<p><a href="/predict">Try another</a></p>
"""


def base_context():
    return {
        "preview_pairs": PAIR_OPTIONS_PREVIEW,
        "total_pairs": TOTAL_PAIRS,
        "total_pairs_minus_one": max(0, TOTAL_PAIRS - 1),
    }


# -------------------------------------------------------------------------
# Flask routes
# -------------------------------------------------------------------------
@app.route("/", methods=["GET"])
def home():
    return render_template_string(HTML_HOME)


@app.route("/predict", methods=["GET", "POST"])
def predict():
    context = base_context()

    if request.method == "GET":
        return render_template_string(HTML_PREDICT, **context)

    # Only handle dataset pairs - no custom uploads
    try:
        manual_value = request.form.get("pair_index_manual")
        selected_value = request.form.get("pair_index")
        raw_index = manual_value if manual_value not in (None, "") else selected_value
        if raw_index in (None, ""):
            raw_index = "0"
        pair_index = int(raw_index)
        result = run_dataset_inference(pair_index)
        result.update(context)
        return render_template_string(HTML_RESULT, **result)

    except Exception as exc:
        error_html = f"""
        <!doctype html>
        <html>
        <head>
            <meta charset="UTF-8">
            <title>Error - Virtual Dressing Room</title>
            <style>
                body {{
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    min-height: 100vh;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    padding: 20px;
                }}
                .error-container {{
                    background: white;
                    padding: 40px;
                    border-radius: 20px;
                    max-width: 600px;
                    box-shadow: 0 20px 60px rgba(0,0,0,0.3);
                    text-align: center;
                }}
                h1 {{ color: #c33; margin-bottom: 20px; }}
                p {{ color: #666; margin-bottom: 30px; }}
                .btn {{
                    display: inline-block;
                    padding: 12px 30px;
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    color: white;
                    text-decoration: none;
                    border-radius: 25px;
                }}
            </style>
        </head>
        <body>
            <div class="error-container">
                <h1>⚠️ Error</h1>
                <p>{str(exc)}</p>
                <a href="/predict" class="btn">Go Back</a>
            </div>
        </body>
        </html>
        """
        return error_html, 500


if __name__ == "__main__":
    print(f"[info] VITON Flask app running on device: {device}")
    app.run(host="0.0.0.0", port=5000, debug=True)
