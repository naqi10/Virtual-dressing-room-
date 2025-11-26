# import torch
# import torch.nn.functional as F
# import torchvision.transforms as transforms
# import numpy as np
# from PIL import Image
# import os
# import sys
# from pathlib import Path

# # SCHP imports
# # ------------------------------------------------------------------
# # These are official SCHP components (ResNet-101 backbone)
# # Import from Self-Correction-Human-Parsing-master/networks, handling the case
# # where root-level networks.py is already loaded
# schp_root = Path(__file__).parent.parent / "Self-Correction-Human-Parsing-master"
# schp_root_str = str(schp_root)

# # Temporarily remove root-level networks from sys.modules if it exists
# # and add SCHP root to path, then import
# original_networks = None
# if "networks" in sys.modules:
#     # Check if it's the root-level networks.py (not a package)
#     networks_module = sys.modules["networks"]
#     if hasattr(networks_module, "__file__") and networks_module.__file__:
#         # If it's a .py file (not a package), temporarily remove it
#         if networks_module.__file__.endswith(".py") and "Self-Correction-Human-Parsing-master" not in networks_module.__file__:
#             original_networks = sys.modules.pop("networks")

# try:
#     # Add SCHP root to path and import
#     if schp_root_str not in sys.path:
#         sys.path.insert(0, schp_root_str)
    
#     # Import from SCHP networks package
#     from networks import init_model
# finally:
#     # Restore original networks module if we removed it
#     if original_networks is not None:
#         sys.modules["networks"] = original_networks
#     # Keep SCHP root in path - init_model may need to access SCHP modules when called
#     # The path is added at position 0, so it takes precedence but won't break other imports


# class SCHPParser:
#     """
#     Loads SCHP human parsing model and generates 20-class segmentation maps.
#     Output matches VITON-HD preprocessing format.
#     """

#     def __init__(self,
#                  checkpoint_path="checkpoints/exp-schp-201908261155-lip.pth",
#                  device="cuda" if torch.cuda.is_available() else "cpu"):

#         self.device = device

#         if not os.path.exists(checkpoint_path):
#             raise FileNotFoundError(
#                 f"SCHP checkpoint not found: {checkpoint_path}"
#             )

#         print(f"[SCHP] Loading checkpoint from {checkpoint_path}")

#         # Build network (ResNet-101 backbone)
#         self.model = init_model('resnet101', num_classes=20, pretrained=None)
#         self.model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"), strict=False)
#         self.model.to(self.device)
#         self.model.eval()

#         # Required input transforms
#         self.transform = transforms.Compose([
#             transforms.Resize((512, 384)),
#             transforms.ToTensor(),
#             transforms.Normalize(mean=[0.406, 0.456, 0.485],
#                                  std=[0.225, 0.224, 0.229])
#         ])

#     def parse(self, img_pil: Image.Image):
#         """
#         Takes a PIL image (person image), returns a segmentation tensor:
#             shape: (1, 20, H, W)
#             values: softmax probabilities per class
#         """
#         img = self.transform(img_pil).unsqueeze(0).to(self.device)

#         with torch.no_grad():
#             # Model returns: [[parsing_result, fusion_result], [edge_result]]
#             # We want fusion_result (the last element of the first list)
#             output = self.model(img)
#             out = output[0][-1]  # Get fusion_result
#             # Handle case where it might still be a list/tuple
#             if isinstance(out, (list, tuple)):
#                 out = out[0]
#             # Ensure it's a tensor
#             if not isinstance(out, torch.Tensor):
#                 raise TypeError(f"Expected tensor, got {type(out)}")
#             out = F.softmax(out, dim=1)

#         return out

#     def get_argmax_mask(self, img_pil: Image.Image):
#         """
#         Returns argmax label mask (H, W) in numpy format (uint8).
#         """
#         prob_map = self.parse(img_pil)  # (1,20,H,W)
        
#         # Debug: Check what the model is actually outputting
#         prob_np = prob_map.squeeze().cpu().numpy()  # (20, H, W) or (1, 20, H, W)
#         if prob_np.ndim == 3 and prob_np.shape[0] == 1:
#             prob_np = prob_np[0]  # Remove batch: (20, H, W)
        
#         if prob_np.ndim == 3:
#             # Check max probabilities per class
#             max_probs = prob_np.max(axis=(1, 2))  # (20,) - max prob per class
#             mean_probs = prob_np.mean(axis=(1, 2))  # (20,) - mean prob per class
#             print(f"[SCHP DEBUG] Output shape: {prob_np.shape}")
#             print(f"[SCHP DEBUG] Top 5 classes by max probability:")
#             top5_max = np.argsort(max_probs)[-5:][::-1]
#             for cls in top5_max:
#                 print(f"[SCHP DEBUG]   Class {cls}: max={max_probs[cls]:.4f}, mean={mean_probs[cls]:.4f}")
            
#             # Check if model output looks valid (should have some variance)
#             std_per_class = prob_np.std(axis=(1, 2))  # (20,) - std per class
#             high_std_classes = np.where(std_per_class > 0.1)[0]
#             print(f"[SCHP DEBUG] Classes with std > 0.1 (have spatial variation): {high_std_classes.tolist()}")
        
#         mask = torch.argmax(prob_map, dim=1).squeeze().cpu().numpy().astype(np.uint8)
        
#         # Debug: Check mask output
#         unique_labels = np.unique(mask)
#         print(f"[SCHP DEBUG] Argmax mask unique labels: {unique_labels}")
#         print(f"[SCHP DEBUG] Argmax mask shape: {mask.shape}")
        
#         return mask


# # Helper function (used in new1.py)
# # -------------------------------------------------------------------

# def run_schp_parsing(image_path, checkpoint="checkpoints/exp-schp-201908261155-lip.pth"):
#     """
#     Loads the image, runs SCHP parsing, returns (prob_map, argmax_mask)
#     """
#     parser = SCHPParser(checkpoint_path=checkpoint)
#     img = Image.open(image_path).convert("RGB")

#     prob_map = parser.parse(img)        # (1,20,H,W)
#     argmax_mask = parser.get_argmax_mask(img)  # (H,W)

#     return prob_map, argmax_mask


import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
import os
import sys
from pathlib import Path
import cv2

# ------------------------------------------------------
# Load official SCHP modules
# ------------------------------------------------------
schp_root = Path(__file__).parent.parent / "Self-Correction-Human-Parsing-master"
schp_root_str = str(schp_root)

original_networks = None
if "networks" in sys.modules:
    networks_module = sys.modules["networks"]
    if hasattr(networks_module, "__file__") and networks_module.__file__:
        if networks_module.__file__.endswith(".py") and "Self-Correction-Human-Parsing-master" not in networks_module.__file__:
            original_networks = sys.modules.pop("networks")

try:
    if schp_root_str not in sys.path:
        sys.path.insert(0, schp_root_str)
    from networks import init_model
finally:
    if original_networks is not None:
        sys.modules["networks"] = original_networks


# ======================================================
# SCHP HUMAN PARSER (Fixed + Official Compatible)
# ======================================================
class SCHPParser:
    def __init__(
            self,
            checkpoint_path="checkpoints/humanparsing/exp-schp-201908261155-lip.pth",
            device="cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.device = device

        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"SCHP checkpoint not found: {checkpoint_path}")

        print(f"[SCHP] Loading checkpoint from: {checkpoint_path}")

        # Detect checkpoint type and set num_classes
        # ATR: 18 classes, LIP: 20 classes (from simple_extractor.py)
        if "atr" in checkpoint_path.lower():
            num_classes = 18
            print(f"[SCHP] Detected ATR checkpoint - using {num_classes} classes")
        else:
            num_classes = 20
            print(f"[SCHP] Detected LIP checkpoint - using {num_classes} classes")

        # Load SCHP network with correct num_classes
        self.model = init_model("resnet101", num_classes=num_classes, pretrained=None)
        self.num_classes = num_classes
        
        # Official loading method (from simple_extractor.py line 106-112)
        state_dict = torch.load(checkpoint_path, map_location="cpu")
        if isinstance(state_dict, dict) and 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']
        
        # Remove 'module.' prefix if present (for DataParallel models)
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:] if k.startswith('module.') else k
            new_state_dict[name] = v
        
        self.model.load_state_dict(new_state_dict, strict=False)
        self.model.to(self.device)
        self.model.eval()

        # ---------------------------
        # OFFICIAL SCHP transform (from simple_extractor.py)
        # ATR: 512x512, LIP: 473x473
        # ---------------------------
        if num_classes == 18:  # ATR
            input_size = (512, 512)
        else:  # LIP
            input_size = (473, 473)
        
        self.input_size = input_size
        self.transform = transforms.Compose([
            transforms.Resize(input_size, interpolation=Image.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.406, 0.456, 0.485], std=[0.225, 0.224, 0.229])
        ])

    # --------------------------------------------------
    # Forward pass through SCHP (correct branch output)
    # --------------------------------------------------
    def parse(self, img_pil):
        img = self.transform(img_pil).unsqueeze(0).to(self.device)

        with torch.no_grad():
            output = self.model(img)

            # Official SCHP output structure (from simple_extractor.py line 138):
            # output[0][-1][0] = fused parsing result
            # Match exact extraction: output[0][-1][0].unsqueeze(0) then upsample
            upsample = torch.nn.Upsample(size=self.input_size, mode='bilinear', align_corners=True)
            upsample_output = upsample(output[0][-1][0].unsqueeze(0))
            # upsample_output is now (1, num_classes, H, W)
            
            # Apply softmax to get probabilities
            prob = F.softmax(upsample_output, dim=1)

        return prob  # (1, num_classes, H, W)

    # --------------------------------------------------
    # Get argmax mask (resized back to original size)
    # --------------------------------------------------
    def get_argmax_mask(self, img_pil):
        orig_w, orig_h = img_pil.size

        prob = self.parse(img_pil)
        mask = torch.argmax(prob, dim=1).squeeze().cpu().numpy().astype(np.uint8)

        # Resize mask back
        mask = cv2.resize(mask, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)

        print("[SCHP DEBUG] Unique labels:", np.unique(mask))

        return mask


# ======================================================
# SIMPLE HELPER FUNCTION
# ======================================================
def run_schp_parsing(image_path, checkpoint=None):
    """
    Run SCHP parsing on image.
    If checkpoint is None, tries ATR checkpoint first, then falls back to LIP.
    """
    if checkpoint is None:
        # Try ATR checkpoint first (better generalization)
        atr_checkpoint = "checkpoints/humanparsing/exp-schp-201908301523-atr.pth"
        lip_checkpoint = "checkpoints/humanparsing/exp-schp-201908261155-lip.pth"
        
        if os.path.exists(atr_checkpoint):
            checkpoint = atr_checkpoint
            print(f"[SCHP] Using ATR checkpoint: {checkpoint}")
        elif os.path.exists(lip_checkpoint):
            checkpoint = lip_checkpoint
            print(f"[SCHP] ATR checkpoint not found, using LIP checkpoint: {checkpoint}")
        else:
            raise FileNotFoundError(
                f"Neither ATR nor LIP checkpoint found. "
                f"Please download exp-schp-201908301523-atr.pth or exp-schp-201908261155-lip.pth "
                f"to checkpoints/humanparsing/ folder"
            )
    
    parser = SCHPParser(checkpoint_path=checkpoint)
    img = Image.open(image_path).convert("RGB")

    prob_map = parser.parse(img)
    argmax_mask = parser.get_argmax_mask(img)

    return prob_map, argmax_mask
