import os
import sys
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np
import torch
from PIL import Image, ImageDraw
from torchvision import transforms
SCHP_PALETTE = [
    0, 0, 0,
    128, 0, 0,
    0, 128, 0,
    128, 128, 0,
    0, 0, 128,
    128, 0, 128,
    0, 128, 128,
    128, 128, 128,
    64, 0, 0,
    192, 0, 0,
    64, 128, 0,
    192, 128, 0,
    64, 0, 128,
    192, 0, 128,
    64, 128, 128,
    192, 128, 128,
    0, 64, 0,
    128, 64, 0,
    0, 192, 0,
    128, 192, 0,
] + [0] * (256 * 3 - 20 * 3)


@dataclass
class KeypointSet:
    points: np.ndarray  # shape: (25, 2)
    confidence: np.ndarray  # shape: (25,)


def _import_schp_modules(root: Path):
    root = root.resolve()
    original_networks = sys.modules.get("networks")
    original_utils = sys.modules.get("utils")

    sys.path.insert(0, str(root))
    try:
        import importlib

        sys.modules.pop("networks", None)
        sys.modules.pop("utils", None)
        schp_networks = importlib.import_module("networks")
        schp_transforms = importlib.import_module("utils.transforms")
    finally:
        sys.path.pop(0)
        if original_networks is not None:
            sys.modules["networks"] = original_networks
        else:
            sys.modules.pop("networks", None)
        if original_utils is not None:
            sys.modules["utils"] = original_utils
        else:
            sys.modules.pop("utils", None)

    sys.modules.setdefault("schp_networks", schp_networks)
    sys.modules.setdefault("schp_utils.transforms", schp_transforms)

    return schp_networks, schp_transforms


class SCHPSegmenter:
    def __init__(self, checkpoint_path: Path, device: torch.device):
        self.device = device
        self.input_size = (473, 473)
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.406, 0.456, 0.485], std=[0.225, 0.224, 0.229]),
            ]
        )
        repo_root = checkpoint_path.resolve().parent.parent / "self-correction-human-parsing-master"
        networks_module, transforms_module = _import_schp_modules(repo_root)
        self.transform_logits = transforms_module.transform_logits
        self.model = networks_module.init_model("resnet101", num_classes=20, pretrained=None)
        self.model = self._load_weights(self.model, checkpoint_path)
        self.model.to(self.device)
        self.model.eval()

    def _load_weights(self, model, checkpoint_path: Path):
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"SCHP checkpoint not found at {checkpoint_path}")
        state = torch.load(checkpoint_path, map_location=self.device)
        if "state_dict" in state:
            state = state["state_dict"]
        cleaned = {k[7:] if k.startswith("module.") else k: v for k, v in state.items()}
        missing, unexpected = model.load_state_dict(cleaned, strict=False)
        if missing and all(".bn." in key for key in missing):
            patched = dict(cleaned)
            for miss_key in missing:
                base_key = miss_key.replace(".bn.", ".")
                if base_key in patched:
                    patched[miss_key] = patched.pop(base_key)
            missing, unexpected = model.load_state_dict(patched, strict=False)
        if missing:
            print(f"[warn] Missing SCHP params: {missing}")
        if unexpected:
            print(f"[warn] Unexpected SCHP params: {unexpected}")
        return model

    @staticmethod
    def _xywh2cs(x, y, w, h, input_size: Tuple[int, int]):
        aspect_ratio = input_size[1] * 1.0 / input_size[0]
        center = np.zeros((2,), dtype=np.float32)
        center[0] = x + w * 0.5
        center[1] = y + h * 0.5
        if w > aspect_ratio * h:
            h = w * 1.0 / aspect_ratio
        elif w < aspect_ratio * h:
            w = h * aspect_ratio
        scale = np.array([w, h], dtype=np.float32)
        return center, scale

    def segment(self, image: Image.Image) -> np.ndarray:
        bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        h, w = bgr.shape[:2]
        center, scale = self._xywh2cs(0, 0, w - 1, h - 1, self.input_size)
        trans = self.transform_logits.__globals__["get_affine_transform"](center, scale, 0, self.input_size)

        input_image = cv2.warpAffine(
            bgr, trans, (int(self.input_size[1]), int(self.input_size[0])), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT
        )
        tensor = self.transform(input_image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            output = self.model(tensor)
            parsing_logits = output[0][-1]
            if isinstance(parsing_logits, (list, tuple)):
                parsing_logits = parsing_logits[0]
            upsample = torch.nn.functional.interpolate(
                parsing_logits, size=self.input_size, mode="bilinear", align_corners=True
            )
            logits = upsample[0].permute(1, 2, 0).cpu().numpy()

        restored_logits = self.transform_logits(logits, center, scale, w, h, self.input_size)
        parsing_result = np.argmax(restored_logits, axis=2).astype(np.uint8)
        return parsing_result


class PoseEstimator:
    def __init__(self):
        self.pose = mp.solutions.pose.Pose(
            static_image_mode=True,
            model_complexity=2,
            enable_segmentation=False,
            min_detection_confidence=0.35,
        )
        self.drawing_color = (255, 255, 255)
        self.dot_radius = 6
        self.stroke = 4

        # OpenPose order mapping to MediaPipe Pose indices
        self.mapping = {
            0: (0,),
            1: (11, 12),
            2: (12,),
            3: (14,),
            4: (16,),
            5: (11,),
            6: (13,),
            7: (15,),
            8: (23, 24),
            9: (24,),
            10: (26,),
            11: (28,),
            12: (23,),
            13: (25,),
            14: (27,),
            15: (5,),
            16: (2,),
            17: (8,),
            18: (7,),
            19: (31,),
            20: (29,),
            21: (29,),
            22: (32,),
            23: (30,),
            24: (30,),
        }
        self.limb_pairs = [
            (0, 1),
            (1, 2),
            (2, 3),
            (3, 4),
            (1, 5),
            (5, 6),
            (6, 7),
            (1, 8),
            (8, 9),
            (9, 10),
            (10, 11),
            (8, 12),
            (12, 13),
            (13, 14),
            (0, 15),
            (15, 17),
            (0, 16),
            (16, 18),
            (14, 19),
            (14, 20),
            (11, 22),
            (11, 23),
        ]

    @staticmethod
    def _aggregate_landmarks(landmarks, indices: Iterable[int]) -> Tuple[Optional[float], Optional[float], float]:
        xs, ys, confs = [], [], []
        for idx in indices:
            landmark = landmarks[idx]
            xs.append(landmark.x)
            ys.append(landmark.y)
            confs.append(landmark.visibility)
        if not confs:
            return None, None, 0.0
        confidence = float(min(confs))
        if confidence <= 0:
            return None, None, 0.0
        avg_x = float(sum(xs) / len(xs))
        avg_y = float(sum(ys) / len(ys))
        return avg_x, avg_y, confidence

    def estimate(self, image: Image.Image) -> Optional[KeypointSet]:
        rgb = np.array(image)
        results = self.pose.process(rgb)
        if not results.pose_landmarks:
            return None
        landmarks = results.pose_landmarks.landmark
        h, w = rgb.shape[:2]
        coords = np.zeros((25, 2), dtype=np.float32)
        confs = np.zeros((25,), dtype=np.float32)

        for idx, mp_indices in self.mapping.items():
            data = self._aggregate_landmarks(landmarks, mp_indices)
            if data[0] is None:
                continue
            x_norm, y_norm, confidence = data
            coords[idx] = [x_norm * w, y_norm * h]
            confs[idx] = confidence

        return KeypointSet(points=coords, confidence=confs)

    def render(self, keypoints: KeypointSet, size: Tuple[int, int]) -> Image.Image:
        width, height = size
        canvas = Image.new("RGB", size, (0, 0, 0))
        draw = ImageDraw.Draw(canvas)

        for start, end in self.limb_pairs:
            if keypoints.confidence[start] <= 0 or keypoints.confidence[end] <= 0:
                continue
            x0, y0 = keypoints.points[start]
            x1, y1 = keypoints.points[end]
            draw.line((x0, y0, x1, y1), fill=self.drawing_color, width=self.stroke)

        for idx, (x, y) in enumerate(keypoints.points):
            if keypoints.confidence[idx] <= 0:
                continue
            draw.ellipse(
                (x - self.dot_radius, y - self.dot_radius, x + self.dot_radius, y + self.dot_radius),
                fill=self.drawing_color,
            )
        return canvas


class ParsingHelper:
    def __init__(self, load_height: int, load_width: int):
        self.load_height = load_height
        self.load_width = load_width

    def get_parse_agnostic(self, parse: Image.Image, pose_data: np.ndarray) -> Image.Image:
        from datasets import VITONDataset  # local import to reuse logic

        dummy_opt = type("dummy", (), {})()
        dummy_opt.load_height = self.load_height
        dummy_opt.load_width = self.load_width
        dummy_opt.semantic_nc = 13
        helper = VITONDataset.__new__(VITONDataset)
        helper.load_height = self.load_height
        helper.load_width = self.load_width
        return VITONDataset.get_parse_agnostic(helper, parse, pose_data.copy())

    def get_img_agnostic(self, img: Image.Image, parse: Image.Image, pose_data: np.ndarray) -> Image.Image:
        from datasets import VITONDataset

        helper = VITONDataset.__new__(VITONDataset)
        helper.load_height = self.load_height
        helper.load_width = self.load_width
        return VITONDataset.get_img_agnostic(helper, img, parse, pose_data.copy())


class ClothMasker:
    def __init__(self, load_height: int, load_width: int):
        self.load_height = load_height
        self.load_width = load_width

    def process(self, cloth_image: Image.Image) -> Tuple[torch.Tensor, torch.Tensor]:
        cloth_rgb = cloth_image.convert("RGB").resize((self.load_width, self.load_height), Image.BICUBIC)
        cloth_np = np.array(cloth_rgb)
        gray = cv2.cvtColor(cloth_np, cv2.COLOR_RGB2GRAY)
        _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        mask = cv2.medianBlur(mask, 5)

        cloth_tensor = transforms.ToTensor()(cloth_rgb)
        cloth_tensor = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(cloth_tensor)

        mask_tensor = torch.from_numpy((mask >= 128).astype(np.float32)).unsqueeze(0)
        return cloth_tensor, mask_tensor


class UploadPreprocessor:
    def __init__(self, opt, device: torch.device, checkpoint_path: Path):
        self.opt = opt
        self.device = device
        self.segmenter = SCHPSegmenter(checkpoint_path, device)
        self.pose_estimator = PoseEstimator()
        self.parsing_helper = ParsingHelper(opt.load_height, opt.load_width)
        self.masker = ClothMasker(opt.load_height, opt.load_width)
        self.tensor_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )
        debug_flag = os.getenv("VDR_UPLOAD_DEBUG", "").lower()
        self.debug_enabled = debug_flag in {"1", "true", "yes", "on"}
        debug_dir_env = os.getenv("VDR_UPLOAD_DEBUG_DIR")
        if debug_dir_env:
            self.debug_dir = Path(debug_dir_env).expanduser().resolve()
        else:
            default_dir = Path(self.opt.save_dir) / "debug"
            self.debug_dir = default_dir.resolve()
        if self.debug_enabled:
            self.debug_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Person preprocessing helpers
    # ------------------------------------------------------------------
    def _compute_person_crop(self, image: Image.Image, keypoints: KeypointSet) -> Tuple[int, int, int, int]:
        width, height = image.size
        valid = keypoints.confidence > 0
        if valid.sum() < 3:
            return 0, 0, width, height

        xs = keypoints.points[valid, 0]
        ys = keypoints.points[valid, 1]

        min_x, max_x = float(xs.min()), float(xs.max())
        min_y, max_y = float(ys.min()), float(ys.max())

        # Expand the bounding box to include some context (head/legs)
        bbox_width = max(max_x - min_x, 1.0)
        bbox_height = max(max_y - min_y, 1.0)
        margin_x = bbox_width * 0.25
        margin_top = bbox_height * 0.35
        margin_bottom = bbox_height * 0.5

        left = min_x - margin_x
        right = max_x + margin_x
        top = min_y - margin_top
        bottom = max_y + margin_bottom

        target_ar = self.opt.load_width / self.opt.load_height

        crop_width = right - left
        crop_height = bottom - top

        if crop_width <= 0 or crop_height <= 0:
            return 0, 0, width, height

        crop_center_x = (left + right) / 2.0
        crop_center_y = (top + bottom) / 2.0

        # Adjust to match target aspect ratio while keeping centre fixed
        if crop_width / crop_height > target_ar:
            crop_height = crop_width / target_ar
        else:
            crop_width = crop_height * target_ar

        half_w = crop_width / 2.0
        half_h = crop_height / 2.0

        left = crop_center_x - half_w
        right = crop_center_x + half_w
        top = crop_center_y - half_h
        bottom = crop_center_y + half_h

        # Clamp to image boundaries while preserving size
        if left < 0:
            right -= left
            left = 0
        if right > width:
            shift = right - width
            left -= shift
            right = width
        if top < 0:
            bottom -= top
            top = 0
        if bottom > height:
            shift = bottom - height
            top -= shift
            bottom = height

        left = max(0, int(round(left)))
        right = min(width, int(round(right)))
        top = max(0, int(round(top)))
        bottom = min(height, int(round(bottom)))

        # Final safety check
        if right <= left or bottom <= top:
            return 0, 0, width, height

        return left, top, right, bottom

    def _crop_and_resize_person(
        self, image: Image.Image, keypoints: KeypointSet
    ) -> Tuple[Image.Image, KeypointSet, Dict[str, float]]:
        left, top, right, bottom = self._compute_person_crop(image, keypoints)
        crop_width = max(right - left, 1)
        crop_height = max(bottom - top, 1)

        cropped = image.crop((left, top, right, bottom))
        resized = cropped.resize((self.opt.load_width, self.opt.load_height), Image.BICUBIC)

        scale_x = self.opt.load_width / crop_width
        scale_y = self.opt.load_height / crop_height

        points = keypoints.points.copy()
        points[:, 0] = (points[:, 0] - left) * scale_x
        points[:, 1] = (points[:, 1] - top) * scale_y

        transformed_keypoints = KeypointSet(points=points, confidence=keypoints.confidence.copy())
        transform = {
            "left": left,
            "top": top,
            "right": right,
            "bottom": bottom,
            "scale_x": scale_x,
            "scale_y": scale_y,
        }
        return resized, transformed_keypoints, transform

    def _create_parse_tensor(self, parse_agnostic: Image.Image) -> torch.Tensor:
        labels = {
            0: [0, 10],
            1: [1, 2],
            2: [4, 13],
            3: [5, 6, 7],
            4: [9, 12],
            5: [14],
            6: [15],
            7: [16],
            8: [17],
            9: [18],
            10: [19],
            11: [8],
            12: [3, 11],
        }
        parse_np = np.array(parse_agnostic, dtype=np.int64)[None]
        parse_tensor = torch.tensor(parse_np, dtype=torch.long)

        full_map = torch.zeros(20, self.opt.load_height, self.opt.load_width, dtype=torch.float32)
        full_map.scatter_(0, parse_tensor, 1.0)

        semantic = torch.zeros(self.opt.semantic_nc, self.opt.load_height, self.opt.load_width, dtype=torch.float32)
        for idx, ids in labels.items():
            for label_idx in ids:
                semantic[idx] += full_map[label_idx]
        return semantic

    def prepare(self, person_image: Image.Image, cloth_image: Image.Image) -> Tuple[Dict, Dict]:
        person_full = person_image.convert("RGB")
        cloth_rgb = cloth_image.convert("RGB")

        keypoints_full = self.pose_estimator.estimate(person_full)
        if keypoints_full is None:
            raise RuntimeError("Pose estimation failed for the provided person image.")

        person_rgb, keypoints, transform = self._crop_and_resize_person(person_full, keypoints_full)

        parsing_result = self.segmenter.segment(person_rgb)
        parse_pil = Image.fromarray(parsing_result)

        pose_array = keypoints.points.copy()
        pose_image = self.pose_estimator.render(keypoints, (self.opt.load_width, self.opt.load_height))
        pose_tensor = self.tensor_transform(pose_image)

        parse_agnostic = self.parsing_helper.get_parse_agnostic(parse_pil, pose_array.copy())
        img_agnostic = self.parsing_helper.get_img_agnostic(person_rgb, parse_pil, pose_array.copy())

        sample = {
            "img_name": "upload_person.jpg",
            "c_name": {"unpaired": "upload_cloth.jpg"},
            "img": self.tensor_transform(person_rgb),
            "img_agnostic": self.tensor_transform(img_agnostic),
            "parse_agnostic": self._create_parse_tensor(parse_agnostic),
            "pose": pose_tensor,
        }

        cloth_tensor, mask_tensor = self.masker.process(cloth_rgb)
        sample["cloth"] = {"unpaired": cloth_tensor}
        sample["cloth_mask"] = {"unpaired": mask_tensor}

        metadata = {
            "pose_keypoints": keypoints,
            "pose_image": pose_image,
            "crop_transform": transform,
            "pose_keypoints_original": keypoints_full,
        }
        if self.debug_enabled:
            debug_id = uuid.uuid4().hex
            metadata["debug_id"] = debug_id
            metadata["debug_assets"] = self._dump_debug_assets(
                debug_id,
                person_full=person_full,
                person_cropped=person_rgb,
                parse_raw=parsing_result,
                parse_pil=parse_pil,
                parse_agnostic=parse_agnostic,
                img_agnostic=img_agnostic,
                pose=pose_image,
                cloth=cloth_rgb,
                cloth_mask=mask_tensor,
            )
        return sample, metadata

    def _dump_debug_assets(
        self,
        debug_id: str,
        *,
        person_full: Image.Image,
        person_cropped: Image.Image,
        parse_raw: np.ndarray,
        parse_pil: Image.Image,
        parse_agnostic: Image.Image,
        img_agnostic: Image.Image,
        pose: Image.Image,
        cloth: Image.Image,
        cloth_mask: torch.Tensor,
    ) -> Dict[str, str]:
        outputs: Dict[str, str] = {}

        def _save(image: Image.Image, suffix: str):
            path = self.debug_dir / f"{debug_id}_{suffix}.png"
            image.save(path, quality=95)
            outputs[suffix] = str(path)

        _save(person_full, "person_full")
        _save(person_cropped, "person_cropped")

        parse_color = Image.fromarray(parse_raw.astype(np.uint8), mode="P")
        parse_color.putpalette(SCHP_PALETTE)
        _save(parse_color.convert("RGB"), "parse_color")
        _save(parse_pil, "parse_labels")
        _save(parse_agnostic, "parse_agnostic")
        _save(img_agnostic, "img_agnostic")
        _save(pose, "pose_render")
        _save(cloth, "cloth_original")

        mask = cloth_mask.detach().cpu().squeeze(0)
        mask_img = Image.fromarray((mask.numpy() * 255.0).clip(0, 255).astype(np.uint8), mode="L")
        _save(mask_img, "cloth_mask")
        return outputs


def tensor_to_image(tensor: torch.Tensor) -> Image.Image:
    if tensor.dim() == 4:
        tensor = tensor[0]
    tensor = tensor.detach().cpu().clamp(-1, 1)
    array = tensor.mul(0.5).add(0.5).mul(255).permute(1, 2, 0).byte().numpy()
    return Image.fromarray(array)


def save_tensors(sample: Dict, result_dir: Path, identifier: str) -> Dict[str, str]:
    os.makedirs(result_dir, exist_ok=True)
    outputs = {}
    for key, tensor in {
        "person": sample["img"].unsqueeze(0),
        "parse": sample["parse_agnostic"].argmax(dim=0, keepdim=True).float(),
        "img_agnostic": sample["img_agnostic"].unsqueeze(0),
    }.items():
        img = tensor_to_image(tensor)
        path = result_dir / f"{key}_{identifier}.jpg"
        img.save(path, quality=95)
        outputs[key] = str(path)
    return outputs

