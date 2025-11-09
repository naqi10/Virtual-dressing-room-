import os
import copy
import uuid
import types
import traceback

import torch
import torch.nn.functional as F
from PIL import Image, ImageOps, ImageFilter
from flask import Flask, request, jsonify, render_template_string

from networks import SegGenerator, GMM, ALIASGenerator
from datasets import VITONDataset

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


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_tensor_type("torch.FloatTensor")

app = Flask(__name__, static_folder="static")
os.makedirs("static", exist_ok=True)
RESULT_DIR = os.path.join("static", "results")
os.makedirs(RESULT_DIR, exist_ok=True)


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

    array = tensor.clamp(-1, 1)
    array = (array + 1) * 0.5
    array = array.mul(255).permute(1, 2, 0).byte().numpy()
    return Image.fromarray(array, mode="RGB")


def save_tensor_image(tensor, path, is_mask=False):
    img = tensor_to_pil(tensor, is_mask=is_mask)
    img.save(path, quality=95)


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

        parse_pred_up = F.interpolate(parse_pred_down, size=(opt.load_height, opt.load_width), mode="bilinear", align_corners=False)
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
        warped_c = F.grid_sample(cloth, warped_grid, padding_mode="border", align_corners=False)
        warped_cm = F.grid_sample(cloth_mask, warped_grid, padding_mode="border", align_corners=False)

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


HTML_HOME = """
<!doctype html>
<title>VITON-HD Try-On</title>
<h1>VITON-HD Virtual Dressing Room</h1>
<p>Select a preprocessed pair from the dataset or upload custom images for a quick overlay demo.</p>
<a href="/predict"><button>Open Try-On Page</button></a>
"""

HTML_PREDICT = """
<!doctype html>
<title>VITON-HD Try-On</title>
<h1>VITON-HD Virtual Dressing Room</h1>

<section>
  <h2>Run on Preprocessed Dataset</h2>
  {% if total_pairs == 0 %}
    <p><strong>Dataset not found.</strong> Please place the preprocessed dataset under <code>datasets/zalando-hd-resized/</code>.</p>
  {% else %}
    <p>Total available pairs: {{ total_pairs }}</p>
    <form method="POST" action="/predict">
      <label for="pair_index">Quick select (first {{ preview_pairs|length }} pairs):</label>
      <select name="pair_index" id="pair_index">
        {% for idx, pair in preview_pairs %}
          <option value="{{ idx }}">{{ idx }} — {{ pair[0] }} ➜ {{ pair[1] }}</option>
        {% endfor %}
      </select>
      <p>Or enter a specific index (0 – {{ total_pairs_minus_one }}):
         <input type="number" name="pair_index_manual" min="0" max="{{ total_pairs_minus_one }}" placeholder="0">
      </p>
      <button type="submit" name="action" value="dataset">Generate Try-On</button>
    </form>
  {% endif %}
</section>

<hr>

<section>
  <h2>Quick Demo Overlay (no preprocessing)</h2>
  <form method="POST" action="/predict" enctype="multipart/form-data">
    <p>Person image: <input type="file" name="person" accept="image/*" required></p>
    <p>Cloth image: <input type="file" name="cloth" accept="image/*" required></p>
    <button type="submit" name="action" value="upload">Generate Demo Overlay</button>
  </form>
  <p>The demo overlay simply pastes the cloth on the torso. For realistic fitting, use the preprocessed dataset option above.</p>
</section>
"""

HTML_RESULT = """
<!doctype html>
<title>Try-On Result</title>
<h1>Try-On Result</h1>
<p>Pair index: {{ pair_index }} ({{ img_name }} ➜ {{ cloth_name }})</p>
<div>
  <h3>Generated Try-On</h3>
  <img src="{{ result_url }}" width="384">
</div>
<div>
  <h3>Original Person</h3>
  <img src="{{ person_url }}" width="256">
</div>
<div>
  <h3>Original Cloth</h3>
  <img src="{{ cloth_url }}" width="256">
</div>
<div>
  <h3>Warped Cloth</h3>
  <img src="{{ warped_url }}" width="256">
</div>
<p><a href="/predict">Run another try-on</a></p>
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


@app.route("/", methods=["GET"])
def home():
    return render_template_string(HTML_HOME)


@app.route("/predict", methods=["GET", "POST"])
def predict():
    context = base_context()

    if request.method == "GET":
        return render_template_string(HTML_PREDICT, **context)

    action = request.form.get("action")

    try:
        if action == "dataset":
            manual_value = request.form.get("pair_index_manual")
            selected_value = request.form.get("pair_index")
            raw_index = manual_value if manual_value not in (None, "") else selected_value
            if raw_index in (None, ""):
                raw_index = "0"
            pair_index = int(raw_index)
            result = run_dataset_inference(pair_index)
            result.update(context)
            return render_template_string(HTML_RESULT, **result)

        if action == "upload":
            if "person" not in request.files or "cloth" not in request.files:
                return jsonify({"error": "Please upload both person and cloth images."}), 400

            person_file = request.files["person"]
            cloth_file = request.files["cloth"]

            if person_file.filename == "" or cloth_file.filename == "":
                return jsonify({"error": "Empty filename supplied for upload."}), 400

            overlay_info = generate_demo_overlay(person_file, cloth_file)
            overlay_info.update(context)
            return render_template_string(HTML_OVERLAY_RESULT, **overlay_info)

        return jsonify({"error": "Unsupported action. Choose dataset or upload."}), 400

    except Exception as exc:
        return jsonify({
            "status": "error",
            "error": str(exc),
            "traceback": traceback.format_exc()
        }), 500


if __name__ == "__main__":
    print(f"[info] VITON Flask app running on device: {device}")
    app.run(host="0.0.0.0", port=5000, debug=True)
