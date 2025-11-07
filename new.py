import os
import torch
import torchvision.transforms as T
from PIL import Image, ImageOps
from flask import Flask, request, jsonify, render_template_string
import types
import traceback

from networks import SegGenerator, GMM, ALIASGenerator

# -----------------------------
# Utility: safe checkpoint loader
# -----------------------------
try:
    from utils import load_checkpoint
except ImportError:
    def load_checkpoint(model, path, map_location="cpu"):
        """Load model weights safely even if utils.py is missing."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        print(f"🔄 Loading checkpoint from: {path}")
        checkpoint = torch.load(path, map_location=map_location)
        model.load_state_dict(checkpoint, strict=False)
        print("✅ Model weights loaded successfully.")
        return model

# -----------------------------
# Device setup (GPU or CPU)
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_tensor_type("torch.FloatTensor")

# -----------------------------
# Flask app setup
# -----------------------------
app = Flask(__name__, static_folder="static")
os.makedirs("static", exist_ok=True)

# -----------------------------
# Home and Upload Templates
# -----------------------------
HTML_HOME = """
<!doctype html>
<title>VITON Virtual Try-On</title>
<h1>🧥 Welcome to the VITON-HD Flask App</h1>
<p>This app lets you upload a person and clothing image to generate a virtual try-on result (demo overlay).</p>
<a href="/predict"><button>Go to Try-On Page</button></a>
"""

HTML_PREDICT = """
<!doctype html>
<title>Virtual Try-On</title>
<h1>👗 Upload Images for VITON Prediction (Demo Overlay)</h1>
<form method="POST" action="/predict" enctype="multipart/form-data">
  <p>Person image: <input type="file" name="person" accept="image/*" required></p>
  <p>Cloth image: <input type="file" name="cloth" accept="image/*" required></p>
  <input type="submit" value="Generate Try-On">
</form>
<p>After submission, the generated result will appear below. This is a demo overlay; real GMM+ALIAS inference can be added next.</p>
"""

# -----------------------------
# Small helpers
# -----------------------------
def pil_open_rgb(fileobj):
    img = Image.open(fileobj).convert("RGB")
    return img

def pil_to_tensor(img, size=(768, 1024)):
    transform = T.Compose([
        T.Resize(size),
        T.ToTensor(),
        T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    return transform(img).unsqueeze(0).to(device)

# -----------------------------
# Model loader (executed once) - kept so we can plug real inference later
# -----------------------------
def load_models():
    checkpoint_dir = "./checkpoints/"
    opt = types.SimpleNamespace()
    opt.load_height = 1024
    opt.load_width = 768
    opt.grid_size = 5
    opt.semantic_nc = 13
    opt.norm_G = "spectralaliasinstance"
    opt.ngf = 64
    opt.num_upsampling_layers = "most"
    opt.init_type = "xavier"
    opt.init_variance = 0.02

    print("🔧 Initializing models (kept for future real inference)...")
    # instantiate but we won't use them in the demo overlay
    seg = SegGenerator(opt, input_nc=opt.semantic_nc + 8, output_nc=opt.semantic_nc).to(device).eval()
    gmm = GMM(opt, inputA_nc=7, inputB_nc=3).to(device).eval()
    opt.semantic_nc = 7
    alias = ALIASGenerator(opt, input_nc=9).to(device).eval()

    # try loading if checkpoints exist, but don't fail hard if they don't
    try:
        load_checkpoint(seg, os.path.join(checkpoint_dir, "seg_final.pth"), map_location=device)
    except Exception as e:
        print(f"⚠️ seg checkpoint not loaded: {e}")
    try:
        load_checkpoint(gmm, os.path.join(checkpoint_dir, "gmm_final.pth"), map_location=device)
    except Exception as e:
        print(f"⚠️ gmm checkpoint not loaded: {e}")
    try:
        load_checkpoint(alias, os.path.join(checkpoint_dir, "alias_final.pth"), map_location=device)
    except Exception as e:
        print(f"⚠️ alias checkpoint not loaded: {e}")

    print("✅ Model initialization done (demo mode).")
    return seg, gmm, alias

SEG_MODEL, GMM_MODEL, ALIAS_MODEL = load_models()

# -----------------------------
# Routes
# -----------------------------
@app.route("/", methods=["GET"])
def home():
    return render_template_string(HTML_HOME)

@app.route("/predict", methods=["GET", "POST"])
def predict():
    if request.method == "GET":
        return render_template_string(HTML_PREDICT)

    try:
        # Validate uploads
        if "person" not in request.files or "cloth" not in request.files:
            return jsonify({"error": "Please upload both 'person' and 'cloth' images."}), 400

        person_file = request.files["person"]
        cloth_file = request.files["cloth"]

        # Open images with PIL (no destructive transforms yet)
        person_pil = pil_open_rgb(person_file)
        cloth_pil = pil_open_rgb(cloth_file)

        # ---------- Demo overlay logic ----------
        # Goal: produce a visible, sensible-looking composite for testing UI.
        # Steps:
        # 1. Resize cloth to fit torso area approx (we use a heuristic).
        # 2. Create a cloth mask by thresholding the grayscale (non-black area).
        # 3. Paste/blend the cloth onto the person at a centered top area.

        # Resize person to a standard canvas for consistency
        canvas_w, canvas_h = 768, 1024
        person_resized = ImageOps.fit(person_pil, (canvas_w, canvas_h), Image.BICUBIC)

        # Resize cloth to be roughly the width of torso (heuristic: 0.6 * person width)
        cloth_target_w = int(canvas_w * 0.6)
        # preserve aspect ratio
        cloth_ratio = cloth_pil.width / max(1, cloth_pil.height)
        cloth_target_h = int(cloth_target_w / max(0.6, cloth_ratio))
        cloth_resized = cloth_pil.resize((cloth_target_w, cloth_target_h), Image.LANCZOS)

        # Create a mask from cloth: non-dark pixels are cloth
        cloth_gray = cloth_resized.convert("L")
        # adaptive threshold: pixels > 20 are considered cloth
        mask = cloth_gray.point(lambda p: 255 if p > 20 else 0).convert("L")

        # Blur mask to smooth edges
        mask = mask.filter(Image.Filter.GaussianBlur(radius=3)) if hasattr(Image, "Filter") else mask

        # Position the cloth roughly at torso: center horizontally, 28% from top vertically
        x = (canvas_w - cloth_target_w) // 2
        y = int(canvas_h * 0.28)

        # Create RGBA layers
        base = person_resized.convert("RGBA")
        cloth_rgba = cloth_resized.convert("RGBA")

        # Apply mask as alpha to cloth
        cloth_with_alpha = cloth_rgba.copy()
        cloth_with_alpha.putalpha(mask)

        # Composite: blend cloth onto base with some opacity
        # convert to an image we can paste with alpha
        composite = Image.alpha_composite(base, Image.new("RGBA", base.size, (0, 0, 0, 0)))
        # Paste cloth onto composite using its alpha
        composite.paste(cloth_with_alpha, (x, y), cloth_with_alpha)

        # Slight global blend to make it look smoother (optional)
        result = Image.blend(person_resized.convert("RGBA"), composite, alpha=0.85).convert("RGB")

        # Save output
        save_path = os.path.join("static", "output.jpg")
        result.save(save_path, quality=95)

        # Return simple HTML with result preview
        return f"""
        <h2>✅ Demo Try-On Generated</h2>
        <img src="/static/output.jpg" width="512">
        <p><a href="/predict">Try another</a></p>
        """

    except Exception as e:
        return jsonify({
            "status": "error",
            "error": str(e),
            "traceback": traceback.format_exc()
        }), 500

# -----------------------------
# Run app
# -----------------------------
if __name__ == "__main__":
    print(f"🔥 VITON Flask app running on device: {device} (demo overlay mode)")
    app.run(host="0.0.0.0", port=5000, debug=True)
