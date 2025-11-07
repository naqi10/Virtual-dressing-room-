# app.py
from flask import Flask, request, jsonify, render_template_string
import os
import types
import traceback
import torch

from networks import SegGenerator, GMM, ALIASGenerator

# --------------------------------------------------------------------
# Safe load_checkpoint (works even if utils.py not provided)
# --------------------------------------------------------------------
try:
    from utils import load_checkpoint
except ImportError:
    def load_checkpoint(model, path, map_location="cpu"):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        checkpoint = torch.load(path, map_location=map_location)
        model.load_state_dict(checkpoint, strict=False)
        return model

# --------------------------------------------------------------------
# Device setup (auto-detects GPU if available)
# --------------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_tensor_type("torch.FloatTensor")

app = Flask(__name__)

# --------------------------------------------------------------------
# Simple HTML form interface
# --------------------------------------------------------------------
HTML = """
<!doctype html>
<title>VITON Model Checker</title>
<h1>🧥 VITON Model Check</h1>
<form action="/test" method="post">
  <p>Checkpoint dir: <input type="text" name="checkpoint_dir" value="./checkpoints/"></p>
  <p>Seg checkpoint: <input type="text" name="seg_checkpoint" value="seg_final.pth"></p>
  <p>GMM checkpoint: <input type="text" name="gmm_checkpoint" value="gmm_final.pth"></p>
  <p>ALIAS checkpoint: <input type="text" name="alias_checkpoint" value="alias_final.pth"></p>
  <p>Batch size: <input type="number" name="batch_size" value="1"></p>
  <p>Load height: <input type="number" name="load_height" value="1024"></p>
  <p>Load width: <input type="number" name="load_width" value="768"></p>
  <p><input type="submit" value="Run Check"></p>
</form>
<p><strong>Note:</strong> Response will be JSON with checkpoint load results and forward pass shapes.</p>
"""

# --------------------------------------------------------------------
# Helper function to mimic argparse options
# --------------------------------------------------------------------
def make_opt(form):
    opt = types.SimpleNamespace()
    opt.name = "test_run"
    opt.batch_size = int(form.get("batch_size", 1))
    opt.workers = 1
    opt.load_height = int(form.get("load_height", 1024))
    opt.load_width = int(form.get("load_width", 768))
    opt.shuffle = False
    opt.dataset_dir = "./datasets/"
    opt.dataset_mode = "test"
    opt.dataset_list = "test_pairs.txt"
    opt.checkpoint_dir = form.get("checkpoint_dir", "./checkpoints/")
    opt.save_dir = form.get("save_dir", "./results/")
    opt.display_freq = 1
    opt.seg_checkpoint = form.get("seg_checkpoint", "seg_final.pth")
    opt.gmm_checkpoint = form.get("gmm_checkpoint", "gmm_final.pth")
    opt.alias_checkpoint = form.get("alias_checkpoint", "alias_final.pth")
    opt.semantic_nc = 13
    opt.init_type = "xavier"
    opt.init_variance = 0.02
    opt.grid_size = 5
    opt.norm_G = "spectralaliasinstance"
    opt.ngf = 64
    opt.num_upsampling_layers = "most"
    return opt


@app.route("/", methods=["GET"])
def index():
    return render_template_string(HTML)


# --------------------------------------------------------------------
# Main /test endpoint — loads checkpoints and runs dummy forward pass
# --------------------------------------------------------------------
@app.route("/test", methods=["POST"])
def run_check():
    form = request.form
    opt = make_opt(form)

    seg_path = os.path.join(opt.checkpoint_dir, opt.seg_checkpoint)
    gmm_path = os.path.join(opt.checkpoint_dir, opt.gmm_checkpoint)
    alias_path = os.path.join(opt.checkpoint_dir, opt.alias_checkpoint)

    report = {
        "device": str(device),
        "seg_checkpoint": seg_path,
        "gmm_checkpoint": gmm_path,
        "alias_checkpoint": alias_path,
        "load": {},
        "forward": {},
        "errors": []
    }

    try:
        # -----------------------------
        # Initialize models
        # -----------------------------
        seg = SegGenerator(opt, input_nc=opt.semantic_nc + 8, output_nc=opt.semantic_nc).to(device)
        gmm = GMM(opt, inputA_nc=7, inputB_nc=3).to(device)
        original_nc = opt.semantic_nc
        opt.semantic_nc = 7
        alias = ALIASGenerator(opt, input_nc=9).to(device)
        opt.semantic_nc = original_nc

        # -----------------------------
        # Load checkpoints
        # -----------------------------
        for model, path, key in [
            (seg, seg_path, "seg"),
            (gmm, gmm_path, "gmm"),
            (alias, alias_path, "alias")
        ]:
            try:
                load_checkpoint(model, path, map_location=device)
                report["load"][key] = "ok"
            except Exception as e:
                report["load"][key] = f"failed: {e}"
                report["errors"].append(f"{key} load: {e}")

        seg.eval(), gmm.eval(), alias.eval()

        bs = max(1, opt.batch_size)
        LH, LW = opt.load_height, opt.load_width
        seg_H, seg_W = 256, 192

        # -----------------------------
        # Forward pass: SegGenerator
        # -----------------------------
        seg_input = torch.randn(bs, opt.semantic_nc + 8, seg_H, seg_W, device=device)
        with torch.no_grad():
            seg_out = seg(seg_input)
        report["forward"]["seg_output_shape"] = list(seg_out.shape)

        # -----------------------------
        # Forward pass: GMM
        # -----------------------------
        a_in = torch.randn(bs, 7, seg_H, seg_W, device=device)
        b_in = torch.randn(bs, 3, seg_H, seg_W, device=device)
        with torch.no_grad():
            theta, warped_grid = gmm(a_in, b_in)
        report["forward"]["gmm_theta_shape"] = list(theta.shape)
        report["forward"]["gmm_warped_grid_shape"] = list(warped_grid.shape)

        # -----------------------------
        # Forward pass: ALIASGenerator
        # -----------------------------
        warped_c = torch.randn(bs, 3, LH, LW, device=device)
        img_agnostic = torch.randn(bs, 3, LH, LW, device=device)
        pose = torch.randn(bs, 3, LH, LW, device=device)
        parse = torch.randn(bs, 7, LH, LW, device=device)
        parse_div = torch.randn(bs, 8, LH, LW, device=device)
        misalign_mask = torch.randn(bs, 1, LH, LW, device=device)

        alias_x = torch.cat((img_agnostic, pose, warped_c), dim=1)
        with torch.no_grad():
            alias_out = alias(alias_x, parse, parse_div, misalign_mask)
        report["forward"]["alias_output_shape"] = list(alias_out.shape)

        # -----------------------------
        # Final status
        # -----------------------------
        report["status"] = "ok" if not report["errors"] else "partial_ok"

    except Exception as e:
        report["status"] = "error"
        report["errors"].append(str(e))
        report["traceback"] = traceback.format_exc()

    return jsonify(report)


if __name__ == "__main__":
    print(f"🔥 Running VITON Model Checker on device: {device}")
    app.run(host="0.0.0.0", port=5000, debug=True)
