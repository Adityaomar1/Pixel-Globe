"""
Cosmic Orbital Change Detection Web Application (Backend Server)
Dual-Model PyTorch Inference Engine with Flask API
Supports: TinyCD & Siamese U-Net Architectures
"""

import os
import io
import time
import base64
import numpy as np
from PIL import Image
import cv2

import torch
import torchvision.transforms.functional as TF
from flask import Flask, render_template, request, jsonify, send_from_directory

# Import model architectures
from models.tinycd import TinyCD
from models.siamese_unet import SiameseUNet

app = Flask(__name__)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHECKPOINT_DIR = os.path.join(BASE_DIR, 'checkpoints')
SAMPLES_DIR = os.path.join(BASE_DIR, 'static', 'samples')

# Check CUDA availability
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DEVICE_NAME = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU Core"

# Model Registry
MODELS_CONFIG = {
    "model_1": {
        "id": "model_1",
        "name": "Orbital-CD Alpha (TinyCD)",
        "type": "tinycd",
        "path": os.path.join(CHECKPOINT_DIR, "best_model.pth"),
        "description": "Optimized TinyCD with EfficientNet-B0 backbone and feature attention.",
        "badge": "EFFICIENTNET-B0"
    },
    "model_2": {
        "id": "model_2",
        "name": "Siam-UNet Beta (Siamese U-Net)",
        "type": "siamese_unet",
        "path": os.path.join(CHECKPOINT_DIR, "siamese_unet.pth"),
        "description": "Siamese U-Net architecture with multi-scale differential skip connections.",
        "badge": "SIAMESE U-NET"
    }
}

class ModelManager:
    def __init__(self):
        self.loaded_models = {}
        self.active_model_id = "model_1"
        self.device = DEVICE
        self.init_default_models()

    def init_default_models(self):
        """Preload models if checkpoint files exist."""
        for m_id, cfg in MODELS_CONFIG.items():
            if os.path.exists(cfg['path']):
                try:
                    self.load_model(m_id, cfg['path'], cfg['type'])
                    print(f"[*] Loaded {cfg['name']} onto {self.device}")
                except Exception as e:
                    print(f"[!] Warning loading {m_id}: {e}")

    def load_model(self, model_id, checkpoint_path, model_type="tinycd"):
        """Loads or reloads a model from a checkpoint path."""
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        if model_type == "siamese_unet":
            model = SiameseUNet().to(self.device)
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            elif isinstance(checkpoint, dict) and 'model_state' in checkpoint:
                model.load_state_dict(checkpoint['model_state'])
            else:
                model.load_state_dict(checkpoint)
            model.eval()
            
            self.loaded_models[model_id] = {
                "model": model,
                "type": "siamese_unet",
                "path": checkpoint_path,
                "backbone": "Siamese DoubleConv (512)",
                "epoch": "Final",
                "best_f1": "Optimized",
                "best_iou": "Optimized"
            }
        else: # tinycd
            args = checkpoint.get('args', {}) if isinstance(checkpoint, dict) else {}
            backbone = args.get('backbone', 'efficientnet_b0')
            model = TinyCD(backbone_name=backbone, pretrained=False, num_classes=1).to(self.device)
            if isinstance(checkpoint, dict) and 'model_state' in checkpoint:
                model.load_state_dict(checkpoint['model_state'])
            else:
                model.load_state_dict(checkpoint)
            model.eval()

            epoch = checkpoint.get('epoch', 'N/A') if isinstance(checkpoint, dict) else 'N/A'
            best_f1 = checkpoint.get('best_f1', None) if isinstance(checkpoint, dict) else None
            best_iou = checkpoint.get('best_iou', None) if isinstance(checkpoint, dict) else None

            self.loaded_models[model_id] = {
                "model": model,
                "type": "tinycd",
                "path": checkpoint_path,
                "backbone": backbone,
                "epoch": epoch,
                "best_f1": f"{best_f1:.4f}" if isinstance(best_f1, (float, int)) else "N/A",
                "best_iou": f"{best_iou:.4f}" if isinstance(best_iou, (float, int)) else "N/A"
            }

        return self.loaded_models[model_id]

    def set_active_model(self, model_id, custom_path=None):
        """Switches active model or loads custom checkpoint."""
        if custom_path:
            m_info = self.load_model("custom", custom_path, "tinycd")
            self.active_model_id = "custom"
            return m_info

        if model_id not in self.loaded_models:
            if model_id in MODELS_CONFIG and os.path.exists(MODELS_CONFIG[model_id]['path']):
                self.load_model(model_id, MODELS_CONFIG[model_id]['path'], MODELS_CONFIG[model_id]['type'])
            else:
                raise ValueError(f"Model ID '{model_id}' is not loaded or available.")

        self.active_model_id = model_id
        return self.loaded_models[model_id]

    def get_active_model_info(self):
        if self.active_model_id not in self.loaded_models:
            if self.loaded_models:
                self.active_model_id = next(iter(self.loaded_models.keys()))
            else:
                raise RuntimeError("No model is currently loaded in memory.")
        return self.loaded_models[self.active_model_id]


model_mgr = ModelManager()


# --- Image Preprocessing & Conversion ---

def pil_to_base64(pil_img, format='PNG'):
    buffered = io.BytesIO()
    pil_img.save(buffered, format=format, optimize=True)
    img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
    return f"data:image/{format.lower()};base64,{img_str}"


def preprocess_image(pil_img, model_type="tinycd", img_size=(256, 256)):
    """Preprocesses PIL image tailored to specific model normalization."""
    img_rgb = pil_img.convert('RGB')
    orig_size = img_rgb.size # (W, H)
    img_resized = TF.resize(img_rgb, img_size, interpolation=TF.InterpolationMode.BILINEAR)

    if model_type == "siamese_unet":
        # Siamese UNet uses [0, 1] normalization (img / 255.0)
        img_np = np.array(img_resized).astype('float32') / 255.0
        tensor = torch.tensor(img_np, dtype=torch.float32).permute(2, 0, 1)
    else:
        # TinyCD uses ImageNet normalization
        tensor = TF.to_tensor(img_resized)
        tensor = TF.normalize(tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    return tensor.unsqueeze(0), img_rgb, orig_size


def hex_to_rgb(hex_str):
    hex_str = hex_str.lstrip('#')
    if len(hex_str) == 3:
        hex_str = ''.join([c*2 for c in hex_str])
    if len(hex_str) != 6:
        return (255, 0, 85)
    return tuple(int(hex_str[i:i+2], 16) for i in (0, 2, 4))


def generate_visualizations(t1_pil, t2_pil, probs_np, threshold=0.5, overlay_hex="#FF0055", alpha=0.5):
    orig_w, orig_h = t2_pil.size

    # 1. Binary Mask
    probs_resized = cv2.resize(probs_np, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
    binary_mask = (probs_resized > threshold).astype(np.uint8) * 255
    mask_pil = Image.fromarray(binary_mask)

    # 2. Glowing Neon Overlay on T2
    t2_np = np.array(t2_pil).astype(np.float32)
    overlay_rgb = hex_to_rgb(overlay_hex)
    overlay_np = t2_np.copy()
    
    change_pixels = binary_mask > 127
    for c in range(3):
        overlay_np[change_pixels, c] = (1 - alpha) * overlay_np[change_pixels, c] + alpha * overlay_rgb[c]
    
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    overlay_uint8 = np.clip(overlay_np, 0, 255).astype(np.uint8)
    cv2.drawContours(overlay_uint8, contours, -1, overlay_rgb, 2, cv2.LINE_AA)
    overlay_pil = Image.fromarray(overlay_uint8)

    # 3. Cosmic Continuous Heatmap (TURBO / JET)
    heatmap_colored = cv2.applyColorMap((probs_resized * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    heatmap_rgb = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    heatmap_pil = Image.fromarray(heatmap_rgb)

    # 4. 3-Panel Synoptic Composite
    target_h = 300
    w1 = int(t1_pil.width * (target_h / t1_pil.height))
    w2 = int(t2_pil.width * (target_h / t2_pil.height))
    w3 = int(overlay_pil.width * (target_h / overlay_pil.height))

    t1_thumb = t1_pil.resize((w1, target_h), Image.Resampling.LANCZOS)
    t2_thumb = t2_pil.resize((w2, target_h), Image.Resampling.LANCZOS)
    ov_thumb = overlay_pil.resize((w3, target_h), Image.Resampling.LANCZOS)

    total_w = w1 + w2 + w3 + 20
    composite = Image.new('RGB', (total_w, target_h + 40), (10, 14, 26))
    composite.paste(t1_thumb, (0, 35))
    composite.paste(t2_thumb, (w1 + 10, 35))
    composite.paste(ov_thumb, (w1 + w2 + 20, 35))

    total_pixels = orig_w * orig_h
    changed_count = int(np.sum(change_pixels))
    changed_pct = round((changed_count / total_pixels) * 100, 2)

    anomaly_level = "LOW"
    if changed_pct > 15:
        anomaly_level = "CRITICAL"
    elif changed_pct > 5:
        anomaly_level = "MODERATE"
    elif changed_pct > 0.5:
        anomaly_level = "ELEVATED"

    return {
        "mask_base64": pil_to_base64(mask_pil),
        "overlay_base64": pil_to_base64(overlay_pil),
        "heatmap_base64": pil_to_base64(heatmap_pil),
        "composite_base64": pil_to_base64(composite),
        "stats": {
            "changed_pixels": changed_count,
            "total_pixels": total_pixels,
            "changed_percentage": changed_pct,
            "anomaly_level": anomaly_level,
            "resolution": f"{orig_w} x {orig_h}"
        }
    }


# --- Routes ---

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/info', methods=['GET'])
def api_info():
    samples = []
    if os.path.exists(SAMPLES_DIR):
        files = os.listdir(SAMPLES_DIR)
        t1_files = sorted([f for f in files if f.startswith('t1_')])
        for t1 in t1_files:
            pair_id = t1[3:]
            t2 = f"t2_{pair_id}"
            if t2 in files:
                samples.append({
                    "id": pair_id.split('.')[0],
                    "name": f"Orbital Sector #{pair_id.split('.')[0]}",
                    "t1_url": f"/static/samples/{t1}",
                    "t2_url": f"/static/samples/{t2}"
                })

    models_info = []
    for m_id, cfg in MODELS_CONFIG.items():
        loaded_info = model_mgr.loaded_models.get(m_id, {})
        models_info.append({
            "id": m_id,
            "name": cfg['name'],
            "description": cfg['description'],
            "badge": cfg['badge'],
            "is_active": (model_mgr.active_model_id == m_id),
            "is_loaded": (m_id in model_mgr.loaded_models),
            "epoch": loaded_info.get('epoch', 'N/A'),
            "best_f1": loaded_info.get('best_f1', 'N/A'),
            "best_iou": loaded_info.get('best_iou', 'N/A'),
            "backbone": loaded_info.get('backbone', cfg['badge'])
        })

    return jsonify({
        "status": "online",
        "device": str(DEVICE).upper(),
        "device_name": DEVICE_NAME,
        "active_model_id": model_mgr.active_model_id,
        "models": models_info,
        "samples": samples
    })


@app.route('/api/switch_model', methods=['POST'])
def switch_model():
    data = request.get_json(silent=True) or request.form
    model_id = data.get('model_id')
    custom_path = data.get('custom_path')

    try:
        m_info = model_mgr.set_active_model(model_id, custom_path)
        return jsonify({
            "success": True,
            "message": f"Successfully activated {model_id}",
            "active_model_id": model_mgr.active_model_id,
            "info": {
                "epoch": m_info.get('epoch'),
                "best_f1": m_info.get('best_f1'),
                "best_iou": m_info.get('best_iou'),
                "backbone": m_info.get('backbone')
            }
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400


@app.route('/api/predict', methods=['POST'])
def predict():
    start_time = time.time()

    req_model_id = request.form.get('model_id')
    if req_model_id and req_model_id in model_mgr.loaded_models:
        model_mgr.active_model_id = req_model_id

    try:
        m_info = model_mgr.get_active_model_info()
        model = m_info['model']
        model_type = m_info['type']
        active_id = model_mgr.active_model_id
    except Exception as e:
        return jsonify({"success": False, "error": f"Model error: {e}"}), 500

    t1_file = request.files.get('image_t1')
    t2_file = request.files.get('image_t2')
    
    if not t1_file or not t2_file:
        data = request.get_json(silent=True) or {}
        t1_b64 = data.get('image_t1')
        t2_b64 = data.get('image_t2')
        if t1_b64 and t2_b64:
            if ',' in t1_b64: t1_b64 = t1_b64.split(',')[1]
            if ',' in t2_b64: t2_b64 = t2_b64.split(',')[1]
            t1_pil = Image.open(io.BytesIO(base64.b64decode(t1_b64))).convert('RGB')
            t2_pil = Image.open(io.BytesIO(base64.b64decode(t2_b64))).convert('RGB')
        else:
            return jsonify({"success": False, "error": "Both 'image_t1' and 'image_t2' images are required."}), 400
    else:
        try:
            t1_pil = Image.open(t1_file.stream).convert('RGB')
            t2_pil = Image.open(t2_file.stream).convert('RGB')
        except Exception as e:
            return jsonify({"success": False, "error": f"Invalid image format: {e}"}), 400

    threshold = float(request.form.get('threshold', 0.5))
    overlay_hex = request.form.get('overlay_color', '#FF0055')
    alpha = float(request.form.get('alpha', 0.5))

    # Preprocess matching active model requirements
    t1_tensor, t1_orig_pil, orig_size = preprocess_image(t1_pil, model_type=model_type)
    t2_tensor, t2_orig_pil, _ = preprocess_image(t2_pil, model_type=model_type)

    t1_tensor = t1_tensor.to(DEVICE)
    t2_tensor = t2_tensor.to(DEVICE)

    try:
        with torch.no_grad():
            logits = model(t1_tensor, t2_tensor)
            probs = torch.sigmoid(logits).squeeze().cpu().numpy()
    except Exception as e:
        return jsonify({"success": False, "error": f"Inference execution failed: {e}"}), 500

    results = generate_visualizations(
        t1_orig_pil, t2_orig_pil, probs,
        threshold=threshold,
        overlay_hex=overlay_hex,
        alpha=alpha
    )

    inference_ms = round((time.time() - start_time) * 1000, 1)
    results['stats']['latency_ms'] = inference_ms
    results['stats']['model_id'] = active_id
    results['stats']['model_name'] = MODELS_CONFIG.get(active_id, {}).get('name', active_id)

    return jsonify({
        "success": True,
        "data": results
    })


if __name__ == '__main__':
    print("="*60)
    print("COSMOS ORBITAL CHANGE DETECTION SERVER")
    print(f"Device: {DEVICE_NAME} ({str(DEVICE).upper()})")
    print(f"Loaded Models: {list(model_mgr.loaded_models.keys())}")
    print("Web Server Running at: http://127.0.0.1:5000")
    print("="*60)
    app.run(host='0.0.0.0', port=5000, debug=False)
