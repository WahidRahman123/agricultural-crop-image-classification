"""
Interactive Agricultural Crop Classifier + Explainable AI + Farming Recommendations
NEW FEATURE for MSc Project.

Upload a crop image → get:
  1. Predicted crop class + confidence
  2. Grad-CAM visualization (where the model looks)
  3. Practical agricultural recommendations for that crop
"""

import sys
from pathlib import Path
import torch
from PIL import Image
import gradio as gr

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from models import create_model
from explainability import run_gradcam
from dataset import CLASS_NAMES, IDX_TO_CLASS

# Farming tips (knowledge base) – practical value for end users
CROP_RECOMMENDATIONS = {
    "almond": {
        "climate": "Mediterranean / temperate, needs chill hours",
        "soil": "Well-drained sandy loam, pH 6.0–7.5",
        "water": "Moderate; critical during flowering & nut development",
        "tips": [
            "Plant in full sun with good air circulation",
            "Prune for open canopy to reduce disease pressure",
            "Watch for navel orangeworm and bacterial leaf spot",
            "Harvest when hulls split; dry properly to avoid aflatoxin",
        ],
    },
    "coconut": {
        "climate": "Tropical, high humidity, 27–32°C",
        "soil": "Sandy coastal soils, good drainage, pH 5.5–7.0",
        "water": "High water requirement; sensitive to drought",
        "tips": [
            "Needs 2000+ mm annual rainfall or irrigation",
            "Apply potash-rich fertilizer for better nut yield",
            "Protect young palms from rhinoceros beetle",
            "Intercrop with bananas or pineapples in early years",
        ],
    },
    "sunflower": {
        "climate": "Temperate to subtropical, full sun",
        "soil": "Well-drained loam, pH 6.0–7.5",
        "water": "Moderate; drought tolerant once established",
        "tips": [
            "Rotate with cereals to break disease cycles",
            "Plant after last frost; spacing 20–30 cm",
            "Bird damage is common – use netting if needed",
            "Harvest when back of head turns yellow-brown",
        ],
    },
    "tomato": {
        "climate": "Warm season, 20–30°C ideal",
        "soil": "Fertile, well-drained, rich in organic matter, pH 6.0–6.8",
        "water": "Consistent moisture; avoid water stress at flowering",
        "tips": [
            "Stake or cage plants for better air flow & fruit quality",
            "Mulch to conserve moisture and suppress weeds",
            "Watch for early blight, late blight, and tomato hornworm",
            "Harvest at breaker stage for longest shelf life",
        ],
    },
    "vigna_radiati_mung": {
        "climate": "Warm tropical / subtropical",
        "soil": "Sandy loam to clay loam, pH 6.2–7.2",
        "water": "Low to moderate; sensitive to waterlogging",
        "tips": [
            "Short duration crop (60–90 days) – good for intercropping",
            "Fix nitrogen – improves soil for following crop",
            "Control aphids and yellow mosaic virus",
            "Harvest pods when they turn black and dry",
        ],
    },
    "wheat": {
        "climate": "Cool season crop, needs cool weather for tillering",
        "soil": "Loam to clay loam, good fertility, pH 6.0–7.5",
        "water": "Critical at crown root initiation, flowering, grain filling",
        "tips": [
            "Use certified seed and recommended varieties for your zone",
            "Apply balanced NPK; split nitrogen application",
            "Monitor for rusts, aphids, and weeds",
            "Harvest at physiological maturity (grain moisture ~20–25%)",
        ],
    },
}


def load_model(checkpoint_path: str, device: str = "cpu"):
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model_name = ckpt.get("model_name", "resnet18")
    img_size = ckpt.get("img_size", 160)
    model = create_model(model_name, num_classes=len(CLASS_NAMES), pretrained=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    return model, img_size, ckpt.get("class_names", CLASS_NAMES)


def format_recommendation(class_name: str) -> str:
    rec = CROP_RECOMMENDATIONS.get(class_name, {})
    if not rec:
        return "No specific recommendations available for this class."
    lines = [
        f"### 🌱 Crop Care Guide: **{class_name.replace('_', ' ').title()}**",
        f"- **Climate**: {rec.get('climate', 'N/A')}",
        f"- **Soil**: {rec.get('soil', 'N/A')}",
        f"- **Water**: {rec.get('water', 'N/A')}",
        "",
        "**Practical Tips:**",
    ]
    for t in rec.get("tips", []):
        lines.append(f"- {t}")
    return "\n".join(lines)


# Global model (loaded once)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = ROOT / "models" / "best_resnet18.pth"
MODEL = None
IMG_SIZE = 160


def ensure_model():
    global MODEL, IMG_SIZE
    if MODEL is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Trained model not found at {MODEL_PATH}. "
                "Please run training first: python src/train.py"
            )
        MODEL, IMG_SIZE, _ = load_model(str(MODEL_PATH), DEVICE)
    return MODEL


def predict(image: Image.Image):
    if image is None:
        return None, "Please upload an image.", ""
    model = ensure_model()
    overlay, pred_idx, conf = run_gradcam(model, image, img_size=IMG_SIZE, device=DEVICE)
    class_name = IDX_TO_CLASS[pred_idx]
    conf_pct = conf * 100
    label = f"**Predicted Crop:** {class_name.replace('_', ' ').title()}  \n**Confidence:** {conf_pct:.1f}%"
    rec = format_recommendation(class_name)
    return overlay, label, rec


def build_demo():
    with gr.Blocks(title="Agricultural Crop Classifier – MSc Project", theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            """
            # 🌾 Agricultural Crops Image Classification
            ### MSc Project – Military Institute of Science and Technology (MIST)
            **New Features:** Grad-CAM Explainability + Crop Care Recommendations
            
            Upload a photo of a crop (leaf / plant / field view). The model will:
            1. Classify the crop type
            2. Show **where** it looked (Grad-CAM heatmap)
            3. Give practical farming advice for that crop
            """
        )
        with gr.Row():
            with gr.Column(scale=1):
                inp = gr.Image(type="pil", label="Upload Crop Image")
                btn = gr.Button("Classify & Explain", variant="primary")
            with gr.Column(scale=1):
                out_img = gr.Image(type="pil", label="Grad-CAM Visualization")
                out_label = gr.Markdown(label="Prediction")
                out_rec = gr.Markdown(label="Recommendations")
        btn.click(fn=predict, inputs=inp, outputs=[out_img, out_label, out_rec])
        gr.Markdown(
            """
            ---
            **Classes supported:** Almond, Coconut, Sunflower, Tomato, Vigna radiata (Mung), Wheat  
            **Model:** ResNet-18 (Transfer Learning) + Advanced Data Augmentation  
            **Explainability:** Grad-CAM  
            """
        )
    return demo


if __name__ == "__main__":
    demo = build_demo()
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
