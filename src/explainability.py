"""
Grad-CAM explainability module – NEW FEATURE for MSc project.
Visualizes which regions of the crop image the model focuses on for classification.
This addresses the 'black-box' limitation of deep learning models and increases trust
for agricultural end-users (farmers / extension officers).
"""

import io
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torchvision import transforms


class GradCAM:
    """
    Gradient-weighted Class Activation Mapping for CNNs.
    Compatible with ResNet, MobileNet, EfficientNet style models.
    """

    def __init__(self, model: torch.nn.Module, target_layer: Optional[torch.nn.Module] = None):
        self.model = model
        self.model.eval()
        self.gradients = None
        self.activations = None

        # Auto-detect a good target layer (last conv block)
        if target_layer is None:
            target_layer = self._find_target_layer(model)
        self.target_layer = target_layer

        self.hook_handles = []
        self.hook_handles.append(
            self.target_layer.register_forward_hook(self._save_activation)
        )
        self.hook_handles.append(
            self.target_layer.register_full_backward_hook(self._save_gradient)
        )

    def _find_target_layer(self, model):
        # ResNet
        if hasattr(model, "layer4"):
            return model.layer4[-1]
        # MobileNet / EfficientNet
        if hasattr(model, "features"):
            return model.features[-1]
        # Fallback: last conv module
        last_conv = None
        for m in model.modules():
            if isinstance(m, torch.nn.Conv2d):
                last_conv = m
        if last_conv is None:
            raise RuntimeError("Could not find a convolutional layer for Grad-CAM")
        return last_conv

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor: torch.Tensor, class_idx: Optional[int] = None) -> np.ndarray:
        """
        Generate Grad-CAM heatmap.
        input_tensor: (1, C, H, W) normalized tensor
        returns: (H, W) heatmap in [0, 1]
        """
        self.model.zero_grad()
        output = self.model(input_tensor)
        if class_idx is None:
            class_idx = output.argmax(dim=1).item()

        score = output[0, class_idx]
        score.backward()

        gradients = self.gradients  # (1, C, h, w)
        activations = self.activations

        weights = gradients.mean(dim=(2, 3), keepdim=True)  # global average pool
        cam = (weights * activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = cam.squeeze().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam

    def remove_hooks(self):
        for h in self.hook_handles:
            h.remove()


def overlay_cam_on_image(
    original_img: Image.Image,
    cam: np.ndarray,
    alpha: float = 0.45,
) -> Image.Image:
    """Overlay heatmap on original RGB image."""
    cam_resized = np.array(
        Image.fromarray((cam * 255).astype(np.uint8)).resize(original_img.size, Image.BILINEAR)
    )
    # Use jet colormap
    cmap = plt.get_cmap("jet")
    heatmap = (cmap(cam_resized / 255.0)[:, :, :3] * 255).astype(np.uint8)
    heatmap_img = Image.fromarray(heatmap)
    blended = Image.blend(original_img.convert("RGB"), heatmap_img, alpha)
    return blended


def preprocess_for_model(img: Image.Image, img_size: int = 160) -> torch.Tensor:
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return transform(img).unsqueeze(0)


def run_gradcam(
    model: torch.nn.Module,
    image: Image.Image,
    img_size: int = 160,
    class_idx: Optional[int] = None,
    device: str = "cpu",
) -> Tuple[Image.Image, int, float]:
    """
    Full pipeline: preprocess -> predict -> Grad-CAM -> overlay.
    Returns: (overlay_image, predicted_class_idx, confidence)
    """
    model.eval()
    tensor = preprocess_for_model(image, img_size).to(device)
    with torch.no_grad():
        logits = model(tensor)
        probs = F.softmax(logits, dim=1)[0]
        pred_idx = probs.argmax().item()
        conf = probs[pred_idx].item()

    if class_idx is None:
        class_idx = pred_idx

    cam_extractor = GradCAM(model)
    cam = cam_extractor.generate(tensor, class_idx=class_idx)
    cam_extractor.remove_hooks()

    # Resize original for nice display
    display_img = image.convert("RGB").resize((320, 320), Image.BILINEAR)
    overlay = overlay_cam_on_image(display_img, cam)
    return overlay, pred_idx, conf
