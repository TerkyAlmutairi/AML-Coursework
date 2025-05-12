import os
import torch
import torch.nn.functional as F
import clip
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
from torchvision import transforms

# === Setup ===
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/16", device=device)

# === Helper: Overlay heatmap on image ===
def overlay_heatmap_on_image(img, heatmap, alpha=0.5):
    heatmap = np.uint8(255 * heatmap)
    heatmap_colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(np.array(img), 1 - alpha, heatmap_colored, alpha, 0)
    return Image.fromarray(overlay)

# === Grad-CAM Visualization ===
def generate_attention(image_path, text_prompt, save_name):
    image = Image.open(image_path).convert("RGB")
    input_tensor = preprocess(image).unsqueeze(0).to(device).requires_grad_()

    # Tokenize text
    text_tokens = clip.tokenize([text_prompt]).to(device)
    text_features = model.encode_text(text_tokens)
    text_features = F.normalize(text_features, dim=-1)

    # Forward pass
    image_features = model.encode_image(input_tensor)
    image_features = F.normalize(image_features, dim=-1)

    similarity = (image_features @ text_features.T)[0][0]

    # Backward pass
    model.zero_grad()
    similarity.backward()

    # Extract gradient and activations
    grads = input_tensor.grad[0].cpu().numpy()
    gradcam = np.mean(grads, axis=0)
    gradcam = np.maximum(gradcam, 0)
    gradcam = cv2.resize(gradcam, image.size)
    gradcam -= gradcam.min()
    gradcam /= gradcam.max()

    # Overlay heatmap
    result_img = overlay_heatmap_on_image(image, gradcam)

    # Save attention map
    plt.figure(figsize=(6, 6))
    plt.imshow(result_img)
    plt.title(f"GradCAM: {text_prompt}\nSimilarity: {similarity.item():.2f}")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(f"{save_name}.png", dpi=300)
    plt.close()
    print(f"✅ Saved: {save_name}.png")

# === Test Samples ===
samples = {
    "caltech101":     ("test_caltech.jpg",     "a photo of an object from Caltech101"),
    "food-101":       ("test_food.jpg",        "a close-up photo of food on a plate"),
    "stanford_cars":  ("test_cars.jpg",        "a side-view photo of a sports car"),
    "fgvc_aircraft":  ("test_aircraft.jpg",    "a detailed photo of a flying aircraft"),
    "cifar-10":       ("test_cifar.png",       "a centered image of an airplane in sky")
}

# === Generate Attention Maps ===
for key, (filename, prompt) in samples.items():
    generate_attention(filename, prompt, f"attention_map_{key}")
