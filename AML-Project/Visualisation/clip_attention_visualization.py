import torch
import clip
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms as T
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/16", device=device)

# === Replace this with your image file
image_path = "test.jpg"  # <-- Make sure this image exists in the same folder
prompt = "a photo of a car"

# Load image and preprocess
image = preprocess(Image.open(image_path).convert("RGB")).unsqueeze(0).to(device).requires_grad_()
text = clip.tokenize([prompt]).to(device)

# Encode
with torch.no_grad():
    text_features = model.encode_text(text)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)

image_features = model.encode_image(image)
image_features = image_features / image_features.norm(dim=-1, keepdim=True)

# Compute similarity
logit = (image_features @ text_features.T).squeeze()
logit.backward()

# Gradients
grads = image.grad[0].cpu().numpy()
heatmap = np.mean(grads, axis=0)
heatmap = np.maximum(heatmap, 0)
heatmap /= heatmap.max()

# Overlay
raw_img = Image.open(image_path).resize((224, 224)).convert("RGB")
plt.imshow(raw_img)
plt.imshow(heatmap, cmap='jet', alpha=0.5)
plt.title(f"Attention: {prompt}")
plt.axis('off')
plt.tight_layout()
plt.savefig("clip_attention_heatmap.png")
plt.show()
