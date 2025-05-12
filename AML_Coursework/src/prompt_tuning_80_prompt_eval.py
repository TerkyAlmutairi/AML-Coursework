import os
import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import clip
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# ==== Setup ====
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/16", device=device)

# ==== Prompt Templates ====
prompt_templates = [
    "a photo of a {}.", "a blurry photo of a {}.", "a black and white photo of a {}.",
    "a bright photo of a {}.", "a dark photo of a {}.", "a cropped photo of a {}.",
    "a close-up photo of a {}.", "a photo of a small {}.", "a photo of a large {}.",
    "a photo of the {}.", "a photo of one {}.", "a photo of many {}.",
    "a photo of a clean {}.", "a photo of a dirty {}.", "a photo of a {} in the wild.",
    "a photo of a {} in the lab.", "a sketch of a {}.", "a painting of a {}.",
    "a cartoon of a {}.", "a plastic {}.", "a toy {}.", "a plushie {}.", "a cartoon {}.",
    "an origami {}.", "a sculpture of a {}.", "a tattoo of a {}.", "a embroidered {}.",
    "a painting of the {}.", "a plastic {} on a table.", "a drawing of the {} on a white background.",
    "a real photo of the {}.", "a rendering of a {}.", "a silhouette of the {}.",
    "a 3d model of a {}.", "a poster of a {}.", "a mural of a {}.", "a photo of a weird {}.",
    "a photo of a {} taken with a DSLR.", "a low resolution photo of the {}.",
    "a photo of a {} with a clean background."
] * 2  # 40 x 2 = 80 prompts

# ==== Dataset Paths ====
data_root = "C:/Users/navee/Downloads/CLIP/data"
dataset_paths = {
    "Caltech101": os.path.join(data_root, "caltech101", "caltech101", "101_ObjectCategories"),
    "Flowers-102": os.path.join(data_root, "flowers-102", "jpg_reorganized"),
    "Food-101": os.path.join(data_root, "food-101", "images"),
    "Stanford Cars": os.path.join(data_root, "stanford_cars", "test"),
    "CIFAR-10": os.path.join(data_root, "cifar10_imagefolder", "test")
}

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
])

class SimpleImageFolder(datasets.ImageFolder):
    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = self.loader(path)
        if self.transform is not None:
            image = self.transform(image)
        return image, label

# ==== Evaluation ====
def evaluate(dataset_name, dataset):
    print(f"\n🔍 Evaluating {dataset_name} with 80 handcrafted prompts...")
    loader = DataLoader(dataset, batch_size=64, shuffle=False)
    classnames = dataset.classes

    with torch.no_grad():
        text_features = []
        for classname in classnames:
            texts = [template.format(classname.replace("_", " ")) for template in prompt_templates]
            tokens = clip.tokenize(texts).to(device)
            embeddings = model.encode_text(tokens)
            embeddings /= embeddings.norm(dim=-1, keepdim=True)
            text_features.append(embeddings.mean(dim=0))
        text_features = torch.stack(text_features).to(device)

    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in tqdm(loader):
            images, labels = images.to(device), labels.to(device)
            image_features = model.encode_image(images)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            logits = image_features @ text_features.T
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    acc = 100 * correct / total
    print(f"✅ {dataset_name} Accuracy (80 prompts avg): {acc:.2f}%")
    return acc

# ==== Main ====
results = {}
for name, path in dataset_paths.items():
    if not os.path.exists(path):
        print(f"❌ Path not found for {name}: {path}")
        continue
    dataset = SimpleImageFolder(path, transform=transform)
    acc = evaluate(name, dataset)
    results[name] = acc

# ==== Plot and Save Results ====
plt.figure(figsize=(10, 5))
plt.bar(results.keys(), results.values(), color="skyblue")
plt.ylabel("Accuracy (%)")
plt.title("Zero-Shot Accuracy using 80 Handcrafted Prompts")
plt.xticks(rotation=45)
plt.grid(axis="y")
plt.tight_layout()
plt.savefig("zero_shot_prompt_ensemble.png")
plt.show()

# Save to log file
log_path = f"80_handcrafted_zero_shot_log.txt"
with open(log_path, "w") as f:
    f.write("Zero-Shot Accuracy Report with 80 Handcrafted Prompts\n")
    f.write("="*50 + "\n")
    for name, acc in results.items():
        f.write(f"{name}: {acc:.2f}%\n")
    f.write("\nChart saved as zero_shot_prompt_ensemble.png\n")

print(f"📄 Results saved to {log_path}")
