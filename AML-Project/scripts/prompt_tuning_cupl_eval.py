import os
import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import clip
import numpy as np
import matplotlib.pyplot as plt

# === Setup ===
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/16", device=device)

# === LLM-generated CuPL-style prompts ===
def generate_llm_prompts(classname):
    return [
        f"a detailed photo of {classname}",
        f"an image that best represents {classname}",
        f"a realistic depiction of {classname}",
        f"a picture of {classname} in natural setting",
        f"a clear image showing {classname}",
        f"a close-up of {classname}",
        f"{classname} shown in its environment",
        f"an example of {classname}"
    ]

# === Dataset Paths ===
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
    transforms.Normalize((0.48145466, 0.4578275, 0.40821073),
                         (0.26862954, 0.26130258, 0.27577711))
])

class SimpleImageFolder(datasets.ImageFolder):
    def __getitem__(self, index):
        path, label = self.samples[index]
        image = self.loader(path)
        if self.transform is not None:
            image = self.transform(image)
        return image, label

# === Evaluation ===
def evaluate(dataset_name, dataset):
    print(f"\n🔍 Evaluating {dataset_name} with CuPL-style LLM prompts...")
    loader = DataLoader(dataset, batch_size=64, shuffle=False)
    classnames = dataset.classes

    with torch.no_grad():
        text_features = []
        for classname in classnames:
            prompts = generate_llm_prompts(classname.replace("_", " "))
            tokens = clip.tokenize(prompts).to(device)
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
    print(f"✅ {dataset_name} Accuracy (CuPL-style prompts avg): {acc:.2f}%")
    return dataset_name, acc

# === Main ===
results = {}
log_lines = []

for name, path in dataset_paths.items():
    dataset = SimpleImageFolder(path, transform=transform)
    ds_name, acc = evaluate(name, dataset)
    results[ds_name] = acc
    log_lines.append(f"{ds_name}: {acc:.2f}%")

# === Logging ===
log_path = "cupl_style_zero_shot_log.txt"
with open(log_path, "w") as f:
    f.write("CuPL-style Zero-Shot Prompt Evaluation Results\n")
    f.write("=" * 50 + "\n")
    for line in log_lines:
        f.write(line + "\n")

# === Plotting ===
plt.figure(figsize=(10, 5))
plt.bar(results.keys(), results.values(), color="orange")
plt.ylabel("Accuracy (%)")
plt.title("Zero-Shot Accuracy using CuPL-style Prompts")
plt.xticks(rotation=45)
plt.grid(axis="y")
plt.tight_layout()
plt.savefig("cupl_zero_shot_results.png")
plt.show()
