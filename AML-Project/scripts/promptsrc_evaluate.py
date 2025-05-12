# promptsrc_evaluate.py

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import clip
import matplotlib.pyplot as plt
import seaborn as sns
import random

device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model, preprocess = clip.load("ViT-B/16", device=device)

class PromptSRC(nn.Module):
    def __init__(self, ctx_len=8, embed_dim=512):
        super().__init__()
        self.ctx = nn.Parameter(torch.randn(ctx_len, embed_dim))
        self.adapter = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)
        )

    def forward(self, image_features):
        image_cond = self.adapter(image_features).unsqueeze(1)
        prompt = self.ctx.unsqueeze(0).expand(image_features.size(0), -1, -1)
        return torch.cat([image_cond, prompt], dim=1)


def load_datasets():
    data_root = "C:/Users/navee/Downloads/CLIP/data"
    paths = {
        "Caltech101": os.path.join(data_root, "caltech101", "caltech101", "101_ObjectCategories"),
        "Food-101": os.path.join(data_root, "food-101", "images"),
        "Stanford Cars": os.path.join(data_root, "stanford_cars", "train"),
        "CIFAR-10": os.path.join(data_root, "cifar10_imagefolder", "train"),
        "FGVC Aircraft": os.path.join(data_root, "fgvc_aircraft")
    }
    datasets_map = {}
    for name, path in paths.items():
        full_ds = datasets.ImageFolder(path, transform=preprocess)
        indices = list(range(len(full_ds)))
        random.shuffle(indices)
        subset = Subset(full_ds, indices[:300])
        datasets_map[name] = subset
    return datasets_map

def evaluate(model, dataset, classnames):
    loader = DataLoader(dataset, batch_size=32, shuffle=False)
    correct = 0
    total = 0
    all_confidences = []

    with torch.no_grad():
        text_inputs = torch.cat([clip.tokenize(f"a photo of a {c}") for c in classnames]).to(device)
        text_features = clip_model.encode_text(text_inputs)
        text_features = F.normalize(text_features, dim=-1)

        for images, labels in tqdm(loader):
            images, labels = images.to(device), labels.to(device)
            image_features = clip_model.encode_image(images)
            image_features = F.normalize(image_features, dim=-1)
            prompts = model(image_features)
            pooled = prompts.mean(dim=1)
            logits = pooled @ text_features.T
            probs = F.softmax(logits, dim=1)
            preds = probs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            all_confidences.extend(probs.max(dim=1).values.tolist())

    acc = 100 * correct / total
    return acc, all_confidences

if __name__ == "__main__":
    torch.manual_seed(42)
    datasets_map = load_datasets()
    model = PromptSRC().to(device)
    model.load_state_dict(torch.load("promptsrc_learned_fast.pth", map_location=device))
    model.eval()

    results = {}
    all_confidences = []

    for name, subset in datasets_map.items():
        print(f"\n🔍 Evaluating {name}...")
        classnames = subset.dataset.classes
        acc, conf = evaluate(model, subset, classnames)
        results[name] = acc
        all_confidences.extend(conf)

    with open("promptsrc_eval_log.txt", "w") as f:
        f.write("Evaluation Results - PromptSRC Learner\n")
        f.write("=" * 50 + "\n")
        for name, acc in results.items():
            f.write(f"{name}: {acc:.2f}%\n")

    # Accuracy Bar Plot
    plt.figure(figsize=(10, 5))
    plt.bar(results.keys(), results.values(), color="cyan")
    plt.ylabel("Accuracy (%)")
    plt.title("PromptSRC Accuracy on 5 Datasets")
    plt.xticks(rotation=45)
    plt.grid(axis="y")
    plt.tight_layout()
    plt.savefig("promptsrc_eval_bar.png")
    plt.show()

    # KDE Plot
    if all_confidences:
        plt.figure(figsize=(8, 4))
        sns.kdeplot(x=all_confidences, fill=True, color='blue', alpha=0.5)
        plt.title("Confidence Distribution (PromptSRC)")
        plt.xlabel("Confidence")
        plt.ylabel("Density")
        plt.grid()
        plt.tight_layout()
        plt.savefig("promptsrc_confidence_kde.png")
        plt.show()
