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

# === Setup ===
device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model, preprocess = clip.load("ViT-B/16", device=device)

# === MaPLe Prompt Learner ===
class MaPLePromptLearner(nn.Module):
    def __init__(self, ctx_len=8, embed_dim=512, num_prompts=4):
        super().__init__()
        self.prompt_ensemble = nn.Parameter(torch.randn(num_prompts, ctx_len, embed_dim))
        self.adapter = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)
        )

    def forward(self, image_features):
        adapted = self.adapter(image_features)  # (B, D)
        idx = torch.arange(adapted.size(0)) % self.prompt_ensemble.size(0)
        return self.prompt_ensemble[idx]  # (B, ctx_len, D)

# === Load datasets ===
def load_all_datasets():
    data_root = "C:/Users/navee/Downloads/CLIP/data"
    dataset_paths = {
        "Caltech101": os.path.join(data_root, "caltech101", "caltech101", "101_ObjectCategories"),
        "Food-101": os.path.join(data_root, "food-101", "images"),
        "Stanford Cars": os.path.join(data_root, "stanford_cars", "train"),
        "CIFAR-10": os.path.join(data_root, "cifar10_imagefolder", "train"),
        "FGVC Aircraft": os.path.join(data_root, "fgvc_aircraft")
    }
    datasets_map = {}
    for name, path in dataset_paths.items():
        dataset = datasets.ImageFolder(path, transform=preprocess)
        indices = list(range(len(dataset)))
        random.shuffle(indices)
        subset = Subset(dataset, indices[:300])
        datasets_map[name] = subset
    return datasets_map

# === Evaluation ===
def evaluate(model, dataset, classnames):
    loader = DataLoader(dataset, batch_size=32, shuffle=False)
    correct, total = 0, 0
    all_confidences = []

    text_inputs = torch.cat([clip.tokenize(f"a photo of a {c}") for c in classnames]).to(device)
    with torch.no_grad():
        text_features = clip_model.encode_text(text_inputs)
        text_features = F.normalize(text_features, dim=-1)

    model.eval()
    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Evaluating"):
            images, labels = images.to(device), labels.to(device)
            image_features = F.normalize(clip_model.encode_image(images), dim=-1)
            prompts = model(image_features)  # (B, ctx_len, D)
            pooled = prompts.mean(dim=1)     # (B, D)
            logits = pooled @ text_features.T
            probs = F.softmax(logits, dim=1)
            preds = probs.argmax(dim=1)
            all_confidences.extend(probs.max(dim=1).values.tolist())
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return 100 * correct / total, all_confidences

# === Main ===
if __name__ == "__main__":
    model = MaPLePromptLearner().to(device)
    model.load_state_dict(torch.load("maple_prompt_learned_fast.pth", map_location=device))

    datasets_map = load_all_datasets()
    results = {}
    all_confidences = []

    for name, dataset in datasets_map.items():
        print(f"\n🔍 Evaluating {name}...")
        acc, confidences = evaluate(model, dataset, dataset.dataset.classes)
        results[name] = acc
        all_confidences.extend(confidences)

    # Log results
    with open("maple_prompt_eval_log.txt", "w") as f:
        f.write("Evaluation Results - MaPLe Prompt Learner\n")
        f.write("=" * 50 + "\n")
        for name, acc in results.items():
            f.write(f"{name}: {acc:.2f}%\n")

    # Bar Plot
    plt.figure(figsize=(10, 5))
    plt.bar(results.keys(), results.values(), color="mediumseagreen")
    plt.ylabel("Accuracy (%)")
    plt.title("MaPLe Prompt Tuning Accuracy")
    plt.xticks(rotation=45)
    plt.grid(axis="y")
    plt.tight_layout()
    plt.savefig("maple_prompt_eval_bar.png")
    plt.show()

    # KDE Plot
    if all_confidences:
        plt.figure(figsize=(8, 4))
        sns.kdeplot(x=all_confidences, fill=True, color='seagreen', alpha=0.4)
        plt.title("Confidence Distribution (MaPLe Prompt Tuning)")
        plt.xlabel("Confidence")
        plt.ylabel("Density")
        plt.grid()
        plt.tight_layout()
        plt.savefig("maple_prompt_confidence_kde.png")
        plt.show()
