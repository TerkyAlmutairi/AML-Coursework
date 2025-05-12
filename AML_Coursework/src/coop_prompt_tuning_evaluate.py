# coop_prompt_tuning_evaluate.py (fully fixed + fast subset eval + confidence KDE)

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, ConcatDataset, Subset
from tqdm import tqdm
import clip
import matplotlib.pyplot as plt
import seaborn as sns
import random

# === Setup ===
device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model, preprocess = clip.load("ViT-B/16", device=device)

# === CoOp Prompt Learner ===
class CoOpPromptLearner(nn.Module):
    def __init__(self, classnames, ctx_len=8):
        super().__init__()
        self.ctx_len = ctx_len
        self.classnames = classnames
        self.tokenizer = clip.tokenize
        self.dtype = clip_model.token_embedding.weight.dtype
        self.device = device

        self.n_cls = len(classnames)
        self.ctx = nn.Parameter(torch.randn(self.n_cls, ctx_len, clip_model.ln_final.weight.shape[0]))
        self.prompts = [f"a photo of a {name.replace('_', ' ')}" for name in classnames]
        self.tokenized_prompts = self.tokenizer(self.prompts).to(self.device)

    def forward(self):
        with torch.no_grad():
            token_embeddings = clip_model.token_embedding(self.tokenized_prompts).type(self.dtype)
        prefix = token_embeddings[:, :1, :]
        suffix = token_embeddings[:, 1 + self.ctx_len:, :]
        return torch.cat([prefix, self.ctx, suffix], dim=1)

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

    all_datasets = []
    all_classnames = []
    class_map = {}

    for name, path in dataset_paths.items():
        dataset = datasets.ImageFolder(path, transform=preprocess)
        offset = len(all_classnames)
        classnames = dataset.classes
        all_classnames.extend(classnames)
        all_datasets.append(dataset)
        class_map[name] = (offset, len(classnames))

    return ConcatDataset(all_datasets), all_classnames, class_map

# === Subset sampling based on dataset index ===
def get_subset_indices(concat_dataset, dataset_idx, max_samples=300):
    start = sum(len(ds) for ds in concat_dataset.datasets[:dataset_idx])
    indices = list(range(start, start + len(concat_dataset.datasets[dataset_idx])))
    random.shuffle(indices)
    return indices[:max_samples]

# === Evaluation ===
def evaluate(loader, learner, dataset_name, valid_indices, confidences):
    learner.eval()
    correct, total = 0, 0
    with torch.no_grad():
        prompts = learner()
        text_features = F.normalize(
            prompts[torch.arange(prompts.shape[0]), learner.tokenized_prompts.argmax(dim=-1)],
            dim=-1
        )

        for images, labels in tqdm(loader, desc=f"Evaluating {dataset_name}"):
            images, labels = images.to(device), labels.to(device)
            image_features = F.normalize(clip_model.encode_image(images), dim=-1)
            logits = image_features @ text_features.T
            logits = logits[:, valid_indices]
            probs = F.softmax(logits, dim=1)
            preds = probs.argmax(dim=1)
            confidences.extend(probs.max(dim=1).values.tolist())
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    if total == 0:
        raise RuntimeError(f"No samples evaluated for {dataset_name}. Check dataset paths and labels.")
    return 100 * correct / total

# === Main ===
if __name__ == "__main__":
    full_dataset, all_classnames, class_map = load_all_datasets()
    learner = CoOpPromptLearner(all_classnames).to(device)
    learner.load_state_dict(torch.load("coop_prompt_learned_fast.pth", map_location=device))

    results = {}
    all_confidences = []

    for idx, (name, (offset, count)) in enumerate(class_map.items()):
        valid_indices = list(range(offset, offset + count))
        subset_indices = get_subset_indices(full_dataset, idx, max_samples=1500)

        if not subset_indices:
            raise RuntimeError(f"No samples found for {name}.")

        subset = Subset(full_dataset, subset_indices)
        loader = DataLoader(subset, batch_size=16, shuffle=False)
        confidences = []
        acc = evaluate(loader, learner, name, valid_indices, confidences)
        results[name] = acc
        all_confidences.extend(confidences)

    with open("coop_prompt_eval_log.txt", "w") as f:
        f.write("Evaluation Results - CoOp Prompt Learner (Subset, Fast Mode)\n")
        f.write("=" * 50 + "\n")
        for name, acc in results.items():
            f.write(f"{name}: {acc:.2f}%\n")

    # Accuracy Bar Plot
    plt.figure(figsize=(10, 5))
    plt.bar(results.keys(), results.values(), color='orange')
    plt.ylabel("Accuracy (%)")
    plt.title("CoOp Prompt Tuning Accuracy (Subset, Fast Mode)")
    plt.xticks(rotation=45)
    plt.grid(axis="y")
    plt.tight_layout()
    plt.savefig("coop_prompt_eval_results_subset_fast.png")
    plt.show()

    # KDE Confidence Plot
    if all_confidences:
        plt.figure(figsize=(8, 4))
        sns.kdeplot(all_confidences, fill=True, color='purple', alpha=0.4)
        plt.title("Confidence Distribution (CoOp Prompt Tuning)")
        plt.xlabel("Confidence")
        plt.ylabel("Density")
        plt.grid()
        plt.tight_layout()
        plt.savefig("coop_prompt_confidence_kde_fast.png")
        plt.show()
