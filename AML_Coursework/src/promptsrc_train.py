import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import clip
import random

# === Setup ===
device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model, preprocess = clip.load("ViT-B/16", device=device)

# === PromptSRC Module (Simplified) ===
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
        adapted = self.adapter(image_features).unsqueeze(1)
        prompt = self.ctx.unsqueeze(0).expand(image_features.size(0), -1, -1)
        return torch.cat([adapted, prompt], dim=1)  # (B, ctx_len+1, D)

# === Load limited samples per dataset ===
def load_dataset(path):
    dataset = datasets.ImageFolder(path, transform=preprocess)
    indices = list(range(len(dataset)))
    random.shuffle(indices)
    subset = Subset(dataset, indices[:300])  # fast training subset
    return subset, dataset.classes

# === Training ===
def train():
    data_root = "C:/Users/navee/Downloads/CLIP/data"
    dataset_paths = [
        os.path.join(data_root, "caltech101", "caltech101", "101_ObjectCategories"),
        os.path.join(data_root, "food-101", "images"),
        os.path.join(data_root, "stanford_cars", "train"),
        os.path.join(data_root, "cifar10_imagefolder", "train"),
        os.path.join(data_root, "fgvc_aircraft")
    ]

    all_datasets = []
    all_classnames = []

    for path in dataset_paths:
        ds, names = load_dataset(path)
        all_datasets.append(ds)
        all_classnames.extend(names)

    full_dataset = torch.utils.data.ConcatDataset(all_datasets)
    loader = DataLoader(full_dataset, batch_size=32, shuffle=True)

    model = PromptSRC(ctx_len=8).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    text_inputs = torch.cat([clip.tokenize(f"a photo of a {c}") for c in all_classnames]).to(device)
    with torch.no_grad():
        text_features = F.normalize(clip_model.encode_text(text_inputs), dim=-1)

    for epoch in range(3):  # faster training
        total_loss = 0
        for images, labels in tqdm(loader, desc=f"Epoch {epoch+1}"):
            images, labels = images.to(device), labels.to(device)
            image_features = F.normalize(clip_model.encode_image(images), dim=-1)

            prompts = model(image_features)  # (B, ctx_len+1, D)
            pooled = prompts.mean(dim=1)
            logits = pooled @ text_features.T
            loss = F.cross_entropy(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"✅ Epoch {epoch+1} - Avg Loss: {total_loss / len(loader):.4f}")

    torch.save(model.state_dict(), "promptsrc_learned_fast.pth")
    print("✅ Saved: promptsrc_learned_fast.pth")

if __name__ == "__main__":
    train()
