import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, ConcatDataset, Subset
from tqdm import tqdm
import clip
import random

# === Setup ===
device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model, preprocess = clip.load("ViT-B/16", device=device)

# === MaPLe Prompt Learner ===
class MaPLePromptLearner(nn.Module):
    def __init__(self, ctx_len=8, embed_dim=512, num_prompts=4):
        super().__init__()
        self.num_prompts = num_prompts
        self.ctx_len = ctx_len
        self.prompt_ensemble = nn.Parameter(torch.randn(num_prompts, ctx_len, embed_dim))
        self.adapter = nn.Sequential(
            nn.Linear(embed_dim, embed_dim), nn.ReLU(), nn.Linear(embed_dim, embed_dim)
        )

    def forward(self, image_features):
        cond = self.adapter(image_features).unsqueeze(1)  # (B, 1, D)
        B = image_features.size(0)
        prompts = self.prompt_ensemble.unsqueeze(0).expand(B, -1, -1, -1)  # (B, P, ctx_len, D)
        prompts = prompts.mean(1)  # average over prompts => (B, ctx_len, D)
        return torch.cat([cond, prompts], dim=1)  # (B, ctx_len+1, D)

# === Load subset from all datasets ===
def load_all_datasets():
    data_root = "C:/Users/navee/Downloads/CLIP/data"
    paths = {
        "Caltech101": os.path.join(data_root, "caltech101", "caltech101", "101_ObjectCategories"),
        "Food-101": os.path.join(data_root, "food-101", "images"),
        "Stanford Cars": os.path.join(data_root, "stanford_cars", "train"),
        "CIFAR-10": os.path.join(data_root, "cifar10_imagefolder", "train"),
        "FGVC Aircraft": os.path.join(data_root, "fgvc_aircraft")
    }
    all_datasets, all_classnames = [], []
    for path in paths.values():
        dataset = datasets.ImageFolder(path, transform=preprocess)
        indices = list(range(len(dataset)))
        random.shuffle(indices)
        subset = Subset(dataset, indices[:300])  # 300 samples per dataset
        all_datasets.append(subset)
        all_classnames.extend(dataset.classes)
    return ConcatDataset(all_datasets), all_classnames

# === Training ===
def train():
    dataset, classnames = load_all_datasets()
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    model = MaPLePromptLearner().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    text_inputs = torch.cat([clip.tokenize(f"a photo of a {c}") for c in classnames]).to(device)
    with torch.no_grad():
        text_features = F.normalize(clip_model.encode_text(text_inputs), dim=-1)

    for epoch in range(1, 4):  # 3 epochs only
        total_loss = 0
        for images, labels in tqdm(loader, desc=f"Epoch {epoch}"):
            images, labels = images.to(device), labels.to(device)
            image_features = F.normalize(clip_model.encode_image(images), dim=-1)

            prompts = model(image_features)  # (B, ctx+1, D)
            pooled = prompts.mean(dim=1)     # (B, D)
            logits = pooled @ text_features.T
            loss = F.cross_entropy(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"\u2705 Epoch {epoch} - Avg Loss: {total_loss / len(loader):.4f}")

    torch.save(model.state_dict(), "maple_prompt_learned_fast.pth")
    print("\u2705 Saved model to maple_prompt_learned_fast.pth")

# === Main ===
if __name__ == "__main__":
    torch.manual_seed(42)
    train()
