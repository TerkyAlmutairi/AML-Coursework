import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, ConcatDataset
from tqdm import tqdm
import clip

device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model, preprocess = clip.load("ViT-B/16", device=device)

# === Learnable Prompt Module ===
class CoOpPromptLearner(nn.Module):
    def __init__(self, classnames, ctx_len=8, clip_model=None):
        super().__init__()
        self.ctx_len = ctx_len
        self.n_classes = len(classnames)
        self.tokenizer = clip.tokenize
        self.dtype = clip_model.token_embedding.weight.dtype
        self.device = clip_model.token_embedding.weight.device

        # Learnable context
        self.ctx = nn.Parameter(torch.randn(ctx_len, clip_model.ln_final.weight.shape[0]))

        # Tokenize classnames
        self.classnames = classnames
        self.prompts = [f"a photo of a {name.replace('_', ' ')}" for name in classnames]
        self.tokenized_prompts = self.tokenizer(self.prompts).to(self.device)

    def forward(self):
        with torch.no_grad():
            token_embeddings = clip_model.token_embedding(self.tokenized_prompts).type(self.dtype)  # (C, 77, D)

        prefix = token_embeddings[:, :1, :]  # [SOS]
        suffix = token_embeddings[:, 1 + self.ctx_len:, :]  # Remaining
        ctx = self.ctx.unsqueeze(0).expand(len(self.classnames), -1, -1)
        return torch.cat([prefix, ctx, suffix], dim=1)  # (C, 77, D)

# === Load ImageFolder with names
def load_imagefolder_dataset(path, name):
    transform = preprocess
    dataset = datasets.ImageFolder(path, transform=transform)
    classnames = dataset.classes
    print(f"📦 {name}: {len(dataset)} images | {len(classnames)} classes")
    return dataset, classnames

# === Training Loop
def train_prompt_learner(loader, classnames, device, epochs=10):
    model = CoOpPromptLearner(classnames, ctx_len=8, clip_model=clip_model).to(device)
    optimizer = torch.optim.Adam([model.ctx], lr=1e-3)

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for images, labels in tqdm(loader, desc=f"Epoch {epoch}"):
            images, labels = images.to(device), labels.to(device)

            prompts = model()  # (C, 77, D)
            x = prompts + clip_model.positional_embedding.unsqueeze(0).type(prompts.dtype)
            x = x.permute(1, 0, 2)  # (77, C, D)
            x = clip_model.transformer(x)
            x = x.permute(1, 0, 2)  # (C, 77, D)
            x = clip_model.ln_final(x).type(prompts.dtype)
            text_features = x[torch.arange(x.shape[0]), model.tokenized_prompts.argmax(dim=-1)]
    text_features = F.normalize(text_features, dim=-1)


            image_features = F.normalize(clip_model.encode_image(images), dim=-1)
            logits = image_features @ text_features.T

            loss = F.cross_entropy(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"✅ Epoch {epoch} - Avg Loss: {total_loss / len(loader):.4f}")

    torch.save(model.state_dict(), "coop_prompt_learned.pth")
    print("✅ Saved model to coop_prompt_learned.pth")

# === Main ===
if __name__ == "__main__":
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

    for name, path in dataset_paths.items():
        dataset, classnames = load_imagefolder_dataset(path, name)
        all_datasets.append(dataset)
        all_classnames.extend(classnames)

    combined_dataset = ConcatDataset(all_datasets)
    loader = DataLoader(combined_dataset, batch_size=64, shuffle=True, num_workers=0)

    train_prompt_learner(loader, all_classnames, device)
