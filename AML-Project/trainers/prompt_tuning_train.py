import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
from torch.utils.data import DataLoader, Dataset
from transformers import BertModel, BertTokenizer
from PIL import Image
from tqdm import tqdm

# ==== MiniCLIP with Learnable Prompts (Improved Version) ====
class MiniCLIPPromptTuning(nn.Module):
    def __init__(self, num_prompts=16):
        super().__init__()
        self.image_encoder = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.image_encoder.fc = nn.Identity()
        self.text_encoder = BertModel.from_pretrained("bert-base-uncased")

        self.image_proj = nn.Sequential(
            nn.Linear(2048, 512),
            nn.LayerNorm(512)
        )
        self.text_proj = nn.Sequential(
            nn.Linear(768, 512),
            nn.LayerNorm(512)
        )

        self.learned_prompts = nn.Parameter(torch.randn(num_prompts, 768))

    def forward(self, image, input_ids, attention_mask):
        image_features = F.normalize(self.image_proj(self.image_encoder(image)), dim=1)
        text_output = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        text_features = F.normalize(self.text_proj(text_output.pooler_output), dim=1)
        return image_features, text_features

# ==== Custom Dataset Loader ====
class SimpleImageFolder(Dataset):
    def __init__(self, root, transform=None):
        self.samples = []
        self.labels = []
        self.classes = sorted([d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))])
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}

        for cls in self.classes:
            cls_folder = os.path.join(root, cls)
            for fname in os.listdir(cls_folder):
                if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                    self.samples.append(os.path.join(cls_folder, fname))
                    self.labels.append(self.class_to_idx[cls])

        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img = Image.open(self.samples[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, self.labels[idx]

# ==== Training Setup ====
if __name__ == "__main__":
    data_root = "C:/Users/navee/Downloads/CLIP/data"

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    # Dataset paths
    caltech_path = os.path.join(data_root, "caltech101", "caltech101", "101_ObjectCategories")
    flowers_path = os.path.join(data_root, "flowers-102", "jpg")
    food_path = os.path.join(data_root, "food-101", "images")
    cars_path = os.path.join(data_root, "stanford_cars", "test")
    cifar_path = os.path.join(data_root, "cifar10_imagefolder", "train")

    all_datasets = []
    for path in [caltech_path, flowers_path, food_path, cars_path, cifar_path]:
        all_datasets.append(SimpleImageFolder(path, transform=transform))

    full_dataset = torch.utils.data.ConcatDataset(all_datasets)
    train_loader = DataLoader(full_dataset, batch_size=32, shuffle=True, num_workers=4)

    model = MiniCLIPPromptTuning(num_prompts=16).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    print("🧾 Total classes:", sum([len(ds.classes) for ds in all_datasets]))
    print("🎯 Starting Enhanced CoOp-style Prompt Tuning...")

    for epoch in range(20):
        model.train()
        total_loss = 0.0

        for images, labels in tqdm(train_loader):
            images, labels = images.cuda(), labels.cuda()
            optimizer.zero_grad()

            class_names = [f"a photo of a {label}" for label in labels.cpu().tolist()]
            tokenized = tokenizer(class_names, padding=True, truncation=True, return_tensors="pt")
            input_ids = tokenized["input_ids"].cuda()
            attention_mask = tokenized["attention_mask"].cuda()

            image_features, text_features = model(images, input_ids, attention_mask)
            logits = image_features @ text_features.T / 0.07  # Temperature scaling

            targets = torch.arange(len(images)).cuda()
            loss = F.cross_entropy(logits, targets)

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch + 1} - Loss: {total_loss / len(train_loader):.4f}")

    torch.save(model.state_dict(), "prompt_tuned_clip_improved.pth")
    print("✅ Improved prompt-tuned MiniCLIP model saved.")
