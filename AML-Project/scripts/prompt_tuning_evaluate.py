import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
from torch.utils.data import DataLoader, Dataset
from transformers import BertModel, BertTokenizer
from PIL import Image
from tqdm import tqdm
from datetime import datetime

# ==== MiniCLIP Model ====
class MiniCLIPPromptTuning(nn.Module):
    def __init__(self, num_prompts=4):
        super().__init__()
        self.image_encoder = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.image_encoder.fc = nn.Identity()
        self.text_encoder = BertModel.from_pretrained("bert-base-uncased")
        self.image_proj = nn.Linear(2048, 512)
        self.text_proj = nn.Linear(768, 512)
        self.learned_prompts = nn.Parameter(torch.randn(num_prompts, 768))

    def forward(self, image, input_ids, attention_mask):
        image_features = F.normalize(self.image_proj(self.image_encoder(image)), dim=1)
        text_output = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        text_features = F.normalize(self.text_proj(text_output.pooler_output), dim=1)
        return image_features, text_features

# ==== Dataset Loader ====
class SimpleImageFolder(Dataset):
    def __init__(self, root, transform=None):
        if not os.path.isdir(root):
            raise RuntimeError(f"Provided root path is not a directory: {root}")

        self.samples = []
        self.labels = []
        self.classes = sorted([d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))])
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}

        if not self.classes:
            raise RuntimeError(f"No class folders found in {root}")

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

# ==== Evaluation ====
def evaluate_dataset(name, dataset, model, tokenizer, device, log_file):
    print(f"\n🔍 Evaluating on {name}...")
    loader = DataLoader(dataset, batch_size=64, shuffle=False)
    class_names = dataset.classes
    prompts = [f"a photo of a {label}" for label in class_names]
    
    tokens = tokenizer(prompts, padding=True, truncation=True, return_tensors="pt")
    input_ids = tokens['input_ids'].to(device)
    attention_mask = tokens['attention_mask'].to(device)

    model.eval()
    with torch.no_grad():
        text_features = model.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        text_features = F.normalize(model.text_proj(text_features.pooler_output), dim=1)

    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in tqdm(loader, desc=f"{name}"):
            images, labels = images.to(device), labels.to(device)
            image_features = F.normalize(model.image_proj(model.image_encoder(images)), dim=1)
            logits = image_features @ text_features.T
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    acc = 100 * correct / total
    result = f"✅ Accuracy on {name}: {acc:.2f}%"
    print(result)
    with open(log_file, "a") as f:
        f.write(result + "\n")

# ==== Main ====
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    data_root = "C:/Users/navee/Downloads/CLIP/data"
    dataset_paths = {
        "Caltech101": os.path.join(data_root, "caltech101", "caltech101", "101_ObjectCategories"),
        "Flowers-102": os.path.join(data_root, "flowers-102", "jpg_reorganized"),
        "Food-101": os.path.join(data_root, "food-101", "images"),
        "Stanford Cars": os.path.join(data_root, "stanford_cars", "test"),
        "CIFAR-10": os.path.join(data_root, "cifar10_imagefolder", "test")
    }

    model = MiniCLIPPromptTuning().to(device)
    model.load_state_dict(torch.load("prompt_tuned_clip.pth", map_location=device))

    log_file = "miniclip_prompt_tuning_results.txt"
    with open(log_file, "w") as f:
        f.write("📝 Prompt-Tuned MiniCLIP Evaluation Log\n")
        f.write(f"📅 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

    for name, path in dataset_paths.items():
        dataset = SimpleImageFolder(path, transform=transform)
        evaluate_dataset(name, dataset, model, tokenizer, device, log_file)

    print(f"\n📄 Evaluation results saved to {log_file}")
