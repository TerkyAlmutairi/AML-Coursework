# Prompt Tuning with CLIP for Multi-Dataset Generalization

This project investigates various prompt tuning strategies for CLIP (Contrastive Language–Image Pretraining) to enhance its performance on multiple datasets from diverse domains. Our work follows the guidance from the [CoOp GitHub repository](https://github.com/KaiyangZhou/CoOp) and expands on it by implementing and comparing multiple techniques.

## 📁 Datasets

We evaluate all methods on five datasets:
- **Caltech101**
- **Food-101**
- **Stanford Cars**
- **CIFAR-10**
- **FGVC Aircraft**

Each dataset was preprocessed into ImageFolder format and stored in data/.

## 🧪 Prompt Tuning Techniques Implemented

We explored the following methods:

| Method         | Description                                                                 |
|----------------|-----------------------------------------------------------------------------|
| **Zero-Shot CLIP**     | Evaluated using 80 handcrafted prompts for each class.                         |
| **CuPL**               | Used LLM-generated prompts (ChatGPT) for each class.                           |
| **CoOp**               | Context-optimized prompts learned per class.                                   |
| **CoCoOp**             | Prompts conditioned on image embeddings to improve generalization.             |
| **PromptSRC**          | Source-consistent regularization to reduce overfitting in learned prompts.     |
| **MaPLe**              | Combines learnable prompts with lightweight adapter networks.                   |

## 🔍 Attention Map Visualizations

We generated attention maps using Grad-CAM for one test image per dataset, visualizing how CLIP attends to regions across different prompting techniques. Maps are stored in attention heatmap/.

## 🛠 How to Run

### 1. Install Dependencies
bash
pip install -r requirements.txt
2. Prepare Datasets
Organize all datasets under data/ in ImageFolder format. See organize_*.py scripts.

3. Train Prompt Learners
bash
Copy
Edit
python coop_prompt_tuning_train.py       # CoOp
python cocoop_prompt_tuning_train.py     # CoCoOp
python promptsrc_train.py                # PromptSRC
python maple_train.py                    # MaPLe
4. Evaluate Models
bash
Copy
Edit
python coop_prompt_tuning_evaluate.py
python cocoop_prompt_tuning_evaluate.py
python promptsrc_evaluate.py
python maple_prompt_evaluate.py
5. Visualize Attention Maps
bash
Copy
Edit
python generate_attention_maps.py

## 🔧 Dataset Setup (Manual)

To run the project, download and extract the following datasets into the `data/` directory:

- **Caltech101**: http://www.vision.caltech.edu/Image_Datasets/Caltech101/
- **Food-101**: http://data.vision.ee.ethz.ch/cvl/food-101.tar.gz
- **Stanford Cars**: https://ai.stanford.edu/~jkrause/cars/car_dataset.html / https://www.kaggle.com/datasets/eduardo4jesus/stanford-cars-dataset
- **CIFAR-10**: https://www.cs.toronto.edu/~kriz/cifar.html
- **FGVC Aircraft**: https://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/

Each dataset must be placed under `data/{dataset_name}/` and formatted using PyTorch’s `ImageFolder` layout.

link for datasets: https://drive.google.com/drive/folders/1qV-km9XAajp5M_Jx36kkVbi_BacGvwpG?usp=sharing

