import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Final clean synthetic data (replace with your real values from log files)
confidences = {
    "Zero-shot (Handcrafted)": [0.82, 0.84, 0.79, 0.87, 0.85, 0.83, 0.86],
    "CuPL (LLM)": [0.83, 0.81, 0.80, 0.85, 0.82, 0.84, 0.86],
    "CoOp": [0.89, 0.91, 0.87, 0.90, 0.88, 0.89, 0.92],
    "CoCoOp": [0.88, 0.87, 0.86, 0.89, 0.90, 0.88, 0.87],
    "PromptSRC": [0.85, 0.83, 0.86, 0.87, 0.84, 0.85, 0.86],
    "MaPLe": [0.90, 0.89, 0.88, 0.91, 0.90, 0.89, 0.92]
}

# Flatten to clean DataFrame
records = []
for method, vals in confidences.items():
    for v in vals:
        try:
            fval = float(v)
            if 0.0 <= fval <= 1.0:
                records.append((method, fval))
        except Exception:
            continue

df = pd.DataFrame(records, columns=["Method", "Confidence"])

# Plot
plt.figure(figsize=(10, 6))
sns.kdeplot(data=df, x="Confidence", hue="Method", fill=True, alpha=0.25)
plt.title("Confidence Distribution Across Prompt Tuning Methods")
plt.xlabel("Confidence")
plt.ylabel("Density")
plt.grid(True)
plt.tight_layout()
plt.savefig("prompt_tuning_kde_all_cleaned_final.png")
plt.show()
