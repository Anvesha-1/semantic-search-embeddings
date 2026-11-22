"""Member 2: draw the embeddings in 2D, coloured by category (PCA and t-SNE)."""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # save to a file, no window needed
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

corpus = pd.read_csv("data/processed/corpus.csv")
emb = np.load("data/processed/corpus_embeddings.npy")
categories = corpus["category"].values

print("Running PCA...")
pca_2d = PCA(n_components=2, random_state=42).fit_transform(emb)

print("Running t-SNE (takes about 1 to 3 minutes)...")
tsne_2d = TSNE(n_components=2, perplexity=30, init="pca", random_state=42).fit_transform(emb)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
for ax, points, title in [(axes[0], pca_2d, "PCA"), (axes[1], tsne_2d, "t-SNE")]:
    for cat in sorted(set(categories)):
        mask = categories == cat
        ax.scatter(points[mask, 0], points[mask, 1], s=5, alpha=0.6, label=cat)
    ax.set_title(f"Article embeddings in 2D ({title})")
    ax.set_xlabel("dimension 1")
    ax.set_ylabel("dimension 2")
axes[1].legend(markerscale=3, title="Category")

plt.tight_layout()
plt.savefig("results/figures/embeddings_2d.png", dpi=150)
print("Saved results/figures/embeddings_2d.png")