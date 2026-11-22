"""Member 2: embed every article in corpus.csv and save the result."""

import time
import numpy as np
import pandas as pd
from embedder import embed_texts

CORPUS_PATH = "data/processed/corpus.csv"
OUTPUT_PATH = "data/processed/corpus_embeddings.npy"

df = pd.read_csv(CORPUS_PATH)
print("Articles loaded:", len(df))

start = time.time()
embeddings = embed_texts(df["text"].tolist(), show_progress=True)
print(f"Done in {time.time() - start:.1f} seconds")

# Row i of the embeddings must match doc_id i in corpus.csv
assert embeddings.shape == (len(df), 384), "Unexpected shape!"

np.save(OUTPUT_PATH, embeddings)
print("Saved to", OUTPUT_PATH)
print("Shape:", embeddings.shape)
print("Length of first vector (should be about 1.0):",
      float(np.linalg.norm(embeddings[0])))