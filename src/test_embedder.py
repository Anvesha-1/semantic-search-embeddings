"""Member 2: quick check that similar meanings get similar embeddings."""

from embedder import embed_texts

pairs = [
    ("cost of fuel goes up", "oil prices rise"),
    ("cost of fuel goes up", "football coach fired"),
]

for a, b in pairs:
    va, vb = embed_texts([a, b])
    score = float(va @ vb)  # vectors are normalized, so this is cosine similarity
    print(f"{score:.3f}  |  '{a}'  vs  '{b}'")