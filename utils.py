from sentence_transformers import SentenceTransformer
import numpy as np
import faiss

_model = SentenceTransformer("all-MiniLM-L6-v2")

def cluster_reviews(texts, similarity_threshold=0.85):
    """
    Cluster reviews using embeddings.
    Returns cluster assignments and representative indices.
    """
    if not texts:
        return [], []

    embeddings = _model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    visited = set()
    clusters = {}
    reps = []
    cluster_id = 0

    for i in range(len(texts)):
        if i in visited:
            continue
        clusters[cluster_id] = [i]
        reps.append(i)
        visited.add(i)

        sims, idxs = index.search(np.expand_dims(embeddings[i], axis=0), len(texts))
        for j, sim in zip(idxs[0], sims[0]):
            if j not in visited and sim >= similarity_threshold:
                clusters[cluster_id].append(j)
                visited.add(j)
        cluster_id += 1

    cluster_assignments = [-1] * len(texts)
    for cid, idxs in clusters.items():
        for j in idxs:
            cluster_assignments[j] = cid

    return cluster_assignments, reps
