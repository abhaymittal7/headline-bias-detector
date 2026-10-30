import sqlite3
import json

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

from src.utils.db import DB_PATH


def cluster_headlines(threshold=0.7):
    """
    Groups headlines into stories based on semantic similarity using Sentence Embeddings.
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("""
    SELECT headline_id, raw_text, source_name, sentiment, bias 
    FROM headlines
    """)

    rows = cursor.fetchall()
    if not rows:
        print("No headlines found to cluster.")
        return [], [], [], []

    ids = []
    texts = []
    sources = []
    biases = []
    sentiments = []

    for row in rows:
        ids.append(row[0])
        texts.append(row[1])
        sources.append(row[2])

        try:
            sent = json.loads(row[3]) if row[3] else {}
        except:
            sent = {}
        
        try:
            b = json.loads(row[4]) if row[4] else {}
        except:
            b = {}

        sentiments.append(sent)
        biases.append(b)

    # ----------------------------------------
    # Sentence Embedding Vectorization
    # ----------------------------------------
    print(f"Generating embeddings for {len(texts)} headlines...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(texts)

    # ----------------------------------------
    # Similarity matrix
    # ----------------------------------------
    sim_matrix = cosine_similarity(embeddings)

    # ----------------------------------------
    # Clustering (Greedy Threshold Clustering)
    # ----------------------------------------
    # Same greedy algorithm as before, vectorised with numpy so it stays
    # fast for thousands of headlines (the pure-Python double loop is O(n^2)).
    clusters = []
    visited = np.zeros(len(texts), dtype=bool)

    for i in range(len(texts)):
        if visited[i]:
            continue

        visited[i] = True
        members = np.where((sim_matrix[i] >= threshold) & ~visited)[0]
        visited[members] = True

        clusters.append([i] + members.tolist())

    conn.close()

    return clusters, texts, sources, biases