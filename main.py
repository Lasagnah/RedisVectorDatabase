#pip install redis numpy
#pip show redis
import redis
import numpy as np

# Connect to Redis
r = redis.Redis(host='localhost', port=6379, decode_responses=False)

# Function to calculate cosine similarity
def cosine_similarity(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

# Clear previous data
r.flushall()

# Cat data
cats = {
    "Explorer": np.array([0.1, 0.8, 0.3, 0.6], dtype=np.float32),
    "Alpha": np.array([0.5, 0.2, 0.3, 0.3], dtype=np.float32),
    "Yoda": np.array([0.8, 0.1, 0.1, 0.4], dtype=np.float32)
}

# Add cats
for cat_name, vector in cats.items():
    r.set(f"cat:{cat_name}", vector.tobytes())
    print(f"Added cat: {cat_name} with vector: {vector}")

# Query vector
query_vector = np.array([0.15, 0.75, 0.35, 0.55], dtype=np.float32)
print(f"\nQuery Vector: {query_vector}")

# Manual similarity search
results = []
for cat_name, vector in cats.items():
    similarity = cosine_similarity(query_vector, vector)
    results.append((cat_name, similarity))

# Sort and display results
results.sort(key=lambda x: x[1], reverse=True)
print("\nSimilarity Search Results:")
for cat_name, similarity in results:
    print(f"Cat: {cat_name}, Similarity: {similarity:.4f}")