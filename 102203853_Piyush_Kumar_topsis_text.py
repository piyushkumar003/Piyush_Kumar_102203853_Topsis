from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Example: Evaluate a model
def evaluate_model(model_name, sentences):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    
    embeddings = []
    for sentence in sentences:
        inputs = tokenizer(sentence, return_tensors="pt", truncation=True, padding=True)
        outputs = model(**inputs)
        embeddings.append(outputs.last_hidden_state.mean(dim=1).detach().numpy())
    
    # Calculate cosine similarities
    similarities = cosine_similarity(np.vstack(embeddings))
    return similarities



import numpy as np

# Example TOPSIS implementation
def topsis(scores, weights):
    scores = np.array(scores)
    # Normalize scores
    norm_scores = scores / np.sqrt((scores**2).sum(axis=0))
    # Identify ideal best and worst
    ideal_best = norm_scores.max(axis=0)
    ideal_worst = norm_scores.min(axis=0)
    # Compute distances
    dist_best = np.sqrt(((norm_scores - ideal_best)**2).sum(axis=1))
    dist_worst = np.sqrt(((norm_scores - ideal_worst)**2).sum(axis=1))
    # Compute rankings
    ranks = dist_worst / (dist_best + dist_worst)
    return ranks





import matplotlib.pyplot as plt

def plot_results(models, scores):
    plt.bar(models, scores)
    plt.title("Model Rankings")
    plt.xlabel("Models")
    plt.ylabel("TOPSIS Score")
    plt.show()

# Sample sentences for similarity evaluation
sentences = [
    "The quick brown fox jumps over the lazy dog.",
    "A fast, brown fox leaps over a sleepy dog.",
    "The weather today is sunny and bright.",
    "It is a bright and sunny day today."
]




# List of pre-trained models to evaluate
models = [
    "sentence-transformers/all-mpnet-base-v2",
    "sentence-transformers/bert-base-nli-mean-tokens",
    "distilbert-base-uncased"
]

# Store similarity scores for each model
model_scores = []
for model_name in models:
    print(f"Evaluating model: {model_name}")
    similarity = evaluate_model(model_name, sentences)
    # Here, you can calculate an aggregate score, e.g., mean similarity
    aggregate_score = similarity.mean()
    model_scores.append({
        "model": model_name,
        "mean_similarity": aggregate_score,
        "inference_speed": 0,  # Placeholder for speed (add real value later)
        "model_size": 0       # Placeholder for model size
    })






# Example scoring matrix
scores = [
    [0.85, 0.9, 0.75],  # Model 1: mean_similarity, inference_speed, model_size
    [0.82, 0.85, 0.8],  # Model 2: same metrics
    [0.78, 0.87, 0.85]  # Model 3: same metrics
]

# Define weights for each criterion (sum should be 1)
weights = [0.5, 0.3, 0.2]  # Adjust based on importance






# Apply TOPSIS to rank models
ranks = topsis(scores, weights)

# Display results
for i, model in enumerate(models):
    print(f"Model: {model}, TOPSIS Rank: {ranks[i]}")
