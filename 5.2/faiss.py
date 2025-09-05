import pickle
import faiss
import numpy as np
from datasets import load_dataset
import warnings

# Suppress a harmless warning from the datasets library regarding a future change
warnings.filterwarnings("ignore", category=FutureWarning)


corpus = load_dataset("scifact", "corpus", trust_remote_code=True)
claims = load_dataset("scifact", "claims", trust_remote_code=True)

print("Constructing ground truth mapping...")
ground_truth = {}
for example in claims['train']:
    claim_id = example['id']
    # Ensure cited_doc_ids are integers for consistent type matching later
    relevant_docs = set(map(int, example['cited_doc_ids']))
    ground_truth[claim_id] = relevant_docs

print("Loading evidence and claim embeddings from pickle files...")
try:
    with open('scifact_evidence_embeddings.pkl', 'rb') as f:
        evidence_embeddings = pickle.load(f)
    with open('scifact_claim_embeddings.pkl', 'rb') as f:
        claim_embeddings = pickle.load(f)
except FileNotFoundError as e:
    print(f"Error: {e}.")
    print("Please make sure 'scifact_evidence_embeddings.pkl' and 'scifact_claim_embeddings.pkl' are in the same directory.")
    exit()

# The keys in evidence_embeddings are tuples (doc_id, sent_id).
# We need an ordered list of these keys to map FAISS indices back to their doc_ids.
evidence_keys = list(evidence_embeddings.keys())
evidence_vectors = np.array(list(evidence_embeddings.values()), dtype='float32')


dimension = evidence_vectors.shape[1]
index = faiss.IndexFlatL2(dimension)
print(f"Is the index trained? {index.is_trained}") # Should be True for IndexFlatL2
index.add(evidence_vectors)
print(f"Number of vectors in the index: {index.ntotal}")

def evaluate_retrieval(evidence_keys, claim_embeddings, ground_truth, faiss_index, k):
    """
    Evaluates the retrieval system using MRR and MAP metrics for a given k.

    Args:
        evidence_keys (list): A list of keys (doc_id, sent_id) corresponding to the indexed vectors.
        claim_embeddings (dict): A dictionary mapping (claim_id, claim_text) to embeddings.
        ground_truth (dict): A dictionary mapping claim_id to a set of relevant doc_ids.
        faiss_index (faiss.Index): The FAISS index containing evidence embeddings.
        k (int): The number of top results to retrieve for each query.

    Returns:
        tuple: A tuple containing the Mean Reciprocal Rank (MRR) and Mean Average Precision (MAP).
    """
    mrr_total = 0.0
    map_total = 0.0
    num_queries = len(claim_embeddings)

    if num_queries == 0:
        return 0.0, 0.0

    for (claim_id, _), claim_embedding in claim_embeddings.items():
        query_vector = np.array([claim_embedding], dtype='float32')

        # Perform the search
        _, indices = faiss_index.search(query_vector, k)

        # Map FAISS indices back to document IDs.
        # The retrieval is for sentences, but evaluation is on documents. We extract the doc_id
        # (the first element of the key tuple).
        # We use dict.fromkeys to get unique doc_ids while preserving order.
        retrieved_doc_ids = list(dict.fromkeys([evidence_keys[idx][0] for idx in indices[0]]))
        
        # Get the set of correct document IDs for the current claim
        correct_doc_ids = ground_truth.get(claim_id, set())

        if not correct_doc_ids:
            continue # Skip claims that have no cited documents in the ground truth

        # --- Calculate Reciprocal Rank (for MRR) ---
        reciprocal_rank = 0.0
        for rank, doc_id in enumerate(retrieved_doc_ids, start=1):
            if doc_id in correct_doc_ids:
                reciprocal_rank = 1.0 / rank
                break  # Stop after finding the first relevant document
        mrr_total += reciprocal_rank

        # --- Calculate Average Precision (for MAP) ---
        relevant_items_found = 0
        sum_of_precisions = 0.0
        for rank, doc_id in enumerate(retrieved_doc_ids, start=1):
            if doc_id in correct_doc_ids:
                relevant_items_found += 1
                precision_at_k = relevant_items_found / rank
                sum_of_precisions += precision_at_k

        num_relevant_docs = len(correct_doc_ids)
        avg_precision = sum_of_precisions / num_relevant_docs if num_relevant_docs > 0 else 0.0
        map_total += avg_precision

    mrr = mrr_total / num_queries
    map_score = map_total / num_queries

    return mrr, map_score

print("\n--- Evaluating IR System Performance ---")
k_values = [1, 10, 50]
results = {}

for k in k_values:
    print(f"Running evaluation for k={k}...")
    mrr, map_score = evaluate_retrieval(
        evidence_keys=evidence_keys,
        claim_embeddings=claim_embeddings,
        ground_truth=ground_truth,
        faiss_index=index,
        k=k
    )
    results[k] = {'mrr': mrr, 'map': map_score}
    print(f"Results @{k}: MRR = {mrr:.4f}, MAP = {map_score:.4f}")

# --- 6. Print Summary Table ---
print("\n" + "="*55)
print("--- Summary of Results ---")
print("="*55)
print(f"{'Approach':<20} | {'Metric':<8} | {'MRR @ 1':<8} | {'MAP @ 1':<8} | {'MRR @ 10':<9} | {'MAP @ 10':<9} | {'MRR @ 50':<9} | {'MAP @ 50':<9}")
print("-"*105)

mrr_1 = results.get(1, {}).get('mrr', 0)
map_1 = results.get(1, {}).get('map', 0)
mrr_10 = results.get(10, {}).get('mrr', 0)
map_10 = results.get(10, {}).get('map', 0)
mrr_50 = results.get(50, {}).get('mrr', 0)
map_50 = results.get(50, {}).get('map', 0)

print(f"{'OpenAI Embeddings':<20} | {'MRR/MAP':<8} | {mrr_1:<8.4f} | {map_1:<8.4f} | {mrr_10:<9.4f} | {map_10:<9.4f} | {mrr_50:<9.4f} | {map_50:<9.4f}")
print("-"*105)