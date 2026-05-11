import json
import re
import numpy as np
from typing import List, Dict
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer

# Global Initialization
# Initialize this globally or via a singleton so it doesn't reload on every function call
encoder = SentenceTransformer('all-MiniLM-L6-v2')

def retrieve(query: str, docs: List[Dict], k: int = 2) -> List[str]:
    """Phase 1: Retrieval Engine - Dense Semantic Vectorized Search"""
    texts = [doc["text"] for doc in docs]
    
    if not texts:
        return []
        
    # Embed the incoming query and the text field of all docs
    query_vec = encoder.encode(query)
    doc_vecs = encoder.encode(texts)
    
    # L2-Normalize both the query vector and the document vectors
    query_vec = query_vec / np.linalg.norm(query_vec)
    doc_norms = np.linalg.norm(doc_vecs, axis=1, keepdims=True)
    doc_norms[doc_norms == 0] = 1 # Avoid division by zero
    doc_vecs = doc_vecs / doc_norms
    
    # Calculate exact cosine similarity using NumPy dot product
    similarities = np.dot(doc_vecs, query_vec)
    
    # Sort indices in descending order based on similarity score
    sorted_indices = np.argsort(similarities)[::-1]
    
    # Return list of exactly k document text strings
    top_k_indices = sorted_indices[:k]
    return [texts[i] for i in top_k_indices]

def build_prompt(query: str, context: List[str]) -> str:
    """Phase 2: Prompt Construction - Serialized JSON Data Contract"""
    joined_context = " ".join(context)
    
    prompt_dict = {
        "instructions": "Extract the most relevant answer from the context based on the query.",
        "context": joined_context,
        "query": query
    }
    
    return json.dumps(prompt_dict)

def generate_answer(prompt: str) -> str:
    """Phase 3: Answer Generation - Sparse Lexical Extraction (Micro TF-IDF)"""
    try:
        prompt_dict = json.loads(prompt)
    except json.JSONDecodeError:
        return "Fallback answer: Error decoding prompt."
        
    context_str = prompt_dict.get("context", "")
    query = prompt_dict.get("query", "")
    
    if not context_str.strip():
        return "Fallback answer: Empty context."
        
    # Segment into individual sentences (splitting by periods)
    # Using list comprehension to filter empty strings after split
    sentences = [s.strip() + "." for s in context_str.split(".") if s.strip()]
    
    if not sentences:
         return "Fallback answer: No sentences found."
         
    # Initialize TfidfVectorizer
    vectorizer = TfidfVectorizer(stop_words='english')
    
    try:
        # Fit and transform strictly on segmented context sentences
        context_vecs = vectorizer.fit_transform(sentences)
        # Transform the query
        query_vec = vectorizer.transform([query])
    except ValueError:
        # This can happen if all words are stop words or vocabulary is empty
        return sentences[0] if sentences else "Fallback answer: Error during vectorization."
        
    # Calculate cosine similarity (dot product between sparse matrices)
    # query_vec is 1 x V, context_vecs is N x V
    similarities = context_vecs.dot(query_vec.T).toarray().flatten()
    
    # Extract the single sentence string with highest TF-IDF overlap
    best_idx = np.argmax(similarities)
    return sentences[best_idx]

def evaluate(answer: str, context: List[str]) -> float:
    """Phase 4: Evaluation - TF-IDF Cosine Similarity"""
    if not answer or not context:
        return 0.0
        
    full_context = " ".join(context)
    corpus = [full_context, answer]
    
    vectorizer = TfidfVectorizer(stop_words='english')
    
    try:
        tfidf_matrix = vectorizer.fit_transform(corpus)
    except ValueError:
        return 0.0
        
    context_vector = tfidf_matrix[0]
    answer_vector = tfidf_matrix[1]
    
    similarity_score = context_vector.dot(answer_vector.T).toarray()[0][0]
    
    return round(float(similarity_score), 4)

if __name__ == "__main__":
    DOCUMENTS = [
        {"id": 1, "text": "India launched Chandrayaan-3 in 2023."},
        {"id": 2, "text": "The mission successfully landed on the moon's south pole."},
        {"id": 3, "text": "ISRO is the Indian Space Research Organisation."},
        {"id": 4, "text": "Chandrayaan-2 had a partial failure during landing."}
    ]

    QUERIES = [
        {"query": "Who launched Chandrayaan-3?"},
        {"query": "What happened in Chandrayaan-2?"},
        {"query": "Which organization is ISRO?"}
    ]

    for q in QUERIES:
        query_text = q["query"]
        
        # 1. Retrieve
        retrieved_context = retrieve(query_text, DOCUMENTS, k=2)
        
        # 2. Build Prompt
        prompt = build_prompt(query_text, retrieved_context)
        
        # 3. Generate Answer
        answer = generate_answer(prompt)
        
        # 4. Evaluate
        score = evaluate(answer, retrieved_context)
        
        # Print output
        print(f"Query: {query_text}")
        print(f"Retrieved Context: {retrieved_context}")
        print(f"Answer: {answer}")
        print(f"Score: {score:.4f}")
        print("-" * 40)
