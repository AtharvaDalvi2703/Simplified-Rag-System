# Thrifty AI: In-Memory Hybrid RAG Pipeline

## 1. Overview
This project implements a localized, in-memory Retrieval-Augmented Generation (RAG) pipeline from scratch. To strictly comply with the constraints prohibiting external vector databases and external LLM APIs, the system utilizes a **Hybrid Architecture**. It leverages deep learning (Dense Embeddings) for semantic document retrieval, and statistical NLP (Sparse Lexical Vectors) for deterministic answer extraction and mathematical evaluation. 

The entire pipeline operates locally, utilizing vectorized `NumPy` operations to maximize in-memory compute efficiency.

---

## 2. Approach & System Architecture

The pipeline is decoupled into four distinct, sequential phases:

### Phase 1: Retrieval (Dense Semantic Search)
* **Approach:** The system uses the `sentence-transformers` library (`all-MiniLM-L6-v2`) to encode both the query and the document corpus into 384-dimensional dense vectors. 
* **Execution:** Instead of iterative similarity loops, both the query and document vectors are L2-Normalized. The system then computes the exact cosine similarity using a single NumPy matrix multiplication (`np.dot`), sorting the resulting array to return the top-$k$ documents.

### Phase 2: Prompt Construction (Serialized Data Contract)
* **Approach:** The retrieved context, user query, and extraction instructions are mapped into a Python dictionary and serialized into a JSON string (`json.dumps`).
* **Execution:** This creates a strict, deterministic boundary between the retrieval and generation phases. By treating the prompt as a JSON payload rather than a concatenated text string, the downstream generation function is immunized against parsing errors or string-injection edge cases caused by unexpected punctuation in the source documents.

### Phase 3: Answer Generation (Sparse Lexical Extraction)
* **Approach:** Because external generative LLMs are prohibited, the system acts as an Extractive QA model. It parses the JSON payload, segments the retrieved context into individual sentences, and builds a localized, on-the-fly TF-IDF matrix using `scikit-learn`.
* **Execution:** The system transforms the user's query into the same vector space and calculates the cosine similarity against the sentence matrix. It surgically extracts and returns the single sentence with the highest lexical overlap. 

### Phase 4: Evaluation (Informational Density Scoring)
* **Approach:** The evaluation function measures the TF-IDF Cosine Similarity between the generated answer and the full retrieved context block.
* **Execution:** The system converts both strings into sparse vectors (filtering out stop-words) and computes their dot product. It returns a normalized `float` between `0.0` and `1.0`.
* **Rationale (Why not Token-Level Matching?):** Standard token-level checks (like Token F1-Score or Jaccard Similarity) rely on Set Theory. When evaluating a highly concise extractive system against a large retrieved context block, these metrics unfairly crush the score because the denominator (the total unique words in the context) is massive. TF-IDF Cosine Similarity solves this by assigning mathematical weight to rare words and measuring angular distance. This provides a smooth, continuous metric that heavily penalizes hallucinations without punishing the answer for being concise.

---

## 3. Design Choices & Rationale

* **Why a Hybrid Search (Dense -> Sparse)?** Dense embeddings are excellent for fuzzy semantic matching (e.g., matching the query "organization" to the document text "ISRO"). However, for the final answer extraction (Phase 3), dense models can lose surgical precision. By switching to TF-IDF (Sparse Lexical Search) for generation, the system assigns mathematical weight to rare, highly specific keywords (like "Chandrayaan-2"), ensuring it extracts the exact factual sentence.
* **Why Extractive Generation?**
  Attempting to generate conversational, fluid text using purely rule-based grammar heuristics (without an LLM) results in brittle code. Surgically extracting the most mathematically relevant sentence guarantees **100% groundedness** and completely eliminates the risk of hallucination.
* **Why TF-IDF Cosine Similarity for Evaluation?**
  Standard set-theory metrics (like Token F1-Score or Jaccard Similarity) unfairly penalize extractive QA models by bloating the denominator with the irrelevant tokens dragged in by the $k$ retrieval parameter. TF-IDF Cosine Similarity creates a smooth, continuous gradient that accurately measures the "informational density" of the answer without penalizing the system for being concise.

---

## 4. Assumptions & Edge Case Handling

1. **Cold-Start Latency:** Loading the PyTorch `MiniLM` weights into memory takes a few seconds during script initialization. It is assumed this one-time cold start is acceptable for the assessment execution. In production, this model would persist in memory.
2. **Zero-Division Failsafes:** Vector normalization arrays (`np.linalg.norm`) and sparse matrix transformations are wrapped in explicit failsafes. If a query contains only stop-words or lacks vocabulary overlap, the system gracefully falls back to returning `0.0` or a fallback string rather than throwing a `ValueError`.
3. **Sentence Tokenization:** For the scope of the provided mock dataset, splitting strings by periods (`.`) followed by `.strip()` is assumed to be a sufficient heuristic for sentence segmentation.

---

## 5. Instructions to Run

### Dependencies
Ensure Python 3.8+ is installed. The system requires standard data science libraries.
```bash
pip install numpy scikit-learn sentence-transformers
```

### Execution
Run the self-contained script from your terminal. The mock documents and queries are hardcoded into the __main__ block to ensure portability without requiring external file paths.

```bash
python main.py
```

## 6. Sample Outputs

```text
Query: Who launched Chandrayaan-3?
Retrieved Context: ['India launched Chandrayaan-3 in 2023.', 'Chandrayaan-2 had a partial failure during landing.']
Answer: India launched Chandrayaan-3 in 2023.
Score: 0.6954
----------------------------------------
Query: What happened in Chandrayaan-2?
Retrieved Context: ['Chandrayaan-2 had a partial failure during landing.', 'India launched Chandrayaan-3 in 2023.']
Answer: Chandrayaan-2 had a partial failure during landing.
Score: 0.6954
----------------------------------------
Query: Which organization is ISRO?
Retrieved Context: ['ISRO is the Indian Space Research Organisation.', 'India launched Chandrayaan-3 in 2023.']
Answer: ISRO is the Indian Space Research Organisation.
Score: 0.6225
----------------------------------------
```