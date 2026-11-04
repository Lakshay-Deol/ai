# Team Roles and Technology Overview

> **Project**: Research Paper Search Engine by Keywords (`find-papers-by-keyword`)  
> **Repository**: [https://github.com/Lakshay-Deol/ai](https://github.com/Lakshay-Deol/ai)

---

## 1. Team Contributions and Roles (5 Students)

### 1. Kshitij — Core Algorithm & Search Architecture Lead
- **Responsibilities**:
  - Designed the end-to-end 3-phase Information Retrieval pipeline: Embeddings $\to$ Keyword Assignment $\to$ Ranked Retrieval.
  - Formulated the multi-keyword similarity expansion using Normalized Pointwise Mutual Information (NPMI).
  - Implemented the database query optimizations (identifying the `INNER JOIN` bottleneck and transitioning to `LEFT JOIN` in `PaperSearchEngine`).
  - Authored the core modules `assign_paper_keywords.py`, `paper_indexer.py`, and `paper_search_engine.py`.

### 2. Lakshay (Lakshay Deol) — Lead Frontend & Full-Stack Systems Engineer
- **Main Project Responsibility (Interactive Frontend Architecture)**:
  - Designed and developed the entire **Interactive Web Frontend** (`app.py`), transforming the command-line search scripts into a production-grade, responsive single-page web application.
  - Implemented the client-side user experience with **Tailwind CSS**, **Lucide Icons**, and asynchronous JavaScript, featuring real-time multi-keyword tag filtering and one-click suggestion pills.
  - Engineered the **Interactive Results Dashboard**, displaying dynamic score badges, citation metrics, categorized tags, author metadata, and expandable abstract previews.
  - Built the live **Trie Text Scanner** interactive playground enabling users to paste arbitrary text or abstracts and benchmark $O(L)$ keyword detection directly in the browser.
  - Designed the **NPMI Semantic Expansion Visualizer** and **Pipeline Architecture** view to provide mathematical transparency for the scoring function:
    $$\text{Score} = \sum \max(\text{NPMI} \times \text{Score}) \times (\text{Citations} + 1)$$
  - Integrated the full-stack REST API layer using **FastAPI** (`/api/search`, `/api/keywords`, `/api/stats`, `/api/match-text`).
  - Engineered the standalone zero-config local SQLite runtime (`local_demo.py`) and managed repository release engineering, dependency packaging, and GitHub deployment.

### 3. Krishna Gupta — NLP & Semantic Embeddings Specialist
- **Responsibilities**:
  - Integrated `SentenceTransformers` using the pre-trained `bert-base-nli-mean-tokens` model to convert research abstracts and titles into 768-dimensional dense vectors.
  - Built the vector normalization routines and cosine similarity calculation workflows in `EmbeddingsGenerator`.
  - Tuned batch encoding for paper text and implemented serialization of embedding matrices into NumPy/Pickle formats.

### 4. Krishna Thakur — Trie Data Structure & Regex Optimization Engineer
- **Responsibilities**:
  - Developed the Prefix Tree (Trie) module in `src/find_papers_by_keyword/trie/Trie.py`.
  - Engineered the algorithm that compiles a Trie of thousands of keywords into an optimized, compact Regular Expression union pattern.
  - Replaced $O(N \cdot M)$ brute-force string matching with $O(L)$ scanning of paper texts, drastically accelerating keyword detection.

### 5. Laksh Gupta — Database Architecture & Citation Ranking Engineer
- **Responsibilities**:
  - Designed the relational database schemas (`Publication`, `FoS`, `Publication_FoS`, and `FoS_npmi_Springer`) with primary and cascading foreign keys.
  - Formulated and implemented the citation-weighted ranking formula:
    $$\text{Total Score} = \sum \max(\text{NPMI} \times \text{Match Score}) \times (\text{Citations} + 1)$$
  - Designed the domain specificity penalization algorithm that reduces the rank of keywords appearing frequently in non-computer-science literature.

---

## 2. Technology Stack Explained (In Short)

### 1. Sentence-BERT (SentenceTransformers)
- **What it is**: A modification of the pre-trained BERT network that uses siamese and triplet network structures to derive fixed-size, semantically meaningful sentence embeddings.
- **Why it's used**: Standard BERT requires passing pairs of sentences together, which is computationally infeasible for searching millions of papers ($O(N^2)$). Sentence-BERT embeds papers and keywords into 768-dimensional vectors independently, allowing instantaneous similarity comparisons via dot products.

### 2. Trie (Prefix Tree) & Regex Union
- **What it is**: A tree data structure where each node represents a character, grouping strings by common prefixes.
- **Why it's used**: Scanning a paper abstract against 40,000 keywords using individual regex patterns would take minutes per paper. By inserting all keywords into a Trie and compiling the Trie into a grouped regex pattern `(\b(?:...)\b)`, keyword matching runs in a single pass in $O(L)$ time, where $L$ is text length.

### 3. Cosine Similarity
- **What it is**: A mathematical metric measuring the cosine of the angle between two multi-dimensional vectors: $\cos(\theta) = \frac{\mathbf{A} \cdot \mathbf{B}}{\|\mathbf{A}\| \|\mathbf{B}\|}$.
- **Why it's used**: It evaluates the conceptual similarity between a paper and a keyword independently of abstract length. Values range from 0 (completely unrelated) to 1 (identical semantic meaning).

### 4. DBSCAN (Density-Based Spatial Clustering of Applications with Noise)
- **What it is**: An unsupervised clustering algorithm that groups points closely packed together in high-density areas, marking points in low-density regions as outliers.
- **Why it's used**: When assigning keywords to a paper, multiple synonyms (e.g., *"neural networks"*, *"deep neural networks"*, *"artificial neural networks"*) often match. DBSCAN clusters their embedding vectors together so the system keeps only the single most representative keyword per cluster, eliminating redundancy.

### 5. Normalized Pointwise Mutual Information (NPMI)
- **What it is**: A statistical correlation metric bounded between $-1$ (never co-occur) and $+1$ (always co-occur), measuring how much more frequently two keywords appear together in academic literature than expected by chance:
  $$\text{NPMI}(x, y) = \frac{\ln \frac{P(x, y)}{P(x)P(y)}}{-\ln P(x, y)}$$
- **Why it's used**: Enables **query expansion**. When a user searches for *"graph decompositions"*, NPMI identifies correlated terms like *"pebble game"* and *"sparsity"*, retrieving relevant papers even if they don't contain the exact query words.

### 6. Relational SQL Database (MySQL & SQLite)
- **What it is**: Structured relational storage engines using ACID-compliant relational tables.
- **Why it's used**:
  - `Publication`: Stores paper ID, title, abstract, and citation counts.
  - `FoS` (Field of Study): Stores keyword IDs and frequencies.
  - `Publication_FoS`: Many-to-many junction table mapping papers to keywords with their computed cosine match scores.
  - `FoS_npmi_Springer`: Stores pairwise NPMI co-occurrence weights between keywords.
  - Allows complex multi-table joins and aggregation queries (`SUM`, `MAX`, `GROUP BY`) to rank millions of papers in milliseconds.

### 7. Citation-Weighted Ranking
- **What it is**: A hybrid scoring function combining topical relevance with historical academic impact:
  $$\text{Score} = \left[ \sum \max(\text{NPMI} \times \text{Match Score}) \right] \times (\text{Citations} + 1)$$
- **Why it's used**: High citation counts indicate influential, peer-reviewed foundations. Adding $+1$ ensures newly published papers with zero citations are not completely disqualified ($0 \times \text{Score} = 0$).

### 8. Python / NumPy / Scikit-Learn
- **What it is**: The industry-standard numerical computing and machine learning stack.
- **Why it's used**: NumPy provides vectorized matrix multiplication for dot-product similarity calculations; Scikit-Learn provides an optimized C-accelerated DBSCAN implementation; Python binds data loading, regex processing, and SQL querying cleanly.

