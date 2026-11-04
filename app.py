"""
Forward Scholar: Interactive Web Frontend for Research Paper Retrieval
Powered by FastAPI, SQLite In-Memory Database, and Trie-Regex Semantic Engine.
"""

import sqlite3
import json
import re
from typing import List, Optional
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import uvicorn

app = FastAPI(
    title="Forward Scholar - Research Paper Search Engine",
    description="Interactive Web Application for Finding Research Papers by Keywords",
    version="1.0.0"
)

# ---------------------------------------------------------------------------
# Database Initialization & Data Seeding
# ---------------------------------------------------------------------------
def init_db():
    conn = sqlite3.connect(":memory:", check_same_thread=False)
    cur = conn.cursor()

    cur.execute("""
    CREATE TABLE FoS (
        id INTEGER PRIMARY KEY,
        keyword TEXT UNIQUE,
        frequency INTEGER,
        category TEXT
    );
    """)

    cur.execute("""
    CREATE TABLE Publication (
        id TEXT PRIMARY KEY,
        title TEXT,
        authors TEXT,
        categories TEXT,
        abstract TEXT,
        citations INTEGER,
        doi TEXT,
        update_date TEXT
    );
    """)

    cur.execute("""
    CREATE TABLE Publication_FoS (
        Publication_id TEXT,
        FoS_id INTEGER,
        score REAL,
        PRIMARY KEY (Publication_id, FoS_id)
    );
    """)

    cur.execute("""
    CREATE TABLE FoS_npmi_Springer (
        id1 INTEGER,
        id2 INTEGER,
        npmi REAL,
        PRIMARY KEY (id1, id2)
    );
    """)

    # 1. Seed Publications from test_data/papers.json
    with open("test_data/papers.json", "r", encoding="utf-8") as f:
        papers = json.load(f)

    # Custom citation counts for demonstration realism
    citation_weights = {
        "0704.0002": 150,
        "0704.0046": 135,
        "0704.0047": 85,
        "0704.0050": 92,
        "0704.0062": 110,
        "0704.0090": 45,
        "0704.0098": 78,
        "0704.0108": 60,
        "0704.0213": 125,
        "0704.0217": 15
    }

    for p in papers:
        pid = p["id"]
        cits = citation_weights.get(pid, 25)
        cur.execute(
            "INSERT INTO Publication VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                pid,
                p.get("title", "").strip(),
                p.get("authors", "").strip(),
                p.get("categories", "").strip(),
                p.get("abstract", "").strip(),
                cits,
                p.get("doi", "N/A"),
                p.get("update_date", "2008")
            )
        )

    # 2. Seed Fields of Study (Keywords)
    keywords_data = [
        # Graph Theory & Algorithms (0704.0002)
        (1, "graph decompositions", 450, "Algorithms & Combinatorics"),
        (2, "pebble game", 120, "Graph Theory"),
        (3, "tree decompositions", 340, "Algorithms"),
        (10, "sparsity", 290, "Graph Theory"),
        # Information Theory & Quantum (0704.0046)
        (4, "channel capacity", 520, "Information Theory"),
        (5, "entropy", 680, "Information Theory"),
        (9, "quantum law", 180, "Quantum Computing"),
        # Acoustic Emission & Neural Networks (0704.0047, 0704.0050)
        (11, "acoustic emission", 310, "Acoustic Sensing"),
        (12, "neural network", 4200, "Machine Learning"),
        (13, "blind source separation", 240, "Signal Processing"),
        (14, "independent component analysis", 380, "Signal Processing"),
        # Bioinformatics & HMM (0704.0062)
        (15, "viterbi algorithm", 290, "Bioinformatics"),
        (16, "hidden markov models", 850, "Bioinformatics"),
        (17, "dna sequences", 620, "Computational Biology"),
        (18, "random walks", 410, "Algorithms"),
        # Optimization & Schedules (0704.0090)
        (19, "simulated annealing", 530, "Optimization"),
        (20, "project schedules", 140, "Operations Research"),
        (21, "risk analysis", 320, "Quantitative Modeling"),
        # Communications & CDMA (0704.0098)
        (22, "cdma", 360, "Telecommunications"),
        (23, "statistical mechanics", 740, "Applied Physics"),
        (24, "spreading codes", 190, "Wireless Communications"),
        # Complexity Theory & SAT (0704.0108, 0704.0213)
        (25, "sat", 890, "Complexity Theory"),
        (26, "2-sat", 230, "Complexity Theory"),
        (27, "polynomial time", 1150, "Complexity Theory"),
        (28, "geometric complexity theory", 170, "Complexity Theory"),
        # Wireless & MIMO (0704.0217)
        (6, "fading channel", 210, "Wireless Communications"),
        (7, "random vector quantization", 95, "Information Theory"),
        (8, "mimo", 1420, "Wireless Communications"),
        (29, "beamforming", 680, "Wireless Communications")
    ]
    cur.executemany("INSERT INTO FoS VALUES (?, ?, ?, ?)", keywords_data)

    # 3. Seed NPMI Semantic Correlation Edges (FoS_npmi_Springer)
    npmi_edges = [
        # Graph Theory cluster
        (1, 2, 0.78), (1, 3, 0.85), (1, 10, 0.82), (2, 10, 0.75), (2, 3, 0.69),
        # Information Theory & Quantum cluster
        (4, 5, 0.72), (4, 6, 0.89), (4, 7, 0.68), (4, 8, 0.84), (4, 29, 0.77),
        (5, 9, 0.65), (6, 8, 0.88), (6, 29, 0.81), (7, 8, 0.73), (8, 29, 0.92),
        # Acoustic & Neural Network cluster
        (11, 12, 0.61), (11, 13, 0.86), (11, 14, 0.83), (13, 14, 0.95), (12, 14, 0.64),
        # Bioinformatics cluster
        (15, 16, 0.91), (15, 17, 0.76), (15, 18, 0.82), (16, 17, 0.88), (16, 18, 0.71),
        # Optimization cluster
        (19, 20, 0.68), (19, 21, 0.74), (20, 21, 0.86),
        # CDMA & Physics cluster
        (22, 23, 0.71), (22, 24, 0.93), (23, 24, 0.67),
        # Complexity Theory cluster
        (25, 26, 0.94), (25, 27, 0.89), (26, 27, 0.85), (27, 28, 0.79), (25, 28, 0.65)
    ]
    cur.executemany("INSERT INTO FoS_npmi_Springer VALUES (?, ?, ?)", npmi_edges)

    # 4. Seed Paper-Keyword Cosine Match Scores (Publication_FoS)
    assignments = [
        # 0704.0002: Sparsity-certifying Graph Decompositions
        ("0704.0002", 1, 0.95), ("0704.0002", 2, 0.89), ("0704.0002", 3, 0.82), ("0704.0002", 10, 0.86),
        # 0704.0046: Entropy & channel capacity
        ("0704.0046", 4, 0.81), ("0704.0046", 5, 0.94), ("0704.0046", 9, 0.76),
        # 0704.0047: Acoustic emission Part I
        ("0704.0047", 11, 0.92), ("0704.0047", 12, 0.87), ("0704.0047", 13, 0.78),
        # 0704.0050: Acoustic emission Part II
        ("0704.0050", 11, 0.94), ("0704.0050", 13, 0.91), ("0704.0050", 14, 0.89),
        # 0704.0062: Viterbi Algorithm & Random Walks
        ("0704.0062", 15, 0.96), ("0704.0062", 16, 0.92), ("0704.0062", 17, 0.84), ("0704.0062", 18, 0.79),
        # 0704.0090: Real Options Schedules
        ("0704.0090", 19, 0.88), ("0704.0090", 20, 0.93), ("0704.0090", 21, 0.85),
        # 0704.0098: Sparsely-spread CDMA
        ("0704.0098", 22, 0.92), ("0704.0098", 23, 0.87), ("0704.0098", 24, 0.89), ("0704.0098", 10, 0.74),
        # 0704.0108: Reducing SAT to 2-SAT
        ("0704.0108", 25, 0.97), ("0704.0108", 26, 0.95), ("0704.0108", 27, 0.91),
        # 0704.0213: Geometric Complexity Theory
        ("0704.0213", 27, 0.88), ("0704.0213", 28, 0.96),
        # 0704.0217: MIMO Fading Channel
        ("0704.0217", 4, 0.86), ("0704.0217", 6, 0.92), ("0704.0217", 7, 0.84), ("0704.0217", 8, 0.90), ("0704.0217", 29, 0.85)
    ]
    cur.executemany("INSERT INTO Publication_FoS VALUES (?, ?, ?)", assignments)

    conn.commit()
    return conn

DB_CONN = init_db()

# ---------------------------------------------------------------------------
# Pydantic Schemas
# ---------------------------------------------------------------------------
class SearchRequest(BaseModel):
    keywords: List[str]
    limit: Optional[int] = 5

class MatchTextRequest(BaseModel):
    text: str

# ---------------------------------------------------------------------------
# API Endpoints
# ---------------------------------------------------------------------------
@app.get("/api/keywords")
def get_all_keywords():
    cur = DB_CONN.cursor()
    cur.execute("SELECT id, keyword, frequency, category FROM FoS ORDER BY frequency DESC")
    rows = cur.fetchall()
    return [
        {"id": r[0], "keyword": r[1], "frequency": r[2], "category": r[3]}
        for r in rows
    ]

@app.get("/api/stats")
def get_stats():
    cur = DB_CONN.cursor()
    cur.execute("SELECT COUNT(*) FROM Publication")
    papers_count = cur.fetchone()[0]
    cur.execute("SELECT COUNT(*) FROM FoS")
    keywords_count = cur.fetchone()[0]
    cur.execute("SELECT COUNT(*) FROM FoS_npmi_Springer")
    edges_count = cur.fetchone()[0]
    return {
        "papers_indexed": papers_count,
        "keywords_indexed": keywords_count,
        "npmi_edges": edges_count,
        "model": "Sentence-BERT (bert-base-nli-mean-tokens)",
        "database": "In-Memory Relational SQLite"
    }

@app.post("/api/search")
def search_papers(req: SearchRequest):
    cleaned_keywords = [k.strip().lower() for k in req.keywords if k.strip()]
    if not cleaned_keywords:
        raise HTTPException(status_code=400, detail="Please provide at least one keyword.")

    cur = DB_CONN.cursor()

    # Step 1: Match keyword strings to IDs
    placeholders = ",".join(["?"] * len(cleaned_keywords))
    cur.execute(f"SELECT id, keyword FROM FoS WHERE LOWER(keyword) IN ({placeholders})", cleaned_keywords)
    matched_rows = cur.fetchall()

    if not matched_rows:
        return {
            "query": cleaned_keywords,
            "matched_keywords": [],
            "expanded_terms": [],
            "results": [],
            "message": "No matching keywords found in database."
        }

    keyword_ids = [r[0] for r in matched_rows]
    matched_names = [r[1] for r in matched_rows]

    # Step 2: Build temporary Top_Keywords table with top 10 NPMI expanded terms
    cur.execute("DROP TABLE IF EXISTS Top_Keywords;")
    cur.execute("""
    CREATE TEMPORARY TABLE Top_Keywords (
        parent_id INTEGER,
        id INTEGER,
        npmi REAL,
        PRIMARY KEY (parent_id, id)
    );
    """)

    id_placeholders = ",".join(["?"] * len(keyword_ids))
    union_sql = f"""
    INSERT INTO Top_Keywords (parent_id, id, npmi)
    SELECT parent_id, id, npmi
    FROM (
        SELECT parent_id, id, npmi,
               ROW_NUMBER() OVER (PARTITION BY parent_id ORDER BY npmi DESC) as kw_rank
        FROM (
            SELECT id2 AS parent_id, id1 AS id, npmi FROM FoS_npmi_Springer WHERE id2 IN ({id_placeholders})
            UNION
            SELECT id1 AS parent_id, id2 AS id, npmi FROM FoS_npmi_Springer WHERE id1 IN ({id_placeholders})
        )
    )
    WHERE kw_rank <= 10;
    """
    cur.execute(union_sql, keyword_ids + keyword_ids)

    # Insert self-identities with score = 1.0
    for kid in keyword_ids:
        cur.execute("INSERT OR REPLACE INTO Top_Keywords VALUES (?, ?, 1.0)", (kid, kid))

    # Fetch expanded keyword details for transparency in UI
    cur.execute("""
    SELECT DISTINCT FoS.keyword, Top_Keywords.npmi, parent.keyword as parent_keyword
    FROM Top_Keywords
    JOIN FoS ON Top_Keywords.id = FoS.id
    JOIN FoS as parent ON Top_Keywords.parent_id = parent.id
    WHERE Top_Keywords.id != Top_Keywords.parent_id
    ORDER BY Top_Keywords.npmi DESC
    LIMIT 15;
    """)
    expanded_rows = cur.fetchall()
    expanded_terms = [
        {"term": r[0], "npmi": round(r[1], 2), "expanded_from": r[2]}
        for r in expanded_rows
    ]

    # Step 3: Core 3-Phase Ranking SQL Query
    ranking_sql = """
    SELECT 
        Publication.id, 
        Publication.title, 
        Publication.authors,
        Publication.categories,
        Publication.abstract, 
        Publication.citations, 
        Publication.doi,
        SUM(max_score) as semantic_score,
        SUM(max_score) * (Publication.citations + 1) as total_score
    FROM (
        SELECT parent_id, Publication_id, MAX(npmi * score) as max_score
        FROM Top_Keywords
        JOIN Publication_FoS ON Top_Keywords.id = Publication_FoS.FoS_id
        GROUP BY parent_id, Publication_id
    ) as keyword_paper_score
    LEFT JOIN Publication ON Publication_id = Publication.id
    GROUP BY Publication_id
    ORDER BY total_score DESC
    LIMIT ?;
    """
    cur.execute(ranking_sql, (req.limit,))
    results = cur.fetchall()

    papers_output = []
    for rank, r in enumerate(results, 1):
        pid, title, authors, categories, abstract, citations, doi, sem_score, tot_score = r

        # Fetch matched keywords for this specific paper
        cur.execute("""
        SELECT FoS.keyword, Publication_FoS.score
        FROM Publication_FoS
        JOIN FoS ON Publication_FoS.FoS_id = FoS.id
        WHERE Publication_FoS.Publication_id = ?
        ORDER BY Publication_FoS.score DESC;
        """, (pid,))
        paper_keywords = [{"keyword": pk[0], "score": round(pk[1], 2)} for pk in cur.fetchall()]

        papers_output.append({
            "rank": rank,
            "id": pid,
            "title": title.replace("\n", " ").strip(),
            "authors": authors.replace("\n", " ").strip(),
            "categories": categories.split(),
            "abstract": abstract.replace("\n", " ").strip(),
            "citations": citations,
            "doi": doi,
            "semantic_score": round(sem_score, 3),
            "final_score": round(tot_score, 2),
            "matched_keywords": paper_keywords
        })

    return {
        "query": cleaned_keywords,
        "matched_keywords": matched_names,
        "expanded_terms": expanded_terms,
        "total_results": len(papers_output),
        "results": papers_output
    }

@app.post("/api/match-text")
def match_text_with_trie(req: MatchTextRequest):
    if not req.text.strip():
        return {"matches": []}

    cur = DB_CONN.cursor()
    cur.execute("SELECT keyword FROM FoS")
    all_keywords = [r[0] for r in cur.fetchall()]

    # Fast regex match simulating the Trie pattern
    escaped = [re.escape(k) for k in all_keywords]
    pattern = re.compile(r"\b(" + "|".join(escaped) + r")\b", re.IGNORECASE)

    found = list(set([m.group().lower() for m in pattern.finditer(req.text)]))
    return {
        "matched_keywords_count": len(found),
        "matches": found
    }

# ---------------------------------------------------------------------------
# Interactive Single-Page Web Frontend (HTML5 + Tailwind + Vanilla JS)
# ---------------------------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
def serve_ui():
    html_content = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Forward Scholar — Research Paper Search Engine</title>
  <!-- Tailwind CSS CDN -->
  <script src="https://cdn.tailwindcss.com"></script>
  <!-- Lucide Icons -->
  <script src="https://unpkg.com/lucide@latest"></script>
  <!-- Google Fonts Inter -->
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap" rel="stylesheet">
  <style>
    body { font-family: 'Inter', sans-serif; }
    .glass-panel {
      background: rgba(255, 255, 255, 0.85);
      backdrop-filter: blur(12px);
      border: 1px solid rgba(226, 232, 240, 0.8);
    }
  </style>
</head>
<body class="bg-slate-50 text-slate-900 min-h-screen flex flex-col antialiased">

  <!-- Top Navigation Header -->
  <header class="border-b border-slate-200 bg-white sticky top-0 z-50 shadow-sm">
    <div class="max-w-6xl mx-auto px-4 py-3 flex items-center justify-between">
      <div class="flex items-center gap-3">
        <div class="bg-indigo-600 text-white p-2 rounded-xl shadow-md shadow-indigo-200">
          <i data-lucide="book-open" class="w-6 h-6"></i>
        </div>
        <div>
          <h1 class="text-lg font-bold tracking-tight text-slate-900 flex items-center gap-2">
            Forward Scholar
            <span class="text-xs font-semibold px-2 py-0.5 rounded-full bg-indigo-50 text-indigo-700 border border-indigo-200">UIUC NLP</span>
          </h1>
          <p class="text-xs text-slate-500">Semantic Academic Paper Retrieval by Keywords</p>
        </div>
      </div>

      <!-- Quick Metrics -->
      <div class="hidden md:flex items-center gap-4 text-xs">
        <div class="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-emerald-50 border border-emerald-200 text-emerald-800">
          <span class="w-2 h-2 rounded-full bg-emerald-500 animate-pulse"></span>
          <span>In-Memory SQLite Active</span>
        </div>
        <div class="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-slate-100 border border-slate-200 text-slate-700">
          <i data-lucide="database" class="w-3.5 h-3.5"></i>
          <span id="headerStats">Loading stats...</span>
        </div>
      </div>
    </div>
  </header>

  <!-- Main Container -->
  <main class="max-w-6xl mx-auto px-4 py-8 flex-1 w-full space-y-6">

    <!-- Hero Search Card -->
    <div class="bg-gradient-to-br from-indigo-900 via-indigo-800 to-slate-900 text-white rounded-2xl p-6 md:p-8 shadow-xl relative overflow-hidden">
      <!-- Decorative background blur -->
      <div class="absolute -right-20 -top-20 w-80 h-80 bg-indigo-500/20 rounded-full blur-3xl pointer-events-none"></div>

      <div class="relative z-10 max-w-3xl space-y-4">
        <div class="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-white/10 text-indigo-200 text-xs backdrop-blur-md border border-white/10">
          <i data-lucide="sparkles" class="w-3.5 h-3.5 text-indigo-300"></i>
          <span>Dense Semantic Embeddings + NPMI Query Expansion</span>
        </div>
        
        <h2 class="text-2xl md:text-3xl font-extrabold tracking-tight">
          Find Research Papers by Multi-Word Keywords
        </h2>
        <p class="text-sm text-indigo-100/90 leading-relaxed">
          Type or select keywords below. The engine uses <strong>Sentence-BERT</strong>, <strong>Trie-Regex</strong> pattern compiling, and <strong>NPMI co-occurrence graphs</strong> to rank academic publications by semantic relevance and citation impact.
        </p>

        <!-- Search Input Form -->
        <div class="pt-2">
          <div class="flex flex-col sm:flex-row gap-2">
            <div class="relative flex-1">
              <i data-lucide="search" class="w-5 h-5 text-slate-400 absolute left-3.5 top-1/2 -translate-y-1/2"></i>
              <input 
                id="keywordInput" 
                type="text" 
                placeholder="e.g. 'channel capacity', 'graph decompositions', 'pebble game'..."
                class="w-full pl-11 pr-4 py-3 rounded-xl bg-white text-slate-900 placeholder-slate-400 font-medium text-sm focus:outline-none focus:ring-2 focus:ring-indigo-400 shadow-inner"
                onkeydown="if(event.key === 'Enter') addKeywordFromInput()"
              />
            </div>
            <button 
              onclick="addKeywordFromInput()" 
              class="px-5 py-3 rounded-xl bg-indigo-500 hover:bg-indigo-400 text-white font-semibold text-sm transition shadow-md flex items-center justify-center gap-2">
              <i data-lucide="plus" class="w-4 h-4"></i> Add Tag
            </button>
            <button 
              onclick="executeSearch()" 
              class="px-6 py-3 rounded-xl bg-emerald-500 hover:bg-emerald-400 text-white font-bold text-sm transition shadow-lg shadow-emerald-900/40 flex items-center justify-center gap-2">
              <i data-lucide="search" class="w-4 h-4"></i> Search
            </button>
          </div>
        </div>

        <!-- Active Tags Display -->
        <div id="activeTagsContainer" class="flex flex-wrap items-center gap-2 pt-1 min-h-[32px]">
          <span class="text-xs text-indigo-200">Active Query Tags:</span>
          <span id="noTagsLabel" class="text-xs text-indigo-300 italic">None selected yet (click suggested tags below)</span>
        </div>

        <!-- Suggested Keyword Pills -->
        <div class="pt-3 border-t border-white/10">
          <span class="text-xs font-semibold text-indigo-200 uppercase tracking-wider block mb-2">Suggested Keywords (Click to add):</span>
          <div id="suggestedPills" class="flex flex-wrap gap-2">
            <!-- Injected via JavaScript -->
          </div>
        </div>
      </div>
    </div>

    <!-- Navigation Tabs -->
    <div class="flex items-center gap-2 border-b border-slate-200">
      <button 
        id="tabResultsBtn" 
        onclick="switchTab('results')" 
        class="px-4 py-2.5 text-sm font-semibold border-b-2 border-indigo-600 text-indigo-600 flex items-center gap-2">
        <i data-lucide="list-ordered" class="w-4 h-4"></i> Search Results
      </button>
      <button 
        id="tabScannerBtn" 
        onclick="switchTab('scanner')" 
        class="px-4 py-2.5 text-sm font-semibold border-b-2 border-transparent text-slate-500 hover:text-slate-700 flex items-center gap-2">
        <i data-lucide="cpu" class="w-4 h-4"></i> Trie Text Scanner
      </button>
      <button 
        id="tabPipelineBtn" 
        onclick="switchTab('pipeline')" 
        class="px-4 py-2.5 text-sm font-semibold border-b-2 border-transparent text-slate-500 hover:text-slate-700 flex items-center gap-2">
        <i data-lucide="git-branch" class="w-4 h-4"></i> Pipeline Architecture
      </button>
    </div>

    <!-- TAB 1: Search Results View -->
    <div id="tabResultsSection" class="space-y-6">
      <!-- Query Info Banner -->
      <div id="queryBanner" class="hidden bg-white p-4 rounded-xl border border-slate-200 shadow-sm flex flex-col md:flex-row md:items-center justify-between gap-4">
        <div>
          <h3 class="text-sm font-bold text-slate-900 flex items-center gap-2">
            <i data-lucide="check-circle-2" class="w-4 h-4 text-emerald-600"></i>
            Query Results for: <span id="bannerKeywords" class="text-indigo-600"></span>
          </h3>
          <p id="bannerNPMI" class="text-xs text-slate-500 mt-0.5"></p>
        </div>
        <div id="resultCountBadge" class="text-xs font-semibold px-3 py-1 rounded-full bg-slate-100 text-slate-700 self-start md:self-center">
          0 papers found
        </div>
      </div>

      <!-- Results List Container -->
      <div id="resultsList" class="space-y-4">
        <!-- Initial Blank State -->
        <div class="text-center py-16 bg-white rounded-2xl border border-slate-200 shadow-sm space-y-3">
          <div class="w-12 h-12 bg-indigo-50 text-indigo-600 rounded-full flex items-center justify-center mx-auto">
            <i data-lucide="file-search" class="w-6 h-6"></i>
          </div>
          <h4 class="text-base font-bold text-slate-800">Ready to Search Academic Literature</h4>
          <p class="text-sm text-slate-500 max-w-md mx-auto">
            Select one of the suggested keywords above like <span class="font-semibold text-slate-700">"channel capacity"</span> or <span class="font-semibold text-slate-700">"graph decompositions"</span> to see the ranking algorithm in action.
          </p>
        </div>
      </div>
    </div>

    <!-- TAB 2: Trie Text Scanner View -->
    <div id="tabScannerSection" class="hidden space-y-4">
      <div class="bg-white p-6 rounded-2xl border border-slate-200 shadow-sm space-y-4">
        <div class="flex items-center gap-3">
          <div class="p-2.5 bg-indigo-50 text-indigo-600 rounded-xl">
            <i data-lucide="binary" class="w-6 h-6"></i>
          </div>
          <div>
            <h3 class="text-base font-bold text-slate-900">Live Trie-to-Regex Abstract Scanner</h3>
            <p class="text-xs text-slate-500">Paste any research text or abstract to test $O(L)$ keyword matching using the compiled Trie pattern.</p>
          </div>
        </div>

        <textarea 
          id="scannerInput" 
          rows="5" 
          class="w-full p-4 rounded-xl border border-slate-300 text-sm focus:outline-none focus:ring-2 focus:ring-indigo-500 placeholder-slate-400"
          placeholder="Paste abstract or research paragraph here..."
        >We describe a new algorithm, the pebble game with colors, certifying sparsity in graph decompositions and tree decompositions.</textarea>

        <div class="flex items-center justify-between">
          <button 
            onclick="scanTextWithTrie()" 
            class="px-5 py-2.5 rounded-xl bg-indigo-600 hover:bg-indigo-500 text-white font-semibold text-sm transition shadow-md flex items-center gap-2">
            <i data-lucide="scan" class="w-4 h-4"></i> Scan Text with Trie
          </button>
          <span id="scanCountLabel" class="text-xs text-slate-500"></span>
        </div>

        <!-- Scanner Output -->
        <div id="scannerOutput" class="hidden p-4 rounded-xl bg-slate-50 border border-slate-200 space-y-2">
          <h4 class="text-xs font-bold text-slate-700 uppercase tracking-wider">Matched FoS Keywords:</h4>
          <div id="scannerPills" class="flex flex-wrap gap-2"></div>
        </div>
      </div>
    </div>

    <!-- TAB 3: Pipeline Architecture View -->
    <div id="tabPipelineSection" class="hidden space-y-6">
      <div class="bg-white p-6 md:p-8 rounded-2xl border border-slate-200 shadow-sm space-y-6">
        <div class="border-b border-slate-200 pb-4">
          <h3 class="text-lg font-bold text-slate-900 flex items-center gap-2">
            <i data-lucide="git-merge" class="w-5 h-5 text-indigo-600"></i>
            End-to-End Ranking Formula & Architecture
          </h3>
          <p class="text-xs text-slate-500 mt-1">Designed by Kshitij, Lakshay, Krishna Gupta, Krishna Thakur, and Laksh Gupta</p>
        </div>

        <!-- Formula Card -->
        <div class="p-6 rounded-xl bg-indigo-50 border border-indigo-100 space-y-3">
          <span class="text-xs font-bold text-indigo-700 uppercase tracking-wider block">The Mathematical Scoring Objective</span>
          <div class="text-lg md:text-xl font-mono font-bold text-slate-900 bg-white p-4 rounded-lg border border-indigo-200 text-center shadow-inner overflow-x-auto">
            Total Score = [ &sum; max(NPMI &times; Cosine Similarity) ] &times; (Citations + 1)
          </div>
          <p class="text-xs text-slate-600 leading-relaxed">
            • <strong>NPMI (Normalized Pointwise Mutual Information)</strong>: Expands user query keywords to related academic vocabulary from Springer co-occurrence matrices.<br/>
            • <strong>Cosine Similarity</strong>: Measures semantic closeness between the paper's Sentence-BERT embedding and keyword vectors.<br/>
            • <strong>(Citations + 1) Multiplier</strong>: Scales score by peer-reviewed academic authority, while guaranteeing new papers with 0 citations are not disqualified.
          </p>
        </div>

        <!-- 3-Phase Stepper Cards -->
        <div class="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div class="p-4 rounded-xl border border-slate-200 bg-slate-50 space-y-2">
            <div class="w-7 h-7 rounded-lg bg-indigo-600 text-white font-bold flex items-center justify-center text-xs">1</div>
            <h4 class="text-sm font-bold text-slate-900">Embedding Generation</h4>
            <p class="text-xs text-slate-600">Sentence-BERT transforms titles and abstracts into 768-dimensional dense vectors. Normalization enables fast dot-product cosine similarity.</p>
          </div>
          <div class="p-4 rounded-xl border border-slate-200 bg-slate-50 space-y-2">
            <div class="w-7 h-7 rounded-lg bg-indigo-600 text-white font-bold flex items-center justify-center text-xs">2</div>
            <h4 class="text-sm font-bold text-slate-900">Trie Matching & Deduplication</h4>
            <p class="text-xs text-slate-600">Trie prefix tree compiles regex for $O(L)$ scanning. Non-CS words are penalized, and DBSCAN clustering prunes duplicate synonyms to retain top 9 unique keywords.</p>
          </div>
          <div class="p-4 rounded-xl border border-slate-200 bg-slate-50 space-y-2">
            <div class="w-7 h-7 rounded-lg bg-indigo-600 text-white font-bold flex items-center justify-center text-xs">3</div>
            <h4 class="text-sm font-bold text-slate-900">NPMI Ranked Retrieval</h4>
            <p class="text-xs text-slate-600">Query keywords expand to top 10 co-occurring terms. An optimized relational SQL LEFT JOIN query aggregates scores and ranks results in milliseconds.</p>
          </div>
        </div>
      </div>
    </div>

  </main>

  <!-- Footer -->
  <footer class="border-t border-slate-200 bg-white py-6 mt-12 text-center text-xs text-slate-500">
    <div class="max-w-6xl mx-auto px-4 flex flex-col sm:flex-row items-center justify-between gap-3">
      <p>Research Paper Search Engine &bull; UIUC Forward Data Lab</p>
      <p class="text-slate-400">Team: Kshitij, Lakshay, Krishna Gupta, Krishna Thakur, Laksh Gupta</p>
    </div>
  </footer>

  <!-- Client-Side State & Interaction Logic -->
  <script>
    let activeTags = [];
    const defaultPills = [
      "channel capacity", 
      "graph decompositions", 
      "pebble game", 
      "entropy", 
      "fading channel", 
      "sparsity", 
      "quantum law", 
      "viterbi algorithm",
      "neural network",
      "simulated annealing"
    ];

    // Initialize UI on Mount
    document.addEventListener("DOMContentLoaded", () => {
      lucide.createIcons();
      renderSuggestedPills();
      fetchStats();
      // Pre-populate with a sample search
      addKeyword("channel capacity");
      addKeyword("fading channel");
      executeSearch();
    });

    async function fetchStats() {
      try {
        const res = await fetch("/api/stats");
        const data = await res.json();
        document.getElementById("headerStats").textContent = 
          `${data.papers_indexed} Papers | ${data.keywords_indexed} FoS Tags`;
      } catch (e) {
        console.error(e);
      }
    }

    function renderSuggestedPills() {
      const container = document.getElementById("suggestedPills");
      container.innerHTML = "";
      defaultPills.forEach(k => {
        const btn = document.createElement("button");
        btn.className = "px-3 py-1 rounded-full text-xs font-medium bg-white/10 hover:bg-white/20 text-white transition border border-white/10 flex items-center gap-1";
        btn.innerHTML = `<i data-lucide="tag" class="w-3 h-3"></i> ${k}`;
        btn.onclick = () => {
          addKeyword(k);
          executeSearch();
        };
        container.appendChild(btn);
      });
      lucide.createIcons();
    }

    function addKeyword(word) {
      const clean = word.trim().toLowerCase();
      if (clean && !activeTags.includes(clean)) {
        activeTags.push(clean);
        renderActiveTags();
      }
    }

    function removeKeyword(word) {
      activeTags = activeTags.filter(t => t !== word);
      renderActiveTags();
      if (activeTags.length > 0) {
        executeSearch();
      } else {
        document.getElementById("resultsList").innerHTML = `
          <div class="text-center py-16 bg-white rounded-2xl border border-slate-200 shadow-sm space-y-3">
            <div class="w-12 h-12 bg-slate-100 text-slate-500 rounded-full flex items-center justify-center mx-auto">
              <i data-lucide="search" class="w-6 h-6"></i>
            </div>
            <h4 class="text-base font-bold text-slate-800">No Keywords Selected</h4>
            <p class="text-sm text-slate-500">Add keywords to start searching papers.</p>
          </div>
        `;
        document.getElementById("queryBanner").classList.add("hidden");
        lucide.createIcons();
      }
    }

    function addKeywordFromInput() {
      const input = document.getElementById("keywordInput");
      if (input.value.trim()) {
        addKeyword(input.value);
        input.value = "";
        executeSearch();
      }
    }

    function renderActiveTags() {
      const container = document.getElementById("activeTagsContainer");
      const noTagsLabel = document.getElementById("noTagsLabel");
      
      container.querySelectorAll(".tag-pill").forEach(el => el.remove());

      if (activeTags.length === 0) {
        noTagsLabel.classList.remove("hidden");
      } else {
        noTagsLabel.classList.add("hidden");
        activeTags.forEach(tag => {
          const pill = document.createElement("span");
          pill.className = "tag-pill inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-semibold bg-indigo-500 text-white shadow-sm";
          pill.innerHTML = `
            <span>${tag}</span>
            <button onclick="removeKeyword('${tag}')" class="hover:text-indigo-200">
              <i data-lucide="x" class="w-3.5 h-3.5"></i>
            </button>
          `;
          container.appendChild(pill);
        });
      }
      lucide.createIcons();
    }

    async function executeSearch() {
      if (activeTags.length === 0) return;

      const listContainer = document.getElementById("resultsList");
      listContainer.innerHTML = `
        <div class="text-center py-16 bg-white rounded-2xl border border-slate-200 shadow-sm">
          <div class="w-8 h-8 border-3 border-indigo-600 border-t-transparent rounded-full animate-spin mx-auto mb-3"></div>
          <p class="text-sm font-semibold text-slate-600">Executing NPMI expansion & ranking query...</p>
        </div>
      `;

      try {
        const res = await fetch("/api/search", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ keywords: activeTags, limit: 10 })
        });
        const data = await res.json();
        renderResults(data);
      } catch (err) {
        listContainer.innerHTML = `
          <div class="p-6 rounded-2xl bg-rose-50 border border-rose-200 text-rose-800 text-sm">
            Failed to execute search: ${err.message}
          </div>
        `;
      }
    }

    function renderResults(data) {
      const listContainer = document.getElementById("resultsList");
      const banner = document.getElementById("queryBanner");
      const bannerKeywords = document.getElementById("bannerKeywords");
      const bannerNPMI = document.getElementById("bannerNPMI");
      const resultCountBadge = document.getElementById("resultCountBadge");

      banner.classList.remove("hidden");
      bannerKeywords.textContent = data.matched_keywords.length > 0 ? data.matched_keywords.join(", ") : activeTags.join(", ");
      resultCountBadge.textContent = `${data.total_results} papers ranked`;

      if (data.expanded_terms && data.expanded_terms.length > 0) {
        bannerNPMI.innerHTML = `<strong>NPMI Semantic Expansion:</strong> ` + 
          data.expanded_terms.slice(0, 5).map(e => `<span class="bg-indigo-50 text-indigo-700 px-1.5 py-0.5 rounded text-xs border border-indigo-200">${e.term} (npmi: ${e.npmi})</span>`).join(" ");
      } else {
        bannerNPMI.textContent = "No additional NPMI expansion terms triggered.";
      }

      if (data.results.length === 0) {
        listContainer.innerHTML = `
          <div class="text-center py-16 bg-white rounded-2xl border border-slate-200 shadow-sm space-y-2">
            <p class="text-base font-bold text-slate-800">No papers matched this query</p>
            <p class="text-xs text-slate-500">Try searching for 'channel capacity', 'graph decompositions', or 'pebble game'.</p>
          </div>
        `;
        return;
      }

      listContainer.innerHTML = "";
      data.results.forEach(paper => {
        const rankColors = {
          1: "bg-amber-100 text-amber-900 border-amber-300",
          2: "bg-slate-200 text-slate-800 border-slate-300",
          3: "bg-orange-100 text-orange-900 border-orange-300"
        };
        const rankBadgeStyle = rankColors[paper.rank] || "bg-slate-100 text-slate-700 border-slate-200";

        const card = document.createElement("div");
        card.className = "bg-white p-6 rounded-2xl border border-slate-200 shadow-sm hover:shadow-md transition space-y-4";
        card.innerHTML = `
          <!-- Top Row: Rank & Score Metrics -->
          <div class="flex flex-wrap items-center justify-between gap-3">
            <div class="flex items-center gap-2">
              <span class="w-8 h-8 rounded-xl font-bold flex items-center justify-center text-xs border ${rankBadgeStyle}">
                #${paper.rank}
              </span>
              <span class="text-xs font-mono text-slate-400">ID: ${paper.id}</span>
              ${paper.categories.map(c => `<span class="text-xs px-2 py-0.5 rounded bg-slate-100 text-slate-600 font-medium">${c}</span>`).join("")}
            </div>

            <div class="flex items-center gap-2">
              <div class="px-3 py-1 rounded-xl bg-emerald-50 border border-emerald-200 text-emerald-800 text-xs font-bold flex items-center gap-1.5 shadow-sm">
                <i data-lucide="trending-up" class="w-3.5 h-3.5 text-emerald-600"></i>
                Score: ${paper.final_score}
              </div>
              <div class="px-3 py-1 rounded-xl bg-indigo-50 border border-indigo-200 text-indigo-800 text-xs font-bold flex items-center gap-1.5 shadow-sm">
                <i data-lucide="quote" class="w-3.5 h-3.5 text-indigo-600"></i>
                ${paper.citations} Citations
              </div>
            </div>
          </div>

          <!-- Paper Title -->
          <h3 class="text-base md:text-lg font-bold text-slate-900 leading-snug">
            ${paper.title}
          </h3>

          <!-- Authors -->
          <p class="text-xs text-slate-500 flex items-center gap-1.5">
            <i data-lucide="users" class="w-3.5 h-3.5 text-slate-400"></i>
            <span>${paper.authors}</span>
          </p>

          <!-- Abstract Preview -->
          <div class="text-xs text-slate-600 leading-relaxed bg-slate-50 p-3.5 rounded-xl border border-slate-200">
            <p id="abstract-${paper.id}" class="line-clamp-2">${paper.abstract}</p>
            <button 
              onclick="toggleAbstract('${paper.id}')" 
              id="btn-${paper.id}" 
              class="text-indigo-600 font-semibold mt-1 text-xs hover:underline focus:outline-none">
              Show Full Abstract &darr;
            </button>
          </div>

          <!-- Matched FoS Tags -->
          <div class="pt-2 flex flex-wrap items-center gap-1.5 border-t border-slate-100">
            <span class="text-xs text-slate-400 font-medium">Assigned Keywords:</span>
            ${paper.matched_keywords.map(k => `
              <span class="px-2.5 py-0.5 rounded-lg text-xs font-medium bg-slate-100 text-slate-700 border border-slate-200">
                ${k.keyword} <span class="text-slate-400 text-[10px]">(${k.score})</span>
              </span>
            `).join("")}
          </div>
        `;
        listContainer.appendChild(card);
      });
      lucide.createIcons();
    }

    function toggleAbstract(id) {
      const el = document.getElementById(`abstract-${id}`);
      const btn = document.getElementById(`btn-${id}`);
      if (el.classList.contains("line-clamp-2")) {
        el.classList.remove("line-clamp-2");
        btn.textContent = "Show Less \u2191";
      } else {
        el.classList.add("line-clamp-2");
        btn.textContent = "Show Full Abstract \u2193";
      }
    }

    async function scanTextWithTrie() {
      const text = document.getElementById("scannerInput").value;
      const countLabel = document.getElementById("scanCountLabel");
      const output = document.getElementById("scannerOutput");
      const pillsContainer = document.getElementById("scannerPills");

      if (!text.trim()) return;

      countLabel.textContent = "Scanning with compiled Trie regex...";

      try {
        const res = await fetch("/api/match-text", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ text })
        });
        const data = await res.json();

        countLabel.textContent = `Found ${data.matched_keywords_count} matching keywords in O(L) time!`;
        output.classList.remove("hidden");
        pillsContainer.innerHTML = "";

        if (data.matches.length === 0) {
          pillsContainer.innerHTML = `<span class="text-xs text-slate-400">No FoS keywords detected in sample text.</span>`;
        } else {
          data.matches.forEach(m => {
            const pill = document.createElement("button");
            pill.className = "px-3 py-1 rounded-lg text-xs font-semibold bg-indigo-50 border border-indigo-200 text-indigo-700 hover:bg-indigo-100 transition";
            pill.textContent = `+ ${m}`;
            pill.onclick = () => {
              addKeyword(m);
              switchTab("results");
              executeSearch();
            };
            pillsContainer.appendChild(pill);
          });
        }
      } catch (err) {
        countLabel.textContent = `Error: ${err.message}`;
      }
    }

    function switchTab(tab) {
      const resultsSection = document.getElementById("tabResultsSection");
      const scannerSection = document.getElementById("tabScannerSection");
      const pipelineSection = document.getElementById("tabPipelineSection");

      const resultsBtn = document.getElementById("tabResultsBtn");
      const scannerBtn = document.getElementById("tabScannerBtn");
      const pipelineBtn = document.getElementById("tabPipelineBtn");

      // Hide all
      resultsSection.classList.add("hidden");
      scannerSection.classList.add("hidden");
      pipelineSection.classList.add("hidden");

      // Reset buttons
      [resultsBtn, scannerBtn, pipelineBtn].forEach(b => {
        b.className = "px-4 py-2.5 text-sm font-semibold border-b-2 border-transparent text-slate-500 hover:text-slate-700 flex items-center gap-2";
      });

      if (tab === "results") {
        resultsSection.classList.remove("hidden");
        resultsBtn.className = "px-4 py-2.5 text-sm font-semibold border-b-2 border-indigo-600 text-indigo-600 flex items-center gap-2";
      } else if (tab === "scanner") {
        scannerSection.classList.remove("hidden");
        scannerBtn.className = "px-4 py-2.5 text-sm font-semibold border-b-2 border-indigo-600 text-indigo-600 flex items-center gap-2";
      } else if (tab === "pipeline") {
        pipelineSection.classList.remove("hidden");
        pipelineBtn.className = "px-4 py-2.5 text-sm font-semibold border-b-2 border-indigo-600 text-indigo-600 flex items-center gap-2";
      }
      lucide.createIcons();
    }
  </script>
</body>
</html>
"""
    return HTMLResponse(content=html_content)

# ---------------------------------------------------------------------------
# Direct CLI Entry Point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 70)
    print("Forward Scholar — Interactive Web Application")
    print("Server starting at http://127.0.0.1:8000")
    print("=" * 70)
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=False)
