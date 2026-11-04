"""
Local Demo Runner for Find-Papers-By-Keyword
Demonstrates the complete end-to-end ranking and search pipeline using local SQLite
and sample papers from test_data/papers.json.

Usage:
    python local_demo.py [keyword1] [keyword2] ...
Example:
    python local_demo.py "channel capacity" "fading channel"
    python local_demo.py "graph decompositions" "sparsity"
"""

import sqlite3
import json
import argparse
import sys

def create_sqlite_database():
    conn = sqlite3.connect(":memory:")
    cur = conn.cursor()

    cur.execute("""
    CREATE TABLE FoS (
        id INTEGER PRIMARY KEY,
        keyword TEXT,
        frequency INTEGER
    );
    """)

    cur.execute("""
    CREATE TABLE Publication (
        id TEXT PRIMARY KEY,
        title TEXT,
        abstract TEXT,
        citations INTEGER
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
        len1 INTEGER,
        len2 INTEGER,
        intersection_len INTEGER,
        npmi REAL,
        PRIMARY KEY (id1, id2)
    );
    """)

    # 1. Populate sample publications from test_data/papers.json
    with open("test_data/papers.json", "r", encoding="utf-8") as f:
        papers = json.load(f)

    for i, p in enumerate(papers):
        citations = (10 - i) * 15
        cur.execute(
            "INSERT INTO Publication VALUES (?, ?, ?, ?)",
            (p["id"], p["title"], p["abstract"], citations)
        )

    # 2. Populate sample Fields of Study (Keywords)
    keywords = [
        (1, "graph decompositions", 450),
        (2, "pebble game", 120),
        (3, "tree decompositions", 340),
        (4, "channel capacity", 520),
        (5, "entropy", 680),
        (6, "fading channel", 210),
        (7, "random vector quantization", 95),
        (8, "machine learning", 2500),
        (9, "quantum law", 180),
        (10, "sparsity", 290)
    ]
    cur.executemany("INSERT INTO FoS VALUES (?, ?, ?)", keywords)

    # 3. Populate NPMI Semantic Co-occurrence similarities
    npmi_data = [
        (1, 2, 450, 120, 80, 0.78),
        (1, 3, 450, 340, 210, 0.85),
        (1, 10, 450, 290, 190, 0.82),
        (4, 5, 520, 680, 310, 0.72),
        (4, 6, 520, 210, 150, 0.89),
        (4, 7, 520, 95, 75, 0.68),
        (5, 9, 680, 180, 90, 0.65),
    ]
    cur.executemany("INSERT INTO FoS_npmi_Springer VALUES (?, ?, ?, ?, ?, ?)", npmi_data)

    # 4. Populate Publication-Keyword match scores (Publication_FoS)
    assignments = [
        ("0704.0002", 1, 0.94),   # graph decompositions
        ("0704.0002", 2, 0.88),   # pebble game
        ("0704.0002", 3, 0.81),   # tree decompositions
        ("0704.0002", 10, 0.85),  # sparsity
        ("0704.0046", 4, 0.79),   # channel capacity
        ("0704.0046", 5, 0.92),   # entropy
        ("0704.0046", 9, 0.74),   # quantum law
        ("0704.0217", 4, 0.86),   # channel capacity
        ("0704.0217", 6, 0.91),   # fading channel
        ("0704.0217", 7, 0.83),   # random vector quantization
    ]
    cur.executemany("INSERT INTO Publication_FoS VALUES (?, ?, ?)", assignments)
    conn.commit()
    return conn

def search_papers(conn, query_keywords, limit=5):
    cur = conn.cursor()

    # Step A: Find keyword IDs for query words
    placeholders = ",".join(["?"] * len(query_keywords))
    cur.execute(f"SELECT id, keyword FROM FoS WHERE keyword IN ({placeholders})", query_keywords)
    matched = cur.fetchall()
    if not matched:
        print(f"\n[!] No matching keywords found in FoS database for query: {query_keywords}")
        print("    Available keywords in sample database: 'graph decompositions', 'pebble game', 'tree decompositions', 'channel capacity', 'entropy', 'fading channel', 'random vector quantization', 'machine learning', 'quantum law', 'sparsity'")
        return []

    keyword_ids = [m[0] for m in matched]
    print(f"\n[*] Query matched {len(keyword_ids)} keywords: {[m[1] for m in matched]} (IDs: {keyword_ids})")

    # Step B: Build Top_Keywords with top-10 similar keywords via NPMI
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

    # Step C: Execute Core Ranking Algorithm
    # total_score = SUM(max(npmi * match_score)) * (citations + 1)
    ranking_sql = f"""
    SELECT Publication_id, title, abstract, citations, SUM(max_score) * (citations + 1) as total_score
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
    cur.execute(ranking_sql, (limit,))
    results = cur.fetchall()
    return results

def print_results(results):
    if not results:
        return
    print(f"\n=== Top {len(results)} Ranked Papers ===")
    for rank, res in enumerate(results, 1):
        pid, title, abstract, citations, score = res
        clean_title = " ".join(title.split())
        clean_abstract = " ".join(abstract.split())
        if len(clean_abstract) > 160:
            clean_abstract = clean_abstract[:160] + "..."
        print(f"\n[Rank {rank}] Final Score: {score:.2f} | Citations: {citations}")
        print(f"Paper ID: {pid}")
        print(f"Title:    {clean_title}")
        print(f"Abstract: {clean_abstract}")

def main():
    parser = argparse.ArgumentParser(description="Find papers by keywords (Local SQLite Runner)")
    parser.add_argument("keywords", type=str, nargs="*", help="Keywords in search query (e.g. 'channel capacity')")
    args = parser.parse_args()

    conn = create_sqlite_database()

    if args.keywords:
        print(f"Running user query: {args.keywords}")
        results = search_papers(conn, args.keywords, limit=5)
        print_results(results)
    else:
        print("No keywords passed. Running automated test suite across sample queries:\n")
        sample_queries = [
            ["graph decompositions"],
            ["channel capacity", "fading channel"],
            ["pebble game", "sparsity"],
            ["entropy", "quantum law"]
        ]
        for query in sample_queries:
            print("-" * 65)
            print(f">>> Search Query: {query}")
            results = search_papers(conn, query, limit=3)
            print_results(results)

    print("\n" + "=" * 65)
    print("Execution completed successfully!")
    print("=" * 65)

if __name__ == "__main__":
    main()
