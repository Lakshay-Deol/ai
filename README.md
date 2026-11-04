# find-papers-by-keyword

This module is responsible for finding research papers that are most relevant to a set of query keywords. The list of papers should be ranked by their relavance to the keywords.

## Interactive Web Frontend (FastAPI + Modern Dashboard) 🚀
Launch the complete interactive web interface locally in one command:
```bash
python app.py
```
Open **[http://127.0.0.1:8000](http://127.0.0.1:8000)** in your browser. Features include:
* **Interactive Keyword Search**: Dynamic multi-keyword tagging with one-click suggestion pills.
* **Citation-Weighted Results**: Ranked paper cards displaying final scores, citation badges, and expandable abstracts.
* **NPMI Expansion Viewer**: Live breakdown of how query terms expand to co-occurring academic vocabulary.
* **Trie Abstract Scanner**: Paste any research abstract to test instant $O(L)$ keyword detection.
* **Algorithmic Pipeline Visualizer**: Interactive architectural diagram illustrating the 3-phase retrieval flow.

---

## Quick Local CLI Run (Zero-Config SQLite Demo)
To test paper search via terminal without configuring external MySQL servers:
```bash
python local_demo.py "channel capacity" "fading channel"
# Or run automated test suite across sample queries:
python local_demo.py
```

---

## Production Setup (with Full MySQL Dataset)
1) Install necessary module dependencies
```bash
pip install -r requirements.txt
```
2) Download `drive_data.zip` from [Forward Shared Data Drive](https://drive.google.com/drive/u/1/folders/1vq72EBXH38lb7qJbJsBIkHZiOW35NByI). Uncompress into a folder named `data/`
3) Install and start MySQL
4) Create an empty MySQL database and populate it using `dump.sql`:
```bash
mysql -u [user] -p [database_name] < data/dump.sql
```
5) Update credentials in `src/sql_creds.py` and run:
```bash
python src/find_papers.py "machine learning" "genetic algorithms"
```

---

## Testing
This module contains a test suite to verify the functionality of the components:
* To run file reader unit tests:
```bash
python -m unittest test/file_readers/test_paper_file_reader.py
```

---

## Team & Student Roles
* **Kshitij**: Core Algorithm & Search Architecture Lead (3-Phase Pipeline, NPMI expansion, SQL Join optimizations).
* **Lakshay (Lakshay Deol)**: Lead Frontend & Full-Stack Systems Engineer (Designed & built the entire Interactive Web Frontend `app.py`, client-side UI/UX dashboard, Trie abstract scanner, NPMI visualization, FastAPI REST API, and local SQLite engine `local_demo.py`).
* **Krishna Gupta**: NLP & Semantic Embeddings Specialist (Sentence-BERT encoding, vector normalization, cosine similarity).
* **Krishna Thakur**: Trie Data Structure & Regex Optimization Engineer (Prefix tree construction, regex union compiler).
* **Laksh Gupta**: Database Architecture & Citation Ranking Engineer (MySQL schema design, citation-weighted ranking score, domain penalties).

*For a detailed breakdown of technologies and roles, see [`TEAM_AND_TECH.md`](./TEAM_AND_TECH.md).*
