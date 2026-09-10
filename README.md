# Tourism development × innovation / entrepreneurship / community: text mining & concept-network analysis

Replication materials for editors and reviewers of the article:

**Mapping the Conceptual Structure of Tourism Development Research at the Nexus of Innovation, Entrepreneurship and Community Development: A Text-Mining and Concept-Network Analysis**

Repository: <https://github.com/mahsatorabi/sport-tourism-innovation-textmining>

## What is included

| Path | Description |
|------|-------------|
| `SEARCH_STRING.md` | Exact Scopus TITLE-ABS-KEY query (Appendix A of the paper) |
| `info.xlsx` | Search metadata (database, export date, filter) |
| `code/pipeline.py` | Cleaning, TF–IDF, NMF topics, concept-network, temporal contrast |
| `outputs/` | Derived tables supporting Results (no raw Scopus abstracts) |
| `figures/` | Publication figures (Fig. 1–6) |
| `requirements.txt` | Python dependencies |

## What is not included

The full Scopus CSV export (titles/abstracts) is **not** redistributed here due to database licence restrictions. Readers with Scopus access can rebuild the corpus from `SEARCH_STRING.md`, save the export as `data.csv` in the repository root, and run the pipeline.

## How to reproduce

```bash
pip install -r requirements.txt
# place your Scopus export as data.csv in the repository root
python code/pipeline.py
```

Default mode analyses the **full cleaned corpus** (Article 1). Optional focused subset:

```bash
# Windows PowerShell
$env:CORPUS_MODE="core"; python code/pipeline.py
```

## Derived outputs (reviewer checklist)

- `outputs/topics.csv` / `topics_labeled.csv` — NMF topics (Table 1)
- `outputs/communities.csv` — Louvain communities (Table 2)
- `outputs/concept_network_metrics.csv` — centrality / bridges (Table 3)
- `outputs/temporal_rising.csv` / `temporal_declining.csv` — post-2019 vocabulary shift
- `outputs/corpus_with_topics.csv` — document-level topic assignment (metadata only; no abstracts)
- `outputs/summary.json` — run summary
- `figures/Fig1_trend.png` … `Fig6_temporal.png`

## Citation

Please cite the published article when available. Until then, cite this repository:

Torabi, M. (2026). Replication materials: Tourism development at the nexus of innovation, entrepreneurship and community development (text mining & concept network). GitHub. https://github.com/mahsatorabi/sport-tourism-innovation-textmining

## Licence

- Code: MIT  
- Derived bibliometric tables and figures: CC BY 4.0  
- Scopus records remain subject to Elsevier/Scopus terms  
