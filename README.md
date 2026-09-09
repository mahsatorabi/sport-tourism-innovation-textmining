# Sport tourism × innovation/entrepreneurship: text mining & concept-network analysis

Replication materials for the article:

**Innovation and Entrepreneurship in Sport Tourism: A Text-Mining and Concept-Network Analysis**

## Contents

| Path | Description |
|------|-------------|
| `SEARCH_STRING.md` | Exact Scopus query (also Appendix A of the paper) |
| `info.xlsx` | Search metadata (database, date, filter) |
| `code/pipeline.py` | Preprocessing, NMF topics, concept-network analysis |
| `code/build_article1_docx.py` | Manuscript/figure build script |
| `code/article1_*.py` | Introduction, literature, discussion text modules |
| `outputs/` | Derived tables (topics, communities, centrality, temporal terms) |
| `figures/` | Publication figures (Fig. 1–6) |

## Raw data note

The full Scopus CSV export (titles/abstracts) is **not** redistributed here due to database licence restrictions. Readers with Scopus access can reproduce the corpus using `SEARCH_STRING.md` and then run `code/pipeline.py`.

## How to reproduce

```bash
pip install pandas numpy scikit-learn nltk networkx matplotlib seaborn python-louvain openpyxl python-docx
# place your Scopus export as data.csv next to the project, or edit DATA path in pipeline.py
python code/pipeline.py
```

## Citation

Please cite the published article when available. Until then, cite this repository:

Torabi, M. (2026). Replication materials: Innovation and entrepreneurship in sport tourism (text mining & concept network). GitHub. https://github.com/mahsatorabi/sport-tourism-innovation-textmining

## Licence

Code: MIT.  
Derived bibliometric tables: CC BY 4.0.  
Scopus records remain subject to Elsevier/Scopus terms.
