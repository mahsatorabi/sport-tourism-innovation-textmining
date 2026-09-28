# Tourism development x innovation / entrepreneurship / community / sport: text mining & concept-network analysis

Replication materials for editors and reviewers of the article:

**Mapping Tourism Development Research at the Intersection of Innovation, Entrepreneurship, Community Development and Sport Tourism: A Text-Mining and Concept-Network Analysis**

Repository: <https://github.com/mahsatorabi/sport-tourism-innovation-textmining>

Reported run: Scopus export of **9 September 2026**, screened corpus **N = 2,265**
(701 records dated 2019 or earlier, 1,564 records dated 2020 or later).

## What is included

| Path | Description |
|------|-------------|
| `SEARCH_STRING.md` | Exact Scopus TITLE-ABS-KEY query and the six alternative arms (Supplementary Table S1) |
| `info.xlsx` | Search metadata (database, export date, filter) |
| `code/pipeline.py` | Screening, deduplication, contamination filter, TF-IDF, NMF, concept network, centrality, Louvain |
| `code/diagnostics.py` | k = 3-16 validation scan, temporal log-odds, cut-point scans, threshold and search-string sensitivity, nexus composition |
| `code/adel_figures.py` | The six figure renderers, imported unchanged by the manuscript build |
| `code/make_figures.py` | Draws all six figures from `outputs/` alone, with no access to abstracts |
| `code/refcheck.py` ... `refcheck5.py` | Crossref reference-verification scripts (Appendix D) |
| `outputs/` | 18 derived tables and JSON records supporting every quantitative claim |
| `figures/` | The six publication figures, byte-identical to those in the article |
| `requirements.txt` | Pinned dependency versions (Supplementary Table S2) |

## What is not included, and why

The raw Scopus export (titles and abstracts) and `screened_corpus.csv` are **not**
redistributed, because Scopus records remain subject to Elsevier database-licence
terms. Everything needed to regenerate them is present:

```bash
pip install -r requirements.txt
# place your own Scopus export as data.csv in the repository root
python code/pipeline.py
python code/diagnostics.py
python code/make_figures.py
```

`make_figures.py` needs no abstracts: it reads only the curated files in
`outputs/`, so the six figures can be rebuilt from redistributable material
alone. The renderers are shared verbatim with the manuscript build, so the
result is byte-identical to the submitted images.

`outputs/documents_with_topics.csv` contains document-level metadata (title, year,
source, document type, citation count, assigned topic) but no abstracts.

`make_figures.py` refuses to run if any of the 18 curated files is missing,
rather than silently drawing a partial figure set.

## Configuration

Paths resolve relative to the repository and can be overridden with environment
variables, so the same code runs here or in the authors' working directory:

| Variable | Default | Meaning |
|----------|---------|---------|
| `ADEL_ROOT` | repository root | Where `data.csv` is expected |
| `ADEL_OUT` | `outputs/` | Derived tables and JSON records |
| `ADEL_FIG` | `.pipeline_figures/` | Exploratory plots drawn by `pipeline.py` |
| `CORPUS_MODE` | `full` | `full` = reported corpus; `core` = stricter sport-tourism subset |

`figures/` is deliberately **not** a pipeline target: it holds the six
publication figures, and `pipeline.py` writes its own exploratory plots to
`.pipeline_figures/` so that directory is never overwritten.

### Rendering backend

`adel_figures.py` calls `matplotlib.use("Agg")` before rendering. This matters:
the Matplotlib default on this machine is `TkAgg`, and the two backends produce
different text metrics, so the same code rendered under Tk yields visibly
different images. Pinning Agg makes the figures reproducible on any machine,
headless or not. If you change the backend, expect the PNGs to differ.

## Random seeds (Supplementary Table S3)

No stochastic step is unseeded.

| Step | Setting |
|------|---------|
| NMF, reported solution (k = 8) | `random_state = 42`, `init = "nndsvda"` |
| NMF, stability scan (k = 3-16) | `init = "random"`, `random_state` in {42, 7, 2023, 101, 555}, 5 refits per k |
| Louvain community detection | `seed = 42` |
| Network layout | `seed = 42` (spring layout, 120 iterations) |
| Figure rendering | `matplotlib.use("Agg")`, `dpi = 300`, `font.family = "DejaVu Sans"` |

A deterministic initialiser (`nndsvda`) returns an identical solution on every run
and is therefore excluded from the stability scan, where it would make the
statistic degenerate at 1.0.

## Derived outputs (referee checklist)

| File | Supports |
|------|----------|
| `corpus_profile.json` | Table 1 screening flow, every stage count, document types, years, sources |
| `nmf_k_diagnostics.csv` | Table 3 and Figure 2: k = 3-16 error, UMass coherence, diversity, five-seed stability |
| `topic_details.csv` | Table 2 and Figure 1: loadings, coherence, assignment confidence, representative documents |
| `topic_share_by_year.csv` | Table 4: topic shares by year |
| `concept_network_edges.csv` | Table 8 and Figure 3: the 543 exported edges with Jaccard weight and co-occurrence count |
| `concept_network_metrics_full.csv` | Table 5: node list with community, all four centralities, ranks |
| `concept_network_metrics_no_generic.csv` | Table 5 (sensitivity row): network rebuilt with generic terms removed |
| `network_threshold_sensitivity.csv` | Table 6: 16-cell node cut-off by Jaccard-threshold grid |
| `network_summary.json` | Network parameters, node and edge counts, density, modularity |
| `temporal_logodds.csv` | Table 7a and Figure 4: log-odds z, presence rate, token rate, relative frequency, Cohen's d |
| `breakpoint_scan.csv` | Table 7b and Figure 6: nine candidate cut-years |
| `breakpoint_year_vs_rest.csv` | Figure 6: each year scored against the remainder of the corpus |
| `search_strategy_sensitivity.csv` | Table 6 and Figure 5: six search arms, topic and bridge overlap |
| `nexus_composition.csv` | Table 8: construct co-presence |
| `yearly_term_relfreq.csv` | Year-by-year relative frequency behind the pre/post shares in Table 2 |
| `documents_with_topics.csv` | Document-level topic assignment and confidence (metadata only) |
| `reference_check.json`, `reference_doi_check.csv` | Appendix D: Crossref verification queries, responses and resolutions |

## Figures

| Figure | Content |
|--------|---------|
| 1 | Highest-loading terms per topic, with normalised NMF weights |
| 2 | Topic-count diagnostics, k = 3 to 16 |
| 3 | Concept co-occurrence network, rendered from `concept_network_edges.csv` |
| 4 | Temporal movement of the concept vocabulary (log-odds z-scores) |
| 5 | Search-string sensitivity: structural overlap per alternative arm |
| 6 | Cut-point scans |

## Citation

Please cite the published article when available. Until then, cite this repository:

Torabi, M. (2026). Mapping tourism development research at the intersection of
innovation, entrepreneurship, community development and sport tourism: A
text-mining and concept-network analysis. Replication materials. GitHub.
https://github.com/mahsatorabi/sport-tourism-innovation-textmining

## Licence

- Code: MIT
- Derived bibliometric tables and figures: CC BY 4.0
- Scopus records remain subject to Elsevier/Scopus terms
