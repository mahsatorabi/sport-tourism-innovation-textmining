"""Render the six publication figures from the curated tables in outputs/.

Reads only the redistributable files in outputs/ and writes the six figures to
figures/. No Scopus abstracts are required. The renderers themselves live in
adel_figures.py, which is the same module the manuscript build imports, so the
images produced here are byte-identical to those in the submitted article.

    python code/make_figures.py

Figure numbering matches the manuscript: 1 loadings, 2 topic-count diagnostics,
3 concept network, 4 temporal log-odds, 5 search-strategy sensitivity,
6 breakpoint scan. Layout and colour constants come from adel_figures.py; the
only seed involved is the spring-layout seed (42), so the network figure is
reproducible.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from adel_figures import (  # noqa: E402
    fig_loadings, fig_kdiagnostics, fig_network, fig_logodds,
    fig_search_sensitivity, fig_breakpoint,
)

OUT = Path(os.environ.get("ADEL_OUT", HERE.parent / "outputs"))
FIG = Path(os.environ.get("ADEL_FIG", HERE.parent / "figures"))

CURATED = [
    "breakpoint_scan.csv", "breakpoint_year_vs_rest.csv", "concept_network_edges.csv",
    "concept_network_metrics_full.csv", "concept_network_metrics_no_generic.csv",
    "corpus_profile.json", "documents_with_topics.csv", "network_summary.json",
    "network_threshold_sensitivity.csv", "nexus_composition.csv", "nmf_k_diagnostics.csv",
    "reference_check.json", "reference_doi_check.csv", "search_strategy_sensitivity.csv",
    "temporal_logodds.csv", "topic_details.csv", "topic_share_by_year.csv",
    "yearly_term_relfreq.csv",
]


def check_inputs() -> None:
    missing = [f for f in CURATED if not (OUT / f).exists()]
    if missing:
        raise SystemExit(
            f"missing {len(missing)} required file(s) in {OUT}:\n  " + "\n  ".join(missing)
            + "\nRun pipeline.py and diagnostics.py first."
        )


def main() -> None:
    check_inputs()
    FIG.mkdir(parents=True, exist_ok=True)

    profile = json.loads((OUT / "corpus_profile.json").read_text(encoding="utf-8"))
    summary = json.loads((OUT / "network_summary.json").read_text(encoding="utf-8"))
    kd = pd.read_csv(OUT / "nmf_k_diagnostics.csv")
    td = pd.read_csv(OUT / "topic_details.csv")
    edges = pd.read_csv(OUT / "concept_network_edges.csv")
    metrics = pd.read_csv(OUT / "concept_network_metrics_full.csv")
    lo = pd.read_csv(OUT / "temporal_logodds.csv")
    bs = pd.read_csv(OUT / "breakpoint_scan.csv")
    by = pd.read_csv(OUT / "breakpoint_year_vs_rest.csv")
    ss = pd.read_csv(OUT / "search_strategy_sensitivity.csv")

    n = int(profile["flow"]["after_token_filter"])
    n_early = int(profile["flow"]["n_early_le2019"])
    n_late = int(profile["flow"]["n_late_gt2019"])

    made = {
        "Fig1_loadings.png": fig_loadings(td, FIG / "Fig1_loadings.png"),
        "Fig2_kdiagnostics.png": fig_kdiagnostics(kd, FIG / "Fig2_kdiagnostics.png"),
        "Fig3_network.png": fig_network(edges, metrics, summary, FIG / "Fig3_network.png"),
        "Fig4_logodds.png": fig_logodds(lo, FIG / "Fig4_logodds.png"),
        "Fig5_search.png": fig_search_sensitivity(ss, FIG / "Fig5_search.png"),
        "Fig6_breakpoint.png": fig_breakpoint(bs, by, FIG / "Fig6_breakpoint.png"),
    }

    print(f"corpus N = {n} ({n_early} <=2019, {n_late} >2019)")
    for name in made:
        print(f"  {name:22s} -> {made[name]}")
    print(f"\nDone -> {FIG}")


if __name__ == "__main__":
    main()
