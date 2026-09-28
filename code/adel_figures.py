"""Figure rendering for the Article1 manuscript.

Extracted verbatim from build_article1_full.py so that the manuscript build and
the public replication package render byte-identical figures. Every function
takes plain DataFrames/dicts plus an output path and has no DOCX dependency.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

try:  # the manuscript build supplies the canonical seed table
    import article1_text as T
except ImportError:  # standalone replication package
    T = None

LAYOUT_SEED = 42 if T is None else T.LAYOUT_SEED

TOPIC_LABELS = {
    0: "Destination innovation & regional markets",
    1: "Sport events & sport tourism",
    2: "Community-based tourism & social development",
    3: "Rural tourism entrepreneurship",
    4: "Cultural heritage tourism",
    5: "Smart / digital tourism transformation",
    6: "Womenâ€™s entrepreneurship & empowerment",
    7: "Sustainable tourism development",
}

PALETTE = ["#1b4f72", "#148f77", "#6c3483", "#b9770e", "#922b21", "#2874a6", "#1abc9c", "#7d3c98"]

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8.5,
    "figure.dpi": 300, "savefig.dpi": 300, "savefig.bbox": "tight",
})
def fig_kdiagnostics(kd, path):
    fig, axes = plt.subplots(1, 3, figsize=(10.6, 3.2))
    k = kd["k"]
    ax = axes[0]
    ax.plot(k, kd["recon_error"], "o-", color="#1b4f72", ms=3.4, lw=1.6)
    ax.set_xlabel("Number of topics (k)")
    ax.set_ylabel("Reconstruction error")
    ax.set_title("(a) Reconstruction error", fontsize=8.5)
    ax.axvline(8, color="#922b21", ls="--", lw=1.0)
    ax2 = ax.twinx()
    ax2.bar(k, kd["delta_error_pct"], color="#c8d6e5", width=0.55, zorder=0)
    ax2.set_ylabel("Incremental error reduction (%)", fontsize=7.4, color="#54606a")
    ax2.tick_params(axis="y", labelsize=6.8, colors="#54606a")
    ax2.set_ylim(0, 1.0)

    ax = axes[1]
    ax.plot(k, kd["umass_coherence"], "s-", color="#148f77", ms=3.4, lw=1.6,
            label="UMass coherence")
    ax.plot(k, kd["topic_diversity"], "^-", color="#b9770e", ms=3.4, lw=1.6,
            label="Topic diversity")
    ax.set_xlabel("Number of topics (k)")
    ax.set_title("(b) Coherence and diversity", fontsize=8.5)
    ax.legend(frameon=False, fontsize=7)
    ax.axvline(8, color="#922b21", ls="--", lw=1.0)

    ax = axes[2]
    mean = kd["stability_top15_mean"]
    lo = kd["stability_top15_mean"] - kd["stability_top15_min"]
    hi = 1.0 - kd["stability_top15_mean"]
    ax.bar(k, mean, color="#6c3483", width=0.6)
    ax.errorbar(k, mean, yerr=[lo, hi], fmt="none", ecolor="#333", elinewidth=0.7, capsize=2)
    ax.plot(k, kd["stability_frac_above_0.5"], "d-", color="#922b21", ms=3.0, lw=1.3,
            label="share of pairs > 0.5")
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Number of topics (k)")
    ax.set_title("(c) Stability across 5 seeds", fontsize=8.5)
    ax.legend(frameon=False, fontsize=7, loc="lower left")
    ax.axvline(8, color="#922b21", ls="--", lw=1.0)

    for a in axes:
        a.spines["top"].set_visible(False)
        a.spines["right"].set_visible(a is axes[0])
    fig.suptitle("Figure 2. Quantitative diagnostics for the topic-count decision "
                 "(k = 3â€“16; dashed line = reported k = 8)", fontsize=9.4, y=1.04)
    fig.tight_layout()
    fig.savefig(path, facecolor="white")
    plt.close(fig)
    return path


def fig_loadings(td, path):
    n = len(td)
    fig, axes = plt.subplots(n, 1, figsize=(7.4, 1.25 * n))
    for ax, (_, row), color in zip(axes, td.iterrows(), PALETTE):
        terms = [t.split(" (")[0] for t in row["top_terms_weighted"].split("; ")[:8]][::-1]
        w = [float(t.split("(")[1].rstrip(")")) for t in
             row["top_terms_weighted"].split("; ")[:8]][::-1]
        ax.barh(terms, w, color=color)
        ax.set_title(f"T{int(row['topic_id'])}: {row['label']}  "
                     f"(n = {int(row['n'])}; confidence = {row['mean_confidence']:.3f}; "
                     f"coherence = {row['coherence_umass']:.3f})",
                     fontsize=7.6, loc="left")
        ax.tick_params(labelsize=6.8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    fig.suptitle("Figure 1. Highest-loading terms per topic (normalised NMF weights)",
                 fontsize=9.4, y=1.005)
    fig.tight_layout()
    fig.savefig(path, facecolor="white")
    plt.close(fig)
    return path


def fig_network(edges, metrics, summary, path):
    """Render the ACTUAL exported graph: nodes from metrics, edges from the edge list."""
    m = metrics.set_index("concept")
    G = nx.Graph()
    for c, r in m.iterrows():
        G.add_node(c, community=int(r["community"]), freq=int(r["freq"]),
                   bet=float(r["betweenness"]), eig=float(r["eigenvector"]))
    wmax = edges["weight_jaccard"].max()
    for r in edges.itertuples():
        if r.source in G and r.target in G:
            G.add_edge(r.source, r.target,
                       weight=float(r.weight_jaccard), cooc=int(r.co_occurrence))
    comm = {c: int(m.loc[c, "community"]) for c in G.nodes()}
    comms = sorted(set(comm.values()))

    # deterministic community-seeded layout
    rng = np.random.default_rng(LAYOUT_SEED)
    init = {}
    for c in G.nodes():
        ang = 2 * np.pi * comm[c] / len(comms) + rng.normal(0, 0.28)
        rad = 0.6 + 0.55 * rng.random()
        init[c] = np.array([rad * np.cos(ang), rad * np.sin(ang)])
    pos = nx.spring_layout(G, pos=init, seed=int(LAYOUT_SEED), iterations=120,
                           k=1.05 / np.sqrt(len(G)), weight="weight")

    fmax = max(d["freq"] for _, d in G.nodes(data=True))
    sizes = [180 + 1500 * (d["freq"] / fmax) for _, d in G.nodes(data=True)]
    widths = [0.35 + 2.6 * (G[u][v]["weight"] / wmax) for u, v in G.edges()]
    colors = [PALETTE[comm[c] % len(PALETTE)] for c in G.nodes()]

    fig, ax = plt.subplots(figsize=(8.4, 7.4))
    nx.draw_networkx_edges(G, pos, ax=ax, width=widths, alpha=0.30,
                           edge_color="#7f8c8d")
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=colors, node_size=sizes,
                           linewidths=0.7, edgecolors="white", alpha=0.96)
    lab = set(metrics.nlargest(18, "eigenvector")["concept"]) | \
          set(metrics.nlargest(12, "betweenness")["concept"])
    nx.draw_networkx_labels(G, pos, labels={n: n for n in G.nodes() if n in lab},
                            ax=ax, font_size=6.6)
    handles = [mpatches.Patch(color=PALETTE[c % len(PALETTE)], label=f"C{c}  (n = "
                             f"{sum(1 for x in comm.values() if x == c)})")
               for c in comms]
    ax.legend(handles=handles, frameon=False, loc="lower left", ncol=3, fontsize=7.4)
    ax.set_title(
        f"Figure 3. Concept co-occurrence network (real exported graph)\n"
        f"n = {summary['n_nodes']} nodes; {summary['n_edges']} edges; "
        f"Jaccard ≥ {summary['jaccard_min']}; co-occurrence ≥ {summary['cooccurrence_min']}; "
        f"min_df = {summary['min_df']}; modularity = {summary['modularity']:.4f}\n"
        f"Node size ∝ document frequency; node colour = Louvain community; "
        f"edge width ∝ Jaccard weight; labels = 18 highest-eigenvector ∪ 12 highest-betweenness",
        fontsize=8.0)
    ax.axis("off")
    fig.savefig(path, facecolor="white")
    plt.close(fig)
    return path


def fig_logodds(lo, path):
    show = [
        ("sustainable", 8.21), ("innovation", 8.00), ("technology", 7.48),
        ("sustainability", 7.45), ("digital", 7.06), ("sustainable development", 5.41),
        ("green", 5.11), ("resilience", 5.02), ("crisis", 3.08),
        ("environmental", 4.60), ("smart tourism", 2.52),
        ("development", 0.54), ("digitalization", 0.76), ("rural tourism", 0.77),
        ("sport", 0.97), ("innovation tourism", -0.40), ("entrepreneurship", -0.28),
        ("social", -0.04), ("community tourism", -3.00), ("community", -4.16),
        ("community development", -6.68),
    ]
    sub = lo.set_index("term")
    rows = []
    for t, _ in show:
        r = sub.loc[t]
        rows.append((t, r["z_log_odds"], r["log_odds"], r["presence_early"],
                     r["presence_late"], r["relfreq_ratio"], r["cohens_d"]))
    rows.sort(key=lambda r: r[1])
    labels = [f"{r[0]}" for r in rows]
    z = [r[1] for r in rows]
    fig, ax = plt.subplots(figsize=(7.6, 6.2))
    cols = ["#922b21" if v < 0 else ("#148f77" if abs(v) >= 1.96 else "#95a5a6") for v in z]
    ax.barh(labels, z, color=cols)
    for y, v in enumerate(z):
        ax.text(v + (0.18 if v >= 0 else -0.18), y, f"{v:+.2f}", va="center",
                ha="left" if v >= 0 else "right", fontsize=6.8)
    ax.axvline(0, color="#333", lw=0.9)
    for thr in (1.96, -1.96):
        ax.axvline(thr, color="#54606a", ls=":", lw=0.9)
    ax.set_xlabel("log-odds z-score (post-2019 vs. â‰¤2019; prior aâ‚€ = 0.01; "
                  "binomial SE)")
    ax.set_title("Figure 4. Temporal movement of the concept vocabulary\n"
                 "Dotted lines = ±1.96. Green = significant increase, red = significant "
                 "decrease, grey = not significant.\nPeriods: n = 701 (â‰¤2019) and n = 1,564 "
                 "(>2019); vectoriser fitted jointly on both periods.",
                 fontsize=8.2)
    ax.tick_params(labelsize=7.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, facecolor="white")
    plt.close(fig)
    return path


def fig_breakpoint(bs, by, path):
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 3.4))
    ax = axes[0]
    ax.plot(bs["cut"], bs["js_divergence"], "o-", color="#1b4f72", ms=4, lw=1.6)
    ax.set_xlabel("Cut year (documents â‰¤ cut vs. > cut)")
    ax.set_ylabel("Jensenâ€“Shannon divergence")
    ax.set_title("(a) Vocabulary contrast between periods", fontsize=8.5)
    ax.axvline(2019, color="#922b21", ls="--", lw=1.2)
    ax.text(2019.1, ax.get_ylim()[1] * 0.92, "2019", color="#922b21", fontsize=7.5)

    ax2 = axes[0].twinx()
    ax2.bar(bs["cut"], bs["n_terms_z_gt3"], color="#c8d6e5", width=0.6, zorder=0)
    ax2.set_ylabel("Terms with |z| > 3", fontsize=7.2, color="#54606a")
    ax2.tick_params(axis="y", labelsize=6.8, colors="#54606a")

    ax = axes[1]
    sub = by[by["year"] >= 2012]
    ax.bar(sub["year"].astype(int).astype(str), sub["mean_abs_z"],
           color=["#922b21" if y == 2019 else "#1b4f72" for y in sub["year"]])
    ax.set_xlabel("Single year scored against the remainder of the corpus")
    ax.set_ylabel("Mean |z| across terms")
    ax.set_title("(b) Year-specific vocabulary deviation", fontsize=8.5)
    ax.tick_params(axis="x", labelsize=6.4, rotation=90)
    for a in (axes[0], axes[1]):
        a.spines["top"].set_visible(False)
    axes[0].spines["right"].set_visible(False)
    fig.suptitle("Figure 6. Cut-point scans: 2019 is not the dominant break in this corpus",
                 fontsize=9.4, y=1.05)
    fig.tight_layout()
    fig.savefig(path, facecolor="white")
    plt.close(fig)
    return path


def fig_search_sensitivity(ss, path):
    arms = ss[ss["mean_topic_overlap_jaccard"].notna()].copy()
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    x = np.arange(len(arms))
    w = 0.38
    ax.bar(x - w / 2, arms["mean_topic_overlap_jaccard"], w,
           label="mean topic-assignment overlap (Jaccard)", color="#1b4f72")
    ax.bar(x + w / 2, arms["bridge_overlap_top10"], w,
           label="top-10 betweenness list overlap", color="#b9770e")
    for xi, (_, r) in zip(x, arms.iterrows()):
        ax.text(xi - w / 2, r["mean_topic_overlap_jaccard"] + 0.02,
                f"{r['mean_topic_overlap_jaccard']:.3f}", ha="center", fontsize=6.6)
        ax.text(xi + w / 2, r["bridge_overlap_top10"] + 0.02,
                f"{r['bridge_overlap_top10']:.2f}", ha="center", fontsize=6.6)
        ax.text(xi, 0.03, f"n = {int(r['n'])}", ha="center", fontsize=6.8,
                color="#333", rotation=90)
    ax.set_xticks(x)
    ax.set_xticklabels(arms["strategy"], fontsize=6.4, rotation=12,
                       ha="right")
    ax.set_ylabel("Overlap with master solution")
    ax.set_ylim(0, 1.12)
    ax.legend(frameon=False, fontsize=7)
    ax.set_title("Figure 5. Search-string sensitivity: structural overlap induced by each "
                 "alternative arm\nS2 (n = 84) and S5 (n = 77) are too small for topic "
                 "estimation and are omitted.", fontsize=8.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, facecolor="white")
    plt.close(fig)
    return path
