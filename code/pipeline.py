"""
Fresh analysis for corrected thesis topic:
تدوین مدل مفهومی توسعه گردشگری ورزشی مبتنی بر کارآفرینی و نوآوری اجتماعی
= Conceptual model of sport tourism development based on entrepreneurship & social innovation
  (text mining + concept network analysis)

Starts from data.csv — independent of prior Article 1/2 pipelines.
"""

from __future__ import annotations

import json
import os
import re
import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
from community import community_louvain
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

warnings.filterwarnings("ignore")

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, OSError):
        pass

HERE = Path(__file__).resolve().parent
ROOT = Path(os.environ.get("ADEL_ROOT", HERE.parent))
DATA = ROOT / "data.csv"
OUT = Path(os.environ.get("ADEL_OUT", HERE.parent / "outputs"))
# pipeline.py renders only its own exploratory plots. These are kept in a
# scratch folder so the curated publication figures in figures/ stay untouched.
FIG = Path(os.environ.get("ADEL_FIG", HERE.parent / ".pipeline_figures"))
OUT_BASE = OUT
FIG_BASE = FIG
OUT.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
sns.set_theme(style="whitegrid", context="talk")

DOMAIN_STOP = {
    "study", "studies", "paper", "research", "result", "results", "finding", "findings",
    "purpose", "aim", "objective", "objectives", "method", "methods", "methodology",
    "approach", "data", "analysis", "analyze", "analyse", "using", "used", "use",
    "based", "also", "however", "therefore", "thus", "moreover", "furthermore",
    "conclusion", "conclusions", "implication", "implications", "suggest", "suggests",
    "show", "shows", "shown", "provide", "provides", "provided", "include", "including",
    "one", "two", "three", "four", "five", "etc", "eg", "ie", "via", "per", "among",
    "within", "across", "toward", "towards", "regarding", "related", "respective",
    "respectively", "new", "author", "authors", "article", "journal", "copyright",
    "elsevier", "springer", "wiley", "taylor", "francis", "all", "right", "reserved",
    "https", "http", "www", "doi", "org", "com", "abstract", "keyword", "keywords",
    "introduction", "discussion", "literature", "review",
    "factor", "factors", "variable", "variables", "significant", "significance",
    "impact", "effect", "effects", "role", "roles", "level", "levels", "type", "types",
    "case", "cases", "area", "areas", "year", "years", "time", "may", "can", "could",
    "would", "should", "must", "many", "much", "well", "first", "second", "third",
    "different", "various", "several", "important", "main", "key", "high", "low",
    "make", "made", "take", "taken", "become", "becomes", "get", "got",
    "people", "participant", "participants", "respondent", "respondents",
    "questionnaire", "survey", "sample", "sampling", "hypothesis", "theories",
    "theoretical", "empirical", "practical",
    "identify", "examine", "explore", "investigate", "propose", "proposed", "present",
    "focus", "contribute", "understand", "understanding", "offer", "consider",
    "indicate", "reveal", "highlight", "demonstrate", "evaluate", "assess",
    "apply", "applied", "implement", "implemented", "obtain", "achieve", "improve",
    "increasing", "increase", "decreased", "decrease", "enhance", "enhancing",
    "find", "found", "seem", "appear", "need", "needed", "require", "required",
    "conduct", "conducted", "perform", "performed", "carry", "according",
    "overall", "particularly", "specifically", "currently", "recently", "highly",
    "additional", "potential", "existing", "current", "future", "previous",
    "process", "processes", "system", "systems", "context", "aspect", "aspects",
    "way", "ways", "part", "parts", "number", "order", "form", "forms",
    "qualitative", "quantitative", "interview", "interviews", "semi", "structured",
    "statistic", "statistical", "regression", "anova", "spss", "lisrel", "amos",
    "smartpls", "pls", "sem", "coding", "theme", "themes", "category", "categories",
    "student", "students", "science", "sciences", "los", "del", "una", "para",
    "train", "training", "application", "applications", "medium", "mediums",
    "diffusion", "conduction", "aggregation", "thermal", "temperature", "heat",
    "particle", "molecules", "equation", "simulation", "nanoparticle",
    "nan", "none", "null",
}

EN_STOP = set(stopwords.words("english")) | DOMAIN_STOP
LEMM = WordNetLemmatizer()
COPYRIGHT_RE = re.compile(
    r"(©|copyright).*?(reserved|elsevier|springer|wiley|sage|taylor|francis|mdpi|emerald).*?$",
    re.I,
)
HTML_RE = re.compile(r"<[^>]+>")
NON_ALPHA_RE = re.compile(r"[^a-z\s\-]")
MULTI_SPACE_RE = re.compile(r"\s+")
TOKEN_RE = re.compile(r"[a-z][a-z\-]{2,}")


def load_raw() -> pd.DataFrame:
    df = pd.read_csv(DATA, encoding="utf-8-sig", low_memory=False)
    df.columns = [c.strip() for c in df.columns]
    return df


def normalize_title(t: str) -> str:
    t = str(t).lower()
    t = re.sub(r"[^a-z0-9\s]", " ", t)
    return MULTI_SPACE_RE.sub(" ", t).strip()


def deduplicate(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["_doi"] = df["DOI"].astype(str).str.lower().str.strip()
    df.loc[df["_doi"].isin(["", "nan", "none"]), "_doi"] = np.nan
    df["_eid"] = df["EID"].astype(str).str.strip()
    df.loc[df["_eid"].isin(["", "nan", "none"]), "_eid"] = np.nan
    df["_title_norm"] = df["Title"].map(normalize_title)
    before = len(df)
    df["_abs_len"] = df["Abstract"].fillna("").astype(str).str.len()
    df = df.sort_values(["Cited by", "_abs_len"], ascending=[False, False])
    df = df.drop_duplicates(subset=["_doi"], keep="first")
    df = df.drop_duplicates(subset=["_eid"], keep="first")
    df = df.drop_duplicates(subset=["_title_norm"], keep="first")
    print(f"Dedup: {before} -> {len(df)}")
    return df


def quality_filter(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    before = len(df)
    bad = {"Retracted", "Erratum", "Letter", "Note", "Editorial", "Conference review"}
    df = df[~df["Document Type"].isin(bad)]
    df = df[df["Language of Original Document"].fillna("").str.contains("English", case=False, na=False)]
    keep = {"Article", "Review", "Conference paper", "Book chapter"}
    df = df[df["Document Type"].isin(keep)]
    df = df[df["Abstract"].fillna("").astype(str).str.len() >= 200]
    df = df[(df["Year"] >= 2000) & (df["Year"] <= 2026)]
    print(f"Quality: {before} -> {len(df)}")
    return df.reset_index(drop=True)


def drop_contamination(df: pd.DataFrame) -> pd.DataFrame:
    raw = (df["Title"].fillna("") + " " + df["Abstract"].fillna("")).str.lower()
    phys = raw.str.contains(
        r"\b(heat\s+conduction|thermal\s+diffusion|nanoparticle|molecular\s+dynamics|"
        r"aggregation[-\s]?diffusion|finite\s+element|reynolds\s+number)\b",
        regex=True,
    )
    weak = ~raw.str.contains(
        r"\b(tourism|tourist|destination|athlete|olympic|stadium|fan|entrepreneur|"
        r"community|host\s+city|sport\s+manag|sport\s+tour)\b",
        regex=True,
    )
    drop = phys | (weak & raw.str.contains(r"\b(diffusion|conduction|aggregation)\b", regex=True))
    print(f"Contamination drop: {int(drop.sum())}")
    return df.loc[~drop].reset_index(drop=True)


def build_corpus_text(row) -> str:
    parts = [
        str(row.get("Title") or ""),
        str(row.get("Abstract") or ""),
        str(row.get("Author Keywords") or "").replace(";", " "),
        str(row.get("Index Keywords") or "").replace(";", " "),
    ]
    text = " . ".join(p for p in parts if p and p.lower() != "nan")
    text = HTML_RE.sub(" ", text)
    text = COPYRIGHT_RE.sub(" ", text)
    return MULTI_SPACE_RE.sub(" ", text.replace("\n", " ")).strip()


def tokenize_lemmatize(text: str) -> list[str]:
    text = NON_ALPHA_RE.sub(" ", text.lower())
    text = MULTI_SPACE_RE.sub(" ", text).strip()
    toks = []
    for tok in word_tokenize(text):
        if not TOKEN_RE.fullmatch(tok) or tok in EN_STOP:
            continue
        lemma = LEMM.lemmatize(LEMM.lemmatize(tok), pos="v")
        if lemma in EN_STOP or len(lemma) < 3 or lemma in {"nan", "none", "null"}:
            continue
        toks.append(lemma)
    return toks


def flag_themes(df: pd.DataFrame) -> pd.DataFrame:
    """Flags aligned with sport tourism development + entrepreneurship + social innovation."""
    df = df.copy()
    raw = (
        df["Title"].fillna("")
        + " "
        + df["Abstract"].fillna("")
        + " "
        + df["Author Keywords"].fillna("")
        + " "
        + df["Index Keywords"].fillna("")
    ).str.lower()

    df["f_sport_tourism"] = raw.str.contains(r"sport[s]?\s+tourism|tourism\s+and\s+sport", regex=True)
    df["f_sport_event"] = raw.str.contains(
        r"sport[s]?\s+event|mega[-\s]?event|olympic|world cup|fifa", regex=True
    )
    df["f_sport"] = raw.str.contains(r"\bsports?\b", regex=True)
    df["f_tourism_dev"] = raw.str.contains(r"tourism\s+develop", regex=True)
    df["f_entrepreneur"] = raw.str.contains(r"entrepreneur", regex=True)
    df["f_social_innovation"] = raw.str.contains(r"social\s+innovation", regex=True)
    df["f_innovation"] = raw.str.contains(r"\binnovation|innovative\b", regex=True)
    df["f_community"] = raw.str.contains(
        r"community\s+develop|local\s+develop|social\s+entrepreneur|resident|community\s+engagement",
        regex=True,
    )
    # Social innovation operationalised broadly (search string + theory): SI + community/local + social entrepreneurship
    df["f_social_innov_broad"] = (
        df["f_social_innovation"]
        | df["f_community"]
        | raw.str.contains(r"social\s+entrepreneur|inclusive\s+innov|social\s+value|social\s+impact", regex=True)
    )

    # Primary analytical corpus for the corrected topic
    sport_ctx = df["f_sport_tourism"] | (df["f_sport"] & (df["f_sport_event"] | df["f_tourism_dev"]))
    pillar = df["f_entrepreneur"] | df["f_social_innov_broad"] | df["f_innovation"]
    df["in_core"] = sport_ctx & pillar

    # Stricter: explicit sport tourism + entrepreneurship/social innovation/innovation
    df["in_strict"] = df["f_sport_tourism"] & (
        df["f_entrepreneur"] | df["f_social_innovation"] | df["f_innovation"]
    )
    return df


def docs_to_strings(token_lists):
    return [" ".join(t) for t in token_lists]


def label_topic(terms: list[str]) -> str:
    t = " ".join(terms).lower()
    rules = [
        (r"social innovation|social entrepreneur|community development|resident", "Social innovation & community value"),
        (r"entrepreneur|entrepreneurial|venture|startup|business", "Sport-tourism entrepreneurship"),
        (r"olympic|mega|host city|host\b", "Mega-events & host destinations"),
        (r"digital|blockchain|fan|virtual|technology", "Digital innovation in sport tourism/events"),
        (r"green|carbon|sustainab", "Sustainable sport tourism development"),
        (r"sport tourism|destination|tourist", "Sport tourism destination development"),
        (r"sport event|event\b", "Sport-event driven tourism development"),
        (r"innovation|policy|industry", "Innovation systems & policy"),
        (r"community|social\b|local", "Community & local development"),
    ]
    for pat, lab in rules:
        if re.search(pat, t):
            return lab
    return "Cross-cutting theme"


def fit_nmf(texts, n_topics=6, max_features=4000):
    vec = TfidfVectorizer(
        max_df=0.75, min_df=3, max_features=max_features,
        ngram_range=(1, 3), stop_words=list(EN_STOP),
    )
    X = vec.fit_transform(texts)
    n_topics = min(n_topics, max(2, X.shape[0] // 8), X.shape[1] // 5)
    nmf = NMF(n_components=n_topics, init="nndsvda", random_state=RANDOM_STATE, max_iter=600)
    W = nmf.fit_transform(X)
    H = nmf.components_
    terms = np.array(vec.get_feature_names_out())
    topics = []
    for i, comp in enumerate(H):
        top_idx = comp.argsort()[::-1][:12]
        top_terms = terms[top_idx].tolist()
        topics.append({
            "topic_id": i,
            "size": int((W.argmax(axis=1) == i).sum()),
            "terms": ", ".join(top_terms),
            "top_terms": top_terms,
            "weights": comp[top_idx].tolist(),
            "label": label_topic(top_terms),
        })
    return W, topics


def top_tfidf(texts, top_n=50):
    vec = TfidfVectorizer(max_df=0.8, min_df=3, max_features=5000, ngram_range=(1, 3), stop_words=list(EN_STOP))
    X = vec.fit_transform(texts)
    scores = np.asarray(X.mean(axis=0)).ravel()
    terms = np.array(vec.get_feature_names_out())
    idx = scores.argsort()[::-1][:top_n]
    return pd.DataFrame({"term": terms[idx], "mean_tfidf": scores[idx]})


def keyword_texts(df, mask):
    rows = []
    for _, r in df.loc[mask].iterrows():
        kw = " ".join([
            str(r.get("Author Keywords") or "").replace(";", " "),
            str(r.get("Index Keywords") or "").replace(";", " "),
            str(r.get("Title") or ""),
        ])
        toks = [t for t in tokenize_lemmatize(kw) if t not in {"nan", "none", "null"}]
        if len(toks) < 8:
            toks = [t for t in r["tokens"] if t not in {"nan", "none", "null"}]
        rows.append(" ".join(toks))
    return rows


def build_network(texts, top_n=55, window_min=3, jaccard_min=0.07):
    min_df = max(3, len(texts) // 40)
    vec = CountVectorizer(
        max_df=0.80, min_df=min_df, max_features=3000,
        ngram_range=(1, 2), stop_words=list(EN_STOP), binary=True,
    )
    X = vec.fit_transform(texts)
    terms = np.array(vec.get_feature_names_out())
    freq = np.asarray(X.sum(axis=0)).ravel()
    top_idx = freq.argsort()[::-1][:top_n]
    X_top = X[:, top_idx]
    terms_top = terms[top_idx]
    C = (X_top.T @ X_top).toarray().astype(float)
    freqs = freq[top_idx].astype(float)
    G = nx.Graph()
    for i, t in enumerate(terms_top):
        G.add_node(t, freq=int(freqs[i]))
    for i in range(len(terms_top)):
        for j in range(i + 1, len(terms_top)):
            co = C[i, j]
            if co < window_min:
                continue
            union = freqs[i] + freqs[j] - co
            jac = co / union if union else 0
            if jac >= jaccard_min:
                G.add_edge(terms_top[i], terms_top[j], weight=float(jac), co=int(co))
    return G


def network_metrics(G):
    if G.number_of_nodes() == 0:
        return pd.DataFrame()
    part = community_louvain.best_partition(G, weight="weight", random_state=RANDOM_STATE)
    deg = dict(G.degree(weight="weight"))
    bet = nx.betweenness_centrality(G, weight="weight", normalized=True)
    try:
        eig = nx.eigenvector_centrality_numpy(G, weight="weight")
    except Exception:
        eig = {n: 0.0 for n in G.nodes()}
    rows = []
    for n in G.nodes():
        rows.append({
            "concept": n, "community": part[n], "degree_w": deg[n],
            "betweenness": bet[n], "eigenvector": eig[n], "freq": G.nodes[n].get("freq", 0),
        })
    return pd.DataFrame(rows).sort_values(["community", "eigenvector"], ascending=[True, False])


def temporal_shift(df, mask, cut=2019):
    sub = df.loc[mask]
    early = docs_to_strings(sub.loc[sub["Year"] <= cut, "tokens"].tolist())
    late = docs_to_strings(sub.loc[sub["Year"] > cut, "tokens"].tolist())
    if len(early) < 12 or len(late) < 12:
        return pd.DataFrame(), len(early), len(late)
    ve = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=2000, stop_words=list(EN_STOP))
    Xe = ve.fit_transform(early)
    e = dict(zip(ve.get_feature_names_out(), np.asarray(Xe.mean(axis=0)).ravel()))
    vl = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=2000, stop_words=list(EN_STOP))
    Xl = vl.fit_transform(late)
    l = dict(zip(vl.get_feature_names_out(), np.asarray(Xl.mean(axis=0)).ravel()))
    rows = [{"term": t, "early": e.get(t, 0), "late": l.get(t, 0), "delta": l.get(t, 0) - e.get(t, 0)}
            for t in (set(e) | set(l))]
    return pd.DataFrame(rows).sort_values("delta", ascending=False), len(early), len(late)


def plot_topics(topics, path):
    n = len(topics)
    fig, axes = plt.subplots(n, 1, figsize=(11, 2.1 * n))
    if n == 1:
        axes = [axes]
    for ax, t in zip(axes, topics):
        terms = t["top_terms"][:8][::-1]
        weights = t["weights"][:8][::-1]
        ax.barh(terms, weights, color="#1b4f72")
        ax.set_title(f"T{t['topic_id']}: {t['label']} (n={t['size']})", fontsize=10, loc="left")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    fig.suptitle("NMF topics — sport tourism development, entrepreneurship & social innovation", y=1.01)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()


def plot_network(G, metrics, path):
    if G.number_of_nodes() < 5 or metrics.empty:
        return
    G2 = G.copy()
    G2.remove_nodes_from(list(nx.isolates(G2)))
    part = dict(zip(metrics["concept"], metrics["community"]))
    communities = sorted(set(part[n] for n in G2.nodes() if n in part))
    cmap = plt.cm.get_cmap("tab10", max(len(communities), 1))
    colors = [cmap(part.get(n, 0) % 10) for n in G2.nodes()]
    sizes = [350 + 50 * np.log1p(G2.nodes[n].get("freq", 1)) for n in G2.nodes()]
    pos = nx.spring_layout(G2, weight="weight", seed=RANDOM_STATE, k=1.7 / np.sqrt(G2.number_of_nodes()))
    plt.figure(figsize=(13, 10))
    widths = [0.4 + 4.0 * G2[u][v]["weight"] for u, v in G2.edges()]
    nx.draw_networkx_edges(G2, pos, alpha=0.25, width=widths, edge_color="#888")
    nx.draw_networkx_nodes(G2, pos, node_color=colors, node_size=sizes, alpha=0.9, edgecolors="white", linewidths=0.5)
    lab_set = set(metrics.nlargest(16, "eigenvector")["concept"]) | set(metrics.nlargest(8, "betweenness")["concept"])
    labels = {n: n for n in G2.nodes() if n in lab_set}
    nx.draw_networkx_labels(G2, pos, labels=labels, font_size=8)
    plt.title("Concept network (Jaccard + Louvain)")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()


def plot_trend(df, mask, path):
    s = df.loc[mask, "Year"].value_counts().sort_index()
    plt.figure(figsize=(10, 4))
    s.plot(kind="line", marker="o", color="#1b4f72")
    plt.axvline(2019.5, color="#922b21", ls="--", lw=1)
    plt.title("Publication trend (analytical corpus)")
    plt.xlabel("Year")
    plt.ylabel("Documents")
    plt.tight_layout()
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()


def main():
    print("=" * 72)
    print("CORRECTED TOPIC — fresh pipeline")
    print("Sport tourism development × entrepreneurship × social innovation")
    print("=" * 72)

    df = load_raw()
    df = deduplicate(df)
    df = quality_filter(df)
    df = drop_contamination(df)
    df["corpus_raw"] = df.apply(build_corpus_text, axis=1)
    print("Tokenizing...")
    df["tokens"] = df["corpus_raw"].map(tokenize_lemmatize)
    df["n_tokens"] = df["tokens"].map(len)
    df = df[df["n_tokens"] >= 40].reset_index(drop=True)
    print(f"After token filter: {len(df)}")
    df = flag_themes(df)

    print("\nTheme counts:")
    for c in ["f_sport_tourism", "f_sport_event", "f_entrepreneur", "f_social_innovation",
              "f_social_innov_broad", "f_innovation", "f_community", "in_strict", "in_core"]:
        print(f"  {c}: {int(df[c].sum())}")

    # Corpus choice:
    #   "full"  = all cleaned docs (~2265) — broader tourism development field
    #   "core"  = sport tourism/event context ∩ entrepreneurship/social innovation/innovation
    import os
    mode = os.environ.get("CORPUS_MODE", "full").lower()
    if mode == "core":
        mask_col = "in_core" if df["in_core"].sum() >= 100 else "in_strict"
        mask = df[mask_col].fillna(False)
    else:
        mask_col = "full_cleaned"
        mask = pd.Series([True] * len(df), index=df.index)
    n = int(mask.sum())
    print(f"\nUsing mask={mask_col}, N={n}")

    # write outputs into the configured output root
    global OUT, FIG
    if mode == "core":
        OUT = OUT_BASE / "core"
        FIG = FIG_BASE / "core"
        OUT.mkdir(parents=True, exist_ok=True)
        FIG.mkdir(parents=True, exist_ok=True)

    keep = ["Authors", "Title", "Year", "Source title", "Cited by", "DOI", "Abstract",
            "Author Keywords", "Index Keywords", "Document Type", "EID", "n_tokens",
            "in_core", "in_strict", "f_sport_tourism", "f_sport_event", "f_entrepreneur",
            "f_social_innovation", "f_social_innov_broad", "f_innovation", "f_community"]
    df[keep].to_csv(OUT / "cleaned_master.csv", index=False)

    texts = docs_to_strings(df.loc[mask, "tokens"].tolist())
    meta = df.loc[mask, ["Title", "Year", "Cited by", "Source title", "Document Type", "DOI"]].copy()

    key_df = top_tfidf(texts, 50)
    key_df.to_csv(OUT / "tfidf_terms.csv", index=False)
    print("\nTop TF-IDF:")
    print(key_df.head(15).to_string(index=False))

    # Topics: scale with corpus size
    n_topics = 8 if n >= 1000 else 6
    W, topics = fit_nmf(texts, n_topics=n_topics)
    meta["topic"] = W.argmax(axis=1)
    meta["topic_label"] = meta["topic"].map({t["topic_id"]: t["label"] for t in topics})
    meta.to_csv(OUT / "corpus_with_topics.csv", index=False)
    pd.DataFrame([{k: v for k, v in t.items() if k != "weights"} for t in topics]).to_csv(
        OUT / "topics.csv", index=False
    )
    print("\nTopics:")
    for t in topics:
        print(f"  T{t['topic_id']} [{t['label']}] n={t['size']}: {t['terms']}")
    plot_topics(topics, FIG / "topics.png")

    kw_texts = keyword_texts(df, mask)
    G = build_network(kw_texts, top_n=55, window_min=3, jaccard_min=0.07)
    if G.number_of_edges() < 25:
        G = build_network(texts, top_n=55, window_min=4, jaccard_min=0.08)
    metrics = network_metrics(G)
    metrics.to_csv(OUT / "concept_network_metrics.csv", index=False)

    if not metrics.empty:
        comm_rows = []
        for cid, g in metrics.groupby("community"):
            comm_rows.append({
                "community": int(cid),
                "size": int(len(g)),
                "top_concepts": ", ".join(g.nlargest(8, "eigenvector")["concept"].tolist()),
            })
        comm = pd.DataFrame(comm_rows)
        comm.to_csv(OUT / "communities.csv", index=False)
        print("\nCommunities:")
        for _, r in comm.iterrows():
            print(f"  C{r['community']} (n={r['size']}): {r['top_concepts']}")
        print("\nBridges:")
        print(metrics.nlargest(12, "betweenness")[["concept", "betweenness", "community"]].to_string(index=False))

    plot_network(G, metrics, FIG / "network.png")
    plot_trend(df, mask, FIG / "trend.png")

    shift, n_early, n_late = temporal_shift(df, mask, 2019)
    if not shift.empty:
        shift.head(25).to_csv(OUT / "temporal_rising.csv", index=False)
        shift.tail(25).to_csv(OUT / "temporal_declining.csv", index=False)
        print(f"\nTemporal early={n_early} late={n_late}")
        print("Rising:")
        print(shift.head(12)[["term", "early", "late", "delta"]].to_string(index=False))

    summary = {
        "topic_fa": "تدوین مدل مفهومی توسعه گردشگری ورزشی مبتنی بر کارآفرینی و نوآوری اجتماعی",
        "topic_en": "Conceptual model of sport tourism development based on entrepreneurship and social innovation",
        "mask": mask_col,
        "n": n,
        "n_topics": len(topics),
        "topic_labels": [t["label"] for t in topics],
        "network_nodes": G.number_of_nodes(),
        "network_edges": G.number_of_edges(),
        "n_communities": int(metrics["community"].nunique()) if not metrics.empty else 0,
        "year_min": int(df.loc[mask, "Year"].min()),
        "year_max": int(df.loc[mask, "Year"].max()),
        "bridges": metrics.nlargest(10, "betweenness")["concept"].tolist() if not metrics.empty else [],
        "topic_terms": [t["terms"] for t in topics],
    }
    with open(OUT / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\nDone ->", OUT)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
