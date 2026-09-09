"""
Option 1 dual-article pipeline:
  Article 1 — Innovation & entrepreneurship in sport tourism (text mining + concept network)
  Article 2 — Governance & community/local development around sport events/tourism

Latest-practice bibliometric NLP: dedup → quality filters → Title+Abs+KW corpus →
lemmatization + domain stopwords + n-grams → NMF topics + TF-IDF keyphrases →
keyword co-occurrence concept network (Louvain) + centrality + temporal slices.
"""

from __future__ import annotations

import json
import re
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

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data.csv"
OUT = Path(__file__).resolve().parent / "outputs"
FIG = Path(__file__).resolve().parent / "figures"
OUT.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
sns.set_theme(style="whitegrid", context="talk")

# Domain stopwords (generic academic + export noise). Keep sport/tourism theory terms.
DOMAIN_STOP = {
    # generic academic boilerplate
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
    # academic verbs that pollute concept networks
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
    # methods / reporting noise
    "qualitative", "quantitative", "interview", "interviews", "semi", "structured",
    "statistic", "statistical", "regression", "anova", "spss", "lisrel", "amos",
    "smartpls", "pls", "sem", "coding", "theme", "themes", "category", "categories",
    "student", "students", "science", "sciences", "los", "del", "una", "para",
    "train", "training", "application", "applications", "medium", "mediums",
    # physics / engineering bleed from false Scopus hits
    "diffusion", "conduction", "aggregation", "thermal", "temperature", "heat",
    "particle", "molecules", "equation", "simulation", "nanoparticle",
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
    # Prefer rows with abstract length / citations when dropping dups
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
    # Drop retracted / errata / notes / letters
    bad_types = {"Retracted", "Erratum", "Letter", "Note", "Editorial", "Conference review"}
    df = df[~df["Document Type"].isin(bad_types)]
    # Prefer English scholarly records
    df = df[df["Language of Original Document"].fillna("").str.contains("English", case=False, na=False)]
    # Keep research-like docs
    keep = {"Article", "Review", "Conference paper", "Book chapter"}
    df = df[df["Document Type"].isin(keep)]
    # Require abstract
    df = df[df["Abstract"].fillna("").astype(str).str.len() >= 200]
    # Year sanity
    df = df[(df["Year"] >= 2000) & (df["Year"] <= 2026)]
    print(f"Quality filter: {before} -> {len(df)}")
    return df.reset_index(drop=True)


def drop_scientific_contamination(df: pd.DataFrame) -> pd.DataFrame:
    """Remove false positives (materials/physics) that pollute sport/tourism topics."""
    raw = (df["Title"].fillna("") + " " + df["Abstract"].fillna("")).str.lower()
    phys = raw.str.contains(
        r"\b(heat\s+conduction|thermal\s+diffusion|nanoparticle|molecular\s+dynamics|"
        r"aggregation[-\s]?diffusion|finite\s+element|reynolds\s+number|phase\s+transition)\b",
        regex=True,
    )
    weak_domain = ~raw.str.contains(
        r"\b(tourism|tourist|destination|athlete|olympic|stadium|fan|coach|"
        r"entrepreneur|governance|resident|host\s+city|sport\s+manag)\b",
        regex=True,
    )
    drop = phys | (weak_domain & raw.str.contains(r"\b(diffusion|conduction|aggregation)\b", regex=True))
    print(f"Contamination drop: {int(drop.sum())} records")
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
    text = text.replace("\n", " ")
    return MULTI_SPACE_RE.sub(" ", text).strip()


def tokenize_lemmatize(text: str) -> list[str]:
    text = text.lower()
    text = NON_ALPHA_RE.sub(" ", text)
    text = MULTI_SPACE_RE.sub(" ", text).strip()
    toks = []
    for tok in word_tokenize(text):
        if not TOKEN_RE.fullmatch(tok):
            continue
        if tok in EN_STOP or len(tok) < 3:
            continue
        lemma = LEMM.lemmatize(tok)
        lemma = LEMM.lemmatize(lemma, pos="v")
        if lemma in EN_STOP or len(lemma) < 3 or lemma in {"nan", "none", "null"}:
            continue
        toks.append(lemma)
    return toks


def docs_to_strings(token_lists: list[list[str]]) -> list[str]:
    return [" ".join(t) for t in token_lists]


def flag_themes(df: pd.DataFrame) -> pd.DataFrame:
    """Binary theme flags on raw Title+Abstract+Keywords (pre-lemma) for subsetting."""
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

    df["f_sport"] = raw.str.contains(r"\bsports?\b", regex=True)
    df["f_sport_tourism"] = raw.str.contains(
        r"sport[s]?\s+tourism|tourism\s+and\s+sport|sports?\s+and\s+tourism", regex=True
    )
    df["f_sport_event"] = raw.str.contains(
        r"sport[s]?\s+event|mega[-\s]?event|olympic|fifa|world cup|championship event",
        regex=True,
    )
    df["f_innovation"] = raw.str.contains(r"innovation|innovative|innovativeness", regex=True)
    df["f_entrepreneur"] = raw.str.contains(r"entrepreneur", regex=True)
    df["f_governance"] = raw.str.contains(r"governance|govern|policy|policymaking", regex=True)
    df["f_community"] = raw.str.contains(
        r"community\s+develop|local\s+develop|social\s+innovation|community\s+engagement|resident",
        regex=True,
    )
    df["f_tourism_dev"] = raw.str.contains(r"tourism\s+develop", regex=True)

    # Article corpora
    df["in_art1"] = df["f_sport"] & (df["f_sport_tourism"] | df["f_sport_event"]) & (
        df["f_innovation"] | df["f_entrepreneur"]
    )
    # Broader art1 if too small: sport + (innovation|entrepreneur) within tourism corpus
    df["in_art1_broad"] = df["f_sport"] & (df["f_innovation"] | df["f_entrepreneur"])

    df["in_art2"] = (df["f_sport_tourism"] | df["f_sport_event"] | df["f_sport"]) & (
        df["f_governance"] | df["f_community"]
    )
    df["in_art2_broad"] = (df["f_governance"] | df["f_community"]) & (
        df["f_sport"] | df["f_tourism_dev"]
    )
    return df


def fit_nmf(texts: list[str], n_topics: int, max_features: int = 4000):
    vec = TfidfVectorizer(
        max_df=0.75,
        min_df=3,
        max_features=max_features,
        ngram_range=(1, 3),
        stop_words=list(EN_STOP),
    )
    X = vec.fit_transform(texts)
    # Adjust topics if matrix too thin
    n_topics = min(n_topics, max(2, X.shape[0] // 8), X.shape[1] // 5)
    nmf = NMF(
        n_components=n_topics,
        init="nndsvda",
        random_state=RANDOM_STATE,
        max_iter=600,
    )
    W = nmf.fit_transform(X)
    H = nmf.components_
    terms = np.array(vec.get_feature_names_out())
    topics = []
    for i, comp in enumerate(H):
        top_idx = comp.argsort()[::-1][:12]
        topics.append(
            {
                "topic_id": i,
                "size": int((W.argmax(axis=1) == i).sum()),
                "terms": ", ".join(terms[top_idx]),
                "top_terms": terms[top_idx].tolist(),
                "weights": comp[top_idx].tolist(),
            }
        )
    return vec, nmf, W, topics


def label_topic(terms: list[str]) -> str:
    """Heuristic intellectual labels for NMF topics (post-hoc interpretability)."""
    t = " ".join(terms).lower()
    rules = [
        (r"olympic|mega|host city|host\b", "Mega-events & Olympic hosting"),
        (r"blockchain|digital|fan|broadcast|reality|virtual|stadium", "Digital technology & fan innovation"),
        (r"entrepreneur|entrepreneurial", "Sport entrepreneurship & event ventures"),
        (r"community development|resident|social responsibility|leverage", "Community outcomes & social legitimacy"),
        (r"carbon|green\b", "Green transition & sustainability"),
        (r"policy|governance|government", "Policy & governance arrangements"),
        (r"innovation|technology|digital innovation", "Innovation systems & digital integration"),
        (r"sport tourism|destination|tourist", "Sport tourism destinations & markets"),
        (r"community|social\b", "Community leverage via sport events"),
        (r"sustainab", "Sustainability-oriented sport development"),
        (r"sport event|event\b", "Sport-event management & value creation"),
        (r"industry|market|business", "Sport industry & market development"),
    ]
    for pat, lab in rules:
        if re.search(pat, t):
            return lab
    return "Cross-cutting / mixed theme"


def keyword_texts(df: pd.DataFrame, mask) -> list[str]:
    """Author+Index keywords prioritized for concept networks (cleaner theoretical terms)."""
    rows = []
    sub = df.loc[mask]
    for _, r in sub.iterrows():
        kw = " ".join(
            [
                str(r.get("Author Keywords") or "").replace(";", " "),
                str(r.get("Index Keywords") or "").replace(";", " "),
                str(r.get("Title") or ""),
            ]
        )
        toks = tokenize_lemmatize(kw)
        toks = [t for t in toks if t not in {"nan", "none", "null"}]
        # fallback if keywords sparse
        if len(toks) < 8:
            toks = [t for t in r["tokens"] if t not in {"nan", "none", "null"}]
        rows.append(" ".join(toks))
    return rows


def top_tfidf_terms(texts: list[str], top_n: int = 40) -> pd.DataFrame:
    vec = TfidfVectorizer(
        max_df=0.8,
        min_df=3,
        max_features=5000,
        ngram_range=(1, 3),
        stop_words=list(EN_STOP),
    )
    X = vec.fit_transform(texts)
    scores = np.asarray(X.mean(axis=0)).ravel()
    terms = np.array(vec.get_feature_names_out())
    idx = scores.argsort()[::-1][:top_n]
    return pd.DataFrame({"term": terms[idx], "mean_tfidf": scores[idx]})


def build_concept_network(texts: list[str], top_n: int = 60, window_min: int = 3, jaccard_min: float = 0.08):
    """Concept network via binary co-occurrence + Jaccard edge weights."""
    min_df = max(3, len(texts) // 40)
    vec = CountVectorizer(
        max_df=0.80,
        min_df=min_df,
        max_features=3000,
        ngram_range=(1, 2),
        stop_words=list(EN_STOP),
        binary=True,
    )
    X = vec.fit_transform(texts)
    terms = np.array(vec.get_feature_names_out())
    df_freq = np.asarray(X.sum(axis=0)).ravel()
    top_idx = df_freq.argsort()[::-1][:top_n]
    X_top = X[:, top_idx]
    terms_top = terms[top_idx]
    C = (X_top.T @ X_top).toarray().astype(float)
    freqs = df_freq[top_idx].astype(float)
    G = nx.Graph()
    for i, t in enumerate(terms_top):
        G.add_node(t, freq=int(freqs[i]))
    for i in range(len(terms_top)):
        for j in range(i + 1, len(terms_top)):
            co = C[i, j]
            if co < window_min:
                continue
            union = freqs[i] + freqs[j] - co
            jac = co / union if union > 0 else 0.0
            if jac >= jaccard_min:
                G.add_edge(terms_top[i], terms_top[j], weight=float(jac), co=int(co))
    return G


def network_metrics(G: nx.Graph) -> pd.DataFrame:
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
        rows.append(
            {
                "concept": n,
                "community": part[n],
                "degree_w": deg[n],
                "betweenness": bet[n],
                "eigenvector": eig[n],
                "freq": G.nodes[n].get("freq", 0),
            }
        )
    return pd.DataFrame(rows).sort_values(["community", "eigenvector"], ascending=[True, False])


def plot_network(G: nx.Graph, metrics: pd.DataFrame, title: str, path: Path):
    if G.number_of_nodes() < 5:
        return
    part = dict(zip(metrics["concept"], metrics["community"]))
    communities = sorted(set(part.values()))
    cmap = plt.cm.get_cmap("tab10", max(len(communities), 1))
    colors = [cmap(part[n] % 10) for n in G.nodes()]
    sizes = [300 + 40 * np.log1p(G.nodes[n].get("freq", 1)) for n in G.nodes()]
    pos = nx.spring_layout(G, weight="weight", seed=RANDOM_STATE, k=1.8 / np.sqrt(G.number_of_nodes()))
    plt.figure(figsize=(14, 11))
    weights = [0.4 + 4.0 * G[u][v]["weight"] for u, v in G.edges()]
    nx.draw_networkx_edges(G, pos, alpha=0.25, width=weights, edge_color="#888")
    nx.draw_networkx_nodes(G, pos, node_color=colors, node_size=sizes, alpha=0.9, linewidths=0.5, edgecolors="white")
    # label top betweenness / eig
    top_labels = set(
        metrics.nlargest(18, "eigenvector")["concept"].tolist()
        + metrics.nlargest(10, "betweenness")["concept"].tolist()
    )
    labels = {n: n for n in G.nodes() if n in top_labels}
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=8)
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()


def plot_topics(topics: list[dict], title: str, path: Path):
    n = len(topics)
    if n == 0:
        return
    fig, axes = plt.subplots(n, 1, figsize=(11, 2.2 * n), sharex=False)
    if n == 1:
        axes = [axes]
    for ax, t in zip(axes, topics):
        terms = t["top_terms"][:8][::-1]
        weights = t["weights"][:8][::-1]
        ax.barh(terms, weights, color="#2c7fb8")
        ax.set_title(f"Topic {t['topic_id']} (n={t['size']})", fontsize=11)
    fig.suptitle(title, y=1.01, fontsize=14)
    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()


def plot_year_trend(df: pd.DataFrame, mask_col: str, title: str, path: Path):
    s = df.loc[df[mask_col], "Year"].value_counts().sort_index()
    plt.figure(figsize=(10, 4))
    s.plot(kind="line", marker="o", color="#2c7fb8")
    plt.title(title)
    plt.xlabel("Year")
    plt.ylabel("Documents")
    plt.tight_layout()
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()


def temporal_term_shift(df: pd.DataFrame, mask, cut: int = 2019) -> pd.DataFrame:
    sub = df.loc[mask].copy()
    early = docs_to_strings(sub.loc[sub["Year"] <= cut, "tokens"].tolist())
    late = docs_to_strings(sub.loc[sub["Year"] > cut, "tokens"].tolist())
    if len(early) < 15 or len(late) < 15:
        return pd.DataFrame()
    ve = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=2000, stop_words=list(EN_STOP))
    Xe = ve.fit_transform(early)
    e_scores = dict(zip(ve.get_feature_names_out(), np.asarray(Xe.mean(axis=0)).ravel()))
    vl = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=2000, stop_words=list(EN_STOP))
    Xl = vl.fit_transform(late)
    l_scores = dict(zip(vl.get_feature_names_out(), np.asarray(Xl.mean(axis=0)).ravel()))
    terms = set(e_scores) | set(l_scores)
    rows = []
    for t in terms:
        a = e_scores.get(t, 0.0)
        b = l_scores.get(t, 0.0)
        rows.append({"term": t, "early": a, "late": b, "delta": b - a})
    return pd.DataFrame(rows).sort_values("delta", ascending=False)


def run_article(
    tag: str,
    title: str,
    df: pd.DataFrame,
    mask_col: str,
    n_topics: int,
    net_top_n: int,
    window_min: int,
):
    print("\n" + "=" * 72)
    print(f"{tag}: {title}")
    print("=" * 72)
    mask = df[mask_col].fillna(False)
    n = int(mask.sum())
    print(f"Corpus size: {n}")
    if n < 40:
        print("WARNING: corpus too small — widen mask.")
        return {"tag": tag, "n": n, "ok": False}

    texts = docs_to_strings(df.loc[mask, "tokens"].tolist())
    meta = df.loc[mask, ["Title", "Year", "Cited by", "Source title", "Document Type", "DOI"]].copy()
    meta.to_csv(OUT / f"{tag}_corpus_meta.csv", index=False)

    # TF-IDF key terms
    key_df = top_tfidf_terms(texts, top_n=50)
    key_df.to_csv(OUT / f"{tag}_tfidf_terms.csv", index=False)
    print("Top TF-IDF terms:")
    print(key_df.head(15).to_string(index=False))

    # Topics
    _, _, W, topics = fit_nmf(texts, n_topics=n_topics)
    for t in topics:
        t["label"] = label_topic(t["top_terms"])
    meta = meta.copy()
    meta["topic"] = W.argmax(axis=1)
    meta["topic_label"] = meta["topic"].map({t["topic_id"]: t["label"] for t in topics})
    meta.to_csv(OUT / f"{tag}_corpus_with_topics.csv", index=False)
    pd.DataFrame(
        [{k: v for k, v in t.items() if k != "weights"} for t in topics]
    ).to_csv(OUT / f"{tag}_topics.csv", index=False)
    print("\nNMF topics:")
    for t in topics:
        print(f"  T{t['topic_id']} [{t['label']}] (n={t['size']}): {t['terms']}")
    plot_topics(topics, f"{tag} — NMF topics", FIG / f"{tag}_topics.png")

    # Concept network from keywords+titles (cleaner theoretical concepts)
    kw_texts = keyword_texts(df, mask)
    G = build_concept_network(kw_texts, top_n=net_top_n, window_min=max(2, window_min - 1), jaccard_min=0.07)
    if G.number_of_edges() < 20:
        # fallback to full texts if keyword graph too sparse
        G = build_concept_network(texts, top_n=net_top_n, window_min=window_min, jaccard_min=0.08)
    metrics = network_metrics(G)
    metrics.to_csv(OUT / f"{tag}_concept_network_metrics.csv", index=False)
    # community summary
    if not metrics.empty:
        comm_rows = []
        for cid, g in metrics.groupby("community"):
            comm_rows.append(
                {
                    "community": int(cid),
                    "size": int(len(g)),
                    "top_concepts": ", ".join(g.nlargest(8, "eigenvector")["concept"].tolist()),
                }
            )
        comm = pd.DataFrame(comm_rows)
        comm.to_csv(OUT / f"{tag}_communities.csv", index=False)
        print("\nConcept communities:")
        for _, r in comm.iterrows():
            print(f"  C{r['community']} (n={r['size']}): {r['top_concepts']}")
        print("\nBridging concepts (betweenness):")
        print(metrics.nlargest(10, "betweenness")[["concept", "betweenness", "community"]].to_string(index=False))
    plot_network(G, metrics, f"{tag} — Concept network", FIG / f"{tag}_network.png")
    plot_year_trend(df, mask_col, f"{tag} — Publication trend", FIG / f"{tag}_trend.png")

    # Temporal
    shift = temporal_term_shift(df, mask, cut=2019)
    if not shift.empty:
        shift.head(25).to_csv(OUT / f"{tag}_temporal_rising.csv", index=False)
        shift.tail(25).to_csv(OUT / f"{tag}_temporal_declining.csv", index=False)
        print("\nRising terms (post-2019 vs <=2019):")
        print(shift.head(12)[["term", "early", "late", "delta"]].to_string(index=False))

    # Quality diagnostics
    diag = {
        "tag": tag,
        "title": title,
        "n": n,
        "n_topics": len(topics),
        "topic_labels": [t["label"] for t in topics],
        "network_nodes": G.number_of_nodes(),
        "network_edges": G.number_of_edges(),
        "n_communities": int(metrics["community"].nunique()) if not metrics.empty else 0,
        "year_min": int(df.loc[mask, "Year"].min()),
        "year_max": int(df.loc[mask, "Year"].max()),
        "ok": True,
        "topic_terms": [t["terms"] for t in topics],
        "bridges": metrics.nlargest(8, "betweenness")["concept"].tolist() if not metrics.empty else [],
    }
    with open(OUT / f"{tag}_summary.json", "w", encoding="utf-8") as f:
        json.dump(diag, f, indent=2, ensure_ascii=False)
    return diag


def main():
    print("Loading...")
    df = load_raw()
    df = deduplicate(df)
    df = quality_filter(df)
    df = drop_scientific_contamination(df)
    df["corpus_raw"] = df.apply(build_corpus_text, axis=1)
    print("Tokenizing / lemmatizing...")
    df["tokens"] = df["corpus_raw"].map(tokenize_lemmatize)
    df["n_tokens"] = df["tokens"].map(len)
    df = df[df["n_tokens"] >= 40].reset_index(drop=True)
    print(f"After token length filter: {len(df)}")
    df = flag_themes(df)

    # Choose masks with enough size & topical purity
    print("\nTheme counts:")
    for c in [
        "f_sport",
        "f_sport_tourism",
        "f_sport_event",
        "f_innovation",
        "f_entrepreneur",
        "f_governance",
        "f_community",
        "in_art1",
        "in_art1_broad",
        "in_art2",
        "in_art2_broad",
    ]:
        print(f"  {c}: {int(df[c].sum())}")

    # Persist cleaned master
    keep_cols = [
        "Authors",
        "Title",
        "Year",
        "Source title",
        "Cited by",
        "DOI",
        "Abstract",
        "Author Keywords",
        "Index Keywords",
        "Document Type",
        "Language of Original Document",
        "EID",
        "n_tokens",
        "in_art1",
        "in_art1_broad",
        "in_art2",
        "in_art2_broad",
        "f_sport",
        "f_sport_tourism",
        "f_sport_event",
        "f_innovation",
        "f_entrepreneur",
        "f_governance",
        "f_community",
    ]
    df[keep_cols].to_csv(OUT / "cleaned_master.csv", index=False)
    df[["Title", "Year", "corpus_raw"]].to_csv(OUT / "cleaned_corpus_text.csv", index=False)

    # Article 1: prefer strict; fall back to broad
    art1_mask = "in_art1" if df["in_art1"].sum() >= 80 else "in_art1_broad"
    art2_mask = "in_art2" if df["in_art2"].sum() >= 80 else "in_art2_broad"
    # If art2_broad is huge tourism noise, intersect with sport OR sport tourism OR events
    if art2_mask == "in_art2_broad" and df["in_art2"].sum() >= 50:
        art2_mask = "in_art2"

    d1 = run_article(
        "A1",
        "Innovation & entrepreneurship in sport tourism — concept structure",
        df,
        art1_mask,
        n_topics=5,
        net_top_n=50,
        window_min=4,
    )
    d2 = run_article(
        "A2",
        "Governance & community development in sport tourism/events — concept structure",
        df,
        art2_mask,
        n_topics=5,
        net_top_n=50,
        window_min=3,
    )

    overview = {
        "cleaned_n": len(df),
        "art1_mask": art1_mask,
        "art2_mask": art2_mask,
        "A1": d1,
        "A2": d2,
    }
    with open(OUT / "run_overview.json", "w", encoding="utf-8") as f:
        json.dump(overview, f, indent=2, ensure_ascii=False)
    print("\nDone. Outputs in", OUT)


if __name__ == "__main__":
    main()
