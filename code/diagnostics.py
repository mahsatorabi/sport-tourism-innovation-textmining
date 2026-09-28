"""
Revision-support diagnostics for Article 1 (full cleaned corpus).

Produces, from data.csv and the exact pipeline code in pipeline.py:
  1. Stage-by-stage screening counts (PRISMA-style flow, reviewer comment 14)
  2. Corpus profile tables (document types, sources, years, keywords)
  3. NMF topic-number diagnostics: reconstruction error, coherence, diversity,
     stability across seeds (comment 3)
  4. Normalised temporal analysis: relative frequency, log-odds ratio with
     informative Dirichlet prior, effect sizes, candidate breakpoints (comments 8, 9)
  5. Concept-network parameter table and threshold sensitivity, multi-centrality
     comparison, generic-term (stopword) sensitivity (comments 5, 6, 7)
  6. Alternative-search-strategy sensitivity: topic/bridge overlap under
     narrower and wider Boolean logic (comments 1, 22)
  7. Topic detail table: loadings, coherence, prevalence, representative
     documents, temporal evolution (comment 15)

All outputs are written to analysis_v2/outputs_diagnostics/ as CSV/JSON so that
the manuscript can quote exact numbers and the repository can ship them.
"""

from __future__ import annotations

import json
import os
import re
import sys
import warnings
from pathlib import Path

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, OSError):
        pass

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

import pipeline as P  # reuse the exact cleaning / preprocessing code

HERE = Path(__file__).resolve().parent
OUT = Path(os.environ.get("ADEL_OUT", HERE.parent / "outputs"))
OUT.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = P.RANDOM_STATE
N_BOOT = 1000


# --------------------------------------------------------------------------
# 1. Screening flow
# --------------------------------------------------------------------------
def screening_flow() -> dict:
    raw = P.load_raw()
    n0 = len(raw)

    stages = []

    # duplicate removal, tracked separately
    df = raw.copy()
    df["_doi"] = df["DOI"].astype(str).str.lower().str.strip()
    df.loc[df["_doi"].isin(["", "nan", "none"]), "_doi"] = np.nan
    df["_eid"] = df["EID"].astype(str).str.strip()
    df.loc[df["_eid"].isin(["", "nan", "none"]), "_eid"] = np.nan
    df["_title_norm"] = df["Title"].map(P.normalize_title)
    df["_abs_len"] = df["Abstract"].fillna("").astype(str).str.len()
    df = df.sort_values(["Cited by", "_abs_len"], ascending=[False, False])
    n_doi = n0 - len(df.drop_duplicates(subset=["_doi"], keep="first"))
    d1 = df.drop_duplicates(subset=["_doi"], keep="first")
    n_eid = len(d1) - len(d1.drop_duplicates(subset=["_eid"], keep="first"))
    d2 = d1.drop_duplicates(subset=["_eid"], keep="first")
    n_title = len(d2) - len(d2.drop_duplicates(subset=["_title_norm"], keep="first"))
    d3 = d2.drop_duplicates(subset=["_title_norm"], keep="first")
    stages += [
        ("Duplicates removed by DOI", int(n_doi), f"{len(d1)} remaining"),
        ("Duplicates removed by EID", int(n_eid), f"{len(d2)} remaining"),
        ("Duplicates removed by normalised title", int(n_title), f"{len(d3)} remaining"),
    ]

    d = d3.copy()
    n_before = len(d)
    bad = {"Retracted", "Erratum", "Letter", "Note", "Editorial", "Conference review"}
    n_doctype_bad = int(d["Document Type"].isin(bad).sum())
    d = d[~d["Document Type"].isin(bad)]
    n_lang = len(d) - len(d[d["Language of Original Document"].fillna("").str.contains("English", case=False, na=False)])
    d = d[d["Language of Original Document"].fillna("").str.contains("English", case=False, na=False)]
    keep = {"Article", "Review", "Conference paper", "Book chapter"}
    n_keep_bad = int((~d["Document Type"].isin(keep)).sum())
    d = d[d["Document Type"].isin(keep)]
    n_thin = len(d) - int((d["Abstract"].fillna("").astype(str).str.len() >= 200).sum())
    d = d[d["Abstract"].fillna("").astype(str).str.len() >= 200]
    n_year = len(d) - int(((d["Year"] >= 2000) & (d["Year"] <= 2026)).sum())
    d = d[(d["Year"] >= 2000) & (d["Year"] <= 2026)]
    stages += [
        ("Records excluded: non-scholarly document types (errata, editorials, letters, notes, retracted)", n_doctype_bad, ""),
        ("Records excluded: non-English language of original document", n_lang, ""),
        ("Records excluded: document type outside Article / Review / Conference paper / Book chapter", n_keep_bad, ""),
        ("Records excluded: abstract shorter than 200 characters", n_thin, ""),
        ("Records excluded: publication year outside 2000-2026", n_year, ""),
    ]
    n_after_screen = len(d)

    raw_txt = (d["Title"].fillna("") + " " + d["Abstract"].fillna("")).str.lower()
    phys = raw_txt.str.contains(
        r"\b(heat\s+conduction|thermal\s+diffusion|nanoparticle|molecular\s+dynamics|"
        r"aggregation[-\s]?diffusion|finite\s+element|reynolds\s+number)\b", regex=True)
    weak = ~raw_txt.str.contains(
        r"\b(tourism|tourist|destination|athlete|olympic|stadium|fan|entrepreneur|"
        r"community|host\s+city|sport\s+manag|sport\s+tour)\b", regex=True)
    drop = phys | (weak & raw_txt.str.contains(r"\b(diffusion|conduction|aggregation)\b", regex=True))
    n_contam = int(drop.sum())
    n_phys = int(phys.sum())
    d = d.loc[~drop].reset_index(drop=True)
    stages.append(("Records excluded: off-topic contamination (materials-science diffusion/conduction leakage)", n_contam, ""))

    d["corpus_raw"] = d.apply(P.build_corpus_text, axis=1)
    d["tokens"] = d["corpus_raw"].map(P.tokenize_lemmatize)
    d["n_tokens"] = d["tokens"].map(len)
    n_thin_tok = len(d) - int((d["n_tokens"] >= 40).sum())
    d = d[d["n_tokens"] >= 40].reset_index(drop=True)
    stages.append(("Records excluded: fewer than 40 analysable tokens after preprocessing", n_thin_tok, ""))

    n = len(d)
    flow = {
        "identified_scopus": int(n0),
        "stages": stages,
        "after_screening": int(n_after_screen),
        "after_contamination": int(n_after_screen - n_contam),
        "after_token_filter": int(n),
        "n_physical_contamination": int(n_phys),
        "n_early_le2019": int((d["Year"] <= 2019).sum()),
        "n_late_gt2019": int((d["Year"] > 2019).sum()),
    }
    d.to_csv(OUT / "screened_corpus.csv", index=False)

    # profile tables
    prof = {
        "document_type": d["Document Type"].value_counts().to_dict(),
        "year": d["Year"].value_counts().sort_index().to_dict(),
        "top_sources": d["Source title"].value_counts().head(20).to_dict(),
        "n_sources": int(d["Source title"].nunique()),
        "open_access": d["Open Access"].value_counts().to_dict(),
        "mean_citations": float(d["Cited by"].mean()),
    }
    with open(OUT / "corpus_profile.json", "w", encoding="utf-8") as f:
        json.dump({"flow": flow, "profile": prof}, f, indent=2, ensure_ascii=False)
    print(json.dumps(flow, indent=2, ensure_ascii=False))
    return flow


# --------------------------------------------------------------------------
# 2. Analytic corpus
# --------------------------------------------------------------------------
def load_corpus() -> pd.DataFrame:
    d = pd.read_csv(OUT / "screened_corpus.csv", low_memory=False)
    d["tokens"] = d["tokens"].map(ast_literal_list)
    return d


def ast_literal_list(s):
    import ast
    try:
        return ast.literal_eval(s)
    except Exception:
        return []


def texts_of(d: pd.DataFrame) -> list[str]:
    return P.docs_to_strings(d["tokens"].tolist())


# --------------------------------------------------------------------------
# 3. NMF diagnostics
# --------------------------------------------------------------------------
def nmf_diagnostics(d: pd.DataFrame, dfa) -> pd.DataFrame:
    texts = texts_of(d)
    vec = P.TfidfVectorizer(
        max_df=0.75, min_df=3, max_features=4000, ngram_range=(1, 3), stop_words=list(P.EN_STOP))
    X = vec.fit_transform(texts)
    terms = np.array(vec.get_feature_names_out())
    Xb = P.CountVectorizer(
        max_df=0.75, min_df=3, max_features=4000, ngram_range=(1, 3), stop_words=list(P.EN_STOP)
    ).fit_transform(texts) if False else None

    # co-occurrence counts on the same features (binary presence per document)
    Xb = (X > 0).astype(np.int8)
    binarised = X.tocsr()
    # Reconstruction error, coherence, diversity: reported for the deterministic
    # nndsvda initialisation actually used in the reported analysis.
    rows = []
    for k in range(3, 17):
        nmf = P.NMF(n_components=k, init="nndsvda", random_state=RANDOM_STATE, max_iter=600)
        nmf.fit_transform(X)
        H = nmf.components_
        rows.append({
            "k": k,
            "recon_error": float(nmf.reconstruction_err_),
            "umass_coherence": round(umass_coherence(H, terms, binarised, top_n=20), 4),
            "topic_diversity": round(topic_diversity(H, terms, top_n=10), 4),
        })
    dfa = pd.DataFrame(rows)
    dfa["delta_error_pct"] = (-dfa["recon_error"].pct_change() * 100).round(2)
    dfa["cum_error_reduction_pct"] = (
        (dfa["recon_error"].iloc[0] - dfa["recon_error"]) / (dfa["recon_error"].iloc[0] - 0) * 100
    ).round(2)

    # Stability: random initialisation across five seeds. nndsvda is deterministic,
    # so a random-init protocol is required for a non-degenerate stability test.
    stab = []
    for k in range(3, 17):
        sets = []
        for seed in [42, 7, 2023, 101, 555]:
            nmf = P.NMF(n_components=k, init="random", random_state=seed, max_iter=600)
            nmf.fit_transform(X)
            sets.append([set(terms[c.argsort()[::-1][:15]]) for c in nmf.components_])
        pairs = []
        for a in range(len(sets)):
            for b in range(a + 1, len(sets)):
                for i in range(k):
                    pairs.append(max(len(sets[a][i] & sets[b][j]) / len(sets[a][i] | sets[b][j])
                                     for j in range(k)))
        stab.append({
            "k": k,
            "stability_top15_mean": round(float(np.mean(pairs)), 3),
            "stability_top15_sd": round(float(np.std(pairs)), 3),
            "stability_top15_min": round(float(np.min(pairs)), 3),
            "stability_frac_above_0.5": round(float(np.mean(np.array(pairs) > 0.5)), 3),
        })
    dfa = dfa.merge(pd.DataFrame(stab), on="k")
    dfa.to_csv(OUT / "nmf_k_diagnostics.csv", index=False)
    print(dfa.to_string(index=False))
    return dfa


def topic_diversity(H, terms, top_n=10) -> float:
    tops = [terms[comp.argsort()[::-1][:top_n]] for comp in H]
    allw = [w for t in tops for w in t]
    return len(set(allw)) / len(allw)


def umass_coherence(H, terms, Xb, top_n=20) -> float:
    """Mean UMass-style topic coherence: mean log((D(w_i,w_j)+1)/D(w_j)) over top pairs."""
    index = {t: i for i, t in enumerate(terms)}
    df_vec = np.asarray(Xb.sum(axis=0)).ravel().astype(float)
    dfs = {terms[i]: df_vec[i] for i in range(len(terms))}
    scores = []
    for comp in H:
        top = comp.argsort()[::-1][:top_n]
        docs = Xb[:, top]
        C = (docs.T @ docs).toarray().astype(float)
        vals = []
        for i in range(1, min(top_n, len(top))):
            wi = terms[top[i]]
            for j in range(i):
                wj = terms[top[j]]
                dij = C[i, j]
                vals.append(np.log((dij + 1) / dfs[wj]))
        if vals:
            scores.append(float(np.mean(vals)))
    return float(np.mean(scores)) if scores else float("nan")


# --------------------------------------------------------------------------
# 4. Topic detail table
# --------------------------------------------------------------------------
def topic_details(d: pd.DataFrame) -> pd.DataFrame:
    texts = texts_of(d)
    vec = P.TfidfVectorizer(
        max_df=0.75, min_df=3, max_features=4000, ngram_range=(1, 3), stop_words=list(P.EN_STOP))
    X = vec.fit_transform(texts)
    terms = np.array(vec.get_feature_names_out())
    Xb = X
    nmf = P.NMF(n_components=8, init="nndsvda", random_state=RANDOM_STATE, max_iter=600)
    W = nmf.fit_transform(X)
    H = nmf.components_
    assign = W.argmax(axis=1)
    d = d.copy()
    d["topic"] = assign
    d["topic_conf"] = W.max(axis=1)
    d["hhi"] = (W ** 2).sum(axis=1)  # concentration of document's weight

    early = d["Year"] <= 2019
    rows = []
    for k in range(8):
        m = d["topic"] == k
        early_share = float((m & early).sum() / max(int(early.sum()), 1))
        late_share = float((m & ~early).sum() / max(int((~early).sum()), 1))
        top_idx = H[k].argsort()[::-1][:12]
        top_terms = terms[top_idx].tolist()
        top_w = H[k][top_idx].tolist()
        # representative documents: highest normalised-weight docs for the topic
        rep = d.loc[m].assign(score=(W[m, k] / (W[m].sum(axis=1) + 1e-9)))
        rep = rep.nlargest(3, "score")
        rep_titles = " | ".join(
            [f"{t.strip()[:110]} ({int(y)})" for t, y in zip(rep["Title"], rep["Year"])])
        rows.append({
            "topic_id": k,
            "label": TOPIC_LABELS[k],
            "n": int(m.sum()),
            "prevalence_pct": round(100 * float(m.mean()), 2),
            "share_pre2019": round(100 * early_share, 2),
            "share_post2019": round(100 * late_share, 2),
            "share_change_pp": round(100 * (late_share - early_share), 2),
            "mean_confidence": round(float(np.mean(W[m, k])), 4),
            "median_hhi": round(float(np.median(d.loc[m, "hhi"])), 4),
            "top_terms_weighted": "; ".join(f"{t} ({w:.3f})" for t, w in zip(top_terms, top_w)),
            "coherence_umass": round(umass_coherence(H[k:k + 1], terms, Xb, top_n=20), 4),
            "repr_docs": rep_titles,
        })
    td = pd.DataFrame(rows)
    td.to_csv(OUT / "topic_details.csv", index=False)
    d[["Title", "Year", "Source title", "Document Type", "Cited by", "topic", "topic_conf", "hhi"]].to_csv(
        OUT / "documents_with_topics.csv", index=False)
    print(td[["topic_id", "label", "n", "prevalence_pct", "share_pre2019", "share_post2019",
              "coherence_umass"]].to_string(index=False))
    return td


# --------------------------------------------------------------------------
# 5. Temporal analysis (normalised)
# --------------------------------------------------------------------------
def temporal_normalised(d: pd.DataFrame) -> pd.DataFrame:
    """Normalised before/after comparison of term salience.

    A single TF-IDF vectoriser is fitted on the union of both periods so that the
    vocabulary and the IDF weights are identical across periods; each period is
    then transformed with that shared model. This avoids the vocabulary
    misalignment that would arise from fitting two independent models.

    Three normalised quantities are reported per term:
      * relative frequency (mean TF-IDF per document) and its ratio;
      * the log-odds ratio with an informative Dirichlet prior alpha0 = 0.01
        (Monroe, Colaresi & Quinn, 2008) and its z-statistic;
      * a standardised effect size for proportions (Cohen's d for the
        document-level presence rate).
    """
    rows = []
    for cut in (2019,):
        early_docs = d.loc[d["Year"] <= cut, "tokens"].tolist()
        late_docs = d.loc[d["Year"] > cut, "tokens"].tolist()
        vec = P.TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=4000,
                                stop_words=list(P.EN_STOP))
        vec.fit(P.docs_to_strings(early_docs + late_docs))  # shared vocabulary + IDF
        Xe = vec.transform(P.docs_to_strings(early_docs))
        Xl = vec.transform(P.docs_to_strings(late_docs))
        terms = np.array(vec.get_feature_names_out())
        pe = np.asarray((Xe > 0).sum(axis=0)).ravel().astype(float) / Xe.shape[0]
        pl = np.asarray((Xl > 0).sum(axis=0)).ravel().astype(float) / Xl.shape[0]
        fe = np.asarray(Xe.sum(axis=0)).ravel().astype(float)
        fl = np.asarray(Xl.sum(axis=0)).ravel().astype(float)
        re_ = np.asarray(Xe.mean(axis=0)).ravel()
        rl = np.asarray(Xl.mean(axis=0)).ravel()
        ne, nl = Xe.shape[0], Xl.shape[0]
        # Document-level presence rates, so the two periods are compared on a
        # common footing despite different corpus sizes. The log-odds estimator
        # follows Monroe, Colaresi & Quinn (2008) with an informative Dirichlet
        # prior a0 = 0.01, and the z statistic uses the Poisson approximation
        # var[log(y + a0)] ~= 1 / (y + a0).
        ce = np.asarray((Xe > 0).sum(axis=0)).ravel().astype(float)
        cl = np.asarray((Xl > 0).sum(axis=0)).ravel().astype(float)
        a0 = 0.01
        se = (ce + a0) / (ne + 2 * a0)
        sl = (cl + a0) / (nl + 2 * a0)
        delta = np.log(sl) - np.log(se)
        # binomial variance of the log-odds: var[log(p)] ~= (1 - p) / (n p)
        z = delta / np.sqrt(
            np.clip((1 - se) / (ne * se), 1e-12, None) + np.clip((1 - sl) / (nl * sl), 1e-12, None))
        tok_e = max(int(sum(len(t) for t in early_docs)), 1)
        tok_l = max(int(sum(len(t) for t in late_docs)), 1)
        rate_e = fe / tok_e * 1000
        rate_l = fl / tok_l * 1000
        pooled = (pe + pl) / 2
        pooled = np.clip(pooled, 1e-9, 1 - 1e-9)
        cohen_d = (pl - pe) / np.sqrt(pooled * (1 - pooled))
        for i, t in enumerate(terms):
            rows.append({
                "term": t, "cut": cut,
                "n_early": ne, "n_late": nl,
                "tokens_early": tok_e, "tokens_late": tok_l,
                "presence_early": round(float(pe[i]), 5),
                "presence_late": round(float(pl[i]), 5),
                "rate_per_1k_early": round(float(rate_e[i]), 4),
                "rate_per_1k_late": round(float(rate_l[i]), 4),
                "rate_ratio": round(float(rate_l[i] / rate_e[i]), 3) if rate_e[i] > 0 else np.nan,
                "relfreq_early": round(float(re_[i]), 6),
                "relfreq_late": round(float(rl[i]), 6),
                "relfreq_ratio": round(float(rl[i] / re_[i]), 3) if re_[i] > 0 else np.nan,
                "log_odds": round(float(delta[i]), 4),
                "z_log_odds": round(float(z[i]), 2),
                "cohens_d": round(float(cohen_d[i]), 4),
            })
    out = pd.DataFrame(rows)
    out["abs_z"] = out["z_log_odds"].abs()
    out = out.sort_values("log_odds", ascending=False)
    out.to_csv(OUT / "temporal_logodds.csv", index=False)
    return out


def breakpoint_scan(d: pd.DataFrame) -> pd.DataFrame:
    """Compare candidate cut years by how strongly normalised vocabulary separates.
    Uses Jensen-Shannon-like divergence of relative term frequency profiles and
    the mean |z| of log-odds over top-|V| terms; purely descriptive, not a
    causal test."""
    rows = []
    vec = P.TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=3000,
                            stop_words=list(P.EN_STOP))
    for cut in range(2014, 2023):
        e = d.loc[d["Year"] <= cut, "tokens"].tolist()
        l = d.loc[d["Year"] > cut, "tokens"].tolist()
        if len(e) < 60 or len(l) < 60:
            continue
        vec.fit(P.docs_to_strings(e + l))
        Xe = vec.transform(P.docs_to_strings(e))
        Xl = vec.transform(P.docs_to_strings(l))
        terms = np.array(vec.get_feature_names_out())
        re_ = np.asarray(Xe.mean(axis=0)).ravel()
        rl = np.asarray(Xl.mean(axis=0)).ravel()
        m = (re_ + rl) > 0
        p = re_[m] / re_[m].sum() if re_[m].sum() else re_[m]
        q = rl[m] / rl[m].sum() if rl[m].sum() else rl[m]
        js = float(0.5 * np.sum(p * np.log((p + 1e-12) / (p / 2 + q / 2 + 1e-12))) +
                   0.5 * np.sum(q * np.log((q + 1e-12) / (p / 2 + q / 2 + 1e-12))))
        fe = np.asarray(Xe.sum(axis=0)).ravel()[m]
        fl = np.asarray(Xl.sum(axis=0)).ravel()[m]
        a0 = 0.01
        z = (np.log(fl + a0) - np.log(fe + a0)) / np.sqrt(1 / (fl + a0) + 1 / (fe + a0))
        rows.append({
            "cut": cut, "n_early": len(e), "n_late": len(l),
            "js_divergence": round(js, 4),
            "mean_abs_z_top200": round(float(np.mean(np.sort(np.abs(z))[-200:])), 3),
            "n_terms_z_gt3": int((np.abs(z) > 3).sum()),
        })
    bp = pd.DataFrame(rows)
    bp.to_csv(OUT / "breakpoint_scan.csv", index=False)
    print(bp.to_string(index=False))
    return bp


def breakpoint_scan_year_vs_rest(d: pd.DataFrame) -> pd.DataFrame:
    """Non-circular breakpoint scan: for each candidate year, compare that single
    year with the pooled remainder of the corpus. A year whose vocabulary departs
    most sharply from the rest is a candidate structural break. This avoids the
    mechanical advantage that late cuts have in cumulative splits."""
    rows = []
    vec = P.TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=3000,
                            stop_words=list(P.EN_STOP))
    vec.fit(P.docs_to_strings(d["tokens"].tolist()))
    years = d["Year"].values
    for y in range(2012, 2026):
        m = years == y
        if m.sum() < 30 or (~m).sum() < 30:
            continue
        Xe = vec.transform(P.docs_to_strings(d.loc[m, "tokens"].tolist()))
        Xl = vec.transform(P.docs_to_strings(d.loc[~m, "tokens"].tolist()))
        terms = np.array(vec.get_feature_names_out())
        ce = np.asarray((Xe > 0).sum(axis=0)).ravel().astype(float)
        cl = np.asarray((Xl > 0).sum(axis=0)).ravel().astype(float)
        ne, nl = Xe.shape[0], Xl.shape[0]
        a0 = 0.01
        se = (ce + a0) / (ne + 2 * a0)
        sl = (cl + a0) / (nl + 2 * a0)
        z = (np.log(sl) - np.log(se)) / np.sqrt(
            np.clip((1 - se) / (ne * se), 1e-12, None) + np.clip((1 - sl) / (nl * sl), 1e-12, None))
        emerging = terms[(ce == 0) & (cl >= 3)]
        rows.append({
            "year": y, "n_year": int(ne), "n_rest": int(nl),
            "n_terms_z_gt3": int((np.abs(z) > 3).sum()),
            "n_terms_z_gt5": int((np.abs(z) > 5).sum()),
            "mean_abs_z": round(float(np.mean(np.abs(z))), 3),
            "n_emerging_terms": int(len(emerging)),
            "emerging_examples": ", ".join(emerging[:8]),
        })
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "breakpoint_year_vs_rest.csv", index=False)
    print(out.to_string(index=False))
    return out


def topic_share_scan(d: pd.DataFrame) -> pd.DataFrame:
    """Share of each topic by year, used to show that the temporal contrast is not
    an artefact of topic-label boundaries."""
    docs = pd.read_csv(OUT / "documents_with_topics.csv")
    tab = pd.crosstab(docs["Year"], docs["topic"], normalize="index").round(4)
    tab.to_csv(OUT / "topic_share_by_year.csv")
    return tab


def year_robustness(d: pd.DataFrame) -> pd.DataFrame:
    """Rolling normalised term change to show that 2019 is one of several candidate
    breakpoints and that the direction of the main terms is stable."""
    vec = P.TfidfVectorizer(ngram_range=(1, 1), min_df=4, max_features=2500,
                            stop_words=list(P.EN_STOP))
    allt = vec.fit_transform(P.docs_to_strings(d["tokens"].tolist()))
    terms = np.array(vec.get_feature_names_out())
    watch = [t for t in ["digital", "technology", "sustainab", "green", "environment",
                         "innovation", "community", "entrepreneur", "sport", "covid",
                         "smart", "tourism", "resilien"] if t in set(terms)]
    rows = []
    for yr in range(2015, 2027):
        m = (d["Year"] == yr).values
        if m.sum() < 10:
            continue
        sub = allt[m]
        rel = np.asarray(sub.mean(axis=0)).ravel()
        rows.append({"year": yr, "n": int(m.sum()),
                     **{t: round(float(rel[list(terms).index(t)]), 5) for t in watch}})
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "yearly_term_relfreq.csv", index=False)
    print(out.to_string(index=False))
    return out


# --------------------------------------------------------------------------
# 6. Network diagnostics
# --------------------------------------------------------------------------
def network_diagnostics(d: pd.DataFrame) -> dict:
    texts = texts_of(d)
    kw_texts = P.keyword_texts(d, slice(None))
    # how many documents contributed their own keyword/title vocabulary vs. the
    # full-token fallback used by pipeline.keyword_texts
    n_fallback = 0
    for _, r in d.iterrows():
        kw = " ".join([str(r.get("Author Keywords") or "").replace(";", " "),
                       str(r.get("Index Keywords") or "").replace(";", " "),
                       str(r.get("Title") or "")])
        if len([t for t in P.tokenize_lemmatize(kw) if t not in {"nan", "none", "null"}]) < 8:
            n_fallback += 1
    summary = {"n_docs_keyword_based": int(len(d) - n_fallback),
               "n_docs_fulltext_fallback": int(n_fallback),
               "min_df_used": int(max(3, len(kw_texts) // 40))}

    # (a) final configuration
    G = P.build_network(kw_texts, top_n=55, window_min=3, jaccard_min=0.07)
    metrics = P.network_metrics(G)
    edges = pd.DataFrame(
        [(u, v, float(a["weight"]), int(a["co"])) for (u, v, a) in G.edges(data=True)],
        columns=["source", "target", "weight_jaccard", "co_occurrence"])
    edges.to_csv(OUT / "concept_network_edges.csv", index=False)
    metrics["degree_rank"] = metrics["degree_w"].rank(ascending=False)
    metrics["betweenness_rank"] = metrics["betweenness"].rank(ascending=False)
    metrics["eigenvector_rank"] = metrics["eigenvector"].rank(ascending=False)
    metrics["closeness_distance"] = [P.nx.closeness_centrality(G, u, distance="weight")
                                     for u in metrics["concept"]]
    metrics["clustering"] = [P.nx.clustering(G, u, weight="weight") for u in metrics["concept"]]
    _br_edges = list(P.nx.bridges(G))
    _br_count = {}
    for u, v in _br_edges:
        _br_count[u] = _br_count.get(u, 0) + 1
        _br_count[v] = _br_count.get(v, 0) + 1
    metrics["n_bridges"] = [_br_count.get(u, 0) for u in metrics["concept"]]
    metrics = metrics.sort_values("betweenness", ascending=False)
    metrics.to_csv(OUT / "concept_network_metrics_full.csv", index=False)

    top_bet = metrics.nlargest(10, "betweenness")["concept"].tolist()
    top_eig = metrics.nlargest(10, "eigenvector")["concept"].tolist()
    top_deg = metrics.nlargest(10, "degree_w")["concept"].tolist()
    top_clo = metrics.nlargest(10, "closeness_distance")["concept"].tolist()

    def jac(a, b):
        return len(set(a) & set(b)) / len(set(a) | set(b))

    # (b) threshold sensitivity
    rows = []
    for top_n in (40, 55, 70, 90):
        for jac_min in (0.05, 0.07, 0.10, 0.15):
            for wmin in (2, 3, 5):
                Gx = P.build_network(kw_texts, top_n=top_n, window_min=wmin, jaccard_min=jac_min)
                if Gx.number_of_edges() == 0:
                    continue
                mx = P.network_metrics(Gx)
                part = dict(zip(mx["concept"], mx["community"]))
                b = mx.nlargest(10, "betweenness")["concept"].tolist()
                sil = modularity(Gx, part)
                rows.append({
                    "top_n": top_n, "jaccard_min": jac_min, "cooccurrence_min": wmin,
                    "n_nodes": Gx.number_of_nodes(), "n_edges": Gx.number_of_edges(),
                    "n_communities": int(mx["community"].nunique()),
                    "modularity": round(float(sil), 4),
                    "density": round(float(P.nx.density(Gx)), 4),
                    "bridge_overlap_top10_with_main": round(jac(top_bet, b), 3),
                })
    sens = pd.DataFrame(rows)
    sens.to_csv(OUT / "network_threshold_sensitivity.csv", index=False)
    print(sens.to_string(index=False))

    # (c) generic-term sensitivity: drop generic methodological terms from the network
    generic = {"model", "study", "research", "analysis", "development", "management",
               "policy", "impact", "effect", "approach", "result", "method", "paper"}
    keep = [c for c in metrics["concept"] if c not in generic]
    Gg = G.subgraph(keep).copy()
    mg = P.network_metrics(Gg)
    bg = mg.nlargest(10, "betweenness")["concept"].tolist()
    summary["generic_terms_in_main"] = sorted(set(metrics["concept"]) & generic)
    summary["bridges_after_generic_removal"] = bg
    summary["bridge_overlap_after_generic_removal"] = round(jac(top_bet, bg), 3)
    mg.sort_values("betweenness", ascending=False).to_csv(
        OUT / "concept_network_metrics_no_generic.csv", index=False)

    # (d) centrality agreement
    agree = {
        "top10_betweenness": top_bet,
        "top10_eigenvector": top_eig,
        "top10_degree": top_deg,
        "top10_closeness": top_clo,
        "jaccard_bet_eig": round(jac(top_bet, top_eig), 3),
        "jaccard_bet_degree": round(jac(top_bet, top_deg), 3),
        "jaccard_bet_closeness": round(jac(top_bet, top_clo), 3),
        "spearman_bet_eig": round(float(pd.Series(
            metrics.set_index("concept")["betweenness"]).corr(
            pd.Series(metrics.set_index("concept")["eigenvector"]),
            method="spearman")), 3),
        "spearman_bet_degree": round(float(pd.Series(
            metrics.set_index("concept")["betweenness"]).corr(
            pd.Series(metrics.set_index("concept")["degree_w"]),
            method="spearman")), 3),
        "spearman_bet_closeness": round(float(pd.Series(
            metrics.set_index("concept")["betweenness"]).corr(
            pd.Series(metrics.set_index("concept")["closeness_distance"]),
            method="spearman")), 3),
        "spearman_bet_clustering": round(float(pd.Series(
            metrics.set_index("concept")["betweenness"]).corr(
            pd.Series(metrics.set_index("concept")["clustering"]),
            method="spearman")), 3),
    }
    with open(OUT / "network_summary.json", "w", encoding="utf-8") as f:
        json.dump({
            "n_nodes": G.number_of_nodes(), "n_edges": G.number_of_edges(),
            "n_communities": int(metrics["community"].nunique()),
            "modularity": round(float(modularity(G, dict(zip(metrics["concept"], metrics["community"])))), 4),
            "density": round(float(P.nx.density(G)), 4),
            "mean_degree": round(float(P.nx.degree(G, weight="weight") and np.mean([d_ for _, d_ in G.degree(weight="weight")])), 2),
            "jaccard_min": 0.07, "cooccurrence_min": 3, "top_n_nodes_considered": 55,
            "min_df": max(3, len(kw_texts) // 40),
            "centrality_agreement": agree,
            "generic_term_check": summary,
        }, f, indent=2, ensure_ascii=False)
    print(json.dumps({"n_nodes": G.number_of_nodes(), "n_edges": G.number_of_edges(),
                      "agree": agree, "generic": summary}, indent=2, ensure_ascii=False))
    return {"metrics": metrics, "sens": sens, "summary": summary}


def modularity(G, part) -> float:
    try:
        return P.community_louvain.modularity(part, G, weight="weight")
    except Exception:
        return float("nan")


# --------------------------------------------------------------------------
# 7. Search-strategy sensitivity
# --------------------------------------------------------------------------
def search_sensitivity(d: pd.DataFrame) -> dict:
    """Alternative Boolean logics approximated on the retrieved export.

    The export already reflects the master query, so alternative strategies are
    approximated by re-applying narrower/wider term logic to title+abstract+
    keywords. Overlap of topics and bridges with the master analysis is reported.
    """
    raw = (d["Title"].fillna("") + " " + d["Abstract"].fillna("") + " " +
           d["Author Keywords"].fillna("") + " " + d["Index Keywords"].fillna("")).str.lower()

    strategies = {
        "S0_master (reported)": pd.Series([True] * len(d), index=d.index),
        "S1_no_social_innovation_phrase": ~(raw.str.contains(r"social\s+innovation", regex=True)),
        "S2_strict_triple_intersection": (
            raw.str.contains(r"entrepreneur", regex=True) &
            raw.str.contains(r"innovat", regex=True) &
            (raw.str.contains(r"community|resident|local", regex=True))),
        "S3_no_sport_branch": ~(raw.str.contains(r"sport", regex=True)),
        "S4_sport_branch_only": raw.str.contains(r"sport", regex=True),
        "S5_harder_social_lens": raw.str.contains(r"social\s+innovation|social\s+entrepreneur", regex=True),
        "S6_broad_entrepreneurship_only": raw.str.contains(r"entrepreneur", regex=True),
    }
    rows = []
    master_topics = None
    master_bridges = None
    for name, mask in strategies.items():
        sub = d[mask]
        if len(sub) < 200:
            rows.append({"strategy": name, "n": len(sub), "note": "too small for NMF"})
            continue
        texts = P.docs_to_strings(sub["tokens"].tolist())
        vec = P.TfidfVectorizer(max_df=0.75, min_df=3, max_features=4000,
                                ngram_range=(1, 3), stop_words=list(P.EN_STOP))
        X = vec.fit_transform(texts)
        terms = np.array(vec.get_feature_names_out())
        k = 8
        nmf = P.NMF(n_components=k, init="nndsvda", random_state=RANDOM_STATE, max_iter=600)
        W = nmf.fit_transform(X)
        H = nmf.components_
        tops = [set(terms[c.argsort()[::-1][:12]]) for c in H]
        kw = P.keyword_texts(sub, slice(None))
        G = P.build_network(kw, top_n=55, window_min=3, jaccard_min=0.07)
        mx = P.network_metrics(G)
        br = mx.nlargest(10, "betweenness")["concept"].tolist()
        if master_topics is None:
            master_topics = [t for t in tops]
            master_bridges = br
        topic_jac = float(np.mean([max(len(a & b) / len(a | b) for b in master_topics) for a in tops]))
        rows.append({
            "strategy": name, "n": len(sub),
            "pct_of_master": round(100 * len(sub) / len(d), 1),
            "mean_topic_overlap_jaccard": round(topic_jac, 3),
            "bridge_overlap_top10": round(len(set(br) & set(master_bridges)) / 10, 3),
            "bridges": ", ".join(br),
        })
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "search_strategy_sensitivity.csv", index=False)
    print(out.to_string(index=False))
    return {"rows": out.to_dict("records")}


# --------------------------------------------------------------------------
def nexus_composition(d: pd.DataFrame) -> pd.DataFrame:
    """How often does the corpus actually address the three constructs jointly?
    This quantifies the operational meaning of the word 'nexus' in the title."""
    raw = (d["Title"].fillna("") + " " + d["Abstract"].fillna("") + " " +
           d["Author Keywords"].fillna("") + " " + d["Index Keywords"].fillna("")).str.lower()
    inn = raw.str.contains(r"innovat", regex=True)
    ent = raw.str.contains(r"entrepreneur", regex=True)
    si = raw.str.contains(r"social\s+innovation", regex=True)
    com = raw.str.contains(r"community\s+develop|local\s+develop|community\s+tourism|"
                           r"community-based\s+tourism|resident|community\s+engage|"
                           r"social\s+entrepreneur", regex=True)
    sport = raw.str.contains(r"sport", regex=True)
    tdev = raw.str.contains(r"tourism\s+develop", regex=True)
    rows = []
    for name, m in [
        ("any innovation term", inn),
        ("any entrepreneurship term", ent),
        ("explicit 'social innovation' phrase", si),
        ("community / local development or social entrepreneurship", com),
        ("sport term", sport),
        ("'tourism development' phrase", tdev),
        ("innovation AND entrepreneurship", inn & ent),
        ("innovation AND community/local", inn & com),
        ("entrepreneurship AND community/local", ent & com),
        ("innovation AND entrepreneurship AND community/local", inn & ent & com),
        ("innovation AND entrepreneurship AND sport", inn & ent & sport),
        ("all three of innovation, entrepreneurship, community/local AND sport",
         inn & ent & com & sport),
        ("at least two of {innovation, entrepreneurship, community/local}",
         (inn.astype(int) + ent.astype(int) + com.astype(int)) >= 2),
    ]:
        rows.append({"construct": name, "n": int(m.sum()),
                     "pct": round(100 * float(m.mean()), 2)})
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "nexus_composition.csv", index=False)
    print(out.to_string(index=False))
    return out


TOPIC_LABELS = {
    0: "Destination innovation and regional markets",
    1: "Sport events and sport tourism",
    2: "Community-based tourism and social development",
    3: "Rural tourism entrepreneurship",
    4: "Cultural heritage tourism",
    5: "Smart / digital tourism transformation",
    6: "Women's entrepreneurship and empowerment",
    7: "Sustainable tourism development",
}


def main():
    print("=" * 70)
    print("1) screening flow")
    screening_flow()
    d = load_corpus()
    print(f"corpus N = {len(d)}")
    print("\n3) NMF k diagnostics")
    nmf_diagnostics(d, None)
    print("\n4) topic details")
    topic_details(d)
    print("\n5) temporal normalised")
    temporal_normalised(d)
    print("\n5b) breakpoint scan")
    breakpoint_scan(d)
    print("\n5b-ii) each year vs the remainder")
    breakpoint_scan_year_vs_rest(d)
    print("\n5b-iii) topic share by year")
    topic_share_scan(d)
    print("\n5c) yearly relfreq")
    year_robustness(d)
    print("\n6) network diagnostics")
    network_diagnostics(d)
    print("\n7) search sensitivity")
    search_sensitivity(d)
    print("\n8) construct co-presence (nexus composition)")
    nexus_composition(d)
    print("\nDone ->", OUT)


if __name__ == "__main__":
    main()
