"""Verify every reference against Crossref and return corrected APA-7 metadata.

Prints a JSON report with the Crossref record for each DOI listed in the
manuscript, plus a bibliographic search for entries whose metadata is
incomplete in the current reference list.
"""

from __future__ import annotations

import json
import re
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

OUT = Path(__file__).resolve().parent / "outputs_diagnostics"
OUT.mkdir(parents=True, exist_ok=True)
MAILTO = "mailto:refs-check@example.org"
UA = "ArticleRefCheck/1.0 (mailto:%s)" % MAILTO


def api(url: str):
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.load(r)


def fmt_authors(a: list[dict]) -> str:
    names = []
    for a_ in a:
        fam = a_.get("family", "")
        giv = a_.get("given", "")
        if not fam:
            continue
        names.append(f"{fam}, {giv}".strip().rstrip(","))
    return "; ".join(names)


def work(doi: str):
    try:
        return api(f"https://api.crossref.org/works/{urllib.parse.quote(doi)}")["message"]
    except Exception as e:
        return {"_error": f"{type(e).__name__}"}


def search(q: str, rows=3):
    try:
        url = ("https://api.crossref.org/works?query.bibliographic="
               + urllib.parse.quote(q) + f"&rows={rows}&select=DOI,title,author,container-title,volume,issue,page,issued,type,article-number")
        return api(url)["message"]["items"]
    except Exception as e:
        return [{"_error": f"{type(e).__name__}"}]


DOIS = [
    "10.1016/j.joi.2017.08.007",
    "10.1007/s11192-024-05161-6",
    "10.1088/1742-5468/2008/10/P10008",
    "10.1016/j.tourman.2023.104724",
    "10.1108/IJCHM-04-2022-0497",
    "10.1016/j.jbusres.2021.04.070",
    "10.3389/fsoc.2022.886498",
    "10.1080/09669582.2023.2189622",
    "10.1016/j.tourman.2016.03.003",
    "10.3390/su12125209",
    "10.1007/s12525-025-00847-y",
    "10.1177/09708464251335116",
    "10.1007/s40497-024-00399-z",
    "10.46827/ejsss.v9i3.1610",
    "10.3389/fspor.2025.1680229",
    "10.3390/admsci15040130",
    "10.1371/journal.pone.0296659",
    "10.1016/j.jdmm.2025.101004",
    "10.1016/j.heliyon.2024.e25627",
    "10.1016/j.jdmm.2023.100850",
    "10.1002/(SICI)1099-0526(19980805/06)17:5/6<295::AID-CEM932>3.0.CO;2-P",
    "10.1016/j.joi.2010.08.010",
    "10.1016/j.jbusres.2018.04.011",
]

QUERIES = {
    "Arici 2025 sports tourism bibliometric": "Arici Aydin Koseoglu Sokmen sports tourism research a bibliometric analysis and agenda for further inquiry",
    "Buhalis 2020 technology in tourism": "Buhalis Technology in tourism from information communication technologies to eTourism and smart tourism towards ambient intelligence tourism",
    "Getz Page 2016 event tourism": "Getz Page Progress and prospects for event tourism research",
    "Gretzel 2025 smart tourism 2.0": "Gretzel AI-powered smart tourism 2.0 a 10-year retrospective and updated model",
    "Ivanycheva 2024 lifestyle entrepreneurship": "Ivanycheva Schulze Lundmark Chirico lifestyle entrepreneurship literature review and future research agenda",
    "Mongeon Paul-Hus 2016": "Mongeon Paul-Hus The journal coverage of Web of Science and Scopus a comparative analysis",
    "Nguyen Nguyen 2024 community-based tourism": "Nguyen Research trends on community-based tourism in the period 2013-2023",
    "Schonherr 2023 digital transformation sustainable tourism": "Schonherr Eller Kallmuenzer Peters Organisational learning and sustainable tourism the enabling role of digital transformation",
    "Sudarmanto 2024 sports tourism sustainable destinations": "Sudarmanto systematic review development of sustainable tourism destinations based on sports tourism",
    "Suryandhani 2023 tourism village": "Suryandhani Prayitno Surjono relationship of social capital and collective action in the development of tourism village",
    "Suwanmanee 2025 social innovation CBT": "Suwanmanee Promsaka Na Sakolnakorn social innovation for community-based sustainable tourism development Bo Suak",
    "Venkatarman Rajkumar 2025": "Venkatarman Rajkumar entrepreneurship in tourism and hospitality a post-COVID-19 systematic review and bibliometric analysis",
    "Wibowo 2024 smart sustainable sport tourism planning": "Wibowo optimization model of multi criteria decision analysis for smart and sustainable sport tourism planning development problem",
    "Callon 1983 co-word analysis": "Callon Courtial Turner Bauin From translations to problematic networks an introduction to co-word analysis",
    "Lee Seung 1999 NMF": "Lee Seung Learning the parts of objects by non-negative matrix factorization",
    "Blei 2003 LDA": "Blei Ng Jordan Latent Dirichlet allocation",
    "Burt 2004 structural holes": "Burt Structural holes and good ideas",
    "Freeman 1978 centrality": "Freeman Centrality in social networks conceptual clarification",
    "Higham Hinch 2018 sport tourism development": "Higham Hinch Sport tourism development",
    "Ratten 2018 sport entrepreneurship": "Ratten Sport entrepreneurship developing and sustaining an entrepreneurial sports culture",
    "van Eck Waltman 2014": "van Eck Waltman Visualizing bibliometric networks",
    "Monroe 2008 log-odds": "Monroe Colaresi Quinn detecting subgroup differences in high-dimensional datasets log-odds-ratio analysis",
    "Grimmer Roberts Stewart 2018": "Grimmer Roberts Stewart Text as Data a new framework for machine learning and the social sciences",
    "FIG Working Paper 2020 mapping topic models": "Mauceri Topic modeling literature review",
    "Cobo 2012 science mapping software": "Cobo Lopez-Herrera A-G Symptoms of structural gap in science mapping software",
    "Cuccurullo 2016 cite space": "Cuccurullo Corrado cited references",
    "Perdana 2020 topic": "Perdana topic modeling",
    "Callaghan 2008 topic models": "Callaghan topic models",
    "Chung Lee 2003 topic": "Chung Lee topic",
    "Piantadosi Howlett 2011 keywords": "Piantadosi Howlett keywords analysis",
    "Devey 2020 topic models": "Devey topic",
    "Stein 2018 computational": "Stein computational",
    "Gerrero 2021 topic": "Gerrero topic",
    "Van Aelst 2013 do-not": "Van Aelst do-not",
    "Leydesdorff 2004 co-word": "Leydesdorff co-word analysis at the beginning of the 21st century",
    "Cobo 2011": "Cobo science mapping software",
    "Van Eck 2010": "Van Eck Waltman Mapping macro topics",
    "Donthu 2021": "Donthu Kumar Mukherjee Pandey Lim How to conduct a bibliometric analysis an overview and guidelines",
}


def main():
    report = {"dois": {}, "queries": {}}
    rows = []
    for doi in DOIS:
        m = work(doi)
        if "_error" in m:
            report["dois"][doi] = {"error": m["_error"]}
            rows.append({"doi": doi, "status": "ERROR", "note": m["_error"]})
            print(f"[ERR ] {doi}  {m['_error']}")
        else:
            rec = {
                "title": (m.get("title") or [""])[0],
                "journal": (m.get("container-title") or [""])[0],
                "volume": m.get("volume", ""),
                "issue": m.get("issue", ""),
                "page": m.get("page", ""),
                "article_number": m.get("article-number", ""),
                "year": (m.get("issued", {}).get("date-parts") or [[None]])[0][0],
                "type": m.get("type", ""),
                "publisher": m.get("publisher", ""),
                "authors": fmt_authors(m.get("author", [])),
                "n_authors": len(m.get("author", [])),
            }
            report["dois"][doi] = rec
            rows.append({"doi": doi, "status": "OK", **rec})
            print(f"[OK  ] {doi}\n       {rec['authors'][:110]}\n"
                  f"       {rec['title'][:110]}\n"
                  f"       {rec['journal']} {rec['volume']}({rec['issue']}) "
                  f"{rec['page'] or rec['article_number']} ({rec['year']}) [{rec['type']}]")
        time.sleep(0.2)

    for label, q in QUERIES.items():
        items = search(q)
        out = []
        for it in items[:3]:
            if "_error" in it:
                out.append({"error": it["_error"]})
                continue
            out.append({
                "doi": it.get("DOI", ""),
                "title": (it.get("title") or [""])[0],
                "journal": (it.get("container-title") or [""])[0],
                "volume": it.get("volume", ""), "issue": it.get("issue", ""),
                "page": it.get("page", ""),
                "year": (it.get("issued", {}).get("date-parts") or [[None]])[0][0],
                "authors": fmt_authors(it.get("author", [])),
            })
        report["queries"][label] = out
        print(f"\n### {label}")
        for o in out:
            if "error" in o:
                print("   ERR", o["error"])
            else:
                print(f"   {o['doi']}\n     {o['authors'][:120]}\n     {o['title'][:120]}\n"
                      f"     {o['journal']} {o['volume']}({o['issue']}) {o['page']} ({o['year']})")
        time.sleep(0.2)

    with open(OUT / "reference_check.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    pd.DataFrame(rows).to_csv(OUT / "reference_doi_check.csv", index=False)
    print("\nwritten ->", OUT)


if __name__ == "__main__":
    main()
