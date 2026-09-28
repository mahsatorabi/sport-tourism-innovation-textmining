"""Final verification pass: remaining method citations + retries."""

from __future__ import annotations

import json
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

UA = "ArticleRefCheck/1.0 (mailto:refs-check@example.org)"
RETRIES = 3


def api(url):
    last = None
    for _ in range(RETRIES):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=40) as r:
                return json.load(r)
        except Exception as e:  # noqa: BLE001
            last = e
            time.sleep(1.5)
    raise last


def dp(m, key):
    return (m.get(key, {}).get("date-parts") or [[None]])[0]


def rec(doi):
    try:
        m = api(f"https://api.crossref.org/works/{urllib.parse.quote(doi)}")["message"]
    except Exception as e:  # noqa: BLE001
        print(f"\nDOI {doi}\n  [FAIL] {type(e).__name__}: {e}")
        return
    print(f"\nDOI {doi}")
    print("  authors:", " | ".join(
        f"{a.get('family','')}; {a.get('given','')}" for a in m.get("author", [])))
    print("  title  :", (m.get("title") or [""])[0].replace("\n", " "))
    print("  journal:", (m.get("container-title") or [""])[0])
    print("  vol/iss/pg:", m.get("volume", ""), m.get("issue", ""),
          m.get("page", "") or m.get("article-number", ""))
    print("  print:", dp(m, "published-print"), "| online:", dp(m, "published-online"))


def search(q, rows=3):
    try:
        items = api("https://api.crossref.org/works?rows=%d&query.bibliographic=%s"
                    % (rows, urllib.parse.quote(q)))["message"]["items"]
    except Exception as e:  # noqa: BLE001
        print(f"\nQUERY {q!r}\n  [FAIL] {type(e).__name__}: {e}")
        return
    print(f"\nQUERY {q!r}")
    for it in items:
        print("  -", it.get("DOI"), "|", (it.get("title") or [""])[0][:90].replace("\n", " "),
          "|", (it.get("container-title") or [""])[0][:45],
          "|", dp(it, "published-print") or dp(it, "published-online"))


DOIS = [
    "10.1111/j.1467-9531.2008.00203.x",
    "10.1093/pan/mps020",
    "10.1017/pan.2018.12.001",
    "10.1002/widm.1237",
    "10.1016/j.joi.2012.09.004",
    "10.1007/s11192-008-0096-0",
    "10.1007/s11336-003-0908-9",
    "10.1016/j.socnet.2010.02.003",
    "10.1080/036109209.08023090",
    "10.1016/j.socnet.2005.01.008",
    "10.1016/j.socnet.2010.01.002",
    "10.1109/MCSE.2014.35",
    "10.1016/j.joi.2010.08.010",
    "10.1016/j.joi.2010.09.002",
    "10.1002/asi.20829",
    "10.1086/421787",
    "10.1371/journal.pone.0294849",
    "10.1016/j.tourman.2021.104057",
    "10.1080/13670016.2020.1730858",
    "10.3390/su13147927",
    "10.3390/su14231323",
    "10.1016/j.jbusres.2021.04.070",
    "10.3389/fsoc.2022.886498",
    "10.1007/s40497-025-00499-4",
    "10.1016/j.annals.2022.103521",
]
for d in DOIS:
    rec(d)
    time.sleep(0.15)

for q in [
    "Monroe Colaresi Quinn Heterogeneity in Latent Classes Sociological Methodology 2008",
    "Grimmer Roberts Stewart Text as Data new framework machine learning social sciences Political Analysis 2018",
    "sports entrepreneurship bibliometric analysis emerging field research",
    "research trends community-based tourism bibliometric 2013 2023",
    "social innovation tourism empirical research agenda",
    "Borgatti Everett node centrality network flow",
    "Opsahl Agnessess Fagerland node centrality weighted networks",
    "Justeson Peh-Zhuang Zhai multi-frequency study of academic text",
    "Hung Wong assessing research performance in science mapping bibliometric review",
    "Bakhshi Cambre Heitor how to interpret positional measures in science mapping",
    "ultrasport event tourism trends sport management bibliometric",
]:
    search(q)
    time.sleep(0.2)
