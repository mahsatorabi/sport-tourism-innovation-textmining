"""Third pass: full author lists and print-issue dates for the entries the
revision will cite, so APA 7 output is exact."""

from __future__ import annotations

import json
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

OUT = Path(__file__).resolve().parent / "outputs_diagnostics"
UA = "ArticleRefCheck/1.0 (mailto:refs-check@example.org)"


def api(url):
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.load(r)


def dp(m, key):
    v = m.get(key, {})
    return (v.get("date-parts") or [[None]])[0]


def rec(doi):
    try:
        m = api(f"https://api.crossref.org/works/{urllib.parse.quote(doi)}")["message"]
    except Exception as e:
        print(f"[ERR] {doi} {type(e).__name__}")
        return
    print(f"\nDOI {doi}")
    print("  authors:", " | ".join(
        f"{a.get('family','')}; {a.get('given','')}" for a in m.get("author", [])))
    print("  title  :", (m.get("title") or [""])[0])
    print("  journal:", (m.get("container-title") or [""])[0])
    print("  vol/iss/pg:", m.get("volume", ""), m.get("issue", ""),
          m.get("page", "") or m.get("article-number", ""))
    print("  issued:", dp(m, "issued"), "| print:", dp(m, "published-print"),
          "| online:", dp(m, "published-online"))
    print("  type  :", m.get("type"), "| publisher:", m.get("publisher", ""))


for d in [
    "10.3389/fspor.2025.1680229",
    "10.1007/s11192-024-05161-6",
    "10.1177/14673584231218106",
    "10.1108/IJCHM-04-2022-0497",
    "10.1111/joms.13000",
    "10.1007/s11192-015-1765-5",
    "10.1108/TR-06-2019-0258",
    "10.1002/asi.21075",
    "10.1007/978-3-319-10377-8_13",
    "10.1111/j.1467-9531.2008.00204.x",
    "10.1016/j.annals.2016.10.006",
    "10.1515/jdis-2017-0006",
    "10.1016/j.eswa.2017.08.047",
    "10.1002/wics.70066",
    "10.1016/j.tourman.2015.03.007",
    "10.1016/j.jbusres.2021.04.070",
    "10.3389/fsoc.2022.886498",
    "10.1016/j.joi.2017.08.007",
    "10.1371/journal.pone.0294849",
    "10.1002/asi.21525",
    "10.1002/asi.22688",
    "10.1080/1742-5468/2008/10/P10008",
    "10.1086/421787",
    "10.1016/0378-8733(78)90021-7",
    "10.1177/053901883022002003",
    "10.1038/44565",
    "10.1007/s12525-025-00847-y",
    "10.1080/09669582.2023.2189622",
    "10.1007/s40497-024-00399-z",
    "10.3390/admsci15040130",
    "10.1016/j.jdmm.2023.100850",
    "10.1016/j.jdmm.2025.101004",
    "10.1016/j.heliyon.2024.e25627",
    "10.1108/JKM-06-2022-0434",
    "10.3390/su12125209",
    "10.1016/j.tourman.2023.104724",
    "10.21832/higham6553",
    "10.1007/978-981-97-8923-8_1",
    "10.1007/s40497-025-00499-4",
    "10.46827/ejsss.v9i3.1610",
    "10.47197/retos.v62.108401",
    "10.21580/prosperity.2023.3.1.14744",
    "10.18280/mmep.111116",
    "10.1177/09708464251335116",
]:
    rec(d)
    time.sleep(0.12)
