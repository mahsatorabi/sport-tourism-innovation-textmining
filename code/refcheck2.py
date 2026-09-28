"""Second, focused Crossref pass: resolve the entries whose metadata in the current
reference list is wrong or incomplete, and resolve the additional methodological
sources the revision needs to cite."""

from __future__ import annotations

import json
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

OUT = Path(__file__).resolve().parent / "outputs_diagnostics"
MAILTO = "refs-check@example.org"
UA = "ArticleRefCheck/1.0 (mailto:%s)" % MAILTO


def api(url: str):
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.load(r)


def fmt_authors(a):
    out = []
    for x in a:
        fam, giv = x.get("family", ""), x.get("given", "")
        if fam:
            out.append(f"{fam}, {giv}".strip().rstrip(","))
    return "; ".join(out)


def show(doi):
    try:
        m = api(f"https://api.crossref.org/works/{urllib.parse.quote(doi)}")["message"]
    except Exception as e:
        print(f"[ERR ] {doi}  {type(e).__name__}")
        return None
    print(f"[OK  ] {doi}")
    print(f"       {fmt_authors(m.get('author', []))}")
    print(f"       {(m.get('title') or [''])[0]}")
    print(f"       {(m.get('container-title') or [''])[0]} {m.get('volume','')}({m.get('issue','')}) "
          f"{m.get('page','') or m.get('article-number','')} ({(m.get('issued',{}).get('date-parts') or [[None]])[0][0]})")
    return m


def q(label, query, rows=3):
    print(f"\n### {label}")
    try:
        items = api("https://api.crossref.org/works?query.bibliographic=" +
                   urllib.parse.quote(query) + f"&rows={rows}")["message"]["items"]
    except Exception as e:
        print("   ERR", type(e).__name__)
        return
    for it in items:
        print(f"   {it.get('DOI')}")
        print(f"     {fmt_authors(it.get('author', []))[:300]}")
        print(f"     {(it.get('title') or [''])[0][:160]}")
        print(f"     {(it.get('container-title') or [''])[0]} {it.get('volume','')}({it.get('issue','')}) "
              f"{it.get('page','') or it.get('article-number','')} "
              f"({(it.get('issued',{}).get('date-parts') or [[None]])[0][0]})")
    time.sleep(0.2)


print("=" * 70)
print("DOIs to verify directly")
for doi in [
    "10.1016/j.tourman.2015.03.007",
    "10.1002/asi.20815",
    "10.1007/978-3-319-10377-8_13",
    "10.1086/421787",
    "10.1016/0378-8733(78)90021-7",
    "10.1177/053901883022002003",
    "10.1038/44565",
    "10.1007/11192-023-09481-7",
    "10.1007/s11192-015-1765-5",
    "10.1108/TR-06-2019-0258",
    "10.1111/joms.13000",
    "10.1108/JKM-06-2022-0434",
    "10.47197/retos.v62.108401",
    "10.21580/prosperity.2023.3.1.14744",
    "10.18280/mmep.111116",
    "10.1177/14673584231218106",
    "10.1177/09708464251335116",
    "10.1002/asi.21525",
    "10.1002/asi.22688",
    "10.1007/978-3-030-02511-3_7",
    "10.1371/journal.pone.0296659",
]:
    show(doi)
    time.sleep(0.15)

print("\n" + "=" * 70)
q("Romero-Medina organizational design CBT",
  "Organizational design for strengthening community-based tourism empowering stakeholders for self-organization and networking")
q("Khoo women tourism entrepreneurs digital competencies",
  "Opportunities and challenges of digital competencies for women tourism entrepreneurs in Latin America a gendered perspective")
q("Rangkuti cultural heritage sports tourism review",
  "Cultural heritage and sports tourism a systematic literature review of sustainable destination management practices")
q("Ul Hassan entrepreneurship tourism hospitality post-COVID bibliometric",
  "Entrepreneurship in tourism and hospitality a post-COVID-19 systematic review and bibliometric analysis Journal of Global Entrepreneurship Research")
q("Ratten 2018 sport entrepreneurship book",
  "Ratten Sport entrepreneurship developing and sustaining an entrepreneurial sports culture")
q("van Eck Waltman 2009 normalize cooccurrence",
  "How to normalize cooccurrence data an analysis of some well-known similarity measures van Eck Waltman")
q("Suwanmanee social innovation community-based sustainable tourism",
  "social innovation community-based sustainable tourism development case study Nan province Thailand")
q("Monroe Colaresi Quinn 2008 log odds",
  "Detecting subgroup differences in high-dimensional datasets log-odds-ratio analysis Monroe")
q("Grimmer Roberts Stewart text as data",
  "Text as data a new framework for machine learning and the social sciences Grimmer")
q("Mauceri topic modeling evaluation literature",
  "Topic modeling literature review Mauceri")
q("Belford topic stability",
  "Belford topic stability in topic models")
q("Aletras coherence topic models",
  "Aletras topic model evaluation coherence topic models")
q("Yu 2016 review of topic models",
  "Yu review of topic modeling in text mining Yu 2016")
q("Stein topic model domain knowledge",
  "Stein topic models what do they mean")
q("Loughnan topic model analysis text",
  "Loughnan McCabe topic model analysis of textual data")
q("Zupic Absecon Kotev science mapping review",
  "Science mapping systematic review of literature on science mapping methodology purpose and characteristics")
q("Boukacem Zhang python-louvain",
  "Boukacem Zhang python-louvain networkx implementation of Louvain clustering algorithm")
q("Fister Boc `corpus stopwords`",
  "Fister Boc stopwords Slovene stopword list")
q("Hobson Khosravi topic modelling text mining",
  "Hobson topic modelling in tourism research")
q("Wei Koh tourism topic modelling",
  "topic modelling tourism research bibliometric")
q("Aydin Koseoglu tourism resilience bibliometric",
  "Aydin Koseoglu tourism resilience bibliometric")
q("Perdana Squire topic modelling bibliographic tourism",
  "Perdana Squire topic modelling")
q("Donthu 2021 how to conduct bibliometric", "How to conduct a bibliometric analysis an overview and guidelines")
q("Aria 2024 comparative science mapping", "Comparative science mapping a novel conceptual structure analysis with metadata")
q("Arici 2025 sports tourism agenda", "Sports tourism research a bibliometric analysis and agenda for further inquiry Tourism and Hospitality Research")
q("Petermann tourism bibliometric review methods", "Petermann tourism bibliometric")
q("Cobo science mapping SciMAT 2012", "SciMAT a new science mapping analysis software tool")
q("Bastow Tinkler topic models bibliometric", "Bastow Tinkler topic modelling for bibliometric review")
q("Lowe Callaghan topic model stability", "topic model stability journals")
q("Griffiths Steinke topic models text", "Griffiths Steinke topic models in the social sciences")
q("Lee Raftery topic model evaluation", "topic model evaluation interpretability coherence")
q("Vaygonin topic models interpretability", "topic models interpretability")
q("EgBERT Roulstone topic model weeds", "Egbert Roulstone topic models weeds in the garden of documents")
q("Blei probabilistic topic models pdf", "Blei Ng Jordan latent Dirichlet allocation JMLR 3 993 1022")
q("Chuang topic model selection", "Chuang topic models in social science research")
q("Lancaster topic modelling social science", "topic modelling in the social sciences Lancaster")
q("Midgett topic model analysis", "Midgett topic modelling in systematic reviews")
q("Asuncion Ambiel topic modelling visualization", "topic modelling visualization Asuncion Ambiel")
q("Salton Buckley term association", "Salton Buckley term-association in automatic text indexing")
q("Schoch clustering TREC", "Schoch cohesion and cohesion in text")
q("van Eck Waltman 2010 mapping macro topics", "Mapping macro topics based on citation networks")
q("Boyack Klavans co-citation", "Boyack Klavans co-citation analysis bibliographic coupling")
q("Klavans Boyack", "Klavans Boyack conceptual structure analysis")
q("Glanzel 2003 science mapping", "Glanzel science mapping network visualisation")
q("Hassan 2016 science mapping", "Hassan science mapping topic modelling")
q("Cobo Lopez-Herrera 2011 science mapping software tools", "Science mapping software tools review analysis and cooperative study among tools")
q("Kern 2020 science mapping", "Kern science mapping bibliometrics")
q("Sudsawad 2020", "Sudsawad science mapping bibliometric")
q("Aria Cuccurullo 2017 bibliometrix", "bibliometrix an R-tool for comprehensive science mapping analysis")
q("Egger Yu 2022 topic modeling comparison", "A topic modeling comparison between LDA NMF Top2Vec and BERTopic")
q("Tosun 2016 social innovation", "Tosun a critical review of approaches to social innovation")
q("Phelan 2016 social innovation", "Phelan what is social innovation")
q("Dawson 2020 social innovation community tourism", "Dawson social innovation tourism")
q("Gawin 2016 social innovation tourism", "Gawin social innovation in tourism")
q("Bessette Kadlec 2012 community tourism", "community-based tourism")
q("Kibert 2012 community based tourism sustainability", "Kibert community based tourism sustainability")
q("Choi Wang 2018 community tourism resilience", "Choi Wang community-based tourism resilience")
q("Nimakayimana 2019 community tourism", "Nimakayimana community-based tourism")
q("Nielsen 2020 community tourism", "Nielsen community tourism")
q("Govaerts 2022", "Govaerts science mapping bibliometrics")
q("Rescalli 2020", "Rescalli tourism bibliometric")
q("Kitson 2020", "Kitson bibliometric tourism")
q("Fyall 2021", "Fyall tourism research bibliometric")
q("Bitektine 2019 bibliometric reviews", "Bitektine Do bibliometric reviews past present and future")
q("Zupic 2019 science mapping", "Science mapping systematic review of literature on science mapping methodology purpose and characteristics")
q("Cobo 2011 science mapping", "science mapping software tools review analysis cooperative study")
q("Aria 2017 bibliometrix", "bibliometrix an R-tool")
q("Cuccurullo 2016 CiteSpace", "CiteSpace text mining and visualization in scientific literature")
q("Chen 2006 CiteSpace", "CiteSpace II tool for visualization and analysis of scientific literature")
q("Scull 2017", "Scull CiteSpace bibliometrics")
q("Aria 2018", "Aria interpretative and descriptive bibliometrics")
q("Garg 2022", "Garg bibliometric")
q("Klavans 2021", "Klavans research fronts")
q("Bornmann 2015", "Bornmann bibliometrics")
q("S引", "test")
