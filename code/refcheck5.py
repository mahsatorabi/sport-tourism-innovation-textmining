import json, sys, time, urllib.parse, urllib.request
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
UA="ArticleRefCheck/1.0 (mailto:refs-check@example.org)"
def api(u):
    for _ in range(3):
        try:
            with urllib.request.urlopen(urllib.request.Request(u, headers={"User-Agent":UA}), timeout=40) as r:
                return json.load(r)
        except Exception as e:
            err=e; time.sleep(1.5)
    raise err
def dp(m,k): return (m.get(k,{}).get("date-parts") or [[None]])[0]
def rec(d):
    try: m=api("https://api.crossref.org/works/"+urllib.parse.quote(d))["message"]
    except Exception as e: print("\nDOI",d,"\n  [FAIL]",type(e).__name__,e); return
    print("\nDOI",d)
    print("  authors:", " | ".join(f"{a.get('family','')}; {a.get('given','')}" for a in m.get("author",[])))
    print("  title  :", (m.get("title") or [""])[0].replace("\n"," "))
    print("  journal:", (m.get("container-title") or [""])[0], "|", m.get("volume",""), m.get("issue",""), m.get("page","") or m.get("article-number",""))
    print("  print:", dp(m,"published-print"), "| online:", dp(m,"published-online"))
def q(s,rows=4):
    try: it=api("https://api.crossref.org/works?rows=%d&query.bibliographic=%s"%(rows,urllib.parse.quote(s)))["message"]["items"]
    except Exception as e: print("\nQ",s,"[FAIL]",e); return
    print("\nQ",s)
    for i in it: print("  -",i.get("DOI"),"|",(i.get("title") or [""])[0][:85].replace("\n"," "),"|",(i.get("container-title") or [""])[0][:40],"|",dp(i,"published-print") or dp(i,"published-online"))
for d in ["10.1017/S1049096513001831","10.1093/pan/mps020","10.1093/pan/mpv005","10.1177/14673584221100719","10.1016/j.ecores.2026.100023","10.1111/j.1467-9531.2008.00203.x","10.1111/j.1467-9531.2007.00352.x"]:
    rec(d); time.sleep(0.15)
for s in ["Grimmer Stewart Text as Data Promise and Pitfalls of Machine Learning Methods for Political Text Political Analysis 2013",
          "Monroe Colaresi Quinn Heterogeneity in Latent Classes What Do We Learn from Community Classifications",
          "social innovation in tourism theoretical framework agenda Annals of Tourism Research",
          "sport tourism innovation knowledge management bibliometric analysis",
          "community based tourism bibliometric analysis 2013 2023 Global Economics Research"]:
    q(s); time.sleep(0.2)
