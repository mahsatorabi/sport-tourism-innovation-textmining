# Scopus search strategy (Appendix A)

Database: **Scopus**
Export date: **9 September 2026**
Filter applied at download: **none**
Fields searched: `TITLE-ABS-KEY`

## Master string (S0, the reported strategy)

```
TITLE-ABS-KEY(
(
"entrepreneurship"
OR "innovation"
OR "social innovation"
OR "community development"
OR "local development"
)
AND
(
"sport tourism"
OR "sports tourism"
OR "sport event*"
OR "tourism development"
)
)
```

Raw records retrieved: **2,815**. Screened analytical corpus: **2,265**.

## Alternative arms (Table A1, evaluated in Section 4.6)

These quantify how much the reported structure depends on the Boolean strategy.

| Arm | Logic | Rationale |
|-----|-------|-----------|
| S0 (master) | construct group AND context branch | the reported strategy |
| S1 | as S0 minus `"social innovation"` | is the phrase doing work? |
| S2 | `"innovation" AND "entrepreneurship" AND "community development" AND "tourism development"` | strict three-way conjunction |
| S3 | construct group AND `"tourism development"` | drop the sport branch |
| S4 | `"sport tourism" OR "sports tourism" OR "sport event*"` | sport branch alone |
| S5 | `"social innovation" AND ("community development" OR "entrepreneurship") AND "tourism development"` | hard social lens |
| S6 | `"entrepreneurship" AND ("tourism development" OR "sport tourism" OR "sports tourism")` | entrepreneurship only |

Corpus size and induced structural overlap for each arm are in
`outputs/search_strategy_sensitivity.csv` and plotted in Figure 5.

## Export fields

title, abstract, author keywords, year, source title, document type, DOI, EID,
language of original document, affiliations, citations, open-access status.

## Reproducing

Place the export as `data.csv` in the repository root and run:

```bash
python code/pipeline.py
python code/diagnostics.py
```

The screening order is fixed in `pipeline.py` and the stage counts are asserted
against the executed run, so a re-run producing different counts fails loudly
rather than silently.
