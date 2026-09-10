# Scopus search string (Appendix A)

Database: **Scopus**  
Export date: **9 September 2026**  
Filter at download: **none**

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

After export, place the CSV as `data.csv` in the repository root and run `python code/pipeline.py`.
The Article 1 analysis uses the **full cleaned** English scholarly corpus after deduplication and quality screening (N = 2,265 in the reported run).
