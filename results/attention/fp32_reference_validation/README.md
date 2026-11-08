Earlier validation run (08 Nov 2026 and before). Its "high-precision truth" was computed by
`sdpa_reference(q.double(), ...)`, but that function cast its inputs with `.float()`, so the truth was
**fp32, not float64**. Kept unchanged as a separately labelled experiment ("fp32 reference").
The float64-reference validation is in `../validation.csv` and `../summary.json`.
