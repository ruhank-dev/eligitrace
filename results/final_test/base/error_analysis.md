# Error analysis — `base` (206 predictions)

Categories are assigned by deterministic rules (`evaluation/errors.py`); one error can have several.

| Category | Count |
|---|---|
| cited evidence where experts found none | 92 |
| temporal qualifier | 78 |
| polarity flip (met vs not met) | 66 |
| negation in evidence | 66 |
| missing information treated as decisive (unsupported decision) | 66 |
| not-applicable confusion | 63 |
| numeric / threshold reasoning | 43 |
| correct label, wrong evidence | 26 |

## Case 1: correct label, wrong evidence
- **Annotation** 159 · patient `sigir-201410` · trial `NCT02097186` · inclusion
- **Criterion:** Age greater than 18 years
- **Gold:** `included` with evidence [0]
- **Predicted:** `included` with evidence [0, 1, 2]
- **Gold evidence text:** [0] A physician is called to see a 67-year-old woman who underwent cardiac catheterization via the right femoral artery earlier in the morning.
- **Predicted evidence text:** [0] A physician is called to see a 67-year-old woman who underwent cardiac catheterization via the right femoral artery earlier in the morning.; [1] She is now complaining of a cool right foot.; [2] Upon examination she has a pulsatile mass in her right groin with loss of distal pulses, and auscultation reveals a bruit over the point at which the right femoral artery was entered.
- **Raw output:** `{"label": "included", "evidence": [0, 1, 2]}`

## Case 2: polarity flip (met vs not met), cited evidence where experts found none
- **Annotation** 161 · patient `sigir-201410` · trial `NCT02097186` · inclusion
- **Criterion:** Patients undergoing elective carotid endarterectomy
- **Gold:** `not included` with evidence []
- **Predicted:** `included` with evidence [0, 1, 2]
- **Gold evidence text:** (none)
- **Predicted evidence text:** [0] A physician is called to see a 67-year-old woman who underwent cardiac catheterization via the right femoral artery earlier in the morning.; [1] She is now complaining of a cool right foot.; [2] Upon examination she has a pulsatile mass in her right groin with loss of distal pulses, and auscultation reveals a bruit over the point at which the right femoral artery was entered.
- **Raw output:** ````json
{
  "label": "included",
  "evidence": [0, 1, 2]
}
````

## Case 3: not-applicable confusion, cited evidence where experts found none
- **Annotation** 165 · patient `sigir-201410` · trial `NCT02097186` · exclusion
- **Criterion:** Pregnancy
- **Gold:** `not applicable` with evidence []
- **Predicted:** `excluded` with evidence [1, 2]
- **Gold evidence text:** (none)
- **Predicted evidence text:** [1] She is now complaining of a cool right foot.; [2] Upon examination she has a pulsatile mass in her right groin with loss of distal pulses, and auscultation reveals a bruit over the point at which the right femoral artery was entered.
- **Raw output:** `{"label": "excluded", "evidence": [1, 2]}`

## Case 4: missing information treated as decisive (unsupported decision), cited evidence where experts found none, numeric / threshold reasoning
- **Annotation** 169 · patient `sigir-201410` · trial `NCT02097186` · exclusion
- **Criterion:** Patients with an estimated pre-operative glomerular filtration rate < 30mls/min/1.73m2
- **Gold:** `not enough information` with evidence []
- **Predicted:** `excluded` with evidence [0, 1, 2]
- **Gold evidence text:** (none)
- **Predicted evidence text:** [0] A physician is called to see a 67-year-old woman who underwent cardiac catheterization via the right femoral artery earlier in the morning.; [1] She is now complaining of a cool right foot.; [2] Upon examination she has a pulsatile mass in her right groin with loss of distal pulses, and auscultation reveals a bruit over the point at which the right femoral artery was entered.
- **Raw output:** `{"label": "excluded", "evidence": [0, 1, 2]}`

## Case 5: polarity flip (met vs not met), cited evidence where experts found none
- **Annotation** 162 · patient `sigir-201410` · trial `NCT02097186` · inclusion
- **Criterion:** Patients undergoing open abdominal aortic aneurysm repair
- **Gold:** `not included` with evidence []
- **Predicted:** `included` with evidence [0, 1, 2]
- **Gold evidence text:** (none)
- **Predicted evidence text:** [0] A physician is called to see a 67-year-old woman who underwent cardiac catheterization via the right femoral artery earlier in the morning.; [1] She is now complaining of a cool right foot.; [2] Upon examination she has a pulsatile mass in her right groin with loss of distal pulses, and auscultation reveals a bruit over the point at which the right femoral artery was entered.
- **Raw output:** ````json
{
  "label": "included",
  "evidence": [0, 1, 2]
}
````

## Case 6: polarity flip (met vs not met), cited evidence where experts found none, negation in evidence
- **Annotation** 163 · patient `sigir-201410` · trial `NCT02097186` · inclusion
- **Criterion:** Patients undergoing endovascular abdominal aneurysm repair
- **Gold:** `not included` with evidence []
- **Predicted:** `included` with evidence [1, 2, 3]
- **Gold evidence text:** (none)
- **Predicted evidence text:** [1] She is now complaining of a cool right foot.; [2] Upon examination she has a pulsatile mass in her right groin with loss of distal pulses, and auscultation reveals a bruit over the point at which the right femoral artery was entered.; [3] The patient will provide informed consent, and will comply with the trial protocol without any practical issues.
- **Raw output:** ````json
{
  "label": "included",
  "evidence": [1, 2, 3]
}
````

## Case 7: polarity flip (met vs not met), cited evidence where experts found none, negation in evidence
- **Annotation** 164 · patient `sigir-201410` · trial `NCT02097186` · inclusion
- **Criterion:** Patients undergoing surgical lower limb revascularisation (suprainguinal or infrainguinal)
- **Gold:** `not included` with evidence []
- **Predicted:** `included` with evidence [1, 2, 3]
- **Gold evidence text:** (none)
- **Predicted evidence text:** [1] She is now complaining of a cool right foot.; [2] Upon examination she has a pulsatile mass in her right groin with loss of distal pulses, and auscultation reveals a bruit over the point at which the right femoral artery was entered.; [3] The patient will provide informed consent, and will comply with the trial protocol without any practical issues.
- **Raw output:** ````json
{
  "label": "included",
  "evidence": [1, 2, 3]
}
````

## Case 8: polarity flip (met vs not met), cited evidence where experts found none, negation in evidence
- **Annotation** 166 · patient `sigir-201410` · trial `NCT02097186` · exclusion
- **Criterion:** Significant upper limb peripheral arterial disease
- **Gold:** `not excluded` with evidence []
- **Predicted:** `excluded` with evidence [1, 2, 3]
- **Gold evidence text:** (none)
- **Predicted evidence text:** [1] She is now complaining of a cool right foot.; [2] Upon examination she has a pulsatile mass in her right groin with loss of distal pulses, and auscultation reveals a bruit over the point at which the right femoral artery was entered.; [3] The patient will provide informed consent, and will comply with the trial protocol without any practical issues.
- **Raw output:** `{"label": "excluded", "evidence": [1, 2, 3]}`

## Case 9: polarity flip (met vs not met), cited evidence where experts found none, temporal qualifier
- **Annotation** 167 · patient `sigir-201410` · trial `NCT02097186` · exclusion
- **Criterion:** Previous history of upper limb deep vein thrombosis
- **Gold:** `not excluded` with evidence []
- **Predicted:** `excluded` with evidence [1, 2]
- **Gold evidence text:** (none)
- **Predicted evidence text:** [1] She is now complaining of a cool right foot.; [2] Upon examination she has a pulsatile mass in her right groin with loss of distal pulses, and auscultation reveals a bruit over the point at which the right femoral artery was entered.
- **Raw output:** `{"label": "excluded", "evidence": [1, 2]}`

## Case 10: polarity flip (met vs not met), cited evidence where experts found none
- **Annotation** 168 · patient `sigir-201410` · trial `NCT02097186` · exclusion
- **Criterion:** Patients on glibenclamide or nicorandil (these medications may interfere with RIPC)
- **Gold:** `not excluded` with evidence []
- **Predicted:** `excluded` with evidence [1, 2]
- **Gold evidence text:** (none)
- **Predicted evidence text:** [1] She is now complaining of a cool right foot.; [2] Upon examination she has a pulsatile mass in her right groin with loss of distal pulses, and auscultation reveals a bruit over the point at which the right femoral artery was entered.
- **Raw output:** `{"label": "excluded", "evidence": [1, 2]}`

## Case 11: polarity flip (met vs not met), cited evidence where experts found none, temporal qualifier
- **Annotation** 170 · patient `sigir-201410` · trial `NCT02097186` · exclusion
- **Criterion:** Patients with a known history of myocarditis, pericarditis or amyloidosis
- **Gold:** `not excluded` with evidence []
- **Predicted:** `excluded` with evidence [0, 1, 2]
- **Gold evidence text:** (none)
- **Predicted evidence text:** [0] A physician is called to see a 67-year-old woman who underwent cardiac catheterization via the right femoral artery earlier in the morning.; [1] She is now complaining of a cool right foot.; [2] Upon examination she has a pulsatile mass in her right groin with loss of distal pulses, and auscultation reveals a bruit over the point at which the right femoral artery was entered.
- **Raw output:** `{"label": "excluded", "evidence": [0, 1, 2]}`

## Case 12: polarity flip (met vs not met), cited evidence where experts found none
- **Annotation** 171 · patient `sigir-201410` · trial `NCT02097186` · exclusion
- **Criterion:** Patients with severe hepatic disease defined as an international normalised ratio >2 in the absence of systemic anticoagulation
- **Gold:** `not excluded` with evidence []
- **Predicted:** `excluded` with evidence [1, 2]
- **Gold evidence text:** (none)
- **Predicted evidence text:** [1] She is now complaining of a cool right foot.; [2] Upon examination she has a pulsatile mass in her right groin with loss of distal pulses, and auscultation reveals a bruit over the point at which the right femoral artery was entered.
- **Raw output:** `{"label": "excluded", "evidence": [1, 2]}`

## Case 13: polarity flip (met vs not met), cited evidence where experts found none
- **Annotation** 172 · patient `sigir-201410` · trial `NCT02097186` · exclusion
- **Criterion:** Patients with severe respiratory disease (for the trial, defined as patients requiring home oxygen therapy)
- **Gold:** `not excluded` with evidence []
- **Predicted:** `excluded` with evidence [1]
- **Gold evidence text:** (none)
- **Predicted evidence text:** [1] She is now complaining of a cool right foot.
- **Raw output:** `{"label": "excluded", "evidence": [1]}`

## Case 14: polarity flip (met vs not met), cited evidence where experts found none
- **Annotation** 173 · patient `sigir-201410` · trial `NCT02097186` · exclusion
- **Criterion:** Patients previously enrolled in the trial representing for a further procedure
- **Gold:** `not excluded` with evidence []
- **Predicted:** `excluded` with evidence [0, 1, 2]
- **Gold evidence text:** (none)
- **Predicted evidence text:** [0] A physician is called to see a 67-year-old woman who underwent cardiac catheterization via the right femoral artery earlier in the morning.; [1] She is now complaining of a cool right foot.; [2] Upon examination she has a pulsatile mass in her right groin with loss of distal pulses, and auscultation reveals a bruit over the point at which the right femoral artery was entered.
- **Raw output:** `{"label": "excluded", "evidence": [0, 1, 2]}`