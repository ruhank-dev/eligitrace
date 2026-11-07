# Error analysis — `finetuned` (206 predictions)

Categories are assigned by deterministic rules (`evaluation/errors.py`); one error can have several.

| Category | Count |
|---|---|
| correct label, wrong evidence | 34 |
| missing information treated as decisive (unsupported decision) | 28 |
| temporal qualifier | 19 |
| false 'not enough information' (evidence was present) | 17 |
| not-applicable confusion | 13 |
| negation in evidence | 7 |
| numeric / threshold reasoning | 6 |
| cited evidence where experts found none | 3 |

## Case 1: false 'not enough information' (evidence was present)
- **Annotation** 161 · patient `sigir-201410` · trial `NCT02097186` · inclusion
- **Criterion:** Patients undergoing elective carotid endarterectomy
- **Gold:** `not included` with evidence []
- **Predicted:** `not enough information` with evidence []
- **Gold evidence text:** (none)
- **Predicted evidence text:** (none)
- **Raw output:** `{"label": "not enough information", "evidence": []}`

## Case 2: missing information treated as decisive (unsupported decision)
- **Annotation** 169 · patient `sigir-201410` · trial `NCT02097186` · exclusion
- **Criterion:** Patients with an estimated pre-operative glomerular filtration rate < 30mls/min/1.73m2
- **Gold:** `not enough information` with evidence []
- **Predicted:** `not excluded` with evidence []
- **Gold evidence text:** (none)
- **Predicted evidence text:** (none)
- **Raw output:** `{"label": "not excluded", "evidence": []}`

## Case 3: correct label, wrong evidence
- **Annotation** 197 · patient `sigir-201411` · trial `NCT01723137` · exclusion
- **Criterion:** Decreased level of consciousness
- **Gold:** `not excluded` with evidence [0, 5]
- **Predicted:** `not excluded` with evidence []
- **Gold evidence text:** [0] A 40-year-old woman with no past medical history presents to the ER with excruciating pain in her right arm that had started 1 hour prior to her admission.; [5] The patient will provide informed consent, and will comply with the trial protocol without any practical issues.
- **Predicted evidence text:** (none)
- **Raw output:** `{"label": "not excluded", "evidence": []}`

## Case 4: not-applicable confusion, temporal qualifier
- **Annotation** 322 · patient `sigir-201417` · trial `NCT01745731` · exclusion
- **Criterion:** Pregnant women or women of childbearing age who do not have adequate contraception.
- **Gold:** `not excluded` with evidence [0]
- **Predicted:** `not applicable` with evidence []
- **Gold evidence text:** [0] A 48-year-old white male with history of common variable immunodeficiency (CVID) with acute abdominal pain, fever, dehydration, HR of 132 bpm, BP 80/40.
- **Predicted evidence text:** (none)
- **Raw output:** `{"label": "not applicable", "evidence": []}`

## Case 5: false 'not enough information' (evidence was present), cited evidence where experts found none
- **Annotation** 162 · patient `sigir-201410` · trial `NCT02097186` · inclusion
- **Criterion:** Patients undergoing open abdominal aortic aneurysm repair
- **Gold:** `not included` with evidence []
- **Predicted:** `not enough information` with evidence [0, 1, 2]
- **Gold evidence text:** (none)
- **Predicted evidence text:** [0] A physician is called to see a 67-year-old woman who underwent cardiac catheterization via the right femoral artery earlier in the morning.; [1] She is now complaining of a cool right foot.; [2] Upon examination she has a pulsatile mass in her right groin with loss of distal pulses, and auscultation reveals a bruit over the point at which the right femoral artery was entered.
- **Raw output:** `{"label": "not enough information", "evidence": [0, 1, 2]}`

## Case 6: false 'not enough information' (evidence was present), cited evidence where experts found none
- **Annotation** 163 · patient `sigir-201410` · trial `NCT02097186` · inclusion
- **Criterion:** Patients undergoing endovascular abdominal aneurysm repair
- **Gold:** `not included` with evidence []
- **Predicted:** `not enough information` with evidence [0, 1, 2]
- **Gold evidence text:** (none)
- **Predicted evidence text:** [0] A physician is called to see a 67-year-old woman who underwent cardiac catheterization via the right femoral artery earlier in the morning.; [1] She is now complaining of a cool right foot.; [2] Upon examination she has a pulsatile mass in her right groin with loss of distal pulses, and auscultation reveals a bruit over the point at which the right femoral artery was entered.
- **Raw output:** `{"label": "not enough information", "evidence": [0, 1, 2]}`

## Case 7: false 'not enough information' (evidence was present), cited evidence where experts found none
- **Annotation** 164 · patient `sigir-201410` · trial `NCT02097186` · inclusion
- **Criterion:** Patients undergoing surgical lower limb revascularisation (suprainguinal or infrainguinal)
- **Gold:** `not included` with evidence []
- **Predicted:** `not enough information` with evidence [0, 1, 2]
- **Gold evidence text:** (none)
- **Predicted evidence text:** [0] A physician is called to see a 67-year-old woman who underwent cardiac catheterization via the right femoral artery earlier in the morning.; [1] She is now complaining of a cool right foot.; [2] Upon examination she has a pulsatile mass in her right groin with loss of distal pulses, and auscultation reveals a bruit over the point at which the right femoral artery was entered.
- **Raw output:** `{"label": "not enough information", "evidence": [0, 1, 2]}`

## Case 8: missing information treated as decisive (unsupported decision), temporal qualifier
- **Annotation** 174 · patient `sigir-201410` · trial `NCT02097186` · exclusion
- **Criterion:** Patients with previous axillary surgery
- **Gold:** `not enough information` with evidence []
- **Predicted:** `not excluded` with evidence []
- **Gold evidence text:** (none)
- **Predicted evidence text:** (none)
- **Raw output:** `{"label": "not excluded", "evidence": []}`

## Case 9: false 'not enough information' (evidence was present), temporal qualifier
- **Annotation** 176 · patient `sigir-201410` · trial `NCT02264964` · inclusion
- **Criterion:** No medical history of central vena catheterization.
- **Gold:** `included` with evidence [0]
- **Predicted:** `not enough information` with evidence [0]
- **Gold evidence text:** [0] A physician is called to see a 67-year-old woman who underwent cardiac catheterization via the right femoral artery earlier in the morning.
- **Predicted evidence text:** [0] A physician is called to see a 67-year-old woman who underwent cardiac catheterization via the right femoral artery earlier in the morning.
- **Raw output:** `{"label": "not enough information", "evidence": [0]}`

## Case 10: false 'not enough information' (evidence was present)
- **Annotation** 177 · patient `sigir-201410` · trial `NCT02264964` · inclusion
- **Criterion:** Maintenance hemodialysis after central vena catheterization.
- **Gold:** `not included` with evidence []
- **Predicted:** `not enough information` with evidence []
- **Gold evidence text:** (none)
- **Predicted evidence text:** (none)
- **Raw output:** `{"label": "not enough information", "evidence": []}`

## Case 11: false 'not enough information' (evidence was present), negation in evidence, temporal qualifier
- **Annotation** 185 · patient `sigir-201411` · trial `NCT01526382` · inclusion
- **Criterion:** Patients were included if they were diagnosed with Shock, Acute respiratory distress syndrome (ARDS), or both.
- **Gold:** `included` with evidence [2, 3]
- **Predicted:** `not enough information` with evidence [0, 1, 2, 3]
- **Gold evidence text:** [2] On examination she is pale and in moderate discomfort, as well as tachypneic and tachycardic.; [3] Her body temperature is normal and her blood pressure is 80/60.
- **Predicted evidence text:** [0] A 40-year-old woman with no past medical history presents to the ER with excruciating pain in her right arm that had started 1 hour prior to her admission.; [1] She denies trauma.; [2] On examination she is pale and in moderate discomfort, as well as tachypneic and tachycardic.; [3] Her body temperature is normal and her blood pressure is 80/60.
- **Raw output:** `{"label": "not enough information", "evidence": [0, 1, 2, 3]}`

## Case 12: false 'not enough information' (evidence was present), numeric / threshold reasoning
- **Annotation** 186 · patient `sigir-201411` · trial `NCT01526382` · inclusion
- **Criterion:** Heart rate of at least 90/min;
- **Gold:** `included` with evidence [2]
- **Predicted:** `not enough information` with evidence [3]
- **Gold evidence text:** [2] On examination she is pale and in moderate discomfort, as well as tachypneic and tachycardic.
- **Predicted evidence text:** [3] Her body temperature is normal and her blood pressure is 80/60.
- **Raw output:** `{"label": "not enough information", "evidence": [3]}`

## Case 13: false 'not enough information' (evidence was present)
- **Annotation** 187 · patient `sigir-201411` · trial `NCT01526382` · inclusion
- **Criterion:** A respiratory rate of at least 20/min or a PaCO2 of 32mmHg or lower or the use of mechanical ventilation;
- **Gold:** `included` with evidence [2]
- **Predicted:** `not enough information` with evidence [2]
- **Gold evidence text:** [2] On examination she is pale and in moderate discomfort, as well as tachypneic and tachycardic.
- **Predicted evidence text:** [2] On examination she is pale and in moderate discomfort, as well as tachypneic and tachycardic.
- **Raw output:** `{"label": "not enough information", "evidence": [2]}`

## Case 14: false 'not enough information' (evidence was present), numeric / threshold reasoning
- **Annotation** 188 · patient `sigir-201411` · trial `NCT01526382` · inclusion
- **Criterion:** The use of vasopressors to maintain a systolic blood pressure of at least 90mmHg despite fluid resuscitation, low dose of dopamine (≤ 5 μg/kg per minute), or dobutamine;
- **Gold:** `not included` with evidence [3]
- **Predicted:** `not enough information` with evidence []
- **Gold evidence text:** [3] Her body temperature is normal and her blood pressure is 80/60.
- **Predicted evidence text:** (none)
- **Raw output:** `{"label": "not enough information", "evidence": []}`