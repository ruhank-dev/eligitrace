from modeling.explain import grounding_check

EV = {1: "His most recent HbA1c was 8.4% despite metformin 1000 mg twice daily."}
CRIT = "HbA1c between 7.0% and 10.5%"


def test_rejects_stray_number():
    ok, why = grounding_check("The HbA1c of 9.1% is in range for this patient.", "included", CRIT, EV)
    assert not ok and "numbers" in why


def test_rejects_verdict_for_undecided_label():
    # real failure seen in the app: NEI label but the rationale asserts the opposite verdict
    text = "The HbA1c was 8.4%, which falls outside the range, so the patient does not meet the criterion."
    assert not grounding_check(text, "not enough information", CRIT, EV)[0]


def test_rejects_undecided_wording_for_definite_label():
    assert not grounding_check("There is not enough information about HbA1c here.", "included", CRIT, EV)[0]


def test_accepts_grounded_rationale():
    assert grounding_check("The note reports an HbA1c of 8.4%, which is within the required band.",
                           "included", CRIT, EV)[0]
