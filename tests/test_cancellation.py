from dataclasses import replace
from datetime import date, timedelta

import pytest

from insurance_assistant.models import DEFAULT_AS_OF, Intent, Outcome, Policy
from insurance_assistant.repository import CsvPolicyRepository, ValidatedPolicyRepository
from insurance_assistant.rules.cancellation import (
    add_years,
    claims_escalation,
    end_of_month,
    evaluate_cancellation,
    full_years_between,
    grant_percent,
    has_open_or_unknown_claims,
)

AS_OF = DEFAULT_AS_OF  # 2026-10-25


class FakeRepository:
    source_name = "fake-api"

    def __init__(self, *policies):
        self.policies = list(policies)

    def get(self, policy_id):
        return next((p for p in self.policies if p.policy_id == policy_id), None)

    def by_customer(self, customer_id):
        return [p for p in self.policies if p.customer_id == customer_id]

    def find_names_in(self, text):
        return [p for p in self.policies if p.customer_name in text]


def make(product="דירה", start=date(2020, 3, 10), premium=100, status="פעילה",
         last_payment=date(2026, 10, 1), open_claims=0, policy_id="7001", customer_id="C-7001"):
    return Policy(policy_id=policy_id, customer_id=customer_id, customer_name="לקוח בדיקה",
                  product=product, start_date=start, monthly_premium=premium, status=status,
                  last_payment_date=last_payment, open_claims=open_claims)


def evaluate(policy, as_of=AS_OF, *others):
    repo = ValidatedPolicyRepository(FakeRepository(policy, *others))
    return evaluate_cancellation(repo.get(policy.policy_id), as_of, repo)


# --- date helpers ----------------------------------------------------------------

def test_add_years():
    assert add_years(date(2020, 2, 29), 1) == date(2021, 2, 28)
    assert add_years(date(2020, 2, 29), 4) == date(2024, 2, 29)
    assert add_years(date(2021, 10, 16), 5) == date(2026, 10, 16)
    with pytest.raises(OverflowError):
        add_years(date(9998, 1, 1), 2)
    with pytest.raises(OverflowError):
        add_years(date(1, 1, 1), -1)


def test_full_years_between():
    assert full_years_between(date(2020, 3, 10), date(2023, 3, 9)) == 2
    assert full_years_between(date(2020, 3, 10), date(2023, 3, 10)) == 3
    assert full_years_between(date(2020, 2, 29), date(2021, 2, 28)) == 1
    assert full_years_between(date(2026, 1, 1), date(2026, 12, 31)) == 0


def test_end_of_month():
    assert end_of_month(date(2026, 10, 25)) == date(2026, 10, 31)
    assert end_of_month(date(2024, 2, 3)) == date(2024, 2, 29)
    assert end_of_month(date(2026, 2, 3)) == date(2026, 2, 28)


# --- grant tiers: both sides of every boundary -------------------------------------------

@pytest.mark.parametrize("as_of,percent", [
    (date(2021, 3, 9), 0), (date(2021, 3, 10), 5),
    (date(2023, 3, 9), 5), (date(2023, 3, 10), 10),
    (date(2025, 3, 10), 10), (date(2025, 3, 11), 15),
])
def test_grant_tier_boundaries(as_of, percent):
    assert grant_percent(date(2020, 3, 10), as_of) == percent
    d = evaluate(make(start=date(2020, 3, 10), premium=100, last_payment=as_of), as_of)
    assert d.code == "CANCEL_ESTIMATE"
    assert d.numbers["grant"] == 100 * 12 * percent / 100
    assert any("מדרגות מענק הוותק" in a for a in d.assumptions)


@pytest.mark.parametrize("as_of,percent", [
    (date(2021, 2, 27), 0), (date(2021, 2, 28), 5),
    (date(2023, 2, 27), 5), (date(2023, 2, 28), 10),
    (date(2025, 2, 28), 10), (date(2025, 3, 1), 15),
])
def test_grant_tiers_feb_29_start(as_of, percent):
    assert grant_percent(date(2020, 2, 29), as_of) == percent


def test_grant_15_percent_after_five_years():
    assert grant_percent(date(2021, 10, 16), date(2026, 10, 25)) == 15
    d = evaluate(make(start=date(2021, 10, 16), premium=200))
    assert d.numbers["grant"] == 360


def test_grant_rounding():
    d = evaluate(make(start=date(2024, 1, 1), premium=99.99))
    assert d.numbers["grant"] == round(99.99 * 12 * 0.05, 2)


# --- fees -------------------------------------------------------------------------

@pytest.mark.parametrize("product", ["דירה", "בריאות"])
@pytest.mark.parametrize("as_of,fee", [(date(2026, 10, 24), 50), (date(2026, 10, 25), 0)])
def test_short_tenure_fee_boundary(product, as_of, fee):
    d = evaluate(make(product=product, start=date(2025, 10, 25)), as_of)
    assert d.code == "CANCEL_ESTIMATE"
    assert d.numbers["fee"] == fee


@pytest.mark.parametrize("start", [date(2025, 10, 25), date(2026, 9, 1), date(2015, 1, 1)])
def test_car_fee_always_150(start):
    d = evaluate(make(product="רכב", start=start))
    assert d.code == "CANCEL_ESTIMATE" and d.numbers["fee"] == 150
    assert any(f.source == "04_חריגים.md §א" for f in d.facts)


# --- cooling off ----------------------------------------------------------------------

@pytest.mark.parametrize("days,code", [(0, "COOLING_OFF"), (14, "COOLING_OFF"), (15, "CANCEL_ESTIMATE")])
@pytest.mark.parametrize("product", ["רכב", "דירה", "בריאות"])
def test_cooling_off_boundary(days, code, product):
    d = evaluate(make(product=product, start=AS_OF - timedelta(days=days)))
    assert d.code == code
    if code == "COOLING_OFF":
        assert d.numbers["fee"] == 0
        assert d.numbers["effective_date"] == AS_OF.isoformat()
        assert d.outcome is Outcome.ANSWER_WITH_GAPS
        assert any("הפרמיה ששולמה בפועל" in m for m in d.missing)
        assert "grant" not in d.numbers and "conditional_estimate" not in d.numbers


# --- effective date ---------------------------------------------------------------------

@pytest.mark.parametrize("start,effective", [
    (date(2023, 12, 31), date(2026, 11, 24)),
    (date(2024, 1, 1), date(2026, 10, 31)),
])
def test_health_legacy_cutoff(start, effective):
    d = evaluate(make(product="בריאות", start=start))
    assert d.numbers["effective_date"] == effective.isoformat()


def test_legacy_health_charges_30_days():
    d = evaluate(make(product="בריאות", start=date(2020, 1, 1)))
    assert any(f.source == "04_חריגים.md §ב" for f in d.facts)
    assert any("30 הימים הללו יחויבו" in s for s in d.say_to_customer)
    assert not any("חודש החיוב הוא החודש הקלנדרי" in a for a in d.assumptions)


def test_billing_month_assumption():
    d = evaluate(make())
    assert "חודש החיוב הוא החודש הקלנדרי" in d.assumptions
    assert any("חודש אוקטובר יחויב במלואו" in s for s in d.say_to_customer)


# --- health grant ------------------------------------------------------------------------

def test_health_never_writes_grant():
    d = evaluate(make(product="בריאות", start=date(2024, 6, 1), premium=200))
    assert "grant" not in d.numbers and "conditional_estimate" not in d.numbers
    assert d.numbers["potential_grant"] == 120
    assert any("תביעה ב-12 החודשים האחרונים" in m for m in d.missing)
    fact = next(f for f in d.facts if "פוטנציאלי" in f.label)
    assert "02_החזרים.md §2ג" in fact.source and "02_החזרים.md §3" in fact.source
    assert not any("120" in s for s in d.say_to_customer)


def test_health_zero_grant_writes_nothing():
    d = evaluate(make(product="בריאות", start=date(2026, 1, 1)))
    assert "grant" not in d.numbers and "potential_grant" not in d.numbers
    assert not any("תביעה ב-12" in m for m in d.missing)


def test_missing_premium_means_missing_grant():
    d = evaluate(make(premium=None))
    assert "grant" not in d.numbers and "conditional_estimate" not in d.numbers
    assert any("פרמיה חודשית" in m for m in d.missing)


# --- arrears and the conditional estimate -------------------------------------------------------

@pytest.mark.parametrize("days,arrears", [(60, False), (61, True)])
def test_arrears_boundary(days, arrears):
    d = evaluate(make(last_payment=AS_OF - timedelta(days=days)))
    assert d.numbers["arrears"] is arrears
    assert ("conditional_estimate" in d.numbers) is (not arrears)
    if arrears:
        assert any("סכום החוב" in m for m in d.missing)
        assert any("גבייה" in w for w in d.warnings)
        assert any("גבייה" in s for s in d.say_to_customer)


def test_unknown_last_payment():
    d = evaluate(make(last_payment=None))
    assert d.numbers["arrears"] is None
    assert "conditional_estimate" not in d.numbers
    assert any("תאריך התשלום האחרון" in m for m in d.missing)


def test_negative_estimate_is_missing_not_shown():
    d = evaluate(make(product="רכב", start=date(2026, 1, 1)))  # grant 0, fee 150
    assert d.numbers["grant"] == 0 and d.numbers["fee"] == 150
    assert "conditional_estimate" not in d.numbers
    assert any("הנוהל אינו מגדיר כיצד נגבה ההפרש" in m for m in d.missing)


def test_estimate_always_has_gaps():
    d = evaluate(make())
    assert d.outcome is Outcome.ANSWER_WITH_GAPS
    assert any("פרמיה ששולמה מראש" in m for m in d.missing)
    assert any("היסטוריית הקפאות" in m for m in d.missing)
    assert any("הוקפאה אי פעם" in w for w in d.warnings)


# --- customer script -----------------------------------------------------------------------

def test_freeze_condition_only_on_zero_fee_and_grant():
    d = evaluate(make(start=date(2020, 3, 10), premium=100))  # home, 15%: grant 180, fee 0
    script = d.say_to_customer
    assert "אין דמי ביטול, בתנאי שהפוליסה לא הוקפאה מעולם." in script
    assert "מענק ותק: 180 ₪, בתנאי שהפוליסה לא הוקפאה מעולם." in script
    assert ("החזר משוער: כ-180 ₪, בתנאי שהפוליסה לא הוקפאה מעולם, בתוספת כל פרמיה ששולמה מראש "
            "לתקופה שאחרי מועד הביטול.") in script
    for fee_line in ("דמי ביטול: 150 ₪.", "דמי ביטול: 50 ₪."):
        product, start = ("רכב", date(2020, 3, 10)) if "150" in fee_line else ("דירה", date(2026, 1, 1))
        assert fee_line in evaluate(make(product=product, start=start)).say_to_customer


def test_arrears_script_has_no_net_estimate():
    d = evaluate(make(last_payment=date(2026, 6, 1)))
    assert not any("החזר משוער" in s for s in d.say_to_customer)
    assert any("מענק ותק" in s and "לא הוקפאה" in s for s in d.say_to_customer)


def test_common_script_lines():
    d = evaluate(make())
    joined = " ".join(d.say_to_customer)
    for phrase in ("בקשה בכתב", "קישור לטופס", "הערכה בלבד", "מחלקת הכספים", "14 ימי עסקים"):
        assert phrase in joined
    sources = {f.source for f in d.facts}
    assert {"01_תנאי_ביטול.md §1", "02_החזרים.md §5", "02_החזרים.md §4"} <= sources


# --- multi-policy -----------------------------------------------------------------------------

def test_multi_policy_discount_warning():
    other = make(policy_id="7002", product="רכב")
    d = evaluate(make(), AS_OF, other)
    assert d.numbers["multi_policy"] is True
    assert any("04_חריגים.md §ג" == f.source for f in d.facts)
    assert any("ריבוי הפוליסות" in s for s in d.say_to_customer)


def test_cancelled_other_policy_does_not_count():
    other = make(policy_id="7002", status="מבוטלת")
    assert evaluate(make(), AS_OF, other).numbers["multi_policy"] is False


# --- stopping rules ----------------------------------------------------------------------------

def test_already_cancelled_shows_notes():
    d = evaluate(replace(make(status="מבוטלת", open_claims=3), notes="בוטלה 03/2026"))
    assert (d.code, d.outcome) == ("ALREADY_CANCELLED", Outcome.ANSWER)
    assert any(f.value == "בוטלה 03/2026" for f in d.facts)
    assert d.numbers == {}


def test_frozen_before_claims():
    d = evaluate(make(status="מוקפאת", open_claims=1))
    assert (d.code, d.outcome) == ("FROZEN", Outcome.ANSWER)
    assert any("גבייה" in s for s in d.say_to_customer)
    assert any("לא נגבית פרמיה ולא נצבר ותק" in s for s in d.say_to_customer)


@pytest.mark.parametrize("status", ["בהקפאה", "", "Active"])
def test_unknown_status(status):
    d = evaluate(make(status=status))
    assert (d.code, d.outcome) == ("UNKNOWN_STATUS", Outcome.CANNOT_ANSWER)


@pytest.mark.parametrize("claims,code", [(1, "OPEN_CLAIM"), (2, "OPEN_CLAIM"), (None, "UNKNOWN_CLAIMS")])
def test_claims_escalate_before_everything_else(claims, code):
    d = evaluate(make(product="אופנוע", start=None, open_claims=claims))
    assert (d.code, d.outcome) == (code, Outcome.ESCALATE)
    assert d.numbers == {}
    assert any("01_תנאי_ביטול.md §6" in w for w in d.warnings)
    assert d.say_to_customer == ["הבקשה מועברת למנהל המשמרת, ותתקבל תשובה תוך 2 ימי עסקים."]


def test_claims_helpers():
    assert has_open_or_unknown_claims(make(open_claims=None))
    assert has_open_or_unknown_claims(make(open_claims=1))
    assert not has_open_or_unknown_claims(make(open_claims=0))
    d = claims_escalation(make(open_claims=1), Intent.CLAIM_STATUS)
    assert d.intent is Intent.CLAIM_STATUS and d.code == "OPEN_CLAIM"


def test_unknown_product():
    assert evaluate(make(product="אופנוע")).code == "UNKNOWN_PRODUCT"


def test_missing_start_date():
    d = evaluate(make(start=None))
    assert (d.code, d.outcome) == ("MISSING_DATA", Outcome.CANNOT_ANSWER)


def test_not_started():
    d = evaluate(make(start=AS_OF + timedelta(days=1)))
    assert (d.code, d.outcome) == ("NOT_STARTED", Outcome.CANNOT_ANSWER)


@pytest.mark.parametrize("product", ["דירה", "בריאות"])
def test_date_overflow_is_out_of_range(product):
    d = evaluate(make(product=product, start=date(9998, 6, 1)), date(9999, 12, 31))
    assert (d.code, d.outcome) == ("DATE_OUT_OF_RANGE", Outcome.CANNOT_ANSWER)
    assert d.missing == [] and d.numbers == {}


def test_data_issues_become_warnings():
    d = evaluate(replace(make(), data_issues=["start_date: נורמל"]))
    assert d.facts[0].label == "סטטוס הפוליסה"
    assert any("start_date: נורמל" in w for w in d.warnings)


# --- travel -----------------------------------------------------------------------------------

@pytest.mark.parametrize("start", [AS_OF, AS_OF - timedelta(days=30)])
def test_trip_started_is_conditional(start):
    d = evaluate(make(product="נסיעות", start=start))
    assert (d.code, d.outcome) == ("TRIP_STARTED", Outcome.ANSWER_WITH_GAPS)
    assert any("תאריך היציאה בפועל" in m for m in d.missing)
    assert any(s.startswith("לפי הפוליסה") and "אם זה מועד היציאה" in s for s in d.say_to_customer)


def test_travel_day_before_trip():
    d = evaluate(make(product="נסיעות", start=AS_OF + timedelta(days=1)))
    assert d.code == "TRAVEL_BEFORE_TRIP"
    assert d.numbers["last_cancel_day"] == AS_OF.isoformat()
    assert any("הנהלים אינם מגדירים איזה מועד גובר" in m for m in d.missing)
    assert any("ייבדק" in s for s in d.say_to_customer)


def test_travel_after_month_end_has_no_conflict():
    d = evaluate(make(product="נסיעות", start=date(2026, 11, 1)))
    assert not any("איזה מועד גובר" in m for m in d.missing)
    assert d.numbers["fee"] == 50 and d.numbers["grant"] == 0
    assert any("תקופת ההתחרטות" in a and "החזר מלא וללא דמי ביטול" in a for a in d.assumptions)


def test_travel_arrears_referral():
    d = evaluate(make(product="נסיעות", start=date(2026, 12, 1), last_payment=date(2026, 7, 1)))
    assert d.numbers["arrears"] is True
    assert any("גבייה" in s for s in d.say_to_customer)


# --- real data, as_of 2026-10-25 ----------------------------------------------------------------

@pytest.fixture(scope="module")
def real():
    repo = ValidatedPolicyRepository(CsvPolicyRepository())
    return lambda pid: evaluate_cancellation(repo.get(pid), AS_OF, repo)


def test_4471(real):
    d = real("4471")
    assert d.code == "CANCEL_ESTIMATE"
    assert d.numbers["fee"] == 150
    assert d.numbers["grant"] == 492
    assert d.numbers["conditional_estimate"] == 342
    assert d.numbers["effective_date"] == "2026-10-31"


def test_4418(real):
    d = real("4418")
    assert d.numbers["conditional_estimate"] == 474
    assert d.numbers["multi_policy"] is True


def test_4408(real):
    d = real("4408")
    assert d.numbers["grant"] == 156 and d.numbers["conditional_estimate"] == 156


def test_4421(real):
    d = real("4421")
    assert d.numbers["grant"] == 288
    assert d.numbers["conditional_estimate"] == 138
    assert d.numbers["arrears"] is False


def test_4413(real):
    d = real("4413")
    assert d.numbers["grant"] == 630 and d.numbers["conditional_estimate"] == 480


def test_4412(real):
    d = real("4412")
    assert d.numbers["arrears"] is True
    assert d.numbers["grant"] == 87 and d.numbers["fee"] == 0
    assert "conditional_estimate" not in d.numbers


def test_4401(real):
    d = real("4401")
    assert d.numbers["effective_date"] == "2026-11-24"
    assert d.numbers["potential_grant"] == 288
    assert "grant" not in d.numbers


def test_4446(real):
    d = real("4446")
    assert d.numbers["effective_date"] == "2026-10-31"
    assert d.numbers["potential_grant"] == 96


def test_4423(real):
    d = real("4423")
    assert d.numbers["fee"] == 50 and "grant" not in d.numbers


def test_4419(real):
    d = real("4419")
    assert d.code == "TRAVEL_BEFORE_TRIP"
    assert d.numbers["last_cancel_day"] == "2026-12-21"
    assert d.numbers["fee"] == 50 and d.numbers["grant"] == 0
    assert d.numbers["effective_date"] == "2026-10-31"
    assert d.numbers["arrears"] is False


def test_4416(real):
    d = real("4416")
    assert d.code == "COOLING_OFF"
    assert d.numbers["fee"] == 0 and d.numbers["effective_date"] == "2026-10-25"


def test_4437(real):
    d = real("4437")
    assert (d.code, d.outcome) == ("TRIP_STARTED", Outcome.ANSWER_WITH_GAPS)


def test_4455(real):
    assert real("4455").code == "FROZEN"


def test_4440(real):
    assert real("4440").code == "ALREADY_CANCELLED"


@pytest.mark.parametrize("pid", ["4425", "4428"])
def test_open_claims_real(real, pid):
    d = real(pid)
    assert (d.outcome, d.code, d.numbers) == (Outcome.ESCALATE, "OPEN_CLAIM", {})
