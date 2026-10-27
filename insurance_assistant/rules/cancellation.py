"""Cancellation rules. Every amount, date and escalation is decided here, deterministically.

04_חריגים overrides 01 and 02. Every Fact cites "<file>.md §<section>" or, for data,
"<policy.source>, פוליסה <id>".
"""
from __future__ import annotations

import calendar
from datetime import date, timedelta

from ..models import Decision, Fact, Intent, Outcome, Policy
from ..repository import PolicyRepository
from .common import (
    ACTIVE, CANCEL, CANCELLED, CAR, EXCEPTIONS, FROZEN, HEALTH, PRODUCTS, REFUNDS, TRAVEL,
    data_source, format_money, start_decision,
)

COOLING_OFF_DAYS = 14
ARREARS_DAYS = 60  # arrears = MORE than 60 days since the last payment
CAR_FEE = 150
SHORT_TENURE_FEE = 50
HEALTH_LEGACY_CUTOFF = date(2024, 1, 1)  # health policies starting BEFORE it
HEALTH_LEGACY_NOTICE_DAYS = 30

NOT_FROZEN = ", בתנאי שהפוליסה לא הוקפאה מעולם"
HEBREW_MONTHS = ("ינואר", "פברואר", "מרץ", "אפריל", "מאי", "יוני", "יולי",
                 "אוגוסט", "ספטמבר", "אוקטובר", "נובמבר", "דצמבר")
TIER_ASSUMPTION = (
    "מדרגות מענק הוותק לפי תאריכים מדויקים: פחות משנה – 0%; משנה בדיוק ועד לפני 3 שנים – 5%; "
    "מ-3 שנים בדיוק ועד 5 שנים בדיוק – 10%; מעל 5 שנים – 15% (פרשנות של הטבלה ב-02_החזרים.md §2ג)"
)


# --- date helpers --------------------------------------------------------------

def add_years(d: date, n: int) -> date:
    """Same day n years later; Feb 29 -> Feb 28. OverflowError outside years 1-9999."""
    year = d.year + n
    if not 1 <= year <= 9999:
        raise OverflowError(f"year {year} is out of range")
    if d.month == 2 and d.day == 29 and not calendar.isleap(year):
        return d.replace(year=year, day=28)
    return d.replace(year=year)


def full_years_between(start: date, end: date) -> int:
    years = end.year - start.year
    if years > 0 and add_years(start, years) > end:
        years -= 1
    elif years < 0 and add_years(start, years) < end:
        years += 1
    return years


def end_of_month(d: date) -> date:
    return d.replace(day=calendar.monthrange(d.year, d.month)[1])


def grant_percent(start: date, as_of: date) -> int:
    if as_of < add_years(start, 1):
        return 0
    if as_of < add_years(start, 3):
        return 5
    if as_of <= add_years(start, 5):
        return 10
    return 15


def grant_amount(monthly_premium: float, percent: int) -> float:
    return round(monthly_premium * 12 * percent / 100, 2)


# --- formatting ----------------------------------------------------------------

def _day(d: date) -> str:
    return d.strftime("%d.%m.%Y")


# --- shared helpers ---------------------------------------------------------------

def has_open_or_unknown_claims(policy: Policy) -> bool:
    return policy.open_claims is None or policy.open_claims > 0


def claims_escalation(policy: Policy, intent: Intent = Intent.CANCELLATION) -> Decision:
    if policy.open_claims is None:
        decision = start_decision(policy, intent, Outcome.ESCALATE, "UNKNOWN_CLAIMS",
                          "לא ידוע אם יש תביעה פתוחה – יש להעביר למנהל משמרת")
        decision.facts.append(Fact("תביעות פתוחות", "לא ידוע", data_source(policy)))
        decision.missing.append("מספר התביעות הפתוחות – לא ניתן לשלול תביעה פתוחה")
    else:
        decision = start_decision(policy, intent, Outcome.ESCALATE, "OPEN_CLAIM",
                          "יש תביעה פתוחה – ביטול דורש אישור מנהל משמרת")
        decision.facts.append(Fact("תביעות פתוחות", str(policy.open_claims), data_source(policy)))
    decision.facts.append(Fact("ביטול פוליסה עם תביעה פתוחה דורש אישור מנהל משמרת",
                               "העברה למנהל משמרת", f"{CANCEL} §6"))
    decision.warnings.append(
        "אין לנקוב בסכומים, לאשר או לדחות את הבקשה – הנציג אינו מוסמך במקרה זה (01_תנאי_ביטול.md §6)"
    )
    decision.say_to_customer.append("הבקשה מועברת למנהל המשמרת, ותתקבל תשובה תוך 2 ימי עסקים.")
    return decision


def _check_arrears(policy: Policy, as_of: date, decision: Decision) -> bool | None:
    if policy.last_payment_date is None:
        decision.numbers["arrears"] = None
        decision.missing.append(f"תאריך התשלום האחרון – לא ניתן לבדוק פיגור בתשלומים ({CANCEL} §5)")
        return None
    decision.facts.append(Fact("תשלום אחרון", policy.last_payment_date.isoformat(), data_source(policy)))
    if policy.last_payment_date > as_of:
        # Impossible data: a payment after the calculation date. Arrears unknown, so no net estimate.
        decision.numbers["arrears"] = None
        decision.warnings.append(
            f"בעיה בנתוני הפוליסה: last_payment_date: {policy.last_payment_date.isoformat()} מאוחר מתאריך "
            f"החישוב {as_of.isoformat()}"
        )
        decision.missing.append(f"תאריך תשלום אחרון תקין – לא ניתן לבדוק פיגור בתשלומים ({CANCEL} §5)")
        return None
    days = (as_of - policy.last_payment_date).days
    arrears = days > ARREARS_DAYS
    decision.numbers["arrears"] = arrears
    if arrears:
        decision.facts.append(Fact("פיגור בתשלומים", f"{days} ימים מהתשלום האחרון (מעל {ARREARS_DAYS})",
                                   f"{CANCEL} §5"))
        decision.warnings.append(
            f"הלקוח בפיגור ({days} ימים מהתשלום האחרון): החוב שנצבר מקוזז מכל החזר, ואם החוב גדול "
            f"מההחזר – הפניה למחלקת גבייה להסדר ({CANCEL} §5)"
        )
        decision.missing.append(f"סכום החוב שנצבר ({CANCEL} §5)")
    return arrears


def _common_block(policy: Policy, repo: PolicyRepository, decision: Decision) -> None:
    others = None
    if policy.customer_id:
        others = [p for p in repo.by_customer(policy.customer_id)
                  if p.policy_id != policy.policy_id and p.status != CANCELLED]
    if others is None:
        decision.numbers["multi_policy"] = None
        decision.missing.append("מספר לקוח – לא ניתן לבדוק אם ללקוח פוליסות נוספות (04_חריגים.md §ג)")
    else:
        decision.numbers["multi_policy"] = bool(others)
    if others:
        ids = ", ".join(p.policy_id for p in others)
        decision.facts.append(Fact("פוליסות נוספות של הלקוח שאינן מבוטלות", ids,
                                   f"{policy.source}, לקוח {policy.customer_id}"))
        decision.facts.append(Fact("ביטול פוליסה מבטל את הנחת ריבוי הפוליסות (5%) על הנותרות",
                                   "מחודש החיוב הבא", f"{EXCEPTIONS} §ג"))
        decision.warnings.append(
            f"ללקוח פוליסות נוספות ({ids}): יש ליידע אותו לפני הביטול שהנחת ריבוי הפוליסות תבוטל "
            f"({EXCEPTIONS} §ג)"
        )
        decision.say_to_customer.append(
            "ביטול הפוליסה יבטל את הנחת ריבוי הפוליסות (5%) על הפוליסות הנותרות, החל מחודש החיוב הבא."
        )
    decision.facts += [
        Fact("ביטול מחייב בקשה בכתב", "בקשה טלפונית אינה מספיקה – נשלח קישור לטופס", f"{CANCEL} §1"),
        Fact("הסכום הסופי", "נקבע ומאושר על ידי מחלקת הכספים", f"{REFUNDS} §5"),
        Fact("מועד ההחזר", "תוך 14 ימי עסקים ממועד כניסת הביטול לתוקף, לאמצעי התשלום המקורי",
             f"{REFUNDS} §4"),
    ]
    decision.say_to_customer += [
        "הביטול מחייב בקשה בכתב – נשלח קישור לטופס.",
        "כל הסכומים שנמסרו הם הערכה בלבד; הסכום הסופי ייקבע על ידי מחלקת הכספים.",
        "החזר, אם מגיע, יועבר תוך 14 ימי עסקים ממועד כניסת הביטול לתוקף, לאמצעי התשלום שממנו שולמה הפרמיה.",
    ]


def _request_date_assumption(as_of: date) -> str:
    return f"מועד קבלת הבקשה בכתב נחשב כ-{as_of.isoformat()} (תאריך החישוב)"


def _finish(decision: Decision) -> Decision:
    if decision.outcome in (Outcome.ANSWER, Outcome.ANSWER_WITH_GAPS):
        decision.outcome = Outcome.ANSWER_WITH_GAPS if decision.missing else Outcome.ANSWER
    return decision


# --- evaluation ----------------------------------------------------------------------

def evaluate_cancellation(policy: Policy, as_of: date, repo: PolicyRepository) -> Decision:
    try:
        return _evaluate(policy, as_of, repo)
    except OverflowError:
        decision = start_decision(policy, Intent.CANCELLATION, Outcome.CANNOT_ANSWER, "DATE_OUT_OF_RANGE",
                          "תאריכי הפוליסה מחוץ לטווח שניתן לחשב")
        decision.warnings.append("חישוב מועדי הביטול חרג מטווח התאריכים הנתמך (שנים 1–9999) – יש לבדוק את התאריכים במערכת")
        return decision


def _evaluate(policy: Policy, as_of: date, repo: PolicyRepository) -> Decision:
    def start(outcome: Outcome, code: str, headline: str) -> Decision:
        return start_decision(policy, Intent.CANCELLATION, outcome, code, headline)

    if policy.status == CANCELLED:
        d = start(Outcome.ANSWER, "ALREADY_CANCELLED", "הפוליסה כבר מבוטלת")
        if policy.notes:
            d.facts.append(Fact("הערות", policy.notes, data_source(policy)))
        d.say_to_customer.append("הפוליסה כבר מבוטלת.")
        return d

    if policy.status == FROZEN:
        d = start(Outcome.ANSWER, "FROZEN", "פוליסה מוקפאת אינה ניתנת לביטול")
        if policy.notes:
            d.facts.append(Fact("הערות", policy.notes, data_source(policy)))
        d.facts.append(Fact("פוליסה מוקפאת אינה ניתנת לביטול כל עוד היא מוקפאת",
                            "יש לבקש הפשרה דרך מחלקת גבייה, ורק לאחר מכן להגיש בקשת ביטול",
                            f"{EXCEPTIONS} §ד"))
        d.facts.append(Fact("במצב הקפאה", "לא נגבית פרמיה ולא נצבר ותק", f"{EXCEPTIONS} §ד"))
        d.say_to_customer += [
            "הפוליסה מוקפאת, ולכן לא ניתן לבטל אותה כל עוד היא מוקפאת.",
            "יש לבקש תחילה הפשרה דרך מחלקת גבייה, ורק לאחר מכן להגיש בקשת ביטול.",
            "בזמן ההקפאה לא נגבית פרמיה ולא נצבר ותק.",
        ]
        return d

    if policy.status != ACTIVE:
        d = start(Outcome.CANNOT_ANSWER, "UNKNOWN_STATUS", "סטטוס הפוליסה אינו מוכר")
        d.missing.append(f"סטטוס פוליסה מוכר (פעילה, מבוטלת או מוקפאת); התקבל: '{policy.status}'")
        return d

    if has_open_or_unknown_claims(policy):
        return claims_escalation(policy)

    if policy.product not in PRODUCTS:
        d = start(Outcome.CANNOT_ANSWER, "UNKNOWN_PRODUCT", "סוג המוצר אינו מוכר")
        d.missing.append(f"סוג מוצר מוכר ({', '.join(PRODUCTS)}); התקבל: '{policy.product}'")
        return d

    if policy.start_date is None:
        d = start(Outcome.CANNOT_ANSWER, "MISSING_DATA", "חסר תאריך תחילת פוליסה")
        d.missing.append("תאריך תחילת הפוליסה – בלעדיו לא ניתן לקבוע ותק, דמי ביטול ומועד ביטול")
        return d

    if policy.product == TRAVEL:
        return _travel(policy, as_of, repo, start)

    if as_of < policy.start_date:
        d = start(Outcome.CANNOT_ANSWER, "NOT_STARTED", "הפוליסה עדיין לא התחילה")
        d.facts.append(Fact("תאריך תחילת הפוליסה", policy.start_date.isoformat(), data_source(policy)))
        d.missing.append("הנהלים אינם מגדירים ביטול של פוליסה (שאינה נסיעות) לפני מועד תחילתה")
        return d

    if (as_of - policy.start_date).days <= COOLING_OFF_DAYS:
        return _cooling_off(policy, as_of, repo, start)

    return _estimate(policy, as_of, repo, start)


def _travel(policy: Policy, as_of: date, repo: PolicyRepository, start) -> Decision:
    trip = policy.start_date
    departure_missing = (
        "תאריך היציאה בפועל לנסיעה – תאריך תחילת הפוליסה אינו בהכרח מועד היציאה (01_תנאי_ביטול.md §4)"
    )
    if as_of >= trip:
        d = start(Outcome.ANSWER_WITH_GAPS, "TRIP_STARTED", "לפי תאריך תחילת הפוליסה, ייתכן שהנסיעה כבר החלה")
        d.facts.append(Fact("תאריך תחילת הכיסוי", trip.isoformat(), data_source(policy)))
        d.facts.append(Fact("פוליסת נסיעות", "לא ניתנת לביטול מרגע שהנסיעה החלה, ולא יינתן החזר",
                            f"{CANCEL} §4"))
        d.missing.append(departure_missing)
        d.say_to_customer.append(
            f"לפי הפוליסה, הכיסוי התחיל ב-{_day(trip)}. אם זה מועד היציאה לנסיעה, הנסיעה כבר החלה – "
            "ולכן לא ניתן לבטל את הפוליסה ולא יינתן החזר."
        )
        return _finish(d)

    d = start(Outcome.ANSWER_WITH_GAPS, "TRAVEL_BEFORE_TRIP", "ביטול פוליסת נסיעות לפני הנסיעה")
    last_cancel_day = trip - timedelta(days=1)
    effective = end_of_month(as_of)
    d.numbers["last_cancel_day"] = last_cancel_day.isoformat()
    d.numbers["effective_date"] = effective.isoformat()
    d.facts += [
        Fact("תאריך תחילת הכיסוי", trip.isoformat(), data_source(policy)),
        Fact("היום האחרון לביטול (אם תחילת הכיסוי היא מועד היציאה)", last_cancel_day.isoformat(),
             f"{CANCEL} §4"),
        Fact("מועד כניסת הביטול לתוקף", f"{effective.isoformat()} (סוף חודש החיוב)", f"{CANCEL} §1"),
    ]
    d.missing.append(departure_missing)
    d.assumptions += [_request_date_assumption(as_of), "חודש החיוב הוא החודש הקלנדרי"]
    d.assumptions.append(
        f"תקופת ההתחרטות של 14 יום ({CANCEL} §2) הונחה כלא חלה לפני תחילת הכיסוי; אם היא חלה – "
        "הביטול הוא בהחזר מלא וללא דמי ביטול"
    )
    d.say_to_customer.append(
        f"פוליסת נסיעות ניתנת לביטול עד יום לפני תחילת הנסיעה. לפי הפוליסה הכיסוי מתחיל ב-{_day(trip)}; "
        f"אם זה מועד היציאה, היום האחרון לביטול הוא {_day(last_cancel_day)}."
    )
    month = HEBREW_MONTHS[as_of.month - 1]
    if trip <= effective:
        d.missing.append(
            f"תחילת הכיסוי ({trip.isoformat()}) חלה לפני סוף חודש החיוב ({effective.isoformat()}) – "
            "הנהלים אינם מגדירים איזה מועד גובר לגבי כניסת הביטול לתוקף והחיוב"
        )
        d.say_to_customer.append(
            f"לפי נוהל הביטול הכללי הביטול נכנס לתוקף בסוף חודש החיוב ({_day(effective)}) וחודש {month} "
            f"מחויב במלואו, אך הנסיעה מתחילה לפני כן – המועד המדויק ייבדק ויימסר בנפרד."
        )
    else:
        d.say_to_customer.append(
            f"הביטול ייכנס לתוקף ב-{_day(effective)} (סוף חודש החיוב), וחודש {month} יחויב במלואו."
        )

    fee = SHORT_TENURE_FEE if as_of < add_years(trip, 1) else 0
    d.numbers["fee"] = fee
    d.numbers["grant"] = 0
    d.facts.append(Fact("דמי ביטול", f"{fee} ₪ (ותק של פחות מ-12 חודשים)", f"{CANCEL} §3"))
    d.facts.append(Fact("מענק ותק", "אין (ותק של פחות משנה)", f"{REFUNDS} §2ג"))
    d.missing.append(f"פרמיה ששולמה מראש לתקופה שאחרי מועד הביטול ({REFUNDS} §2א) – לא קיימת בנתונים")
    d.say_to_customer.append(f"דמי ביטול: {fee} ₪.")
    arrears = _check_arrears(policy, as_of, d)
    d.say_to_customer.append(
        "ההחזר יחושב כפרמיה ששולמה מראש לתקופה שאחרי מועד הביטול, בניכוי דמי הביטול ובניכוי חוב אם קיים, "
        "בתוספת מענק ותק אם מגיע."
    )
    if arrears:
        d.say_to_customer.append(
            "קיים פיגור בתשלומים: החוב שנצבר יקוזז מההחזר, ואם החוב גדול מההחזר – תתבצע הפניה למחלקת גבייה להסדר."
        )
    _common_block(policy, repo, d)
    return _finish(d)


def _cooling_off(policy: Policy, as_of: date, repo: PolicyRepository, start) -> Decision:
    d = start(Outcome.ANSWER_WITH_GAPS, "COOLING_OFF", "ביטול בתקופת ההתחרטות – החזר מלא")
    days = (as_of - policy.start_date).days
    d.numbers["fee"] = 0
    d.numbers["effective_date"] = as_of.isoformat()
    d.facts += [
        Fact("תאריך תחילת הפוליסה", policy.start_date.isoformat(), data_source(policy)),
        Fact("תקופת התחרטות", f"{days} ימים מתחילת הפוליסה (עד {COOLING_OFF_DAYS})", f"{CANCEL} §2"),
        Fact("דמי ביטול", "אין – גם בפוליסת רכב, דמי 150 ₪ אינם חלים בתקופת ההתחרטות",
             f"{CANCEL} §2, {EXCEPTIONS} §א"),
        Fact("מועד כניסת הביטול לתוקף", f"{as_of.isoformat()} (מיידית)", f"{CANCEL} §2"),
        Fact("החזר", "החזר מלא של הפרמיה ששולמה", f"{CANCEL} §2"),
    ]
    d.missing.append("הפרמיה ששולמה בפועל – קובעת את סכום ההחזר המלא")
    d.assumptions += [
        _request_date_assumption(as_of),
        f"'בתוך 14 ימים' כולל את היום ה-{COOLING_OFF_DAYS} מתחילת הפוליסה",
    ]
    d.say_to_customer += [
        "הבקשה בתוך תקופת ההתחרטות: הביטול נכנס לתוקף מיידית, ללא דמי ביטול, עם החזר מלא של הפרמיה ששולמה.",
    ]
    _common_block(policy, repo, d)
    return _finish(d)


def _estimate(policy: Policy, as_of: date, repo: PolicyRepository, start) -> Decision:
    d = start(Outcome.ANSWER_WITH_GAPS, "CANCEL_ESTIMATE", "הערכת ביטול")
    begin = policy.start_date
    src = data_source(policy)
    d.facts += [
        Fact("מוצר", policy.product, src),
        Fact("תאריך תחילת הפוליסה", begin.isoformat(), src),
    ]
    d.assumptions.append(_request_date_assumption(as_of))

    # Effective date
    month = HEBREW_MONTHS[as_of.month - 1]
    if policy.product == HEALTH and begin < HEALTH_LEGACY_CUTOFF:
        effective = as_of + timedelta(days=HEALTH_LEGACY_NOTICE_DAYS)
        d.facts.append(Fact("מועד כניסת הביטול לתוקף",
                            f"{effective.isoformat()} – 30 יום ממועד קבלת הבקשה (פוליסת בריאות מלפני 1.1.2024)",
                            f"{EXCEPTIONS} §ב"))
        d.say_to_customer.append(
            f"הביטול ייכנס לתוקף ב-{_day(effective)}, 30 יום ממועד קבלת הבקשה, ו-30 הימים הללו יחויבו."
        )
    else:
        effective = end_of_month(as_of)
        d.facts.append(Fact("מועד כניסת הביטול לתוקף", f"{effective.isoformat()} (סוף חודש החיוב)",
                            f"{CANCEL} §1"))
        d.assumptions.append("חודש החיוב הוא החודש הקלנדרי")
        d.say_to_customer.append(
            f"הביטול ייכנס לתוקף ב-{_day(effective)} (סוף חודש החיוב), וחודש {month} יחויב במלואו."
        )
    d.numbers["effective_date"] = effective.isoformat()

    # Fee
    if policy.product == CAR:
        fee = CAR_FEE
        d.facts.append(Fact("דמי ביטול", f"{CAR_FEE} ₪ בכל מקרה (פוליסת רכב)", f"{EXCEPTIONS} §א"))
        d.say_to_customer.append(f"דמי ביטול: {CAR_FEE} ₪.")
    elif as_of < add_years(begin, 1):
        fee = SHORT_TENURE_FEE
        d.facts.append(Fact("דמי ביטול", f"{SHORT_TENURE_FEE} ₪ (ותק של פחות מ-12 חודשים)", f"{CANCEL} §3"))
        d.say_to_customer.append(f"דמי ביטול: {SHORT_TENURE_FEE} ₪.")
    else:
        fee = 0
        d.facts.append(Fact("דמי ביטול", "אין (ותק של 12 חודשים ומעלה)", f"{CANCEL} §3"))
        d.say_to_customer.append(f"אין דמי ביטול{NOT_FROZEN}.")
    d.numbers["fee"] = fee

    # Seniority grant
    grant = None
    if policy.monthly_premium is None:
        d.missing.append(f"פרמיה חודשית – לא ניתן לחשב את מענק הוותק ({REFUNDS} §2ג)")
    else:
        percent = grant_percent(begin, as_of)
        amount = grant_amount(policy.monthly_premium, percent)
        years = full_years_between(begin, as_of)
        d.assumptions.append(TIER_ASSUMPTION)
        d.facts.append(Fact("פרמיה חודשית", f"{format_money(policy.monthly_premium)} ₪", src))
        if policy.product == HEALTH:
            if amount > 0:
                d.numbers["potential_grant"] = amount
                d.facts.append(Fact(
                    "מענק ותק פוטנציאלי (לנציג בלבד)",
                    f"{format_money(amount)} ₪ ({percent}% × פרמיה שנתית, ותק {years} שנים) – רק אם לא שולמה "
                    "ללקוח תביעה ב-12 החודשים האחרונים",
                    f"{REFUNDS} §2ג, {REFUNDS} §3"))
                d.missing.append(
                    f"האם שולמה ללקוח תביעה ב-12 החודשים האחרונים ({REFUNDS} §3) – קובע אם מגיע מענק ותק"
                )
        else:
            grant = amount
            d.numbers["grant"] = amount
            d.facts.append(Fact("מענק ותק", f"{format_money(amount)} ₪ ({percent}% × פרמיה שנתית, ותק {years} שנים)",
                                f"{REFUNDS} §2ג"))

    arrears = _check_arrears(policy, as_of, d)

    d.missing += [
        f"פרמיה ששולמה מראש לתקופה שאחרי מועד הביטול ({REFUNDS} §2א) – לא קיימת בנתונים",
        f"היסטוריית הקפאות ({EXCEPTIONS} §ד) – לא קיימת בנתונים",
    ]
    d.warnings.append(
        "יש לשאול את הלקוח אם הפוליסה הוקפאה אי פעם: הוותק, דמי הביטול ומענק הוותק מניחים שלא הייתה הקפאה "
        f"({EXCEPTIONS} §ד)"
    )

    if grant is not None:
        if grant > 0:
            d.say_to_customer.append(f"מענק ותק: {format_money(grant)} ₪{NOT_FROZEN}.")
        if arrears is False:
            estimate = round(grant - fee, 2)
            if estimate >= 0:
                d.numbers["conditional_estimate"] = estimate
                d.facts.append(Fact("החזר משוער מותנה", f"{format_money(estimate)} ₪ (מענק ותק פחות דמי ביטול), "
                                    "בתוספת פרמיה ששולמה מראש", f"{REFUNDS} §2"))
                d.say_to_customer.append(
                    f"החזר משוער: כ-{format_money(estimate)} ₪{NOT_FROZEN}, בתוספת כל פרמיה ששולמה מראש לתקופה "
                    "שאחרי מועד הביטול."
                )
            else:
                d.missing.append(
                    f"דמי הביטול ({format_money(fee)} ₪) גבוהים ממענק הוותק ({format_money(grant)} ₪) – הנוהל אינו מגדיר "
                    "כיצד נגבה ההפרש"
                )
    if arrears:
        d.say_to_customer.append(
            "קיים פיגור בתשלומים: החוב שנצבר יקוזז מכל החזר, ואם החוב גדול מההחזר – תתבצע הפניה למחלקת גבייה להסדר."
        )

    _common_block(policy, repo, d)
    return _finish(d)
