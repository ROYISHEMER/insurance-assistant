"""Claim, refund-timing and policy-info answers. Deterministic; every Fact cites its source."""
from __future__ import annotations

from ..models import Decision, Fact, Intent, Outcome, Policy
from .common import CLAIMS, MISSING_VALUE, PRODUCTS, REFUNDS, data_source, format_money, start_decision

NO_NOTES = "אין הערות"

CLAIM_STAGES = (
    ("שלב 1 – אישור קבלה", "נשלח ללקוח תוך 3 ימי עסקים מהגשת התביעה"),
    ("שלב 2 – מינוי מעריך", "תוך 10 ימי עסקים; המעריך יוצר קשר עם הלקוח במידת הצורך"),
    ("שלב 3 – החלטה", "אישור מלא, אישור חלקי או דחייה; נשלחת ללקוח בכתב עם נימוק – אין לוח זמנים בנוהל"),
    ("שלב 4 – תשלום", "הסכום המאושר מועבר לאחר אישור התביעה – אין לוח זמנים בנוהל"),
)
DOCUMENTS = {
    "רכב": "דוח משטרה (אם רלוונטי), הצעת מחיר מוסך, צילומי הנזק",
    "דירה": "צילומי הנזק, קבלות על רכוש שניזוק, דוח שמאי אם הנזק מעל 10,000 ₪",
    "בריאות": "מסמכים רפואיים, קבלות, הפניית רופא",
    "נסיעות": "כרטיסי טיסה, אישורים רפואיים מחו\"ל, קבלות",
}
CLAIM_WARNING = (
    f"אין לנקוב בסכומים או להעריך סיכויי אישור, ואין למסור את המלצת המעריך לפני שההחלטה נשלחה בכתב "
    f"({CLAIMS} §3, §6)"
)
POLICY_FIELDS = (
    ("policy_id", "מספר פוליסה"),
    ("customer_id", "מספר לקוח"),
    ("customer_name", "שם הלקוח"),
    ("product", "מוצר"),
    ("start_date", "תאריך תחילה"),
    ("monthly_premium", "פרמיה חודשית"),
    ("status", "סטטוס"),
    ("last_payment_date", "תשלום אחרון"),
    ("open_claims", "תביעות פתוחות"),
    ("notes", "הערות"),
)


def _claim_count_unknown(policy: Policy, intent: Intent) -> Decision:
    d = start_decision(policy, intent, Outcome.CANNOT_ANSWER, "CLAIM_COUNT_UNKNOWN",
               "לא ידוע אם לפוליסה יש תביעה פתוחה")
    d.facts.append(Fact("תביעות פתוחות", "לא ידוע", data_source(policy)))
    d.missing.append("מספר התביעות הפתוחות בפוליסה")
    return d


def _no_open_claim(policy: Policy, intent: Intent) -> Decision:
    d = start_decision(policy, intent, Outcome.ANSWER, "NO_OPEN_CLAIM", "אין לפוליסה תביעה פתוחה")
    d.facts.append(Fact("תביעות פתוחות", "0", data_source(policy)))
    d.say_to_customer.append("לפי הנתונים אין בפוליסה תביעה פתוחה.")
    return d


def claim_payment_date(policy: Policy) -> Decision:
    intent = Intent.CLAIM_PAYMENT_DATE
    if policy.open_claims is None:
        return _claim_count_unknown(policy, intent)
    if policy.open_claims == 0:
        return _no_open_claim(policy, intent)
    d = start_decision(policy, intent, Outcome.CANNOT_ANSWER, "CLAIM_PAYMENT_UNKNOWN",
               "לא ניתן לקבוע מועד תשלום לתביעה")
    d.facts.append(Fact("תביעות פתוחות", str(policy.open_claims), data_source(policy)))
    d.facts += [Fact(label, value, f"{CLAIMS} §2") for label, value in CLAIM_STAGES]
    d.missing += [
        "השלב הנוכחי של התביעה – אינו קיים בנתוני הפיילוט (הנציג רואה אותו במערכת התביעות)",
        f"לוח זמנים להחלטה ולתשלום – הנוהל אינו מגדיר ({CLAIMS} §2)",
    ]
    d.warnings.append(CLAIM_WARNING)
    d.say_to_customer += [
        "אין לנו מועד תשלום מוגדר: הנוהל קובע אישור קבלה תוך 3 ימי עסקים ומינוי מעריך תוך 10 ימי עסקים.",
        "לאחר מכן מתקבלת החלטה שנשלחת אליך בכתב עם נימוק, והתשלום מועבר לאחר אישור התביעה.",
    ]
    return d


def claim_status(policy: Policy) -> Decision:
    intent = Intent.CLAIM_STATUS
    if policy.open_claims is None:
        return _claim_count_unknown(policy, intent)
    if policy.open_claims == 0:
        return _no_open_claim(policy, intent)
    d = start_decision(policy, intent, Outcome.ANSWER_WITH_GAPS, "CLAIM_OPEN_STAGE_UNKNOWN",
               "יש תביעה פתוחה; שלב הטיפול אינו בנתוני הפיילוט")
    d.facts.append(Fact("תביעות פתוחות", str(policy.open_claims), data_source(policy)))
    d.facts.append(Fact("שלב הטיפול הנוכחי", "הנציג יכול לראות אותו במערכת התביעות", f"{CLAIMS} §3"))
    d.missing.append("השלב הנוכחי של התביעה – אינו קיים בנתוני הפיילוט; יש לבדוק במערכת התביעות")
    d.warnings.append(CLAIM_WARNING)
    d.say_to_customer.append("יש בפוליסה תביעה פתוחה; אבדוק עבורך באיזה שלב הטיפול נמצא.")
    return d


def claim_documents(product: str | None, policy_id: str | None = None) -> Decision:
    intent = Intent.CLAIM_DOCUMENTS
    if product not in PRODUCTS:
        d = Decision(Outcome.CANNOT_ANSWER, intent, "NO_PRODUCT", "לא ידוע סוג המוצר", policy_id=policy_id)
        d.missing.append(f"סוג המוצר ({', '.join(PRODUCTS)}) – המסמכים הנדרשים תלויים בו")
        return d
    d = Decision(Outcome.ANSWER, intent, "CLAIM_DOCS", f"מסמכים להגשת תביעה – {product}", policy_id=policy_id)
    d.facts += [
        Fact("ערוצי הגשה", "האתר, האפליקציה או מייל למחלקת התביעות", f"{CLAIMS} §1"),
        Fact("פרטי חובה בתביעה", "מספר פוליסה, תיאור האירוע ומועדו, מסמכים תומכים", f"{CLAIMS} §1"),
        Fact(f"מסמכים נדרשים – {product}", DOCUMENTS[product], f"{CLAIMS} §4"),
    ]
    d.say_to_customer += [
        "ניתן להגיש את התביעה דרך האתר, האפליקציה או במייל למחלקת התביעות.",
        "יש לכלול מספר פוליסה, תיאור האירוע ומועדו, ומסמכים תומכים.",
        f"המסמכים הנדרשים: {DOCUMENTS[product]}.",
    ]
    return d


def refund_timing(policy_id: str | None = None) -> Decision:
    d = Decision(Outcome.ANSWER, Intent.REFUND_TIMING, "REFUND_TIMING", "מועד העברת החזר", policy_id=policy_id)
    d.facts += [
        Fact("מועד ההחזר", "תוך 14 ימי עסקים ממועד כניסת הביטול לתוקף", f"{REFUNDS} §4"),
        Fact("אמצעי התשלום", "לאמצעי התשלום שממנו שולמה הפרמיה", f"{REFUNDS} §4"),
        Fact("אמצעי תשלום אחר", "רק לאחר אימות זהות במחלקת הכספים", f"{REFUNDS} §4"),
    ]
    d.say_to_customer += [
        "ההחזר מועבר תוך 14 ימי עסקים ממועד כניסת הביטול לתוקף, לאמצעי התשלום שממנו שולמה הפרמיה.",
        "העברה לאמצעי תשלום אחר אפשרית רק לאחר אימות זהות במחלקת הכספים.",
    ]
    return d


def _shown(name: str, value: object) -> str:
    if value is None or value == "":
        return NO_NOTES if name == "notes" else MISSING_VALUE  # notes are optional, not missing data
    if hasattr(value, "isoformat"):
        return value.isoformat()
    if isinstance(value, float):
        return f"{format_money(value)} ₪"
    return str(value)


def policy_info(policy: Policy) -> Decision:
    d = Decision(Outcome.ANSWER, Intent.POLICY_INFO, "POLICY_INFO", f"פרטי פוליסה {policy.policy_id}",
                 policy_id=policy.policy_id)
    source = data_source(policy)
    d.facts += [Fact(label, _shown(name, getattr(policy, name)), source) for name, label in POLICY_FIELDS]
    d.warnings += [f"בעיה בנתוני הפוליסה: {issue}" for issue in policy.data_issues]
    return d
