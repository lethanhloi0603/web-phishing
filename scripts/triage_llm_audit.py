import argparse
import re
import sys
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.preprocessing import safe_soup


PHISHING_TERMS = {
    "account", "authenticate", "confirm", "credential", "login", "password",
    "payment", "security", "sign in", "signin", "suspended", "unlock", "update",
    "verification", "verify", "wallet",
}
URGENCY_TERMS = {
    "act now", "immediately", "required", "urgent", "within 24", "expired",
    "suspended", "limited time",
}
SENSITIVE_TERMS = {
    "card", "cvv", "credit", "dob", "email", "otp", "passcode", "password",
    "pin", "social security", "ssn", "username",
}


def hostname(value):
    value = str(value or "").strip()
    if value.startswith("//"):
        value = "https:" + value
    try:
        return (urlparse(value).hostname or "").lower()
    except ValueError:
        return ""


def text_value(tag):
    fields = [tag.get_text(" ", strip=True)]
    for key in ("type", "name", "id", "placeholder", "autocomplete", "aria-label"):
        value = tag.attrs.get(key)
        if isinstance(value, str):
            fields.append(value)
    return " ".join(fields).lower()


def triage_html(raw_html, page_url=""):
    raw_html = str(raw_html or "")
    soup = safe_soup(raw_html)
    visible = re.sub(r"\s+", " ", soup.get_text(" ", strip=True)).strip()
    visible_lower = visible.lower()
    title = soup.title.get_text(" ", strip=True) if soup.title else ""
    forms = soup.find_all("form")
    inputs = soup.find_all("input")
    buttons = soup.find_all("button")
    page_host = hostname(page_url)

    score = 0
    reasons = []
    password_fields = [tag for tag in inputs if str(tag.attrs.get("type", "")).lower() == "password"]
    if password_fields:
        score += 3
        reasons.append(f"password_field={len(password_fields)}")
    if forms:
        score += 1
        reasons.append(f"form={len(forms)}")

    sensitive_inputs = [
        tag for tag in inputs if any(term in text_value(tag) for term in SENSITIVE_TERMS)
    ]
    if sensitive_inputs:
        score += 2 if len(sensitive_inputs) >= 2 else 1
        reasons.append(f"sensitive_input={len(sensitive_inputs)}")

    absolute_actions = 0
    external_actions = 0
    for form in forms:
        action_host = hostname(form.attrs.get("action", ""))
        if action_host:
            absolute_actions += 1
            if page_host and action_host != page_host:
                external_actions += 1
    if external_actions:
        score += 3
        reasons.append(f"external_form_action={external_actions}")
    elif absolute_actions:
        score += 1
        reasons.append(f"absolute_form_action={absolute_actions}")

    phishing_hits = sorted(term for term in PHISHING_TERMS if term in visible_lower)
    if phishing_hits:
        score += 2 if len(phishing_hits) >= 3 else 1
        reasons.append("phishing_terms=" + "|".join(phishing_hits[:6]))

    urgency_hits = sorted(term for term in URGENCY_TERMS if term in visible_lower)
    if urgency_hits:
        score += 1
        reasons.append("urgency=" + "|".join(urgency_hits[:4]))

    raw_lower = raw_html.lower()
    collection_indicators = sum(
        raw_lower.count(term)
        for term in ("fetch(", "xmlhttprequest", "formdata(", "localstorage", "sendbeacon")
    )
    if collection_indicators:
        score += 1
        reasons.append(f"script_collection_indicator={collection_indicators}")

    complete_interaction = bool(
        forms and inputs and (buttons or soup.find("input", {"type": "submit"}))
    )
    if complete_interaction:
        score += 1
        reasons.append("complete_interaction_flow")
    if not forms and not inputs:
        score -= 2
        reasons.append("no_form_or_input")
    if len(visible) < 30:
        score -= 1
        reasons.append("very_little_visible_text")

    score = max(0, score)
    if score >= 7:
        level, recommendation = "high", "review_first"
    elif score >= 4:
        level, recommendation = "medium", "review"
    else:
        level, recommendation = "low", "sample_check"
    return {
        "triage_score": score,
        "triage_level": level,
        "triage_recommendation": recommendation,
        "triage_reasons": "; ".join(reasons) if reasons else "no_strong_signal",
        "page_title": title[:200],
        "visible_text_preview": visible[:350],
        "form_count": len(forms),
        "input_count": len(inputs),
        "password_count": len(password_fields),
    }


def automated_screening_decision(raw_html, triage):
    """Apply deterministic screening rules without creating a manual label."""
    visible = safe_soup(str(raw_html or "")).get_text(" ", strip=True).lower()
    disclaimer_terms = (
        "educational example", "educational security demonstration", "for learning purposes",
        "not for production", "course materials", "demo credentials", "demo accounts",
    )
    disclaimer_hits = [term for term in disclaimer_terms if term in visible]
    if disclaimer_hits:
        return "rejected", 0, "explicit_demo_or_educational_disclaimer=" + "|".join(disclaimer_hits)

    credential_page = (
        triage["password_count"] >= 1
        and triage["input_count"] >= 2
        and (
            triage["form_count"] >= 1
            or any(term in visible for term in ("login", "sign in", "identification", "register"))
        )
    )
    if credential_page:
        return "accepted", 1, "credential_entry_flow_with_password"

    if triage["triage_score"] >= 7:
        return "needs_human_review", 0, "strong_signals_but_no_clear_password_flow"
    return "rejected", 0, "page_component_without_clear_credential_capture"


def apply_existing_audit(frame, audit_path):
    if not audit_path:
        return frame
    audit = pd.read_csv(audit_path)
    if "sample_id" not in audit:
        raise ValueError(f"{audit_path}: missing sample_id")
    audit = audit.drop_duplicates("sample_id", keep="last").set_index("sample_id")
    result = frame.copy()
    protected = {"html", "label", "source", "html_hash", "dom_hash"}
    for column in audit.columns:
        if column in protected:
            continue
        mapped = result["sample_id"].map(audit[column])
        result[column] = mapped.where(mapped.notna(), result[column]) if column in result else mapped
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Rank LLM HTML samples for fast human audit; never auto-validates labels."
    )
    parser.add_argument("--llm", required=True)
    parser.add_argument("--audit", default="", help="Optional existing audit CSV to preserve edits")
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--automated-screening", "--ai-review", dest="automated_screening",
        action="store_true",
        help="Add deterministic screening decisions without changing manual_validated",
    )
    args = parser.parse_args()

    frame = pd.read_csv(args.llm)
    missing = {"sample_id", "html"} - set(frame.columns)
    if missing:
        raise ValueError(f"{args.llm}: missing columns {sorted(missing)}")
    frame = apply_existing_audit(frame, args.audit)
    if "manual_validated" not in frame:
        frame["manual_validated"] = 0
    if "url" not in frame:
        frame["url"] = ""

    triage = pd.DataFrame([
        triage_html(row.html, row.url)
        for row in frame[["html", "url"]].fillna("").itertuples(index=False)
    ], index=frame.index)
    for column in triage.columns:
        frame[column] = triage[column]
    if "review_status" not in frame:
        frame["review_status"] = "pending"
    frame["manual_validated"] = (
        pd.to_numeric(frame["manual_validated"], errors="coerce").fillna(0).astype(int)
    )
    if args.automated_screening:
        decisions = [
            automated_screening_decision(raw_html, triage_row)
            for raw_html, triage_row in zip(frame["html"], triage.to_dict("records"))
        ]
        frame["validation_status"] = [decision[0] for decision in decisions]
        frame["ai_validated"] = [decision[1] for decision in decisions]
        frame["validated_by"] = "deterministic_rules_v1"
        frame["review_method"] = "automated_screening"
        frame["validation_reason"] = [decision[2] for decision in decisions]
        frame["review_status"] = [
            "automated_screened" if decision[0] != "needs_human_review" else "pending_human"
            for decision in decisions
        ]

    preferred = [
        "sample_id", "manual_validated", "ai_validated", "validation_status", "validated_by",
        "review_method", "validation_reason", "review_status", "triage_score", "triage_level",
        "triage_recommendation", "triage_reasons", "page_title", "visible_text_preview",
        "generator", "generator_family", "brand", "template_id", "quality_level",
        "form_count", "input_count", "password_count", "file_path", "html_hash", "dom_hash",
    ]
    output_columns = [column for column in preferred if column in frame]
    output_columns.extend(
        column for column in frame.columns if column not in output_columns and column != "html"
    )
    ranked = frame.sort_values(
        ["manual_validated", "triage_score"], ascending=[True, False]
    )[output_columns]

    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    ranked.to_csv(output, index=False)
    summary = ranked["triage_level"].value_counts().reindex(
        ["high", "medium", "low"], fill_value=0
    )
    print(f"Saved ranked audit: {output}")
    print(summary.to_string())
    print("No row was automatically marked manual_validated=1.")
    if args.automated_screening:
        print("\nAutomated screening decisions:")
        print(ranked["validation_status"].value_counts().to_string())
    print("Review high -> medium, then sample-check low; edit this output file directly.")


if __name__ == "__main__":
    main()
