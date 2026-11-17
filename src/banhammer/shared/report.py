"""Render a BalanceReport as Markdown and resolve the sources it cites."""
from __future__ import annotations

import re

from banhammer.models import BalanceReport, CardInput, Hit


def cited_ids(report: BalanceReport) -> set[str]:
    ids = set(report.power_assessment.comparable_ids)
    for issue in report.wording_issues:
        ids.update(issue.rule_ids)
    for combo in report.combos:
        ids.update(combo.card_ids)
    return ids


def cited_sources(report: BalanceReport, hits: list[Hit]) -> dict[str, dict]:
    """The text behind every id the report cites, so a reader sees what each citation points to."""
    cited = cited_ids(report)
    return {
        hit.id: {"namespace": hit.namespace, "name": hit.metadata.get("name") or hit.id, "text": hit.text}
        for hit in hits
        if hit.id in cited
    }


def report_stem(name: str) -> str:
    stem = name.replace("//", "-")
    stem = re.sub(r"[\\/:*?\"<>|]+", "", stem).strip().rstrip(".")
    return stem or "card"


def render_markdown(card: CardInput, report: BalanceReport, fmt: str = "modern", dropped: int = 0) -> str:
    lines = [
        f"# {card.name}",
        "",
        f"**Format:** {fmt}",
        "",
        f"**Risk:** {report.risk} (confidence {report.confidence:.2f})",
        "",
        "## Wording issues",
        "",
    ]
    lines.extend(_issues(report))
    lines.extend(["", "## Combos", ""])
    lines.extend(_combos(report))
    lines.extend(["", "## Power assessment", "", report.power_assessment.summary or "None."])
    if report.power_assessment.comparable_ids:
        lines.append("")
        lines.append("Comparable cards: " + ", ".join(report.power_assessment.comparable_ids))
    if dropped:
        lines.extend(["", f"_{dropped} claim(s) discarded for lacking a valid citation._"])
    lines.append("")
    return "\n".join(lines)


def _issues(report: BalanceReport) -> list[str]:
    if not report.wording_issues:
        return ["None."]
    rendered = []
    for issue in report.wording_issues:
        rendered.append(f"- {issue.description} — {', '.join(issue.rule_ids)}")
    return rendered


def _combos(report: BalanceReport) -> list[str]:
    if not report.combos:
        return ["None."]
    rendered = []
    for combo in report.combos:
        rendered.append(f"- {combo.description} — {', '.join(combo.card_ids)}")
        for step in combo.sequence:
            rendered.append(f"  - {step}")
    return rendered
