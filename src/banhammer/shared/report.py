"""Render a BalanceReport as Markdown."""
from __future__ import annotations

import re

from banhammer.models import BalanceReport, CardInput


def report_stem(name: str) -> str:
    stem = name.replace("//", "-")
    stem = re.sub(r"[\\/:*?\"<>|]+", "", stem).strip().rstrip(".")
    return stem or "card"


def render_markdown(card: CardInput, report: BalanceReport) -> str:
    lines = [
        f"# {card.name}",
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
    lines.append("")
    return "\n".join(lines)


def _issues(report: BalanceReport) -> list[str]:
    if not report.wording_issues:
        return ["None."]
    rendered = []
    for issue in report.wording_issues:
        note = " (speculation)" if issue.speculation else ""
        cites = ", ".join(issue.rule_ids) if issue.rule_ids else "no citation"
        rendered.append(f"- {issue.description}{note} — {cites}")
    return rendered


def _combos(report: BalanceReport) -> list[str]:
    if not report.combos:
        return ["None."]
    rendered = []
    for combo in report.combos:
        note = " (speculation)" if combo.speculation else ""
        cites = ", ".join(combo.card_ids) if combo.card_ids else "no citation"
        rendered.append(f"- {combo.description}{note} — {cites}")
        for step in combo.sequence:
            rendered.append(f"  - {step}")
    return rendered
