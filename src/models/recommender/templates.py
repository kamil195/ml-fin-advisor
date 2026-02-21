"""
Natural-language explanation templates (SPEC §8.4).

Template strings for budget recommendations, regime-change alerts,
positive reinforcement, and savings opportunities.
"""

from __future__ import annotations

TEMPLATES = {
    "over_budget_habit": (
        "You've spent ${amount} on {category} this month, which is "
        "${over_amount} over your budget. This appears to be a recurring "
        "pattern — you've exceeded this budget in {n_months} of the last "
        "6 months. Consider {suggestion}."
    ),
    "savings_opportunity": (
        "Reducing {category} by {pct}% (about ${save_amount}/month) "
        "would help you reach your {goal_name} goal {time_saved} sooner. "
        "Your spending on {category} is in the {percentile}th percentile "
        "compared to similar users."
    ),
    "regime_change_alert": (
        "Your {category} spending has {direction} by {change_pct}% since "
        "{change_date}. At this rate, you'll spend an estimated "
        "${projected} this month — ${delta} {over_under} your budget."
    ),
    "positive_reinforcement": (
        "Great job! You've stayed within your {category} budget for "
        "{streak} consecutive months, saving a total of ${total_saved}."
    ),
    "low_confidence_disclaimer": (
        "This suggestion is based on limited data — please review carefully."
    ),
    "general_recommendation": (
        "Based on your spending patterns, we recommend a monthly budget of "
        "${budget} for {category}. {explanation}"
    ),
}


def render_template(
    template_key: str,
    **kwargs: str | float | int,
) -> str:
    """
    Render a template with the given keyword arguments.

    Parameters
    ----------
    template_key : str
        Key into the TEMPLATES dictionary.
    **kwargs
        Template variables.

    Returns
    -------
    str
        Rendered template string.

    Raises
    ------
    KeyError
        If template_key is not found.
    """
    template = TEMPLATES[template_key]
    try:
        return template.format(**kwargs)
    except KeyError as e:
        return template  # return unformatted if variables missing


def render_recommendation(
    category: str,
    budget: float,
    baseline: float,
    confidence: float,
    explanation: str = "",
) -> str:
    """
    Render a full budget recommendation with appropriate template
    and confidence disclaimers.
    """
    parts = []

    if budget < baseline:
        cut_pct = (baseline - budget) / baseline * 100
        parts.append(
            render_template(
                "savings_opportunity",
                category=category,
                pct=f"{cut_pct:.0f}",
                save_amount=f"{baseline - budget:.0f}",
                goal_name="savings",
                time_saved="faster",
                percentile="50",
            )
        )
    else:
        parts.append(
            render_template(
                "general_recommendation",
                budget=f"{budget:.0f}",
                category=category,
                explanation=explanation,
            )
        )

    if confidence < 0.6:
        parts.append(TEMPLATES["low_confidence_disclaimer"])

    return " ".join(parts)
