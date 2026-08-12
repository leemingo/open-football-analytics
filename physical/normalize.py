"""Eligibility filters and normalisation for physical performances.

The functions here operate on *player-match* rows and deliberately do not infer
columns or silently repair invalid exposure:

* a performance is eligible when ``played_minutes >= 60`` (inclusive), which is
  the article's stated sample filter;
* P60 BIP rates use observed ball-in-play minutes, P30 TIP rates use observed
  team-in-possession minutes, per-90 uses played minutes — never frame counts;
* rows with non-positive or non-finite exposure, negative counts or distances,
  or ball-in-play time materially above played time are rejected before
  normalisation rather than coerced to zero.

The article's footnote "5 datapoints in sample" counts the **five competitions**
plotted, not five matches per player, so the default league estimand is the
unweighted mean over eligible player-matches with no per-player minimum.
``minimum_eligible_performances`` and ``weighting="player"`` remain available as
sensitivities.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np
import pandas as pd

DEFAULT_GOALKEEPER_LABELS = frozenset(
    {"gk", "goalkeeper", "goal keeper", "keeper", "골키퍼"}
)

#: Article benchmark metric -> our player-match column.
BENCHMARK_COLUMNS = {
    "distance_m": "total_distance_m_p60_bip",
    "running_distance_m": "running_distance_m_p60_bip",
    "high_speed_running_dist_m": "hsr_distance_m_p60_bip",
    "sprinting_distance_m": "sprint_distance_m_p60_bip",
    "sprints_count": "sprint_count_p60_bip",
    "high_intensity_events": "high_intensity_count_p60_bip",
    "high_accelerations": "high_acceleration_count_p60_bip",
    "psv_99_kmh": "psv99_kmh",
}

#: Metrics that are peaks rather than volumes and must never be rate-normalised.
PEAK_COLUMNS = ("psv99_kmh", "psv99_frame_kmh", "psv99_activity_min07_kmh", "max_speed_kmh")


def _require_columns(frame: pd.DataFrame, columns: Iterable[str]) -> None:
    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def _as_columns(columns: str | Sequence[str]) -> list[str]:
    return [columns] if isinstance(columns, str) else list(columns)


def eligible_performances(
    performances: pd.DataFrame,
    *,
    played_minutes_col: str = "played_minutes",
    minimum_played_minutes: float = 60.0,
) -> pd.DataFrame:
    """Return finite player-match performances meeting the minutes threshold."""

    _require_columns(performances, [played_minutes_col])
    minutes = pd.to_numeric(performances[played_minutes_col], errors="coerce")
    return performances.loc[np.isfinite(minutes) & (minutes >= minimum_played_minutes)].copy()


def _add_rates(
    performances: pd.DataFrame,
    value_columns: Sequence[str],
    exposure_minutes: pd.Series,
    *,
    window_minutes: float,
    output_suffix: str,
    valid: pd.Series,
    invalid: str,
    error_label: str = "exposure or metric",
) -> pd.DataFrame:
    frame = performances.copy()
    numeric = frame[list(value_columns)].apply(pd.to_numeric, errors="coerce")
    valid = valid & np.isfinite(numeric).all(axis=1) & (numeric >= 0).all(axis=1)
    if not bool(valid.all()):
        bad = frame.index[~valid].tolist()
        if invalid == "raise":
            preview = bad[:10]
            suffix = "..." if len(bad) > 10 else ""
            raise ValueError(
                f"Invalid {error_label} values at row indices {preview}{suffix}"
            )
        frame = frame.loc[valid].copy()
        numeric = numeric.loc[valid]
        exposure_minutes = exposure_minutes.loc[valid]
    for column in value_columns:
        frame[f"{column}{output_suffix}"] = (
            numeric[column].to_numpy(dtype=float)
            * window_minutes
            / exposure_minutes.to_numpy(dtype=float)
        )
    return frame


def add_p60_bip_rates(
    performances: pd.DataFrame,
    value_columns: str | Sequence[str],
    *,
    bip_minutes_col: str = "bip_minutes",
    played_minutes_col: str = "played_minutes",
    output_suffix: str = "_p60_bip",
    maximum_bip_overrun_minutes: float = 1.0,
    invalid: str = "raise",
) -> pd.DataFrame:
    """Add per-60-minutes-of-ball-in-play rates with exposure guardrails.

    ``value_columns`` must be additive quantities such as metres or event
    counts.  Peak or percentile speeds must not be normalised — see
    :data:`PEAK_COLUMNS`.

    Args:
        maximum_bip_overrun_minutes: Tolerance for provider exposure rounded to
            whole minutes.  Set to zero when both clocks keep sub-second precision.
        invalid: ``"raise"`` reports offending row indices; ``"drop"`` removes
            them.  Invalid rows are never converted to zero.
    """

    values = _as_columns(value_columns)
    _require_columns(performances, [bip_minutes_col, played_minutes_col, *values])
    if invalid not in {"raise", "drop"}:
        raise ValueError("invalid must be either 'raise' or 'drop'")
    if maximum_bip_overrun_minutes < 0:
        raise ValueError("maximum_bip_overrun_minutes must be non-negative")
    overlap = set(values).intersection(PEAK_COLUMNS)
    if overlap:
        raise ValueError(f"peak metrics must not be rate-normalised: {sorted(overlap)}")

    bip = pd.to_numeric(performances[bip_minutes_col], errors="coerce")
    played = pd.to_numeric(performances[played_minutes_col], errors="coerce")
    valid = (
        np.isfinite(bip)
        & np.isfinite(played)
        & (bip > 0)
        & (played > 0)
        & (bip <= played + maximum_bip_overrun_minutes)
    )
    return _add_rates(
        performances, values, bip, window_minutes=60.0,
        output_suffix=output_suffix, valid=valid, invalid=invalid,
        error_label="P60-BIP exposure or metric",
    )


def add_p30_tip_rates(
    performances: pd.DataFrame,
    value_columns: str | Sequence[str],
    *,
    tip_minutes_col: str = "tip_minutes",
    output_suffix: str = "_p30_tip",
    invalid: str = "drop",
) -> pd.DataFrame:
    """Add per-30-minutes-of-team-in-possession rates.

    This is the normalisation SkillCorner uses for off-ball-run and pressure
    metrics.  Team-in-possession time is strictly narrower than ball-in-play, so
    the two rates are not interchangeable.
    """

    values = _as_columns(value_columns)
    _require_columns(performances, [tip_minutes_col, *values])
    tip = pd.to_numeric(performances[tip_minutes_col], errors="coerce")
    return _add_rates(
        performances, values, tip, window_minutes=30.0,
        output_suffix=output_suffix, valid=np.isfinite(tip) & (tip > 0), invalid=invalid,
    )


def add_per90_rates(
    performances: pd.DataFrame,
    value_columns: str | Sequence[str],
    *,
    played_minutes_col: str = "played_minutes",
    output_suffix: str = "_p90",
    invalid: str = "drop",
) -> pd.DataFrame:
    """Add conventional per-90-played-minutes rates."""

    values = _as_columns(value_columns)
    _require_columns(performances, [played_minutes_col, *values])
    played = pd.to_numeric(performances[played_minutes_col], errors="coerce")
    return _add_rates(
        performances, values, played, window_minutes=90.0,
        output_suffix=output_suffix, valid=np.isfinite(played) & (played > 0),
        invalid=invalid,
    )


def _outfield_mask(positions: pd.Series, goalkeeper_labels: Iterable[str]) -> pd.Series:
    labels = {str(value).strip().casefold() for value in goalkeeper_labels}
    normalized = positions.astype("string").str.strip().str.casefold()
    # Missing position is not safely classifiable as outfield.
    return normalized.notna() & ~normalized.isin(labels)


def select_population(
    performances: pd.DataFrame,
    *,
    population: str,
    position_col: str = "position_group",
    raw_position_col: str = "position",
    goalkeeper_labels: Iterable[str] = DEFAULT_GOALKEEPER_LABELS,
) -> pd.DataFrame:
    """Return the requested player population with defensive GK detection.

    Older materialised SkillCorner tables labelled the broad position group of
    goalkeepers as ``Other`` while retaining ``GK`` in the raw position. The
    raw field is therefore checked as a second goalkeeper signal so historical
    tables can be aggregated correctly without rewriting tracking metrics.
    """

    if population not in {"all", "outfield"}:
        raise ValueError("population must be 'all' or 'outfield'")
    if population == "all":
        return performances.copy()
    _require_columns(performances, [position_col])
    mask = _outfield_mask(performances[position_col], goalkeeper_labels)
    if raw_position_col in performances:
        raw = performances[raw_position_col]
        raw_known = raw.astype("string").notna()
        mask &= ~(raw_known & ~_outfield_mask(raw, goalkeeper_labels))
    return performances.loc[mask].copy()


def aggregate_league(
    performances: pd.DataFrame,
    metric_columns: str | Sequence[str],
    *,
    group_columns: Sequence[str] = ("season",),
    match_id_col: str = "match_id",
    player_id_col: str = "player_id",
    played_minutes_col: str = "played_minutes",
    position_col: str = "position_group",
    population: str = "outfield",
    weighting: str = "performance",
    minimum_played_minutes: float = 60.0,
    minimum_eligible_performances: int = 1,
    goalkeeper_labels: Iterable[str] = DEFAULT_GOALKEEPER_LABELS,
    dispersion: bool = False,
) -> pd.DataFrame:
    """Aggregate qualified player-match rows into a league summary.

    Qualification is recomputed *after* the population filter, and the returned
    counts make the denominator auditable.  ``dispersion=True`` adds the
    standard deviation and standard error across player-matches, which matters
    because the article's per-league figures rest on small samples.
    """

    metrics = _as_columns(metric_columns)
    groups = list(group_columns)
    if not groups:
        raise ValueError("group_columns must contain at least one column")
    if minimum_eligible_performances < 1:
        raise ValueError("minimum_eligible_performances must be at least one")
    if population not in {"all", "outfield"}:
        raise ValueError("population must be 'all' or 'outfield'")
    if weighting not in {"player", "performance"}:
        raise ValueError("weighting must be 'player' or 'performance'")

    required = [*groups, match_id_col, player_id_col, played_minutes_col, *metrics]
    if population == "outfield":
        required.append(position_col)
    _require_columns(performances, required)

    frame = eligible_performances(
        performances,
        played_minutes_col=played_minutes_col,
        minimum_played_minutes=minimum_played_minutes,
    )
    frame = select_population(
        frame,
        population=population,
        position_col=position_col,
        goalkeeper_labels=goalkeeper_labels,
    )

    if frame.duplicated([*groups, match_id_col, player_id_col]).any():
        raise ValueError("Expected one row per player-match within each aggregation group")

    numeric = frame[metrics].apply(pd.to_numeric, errors="coerce")
    keep = np.isfinite(numeric).all(axis=1)
    frame = frame.loc[keep].copy()
    frame[metrics] = numeric.loc[keep]

    player_groups = [*groups, player_id_col]
    if minimum_eligible_performances > 1:
        counts = (
            frame.groupby(player_groups, dropna=False)[match_id_col]
            .nunique()
            .rename("_n")
            .reset_index()
        )
        qualified = counts.loc[counts["_n"] >= minimum_eligible_performances, player_groups]
        frame = frame.merge(qualified, how="inner", on=player_groups, validate="many_to_one")

    summary = (
        frame.groupby(groups, dropna=False)
        .agg(n_performances=(player_id_col, "size"), n_players=(player_id_col, "nunique"))
        .reset_index()
    )
    if weighting == "performance":
        estimates = frame.groupby(groups, dropna=False)[metrics].mean().reset_index()
    else:
        player_means = frame.groupby(player_groups, dropna=False)[metrics].mean().reset_index()
        estimates = player_means.groupby(groups, dropna=False)[metrics].mean().reset_index()

    result = estimates.merge(summary, on=groups, validate="one_to_one")
    if dispersion:
        spread = frame.groupby(groups, dropna=False)[metrics].std(ddof=1).reset_index()
        spread = spread.rename(columns={m: f"{m}_sd" for m in metrics})
        result = result.merge(spread, on=groups, validate="one_to_one")
        for metric in metrics:
            result[f"{metric}_se"] = result[f"{metric}_sd"] / np.sqrt(result["n_performances"])
    result["population"] = population
    result["weighting"] = f"{weighting}_weighted"
    ordered = [*groups, "population", "weighting", "n_players", "n_performances", *metrics]
    return result[ordered + [c for c in result.columns if c not in ordered]]


def aggregate_all_variants(
    performances: pd.DataFrame,
    metric_columns: str | Sequence[str],
    **kwargs: object,
) -> pd.DataFrame:
    """Return all-player/outfield x player/performance-weighted summaries."""

    return pd.concat(
        [
            aggregate_league(
                performances, metric_columns, population=population,
                weighting=weighting, **kwargs,
            )
            for population in ("all", "outfield")
            for weighting in ("player", "performance")
        ],
        ignore_index=True,
    )


def benchmark_table(
    performances: pd.DataFrame,
    article_values: dict[str, float],
    *,
    group_columns: Sequence[str] = ("season",),
    **kwargs: object,
) -> pd.DataFrame:
    """Article value / our value / absolute delta / percent delta, per metric."""

    columns = [BENCHMARK_COLUMNS[name] for name in article_values if name in BENCHMARK_COLUMNS]
    summary = aggregate_league(
        performances, columns, group_columns=group_columns, dispersion=True, **kwargs
    )
    if summary.empty:
        raise ValueError("no eligible performances to compare")
    row = summary.iloc[0]
    records = []
    for name, article in article_values.items():
        column = BENCHMARK_COLUMNS.get(name)
        if column is None:
            continue
        ours = float(row[column])
        records.append(
            {
                "metric": name,
                "column": column,
                "article": float(article),
                "ours": ours,
                "abs_delta": ours - float(article),
                "pct_delta": 100.0 * (ours - float(article)) / float(article),
                "sd": float(row.get(f"{column}_sd", np.nan)),
                "se": float(row.get(f"{column}_se", np.nan)),
                "n_performances": int(row["n_performances"]),
                "n_players": int(row["n_players"]),
            }
        )
    return pd.DataFrame(records)


def aggregate_p60(
    player_matches: pd.DataFrame,
    *,
    min_played_minutes: float = 60.0,
    min_player_matches: int = 1,
    player_weighted: bool = False,
) -> pd.DataFrame:
    """Aggregate eligible performances into a single league row.

    Thin wrapper over :func:`aggregate_league`.  The default estimand is the
    unweighted mean over eligible player-matches, which is the unit of
    observation the published physical benchmarks use.
    """

    metrics = [
        column
        for column in player_matches.columns
        if column.endswith(("_p60_bip", "_p30_tip")) or column == "psv99_kmh"
    ]
    frame = player_matches.copy()
    if "season" not in frame:
        frame["season"] = 0
    summary = aggregate_league(
        frame,
        metrics,
        group_columns=("season",),
        population="all",
        weighting="player" if player_weighted else "performance",
        minimum_played_minutes=min_played_minutes,
        minimum_eligible_performances=min_player_matches,
    )
    return summary.rename(
        columns={"n_players": "player_count", "n_performances": "performance_count"}
    )
