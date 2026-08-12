"""Physical metrics from tracking data — distance, speed bands, sprints, PSV-99.

The public API operates on pandas data frames rather than vendor objects, so a
provider adapter only has to produce the small canonical tracking schema
described in :mod:`physical.skillcorner_tracking`. Every threshold and counting
rule lives in :mod:`physical.definitions`.

Typical use on SkillCorner Open Data::

    from physical import PhysicalConfig, compute_player_match_metrics
    from physical.skillcorner_tracking import (
        download_skillcorner_opendata, iter_skillcorner_tracking,
        load_skillcorner_bundle, resolve_match_files,
    )

    download_skillcorner_opendata(match_ids=[1886347])
    files = resolve_match_files(1886347)
    bundle = load_skillcorner_bundle(files["match"], season="2024/25")
    tracking = pd.concat(iter_skillcorner_tracking(files["tracking"], bundle))
    player_matches = compute_player_match_metrics(tracking, bundle.lineup)
"""

from .definitions import (
    PHYSICAL_DEFINITION_VERSION,
    POSITION_GROUPS,
    PhysicalConfig,
    SpeedThresholds,
)
from .kinematics import normalize_position_group, prepare_kinematics
from .normalize import aggregate_league, aggregate_p60
from .physical_features import (
    compute_player_match_metrics,
    compute_rate_sensitivity,
)

__all__ = [
    "PHYSICAL_DEFINITION_VERSION",
    "POSITION_GROUPS",
    "PhysicalConfig",
    "SpeedThresholds",
    "aggregate_league",
    "aggregate_p60",
    "compute_player_match_metrics",
    "compute_rate_sensitivity",
    "normalize_position_group",
    "prepare_kinematics",
]
