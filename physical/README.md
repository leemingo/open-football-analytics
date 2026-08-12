# Physical Metrics

Tracking-derived **physical metrics** — distance, speed bands, sprints, accelerations,
PSV-99, and in/out-of-possession splits — built from SkillCorner Open Data.

Total distance is the metric everyone has and almost nobody can act on: two players can
cover the same 10 km while one jogged between positions and the other repeatedly
accelerated to full speed. This package separates them by splitting movement into speed
bands, counting discrete efforts, and normalising by the time the ball was actually in
play.

## Tutorial And Analysis

| Resource | Link |
|---|---|
| Tutorial notebook | `physical/notebooks/physical_tutorial.ipynb` |
| Example analysis | [Week 5: physical metrics analysis](https://kaisport.github.io/posts/week5-physical-en.html) |

The notebook goes from raw tracking coordinates to player leaderboards: speed estimation
and segmentation, speed bands and the sustained-effort rule, PSV-99, why the smoothing
choice changes acceleration counts but not distance, the ball-in-play denominator,
TIP/OTIP splits, off-ball runs by phase of play, and what this feed cannot support.

## Input Data

The default input is the public
[SkillCorner Open Data](https://github.com/SkillCorner/opendata) sample — 10 A-League
2024/25 matches with extrapolated tracking at 10 Hz.

Unlike the other tutorials here, this one needs the **tracking** files. They are stored
with **Git LFS** (~90 MB per match, ~915 MB for all ten) and must be fetched from the
media endpoint — `raw.githubusercontent.com` returns a ~133-byte pointer file instead:

```bash
python -m physical.build_skillcorner_physical --download
```

`download_skillcorner_opendata` compares each local file against the remote
`Content-Length` rather than merely checking existence, so an interrupted run — or a
pointer file downloaded from the wrong endpoint — is re-fetched instead of being mistaken
for real tracking data.

For your own licensed data, point at a SkillCorner match-bundle root:

```bash
export SKILLCORNER_ROOT=/path/to/skillcorner/matches
```

## Metric Definitions

Every threshold lives in `physical/definitions.py`. Nothing here is derived — these are
conventions, and a number is only comparable with another source that shares them.

| Band | Definition |
|---|---|
| Walking / jogging | < 15 km/h |
| Running | 15 – 20 km/h |
| High-speed running (HSR) | 20 – 25 km/h |
| Sprint | > 25 km/h |
| High intensity | > 20 km/h (HSR ∪ sprint) |

| Rule | Value |
|---|---|
| Sampling grid | 10 Hz (`dt = 0.1 s`) |
| Minimum sustained duration for one effort | **0.7 s** |
| High acceleration / deceleration | `\|a\| > 3.0 m/s²`, sustained ≥ 0.7 s |
| PSV-99 | 99th percentile of **per-activity peak speeds** above 15 and at or below 40 km/h; the season profile keeps the strongest qualifying performance |
| Plausibility caps | speed > 40 km/h, `\|a\|` > 8 m/s² → marked invalid, **never clipped** |
| Segment break | player/period change, or a gap > 0.5 s |
| Effort possession attribution | counts if ≥ 50% of the effort is ball-in-play |
| High-intensity **count** | `hsr_count + sprint_count`, not a re-detection above 20 km/h |

Rates:

```
per 60 ball-in-play minutes:  volume x 3600 / bip_seconds
per 30 TIP minutes:           volume x 1800 / tip_seconds
per 30 OTIP minutes:          volume x 1800 / otip_seconds
```

TIP and OTIP are each normalised by *their own* exposure, so a player who spends most of
the match defending is not credited with more out-of-possession work simply for having
more of it.

## Ball-In-Play

For the public SkillCorner physical workflow, ball-in-play is the TIP + OTIP exposure:
frames with a possession label for either team are live. This is the same denominator as
the published aggregate fields `minutes_full_tip + minutes_full_otip`. Frames with no
owning team remain available as tracking-quality diagnostics, but are not included in
the canonical P60 BIP denominator. Ball-in-play is the denominator of every rate above.

Two diagnostic percentages come out of the same file and are **not** interchangeable:

| Weighting | Stored in | What it is for |
|---|---|---|
| Frame-weighted TIP + OTIP | `bip_frame_weighted_pct` | match-level exposure diagnostic |
| Player-observation-weighted TIP + OTIP | `bip_player_weighted_pct` | player-level exposure diagnostic |

They differ because the camera follows live play, so live frames carry more tracked
players than dead ones. Both are recorded in `coverage.parquet`.

## Eligibility

Defaults in `physical/build_skillcorner_physical.py`, tuned for a ten-match sample:

| Rule | Value | Why it differs from the published analysis |
|---|---|---|
| Minimum played minutes | 60 per performance | same |
| Minimum matches per player | **2** | the published rule asks for 5, which empties a 10-match table |
| Population | outfield | same |
| Coverage gate | **none** | the licensed 80% gate rejects 25 of 26 players here, for dead-ball time |

A season profile pools raw volume over pooled exposure — `sum(volume) / sum(exposure)` —
rather than averaging per-match rates, which would weight a 62-minute appearance the same
as a full match. Peak measures (PSV-99, top speed) keep the strongest qualifying
performance, matching the public SkillCorner Open Data workflow. Players are grouped by `player_id` alone with the modal position
group, so anyone who changed role between matches is not split into two half-profiles.

## Workflow

```
physical.skillcorner_tracking        download, stream JSONL -> canonical frame, lineup
physical.definitions                 thresholds and PhysicalConfig
physical.kinematics                  segmentation, resampling, velocity, plausibility caps
physical.physical_features           bands, efforts, PSV-99, TIP/OTIP, rates
physical.normalize                   eligibility, rate helpers, league aggregation
physical.explosiveness               reconstructed time-to-speed and post-turn efforts
physical.build_skillcorner_physical  match loop, season profiles, percentiles, CLI
physical.physical_plots              figures
```

Outputs land under `tmp/data/physical_opendata/`: `player_matches.parquet` (one row per
match-player), `coverage.parquet` (per-match tracking quality), and `manifest.json`
(definition version, config, and every eligibility choice, so a table can be traced back
to the rules that produced it).

## Velocity Estimation

Two estimators ship here: `trailing_mean` (the default — a 1-second trailing average of
speed) and `savgol` (smooth position, then differentiate analytically). The choice barely
touches distance and roughly **triples** the high-acceleration count, because
acceleration is the second derivative and a pre-averaged speed signal has flattened
peaks. Distance and speed-band metrics port between sources reasonably well; effort counts
port only when the sustain rule matches; acceleration counts barely port at all.

`kleague-insights` additionally implements Butterworth and Kalman estimators for its
calibration work against published figures.

## Explosiveness

`physical.explosiveness` measures time from a walking-speed onset (9 km/h) to HSR (20
km/h) and sprint (25 km/h) after a qualifying acceleration, and after a direction change
of ≥ 60° within a 0.5 s window. It is a **transparent reconstruction**, fully specified in
code — not a vendor's proprietary explosive-acceleration product, and it should never be
presented as one.

**Run it with `estimator="savgol"`**, not the default. It triggers off accelerations above
3 m/s², and `trailing_mean` averages *speed* over a full second, which flattens exactly
the peaks it needs. On match 1886347 with `savgol` it finds 227 efforts — mean 1.30 s to
reach 20 km/h and 2.05 s to 25 km/h, and 1.17 s / 2.13 s measured after a direction change.

## Tests

Behaviour is pinned by unit tests on synthetic signals, requiring no data:

```bash
python -m pytest physical/tests -q
```

They cover band edges (14.9 vs 15.0 vs 20.0 vs 25.0 km/h), the 0.7 s rule at and below
threshold, strict `> 3.0 m/s²` acceleration, PSV-99 percentile interpolation and the
40 km/h ceiling, segment isolation across gaps and period breaks, and the invariance of a
per-60-BIP rate to dead-ball time.
