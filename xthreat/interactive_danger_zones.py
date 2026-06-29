"""Standalone interactive danger-zone views for xT analysis."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from string import Template
from typing import Iterable

import numpy as np
import pandas as pd

try:
    from football_cdf.constants import PITCH_X, PITCH_Y
except ModuleNotFoundError:
    PITCH_X = 105.0
    PITCH_Y = 68.0

from xthreat.xthreat_model import get_cell_indexes_center


DEFAULT_OUTPUT = Path("tmp/data/bepro_drive_xthreat_k1/interactive/team_danger_zones_2025.html")
MOVE_ACTIONS = ["pass", "carry"]


def _round(value, ndigits: int = 4):
    if pd.isna(value):
        return None
    return round(float(value), ndigits)


def _valid_team_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if "team_shortname" not in frame.columns:
        return frame.copy()
    team = frame["team_shortname"].astype("string")
    return frame[team.notna() & ~team.astype(str).isin(["<NA>", "nan", "None", ""])].copy()


def _required_columns(value_col: str) -> set[str]:
    return {
        "team_shortname",
        "action_type",
        "start_x",
        "start_y",
        "end_x",
        "end_y",
        value_col,
    }


def build_team_danger_zone_payload(
    scored: pd.DataFrame,
    *,
    season_name: str | None = "2025",
    value_col: str = "custom_xT_added",
    l: int = 16,
    w: int = 12,
    positive_only: bool = True,
    top_players: int = 5,
    top_routes: int = 8,
    top_end_zones: int = 5,
    max_destination_zones: int | None = 24,
    player_name_map: dict[str, str] | None = None,
) -> dict:
    """Aggregate xT created by start zone for a team-level interactive view.

    The view uses a common xT value column for every team. This makes the zone
    totals comparable across teams while still allowing each team to highlight
    its own strongest creation zones.
    """
    missing = sorted(_required_columns(value_col) - set(scored.columns))
    if missing:
        raise KeyError(f"Missing columns for danger-zone payload: {missing}")

    player_name_map = {str(key): str(value) for key, value in (player_name_map or {}).items()}
    frame = scored.copy()
    if season_name is not None and "season_name" in frame.columns:
        frame = frame[frame["season_name"].astype(str).eq(str(season_name))].copy()
    frame = _valid_team_frame(frame)
    frame = frame[frame["action_type"].astype(str).isin(MOVE_ACTIONS)].copy()
    frame = frame.dropna(subset=["start_x", "start_y", "end_x", "end_y"]).copy()
    if "player_name" not in frame.columns:
        frame["player_name"] = ""
    raw_player_name = frame["player_name"].astype(str)
    if "player_display_name" in frame.columns:
        display_name = frame["player_display_name"].astype("string")
        frame["player_display_name"] = display_name.fillna(raw_player_name).astype(str)
    else:
        frame["player_display_name"] = raw_player_name.map(player_name_map).fillna(raw_player_name).astype(str)
    frame[value_col] = pd.to_numeric(frame[value_col], errors="coerce").fillna(0.0)
    frame["value"] = frame[value_col].clip(lower=0.0) if positive_only else frame[value_col]
    frame["net_value"] = frame[value_col]
    frame["progression"] = pd.to_numeric(frame["end_x"], errors="coerce") - pd.to_numeric(frame["start_x"], errors="coerce")

    start_xi, start_yj = get_cell_indexes_center(frame["start_x"], frame["start_y"], l=l, w=w)
    end_xi, end_yj = get_cell_indexes_center(frame["end_x"], frame["end_y"], l=l, w=w)
    frame["x_cell"] = start_xi.to_numpy()
    frame["y_cell"] = start_yj.to_numpy()
    frame["cell_id"] = frame["y_cell"].astype(str) + "-" + frame["x_cell"].astype(str)
    frame["end_x_cell"] = end_xi.to_numpy()
    frame["end_y_cell"] = end_yj.to_numpy()
    frame["end_cell_id"] = frame["end_y_cell"].astype(str) + "-" + frame["end_x_cell"].astype(str)

    team_totals = (
        frame.groupby("team_shortname", dropna=False)
        .agg(
            actions=("action_type", "size"),
            xT=("value", "sum"),
            net_xT=("net_value", "sum"),
            avg_xT=("value", "mean"),
        )
        .reset_index()
        .sort_values("xT", ascending=False)
    )

    dx = PITCH_X / float(l)
    dy = PITCH_Y / float(w)
    all_cells = pd.DataFrame(
        [
            {
                "x_cell": xi,
                "y_cell": yj,
                "cell_id": f"{yj}-{xi}",
                "x0": -PITCH_X / 2 + xi * dx,
                "x1": -PITCH_X / 2 + (xi + 1) * dx,
                "y0": -PITCH_Y / 2 + yj * dy,
                "y1": -PITCH_Y / 2 + (yj + 1) * dy,
                "center_x": -PITCH_X / 2 + (xi + 0.5) * dx,
                "center_y": -PITCH_Y / 2 + (yj + 0.5) * dy,
            }
            for yj in range(w)
            for xi in range(l)
        ]
    )

    teams: dict[str, dict] = {}
    for team_name, team_frame in frame.groupby("team_shortname", dropna=False):
        team_name = str(team_name)
        grouped = (
            team_frame.groupby(["x_cell", "y_cell", "cell_id"], dropna=False)
            .agg(
                actions=("action_type", "size"),
                passes=("action_type", lambda s: int((s == "pass").sum())),
                carries=("action_type", lambda s: int((s == "carry").sum())),
                xT=("value", "sum"),
                net_xT=("net_value", "sum"),
                avg_xT=("value", "mean"),
                avg_progression=("progression", "mean"),
            )
            .reset_index()
        )
        cells = all_cells.merge(grouped, on=["x_cell", "y_cell", "cell_id"], how="left")
        for col in ["actions", "passes", "carries"]:
            cells[col] = cells[col].fillna(0).astype(int)
        for col in ["xT", "net_xT", "avg_xT", "avg_progression"]:
            cells[col] = pd.to_numeric(cells[col], errors="coerce").fillna(0.0)
        cells["xT_per_100_actions"] = np.where(cells["actions"] > 0, cells["xT"] / cells["actions"] * 100.0, 0.0)

        cell_details = {}
        for cell_id, cell_frame in team_frame.groupby("cell_id", dropna=False):
            player = (
                cell_frame.groupby("player_display_name", dropna=False)
                .agg(actions=("action_type", "size"), xT=("value", "sum"), net_xT=("net_value", "sum"))
                .reset_index()
                .sort_values("xT", ascending=False)
                .head(int(top_players))
            )
            destination_zones = (
                cell_frame.groupby(["end_x_cell", "end_y_cell", "end_cell_id"], dropna=False)
                .agg(
                    actions=("action_type", "size"),
                    xT=("value", "sum"),
                    net_xT=("net_value", "sum"),
                    avg_xT=("value", "mean"),
                    avg_progression=("progression", "mean"),
                )
                .reset_index()
                .sort_values("xT", ascending=False)
            )
            if positive_only:
                destination_zones = destination_zones[destination_zones["xT"].gt(0)].copy()
            end_zones = destination_zones.head(int(top_end_zones))
            destination_heat = destination_zones
            if max_destination_zones is not None:
                destination_heat = destination_heat.head(int(max_destination_zones))
            routes = (
                cell_frame.sort_values("value", ascending=False)
                .head(int(top_routes))
                .copy()
            )
            route_cols = [
                "player_name",
                "action_type",
                "start_x",
                "start_y",
                "end_x",
                "end_y",
                "value",
                "net_value",
                "match_id",
                "phase_index",
                "player_display_name",
            ]
            for col in route_cols:
                if col not in routes.columns:
                    routes[col] = pd.NA
            cell_details[str(cell_id)] = {
                "players": [
                    {
                        "name": str(row["player_display_name"]),
                        "actions": int(row["actions"]),
                        "xT": _round(row["xT"], 4),
                        "netX": _round(row["net_xT"], 4),
                    }
                    for _, row in player.iterrows()
                ],
                "endZones": [
                    {
                        "cell": str(row["end_cell_id"]),
                        "xCell": int(row["end_x_cell"]),
                        "yCell": int(row["end_y_cell"]),
                        "actions": int(row["actions"]),
                        "xT": _round(row["xT"], 4),
                        "netX": _round(row["net_xT"], 4),
                    }
                    for _, row in end_zones.iterrows()
                ],
                "destinationZones": [
                    {
                        "cell": str(row["end_cell_id"]),
                        "xCell": int(row["end_x_cell"]),
                        "yCell": int(row["end_y_cell"]),
                        "actions": int(row["actions"]),
                        "xT": _round(row["xT"], 4),
                        "netX": _round(row["net_xT"], 4),
                        "avgX": _round(row["avg_xT"], 6),
                        "avgProg": _round(row["avg_progression"], 3),
                    }
                    for _, row in destination_heat.iterrows()
                ],
                "routes": [
                    {
                        "player": str(row["player_display_name"]),
                        "type": str(row["action_type"]),
                        "sx": _round(row["start_x"], 3),
                        "sy": _round(row["start_y"], 3),
                        "ex": _round(row["end_x"], 3),
                        "ey": _round(row["end_y"], 3),
                        "xT": _round(row["value"], 4),
                        "netX": _round(row["net_value"], 4),
                        "match": str(row["match_id"]),
                        "phase": str(row["phase_index"]),
                    }
                    for _, row in routes.iterrows()
                ],
            }

        team_total = team_totals[team_totals["team_shortname"].astype(str).eq(team_name)].iloc[0]
        teams[team_name] = {
            "summary": {
                "actions": int(team_total["actions"]),
                "xT": _round(team_total["xT"], 4),
                "netX": _round(team_total["net_xT"], 4),
                "avgX": _round(team_total["avg_xT"], 6),
                "maxCellX": _round(cells["xT"].max(), 4),
            },
            "cells": [
                {
                    "cell": str(row["cell_id"]),
                    "xCell": int(row["x_cell"]),
                    "yCell": int(row["y_cell"]),
                    "x0": _round(row["x0"], 3),
                    "x1": _round(row["x1"], 3),
                    "y0": _round(row["y0"], 3),
                    "y1": _round(row["y1"], 3),
                    "cx": _round(row["center_x"], 3),
                    "cy": _round(row["center_y"], 3),
                    "actions": int(row["actions"]),
                    "passes": int(row["passes"]),
                    "carries": int(row["carries"]),
                    "xT": _round(row["xT"], 4),
                    "netX": _round(row["net_xT"], 4),
                    "avgX": _round(row["avg_xT"], 6),
                    "xTPer100": _round(row["xT_per_100_actions"], 4),
                    "avgProg": _round(row["avg_progression"], 3),
                }
                for _, row in cells.iterrows()
            ],
            "details": cell_details,
        }

    return {
        "meta": {
            "season": str(season_name) if season_name is not None else "all",
            "valueCol": value_col,
            "positiveOnly": bool(positive_only),
            "pitchX": PITCH_X,
            "pitchY": PITCH_Y,
            "gridL": int(l),
            "gridW": int(w),
        },
        "teams": teams,
        "teamOrder": [str(t) for t in team_totals["team_shortname"].tolist()],
    }


def _html_template() -> Template:
    return Template(
        r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>$title</title>
  <style>
    :root {
      --text: #202124;
      --muted: #5f6368;
      --line: #d7ddd6;
      --green: #2f8f46;
      --red: #e45745;
      --blue: #4e79a7;
      --orange: #f28e2b;
      --bg: #ffffff;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Arial, sans-serif;
      color: var(--text);
      background: var(--bg);
    }
    main {
      width: min(1180px, calc(100vw - 32px));
      margin: 24px auto 34px;
    }
    h1 {
      margin: 0 0 18px;
      text-align: center;
      font-size: 26px;
      line-height: 1.2;
    }
    .controls {
      display: flex;
      align-items: center;
      justify-content: center;
      gap: 12px;
      margin-bottom: 20px;
      flex-wrap: wrap;
    }
    select {
      font-size: 16px;
      padding: 9px 36px 9px 12px;
      border: 1px solid #c8c8c8;
      border-radius: 6px;
      background: white;
      min-width: 260px;
    }
    .layout {
      display: grid;
      grid-template-columns: minmax(520px, 1fr) 330px;
      gap: 28px;
      align-items: start;
    }
    .pitch-wrap {
      border-bottom: 1px solid var(--line);
      padding-bottom: 12px;
    }
    svg {
      width: 100%;
      height: auto;
      display: block;
      background: #f9fbf7;
      border: 1px solid #8f978d;
    }
    .cell {
      cursor: pointer;
      stroke: rgba(255,255,255,0.42);
      stroke-width: 0.12;
      transition: opacity 120ms ease, stroke 120ms ease;
    }
    .cell:hover {
      stroke: #1e5b2e;
      stroke-width: 0.35;
    }
    .selected {
      stroke: #111 !important;
      stroke-width: 0.45 !important;
    }
    #destination-layer, #source-layer {
      pointer-events: none;
    }
    .destination-cell {
      pointer-events: none;
      stroke: rgba(255,255,255,0.55);
      stroke-width: 0.1;
    }
    .source-highlight {
      pointer-events: none;
      fill: none;
      stroke: var(--orange);
      stroke-width: 0.7;
    }
    .pitch-line {
      fill: none;
      stroke: rgba(80, 94, 80, 0.38);
      stroke-width: 0.22;
    }
    .route-pass {
      stroke: var(--blue);
      fill: none;
      stroke-width: 0.55;
      opacity: 0.75;
    }
    .route-carry {
      stroke: var(--orange);
      fill: none;
      stroke-width: 0.55;
      opacity: 0.75;
    }
    aside h2 {
      margin: 0 0 14px;
      font-size: 24px;
      line-height: 1.2;
    }
    .lede {
      margin: 0 0 18px;
      color: var(--muted);
      line-height: 1.45;
      font-size: 15px;
    }
    .metric-grid {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 8px;
      margin: 14px 0 18px;
    }
    .metric {
      border: 1px solid #e2e5e0;
      border-radius: 6px;
      padding: 9px 10px;
      background: #fbfcfb;
    }
    .metric span {
      display: block;
      font-size: 12px;
      color: var(--muted);
      margin-bottom: 3px;
    }
    .metric strong {
      font-size: 18px;
    }
    h3 {
      margin: 17px 0 7px;
      font-size: 15px;
    }
    table {
      width: 100%;
      border-collapse: collapse;
      font-size: 13px;
    }
    th, td {
      text-align: left;
      padding: 5px 4px;
      border-bottom: 1px solid #ecefec;
      vertical-align: top;
    }
    th {
      color: var(--muted);
      font-weight: 600;
    }
    .hint {
      color: var(--muted);
      font-size: 13px;
      margin-top: 10px;
      line-height: 1.45;
    }
    .legend {
      display: flex;
      align-items: center;
      gap: 12px;
      flex-wrap: wrap;
      margin-top: 10px;
      color: var(--muted);
      font-size: 13px;
    }
    .swatch {
      display: inline-block;
      width: 13px;
      height: 13px;
      border-radius: 2px;
      margin-right: 5px;
      vertical-align: -2px;
    }
    @media (max-width: 860px) {
      .layout { grid-template-columns: 1fr; }
      main { width: min(100vw - 20px, 760px); margin-top: 16px; }
    }
  </style>
</head>
<body>
<main>
  <h1>Who creates danger from where?</h1>
  <div class="controls">
    <label for="team-select">Team</label>
    <select id="team-select"></select>
  </div>
  <div class="layout">
    <section class="pitch-wrap">
      <svg id="pitch" viewBox="0 0 105 68" role="img" aria-label="Team danger zone pitch">
        <defs>
          <marker id="arrow-pass" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="4" markerHeight="4" orient="auto-start-reverse">
            <path d="M 0 0 L 10 5 L 0 10 z" fill="#4e79a7"></path>
          </marker>
          <marker id="arrow-carry" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="4" markerHeight="4" orient="auto-start-reverse">
            <path d="M 0 0 L 10 5 L 0 10 z" fill="#f28e2b"></path>
          </marker>
        </defs>
        <g id="heat-layer"></g>
        <g id="destination-layer"></g>
        <g id="pitch-lines"></g>
        <g id="source-layer"></g>
        <g id="route-layer"></g>
      </svg>
      <div class="legend">
        <span><span class="swatch" style="background:#2f8f46"></span>Higher cumulative <span id="value-label-legend">positive xT</span> created from start zone</span>
        <span><span class="swatch" style="background:#e45745"></span>Destination zones from selected start zone</span>
        <span><span class="swatch" style="background:#4e79a7"></span>Pass route</span>
        <span><span class="swatch" style="background:#f28e2b"></span>Carry route</span>
      </div>
      <p class="hint">Hover a zone to preview destination danger. Click locks the panel, destination heatmap, and representative routes.</p>
    </section>
    <aside>
      <h2 id="team-title"></h2>
      <p class="lede" id="team-summary"></p>
      <div class="metric-grid">
        <div class="metric"><span id="cell-value-label">Zone positive xT</span><strong id="cell-xT">-</strong></div>
        <div class="metric"><span>Actions</span><strong id="cell-actions">-</strong></div>
        <div class="metric"><span>xT / 100 actions</span><strong id="cell-rate">-</strong></div>
        <div class="metric"><span>Avg progression</span><strong id="cell-prog">-</strong></div>
      </div>
      <h3>Top players from this zone</h3>
      <table id="players-table"></table>
      <h3>Top destination zones</h3>
      <table id="endzones-table"></table>
      <h3>Representative high-xT routes</h3>
      <table id="routes-table"></table>
    </aside>
  </div>
</main>
<script id="payload" type="application/json">$payload</script>
<script>
const payload = JSON.parse(document.getElementById("payload").textContent);
const select = document.getElementById("team-select");
const heatLayer = document.getElementById("heat-layer");
const destinationLayer = document.getElementById("destination-layer");
const sourceLayer = document.getElementById("source-layer");
const routeLayer = document.getElementById("route-layer");
const pitchLines = document.getElementById("pitch-lines");
const valueLabel = payload.meta.positiveOnly ? "positive xT" : "net xT";
let currentTeam = payload.teamOrder[0];
let lockedCell = null;
document.getElementById("value-label-legend").textContent = valueLabel;
document.getElementById("cell-value-label").textContent = `Zone ${valueLabel}`;

function fmt(value, digits = 3) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return "-";
  return Number(value).toFixed(digits);
}
function sx(x) { return Number(x) + 52.5; }
function sy(y) { return 34 - Number(y); }
function rectY(y0, y1) { return sy(y1); }
function rectH(y0, y1) { return Math.abs(Number(y1) - Number(y0)); }
function cellBounds(xCell, yCell) {
  const dx = Number(payload.meta.pitchX) / Number(payload.meta.gridL);
  const dy = Number(payload.meta.pitchY) / Number(payload.meta.gridW);
  const x0 = -Number(payload.meta.pitchX) / 2 + Number(xCell) * dx;
  const x1 = x0 + dx;
  const y0 = -Number(payload.meta.pitchY) / 2 + Number(yCell) * dy;
  const y1 = y0 + dy;
  return {x0, x1, y0, y1};
}
function clear(node) { while (node.firstChild) node.removeChild(node.firstChild); }
function el(name, attrs = {}) {
  const node = document.createElementNS("http://www.w3.org/2000/svg", name);
  for (const [k, v] of Object.entries(attrs)) node.setAttribute(k, v);
  return node;
}
function drawPitchLines() {
  clear(pitchLines);
  pitchLines.appendChild(el("rect", {x: 0, y: 0, width: 105, height: 68, class: "pitch-line"}));
  pitchLines.appendChild(el("line", {x1: 52.5, y1: 0, x2: 52.5, y2: 68, class: "pitch-line"}));
  pitchLines.appendChild(el("circle", {cx: 52.5, cy: 34, r: 9.15, class: "pitch-line"}));
  pitchLines.appendChild(el("circle", {cx: 52.5, cy: 34, r: 0.45, fill: "rgba(80,94,80,0.45)"}));
  for (const side of [0, 1]) {
    const left = side === 0;
    const goalX = left ? 0 : 105;
    const boxX = left ? 0 : 105 - 16.5;
    const sixX = left ? 0 : 105 - 5.5;
    pitchLines.appendChild(el("rect", {x: boxX, y: 34 - 20.16, width: 16.5, height: 40.32, class: "pitch-line"}));
    pitchLines.appendChild(el("rect", {x: sixX, y: 34 - 9.16, width: 5.5, height: 18.32, class: "pitch-line"}));
    pitchLines.appendChild(el("circle", {cx: left ? 11 : 94, cy: 34, r: 0.45, fill: "rgba(80,94,80,0.45)"}));
    pitchLines.appendChild(el("rect", {x: left ? -1.2 : 105, y: 34 - 3.66, width: 1.2, height: 7.32, class: "pitch-line"}));
  }
}
function table(rows, columns) {
  if (!rows || rows.length === 0) return "<tbody><tr><td>No data</td></tr></tbody>";
  const head = "<thead><tr>" + columns.map(c => `<th>${c.label}</th>`).join("") + "</tr></thead>";
  const body = "<tbody>" + rows.map(row => "<tr>" + columns.map(c => `<td>${c.format ? c.format(row[c.key], row) : (row[c.key] ?? "-")}</td>`).join("") + "</tr>").join("") + "</tbody>";
  return head + body;
}
function populateSelect() {
  payload.teamOrder.forEach(team => {
    const opt = document.createElement("option");
    opt.value = team;
    opt.textContent = team;
    select.appendChild(opt);
  });
  select.value = currentTeam;
  select.addEventListener("change", () => {
    currentTeam = select.value;
    lockedCell = null;
    drawTeam();
  });
}
function cellById(team, cellId) {
  return payload.teams[team].cells.find(c => c.cell === cellId);
}
function updatePanel(cellId) {
  const teamData = payload.teams[currentTeam];
  const cell = cellById(currentTeam, cellId) || teamData.cells.slice().sort((a,b) => b.xT - a.xT)[0];
  const detail = teamData.details[cell.cell] || {players: [], routes: [], endZones: [], destinationZones: []};
  document.getElementById("team-title").textContent = currentTeam;
  const netText = payload.meta.positiveOnly ? ` (net xT ${fmt(teamData.summary.netX, 2)})` : "";
  document.getElementById("team-summary").textContent =
    `${currentTeam} creates ${fmt(teamData.summary.xT, 2)} ${valueLabel}${netText} from ${teamData.summary.actions.toLocaleString()} pass/carry actions. The orange cell is the selected start zone; red cells show where those moves created danger.`;
  document.getElementById("cell-xT").textContent = fmt(cell.xT, 3);
  document.getElementById("cell-actions").textContent = cell.actions.toLocaleString();
  document.getElementById("cell-rate").textContent = fmt(cell.xTPer100, 3);
  document.getElementById("cell-prog").textContent = fmt(cell.avgProg, 2);
  document.getElementById("players-table").innerHTML = table(detail.players, [
    {key: "name", label: "Player"},
    {key: "xT", label: valueLabel, format: v => fmt(v, 3)},
    {key: "actions", label: "Actions"}
  ]);
  document.getElementById("endzones-table").innerHTML = table(detail.endZones, [
    {key: "cell", label: "End cell"},
    {key: "xT", label: valueLabel, format: v => fmt(v, 3)},
    {key: "actions", label: "Actions"}
  ]);
  document.getElementById("routes-table").innerHTML = table(detail.routes.slice(0, 6), [
    {key: "player", label: "Player"},
    {key: "type", label: "Type"},
    {key: "xT", label: valueLabel, format: v => fmt(v, 3)}
  ]);
  drawDestinationHeat(detail.destinationZones || [], cell);
  drawRoutes(detail.routes || []);
}
function drawDestinationHeat(destinationZones, sourceCell) {
  clear(destinationLayer);
  clear(sourceLayer);
  const zones = (destinationZones || []).filter(zone => Number(zone.xT) > 0);
  const maxDestination = Math.max(...zones.map(zone => Number(zone.xT) || 0), 0.00001);
  zones.forEach(zone => {
    const bounds = cellBounds(zone.xCell, zone.yCell);
    const value = Math.max(Number(zone.xT) || 0, 0);
    const rect = el("rect", {
      x: sx(bounds.x0),
      y: rectY(bounds.y0, bounds.y1),
      width: Math.abs(bounds.x1 - bounds.x0),
      height: rectH(bounds.y0, bounds.y1),
      fill: "#e45745",
      opacity: Math.min(0.82, 0.08 + 0.72 * Math.sqrt(value / maxDestination)),
      class: "destination-cell"
    });
    const title = document.createElementNS("http://www.w3.org/2000/svg", "title");
    title.textContent = `${zone.cell} | destination ${valueLabel} ${fmt(zone.xT,3)} | actions ${zone.actions}`;
    rect.appendChild(title);
    destinationLayer.appendChild(rect);
  });
  if (sourceCell) {
    const outline = el("rect", {
      x: sx(sourceCell.x0),
      y: rectY(sourceCell.y0, sourceCell.y1),
      width: Math.abs(sourceCell.x1 - sourceCell.x0),
      height: rectH(sourceCell.y0, sourceCell.y1),
      class: "source-highlight"
    });
    sourceLayer.appendChild(outline);
  }
}
function drawRoutes(routes) {
  clear(routeLayer);
  routes.slice(0, 8).forEach(route => {
    if ([route.sx, route.sy, route.ex, route.ey].some(v => v === null || v === undefined)) return;
    const klass = route.type === "carry" ? "route-carry" : "route-pass";
    const marker = route.type === "carry" ? "url(#arrow-carry)" : "url(#arrow-pass)";
    routeLayer.appendChild(el("line", {
      x1: sx(route.sx),
      y1: sy(route.sy),
      x2: sx(route.ex),
      y2: sy(route.ey),
      class: klass,
      "marker-end": marker
    }));
  });
}
function drawTeam() {
  clear(heatLayer);
  clear(destinationLayer);
  clear(sourceLayer);
  clear(routeLayer);
  const teamData = payload.teams[currentTeam];
  const maxCell = Math.max(teamData.summary.maxCellX || 0, 0.00001);
  document.getElementById("team-title").textContent = currentTeam;
  const best = teamData.cells.slice().sort((a,b) => b.xT - a.xT)[0];
  teamData.cells.forEach(cell => {
    const rect = el("rect", {
      x: sx(cell.x0),
      y: rectY(cell.y0, cell.y1),
      width: Math.abs(cell.x1 - cell.x0),
      height: rectH(cell.y0, cell.y1),
      fill: "#2f8f46",
      opacity: Math.min(0.86, 0.03 + 0.78 * Math.sqrt(Math.max(cell.xT, 0) / maxCell)),
      class: "cell",
      "data-cell": cell.cell
    });
    const tip = `${cell.cell} | ${valueLabel} ${fmt(cell.xT,3)} | actions ${cell.actions} | xT/100 ${fmt(cell.xTPer100,3)}`;
    const title = document.createElementNS("http://www.w3.org/2000/svg", "title");
    title.textContent = tip;
    rect.appendChild(title);
    rect.addEventListener("mouseenter", () => { if (!lockedCell) updatePanel(cell.cell); });
    rect.addEventListener("click", () => {
      lockedCell = cell.cell;
      document.querySelectorAll(".cell").forEach(n => n.classList.remove("selected"));
      rect.classList.add("selected");
      updatePanel(cell.cell);
    });
    heatLayer.appendChild(rect);
  });
  updatePanel(lockedCell || best.cell);
}
drawPitchLines();
populateSelect();
drawTeam();
</script>
</body>
</html>
"""
    )


def write_team_danger_zone_html(
    scored: pd.DataFrame,
    *,
    output_path: str | Path = DEFAULT_OUTPUT,
    season_name: str | None = "2025",
    value_col: str = "custom_xT_added",
    l: int = 16,
    w: int = 12,
    positive_only: bool = True,
    max_destination_zones: int | None = 24,
    player_name_map: dict[str, str] | None = None,
    title: str | None = None,
) -> Path:
    payload = build_team_danger_zone_payload(
        scored,
        season_name=season_name,
        value_col=value_col,
        l=l,
        w=w,
        positive_only=positive_only,
        max_destination_zones=max_destination_zones,
        player_name_map=player_name_map,
    )
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    page_title = title or f"Team danger zones | {payload['meta']['season']}"
    payload_json = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    html = _html_template().safe_substitute(
        title=page_title,
        payload=payload_json,
    )
    output_path.write_text(html, encoding="utf-8")
    return output_path


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a standalone team danger-zone xT HTML.")
    parser.add_argument("--scored-path", type=Path, default=Path("tmp/data/bepro_drive_xthreat_k1/scored_actions_set_piece_included.parquet"))
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--season", default="2025")
    parser.add_argument("--value-col", default="custom_xT_added")
    parser.add_argument("--grid-l", type=int, default=16)
    parser.add_argument("--grid-w", type=int, default=12)
    parser.add_argument("--destination-zones", type=int, default=24, help="Maximum destination heatmap zones stored per start cell.")
    parser.add_argument("--net", action="store_true", help="Use net xT-added instead of positive xT-created.")
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    scored = pd.read_parquet(args.scored_path)
    path = write_team_danger_zone_html(
        scored,
        output_path=args.output,
        season_name=args.season,
        value_col=args.value_col,
        l=args.grid_l,
        w=args.grid_w,
        positive_only=not args.net,
        max_destination_zones=args.destination_zones,
    )
    print(path)


if __name__ == "__main__":
    main()
