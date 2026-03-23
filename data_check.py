#!/usr/bin/env python3
"""Validate local goal annotations against external goal events.

This script is designed for SoccerNet-style folders:
  dataset/<league>/<season>/<match>/Labels-v2.json

Supported providers:
  1) statsbomb-open-data (offline, local JSON files)
  2) api-football (RapidAPI)

By default it uses statsbomb-open-data if ``open-data`` exists,
otherwise falls back to api-football.

API-Football endpoints used:
  - GET /fixtures?date=YYYY-MM-DD
  - GET /fixtures/events?fixture=<fixture_id>

Required environment variables for API-Football mode:
  - RAPIDAPI_KEY
  - RAPIDAPI_HOST (default: v3.football.api-sports.io)

Example:
  python data_check.py \
      --dataset-root dataset \
      --out report_goal_check.csv \
      --max-matches 50
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

MATCH_DIR_RE = re.compile(
    r"^(?P<date>\d{4}-\d{2}-\d{2})\s-\s\d{2}-\d{2}\s"
    r"(?P<home>.+?)\s\d+\s-\s\d+\s(?P<away>.+)$"
)


@dataclass
class AnnotatedGoal:
    half: int
    minute_in_half: int
    second_in_minute: int
    absolute_minute: float
    raw_game_time: str


@dataclass
class ApiGoal:
    elapsed: int
    extra: int
    absolute_minute: int
    team: str


@dataclass
class MatchCheckResult:
    match_dir: str
    date: str
    home: str
    away: str
    annotation_goals: int
    api_goals: int
    matched_goals: int
    unmatched_annotations: int
    unmatched_api_goals: int
    max_abs_minute_diff: float
    status: str
    notes: str


class APIFootballClient:
    def __init__(self, api_key: str, host: str, timeout_sec: float = 15.0, sleep_sec: float = 0.25):
        self.api_key = api_key
        self.host = host
        self.timeout_sec = timeout_sec
        self.sleep_sec = sleep_sec

    def _request_json(self, path: str, params: Dict[str, Any]) -> Dict[str, Any]:
        url = f"https://{self.host}{path}?{urlencode(params)}"
        req = Request(
            url,
            headers={
                "x-rapidapi-key": self.api_key,
                "x-rapidapi-host": self.host,
            },
            method="GET",
        )
        try:
            with urlopen(req, timeout=self.timeout_sec) as resp:
                body = resp.read().decode("utf-8")
                return json.loads(body)
        except HTTPError as exc:
            raise RuntimeError(f"HTTP {exc.code} calling {url}") from exc
        except URLError as exc:
            raise RuntimeError(f"Network error calling {url}: {exc}") from exc

    def find_fixture(self, date_str: str, home: str, away: str) -> Optional[int]:
        data = self._request_json("/fixtures", {"date": date_str})
        fixtures = data.get("response", [])
        if not fixtures:
            return None

        home_fold = fold_team_name(home)
        away_fold = fold_team_name(away)

        # First pass: exact normalized name match.
        for item in fixtures:
            h = fold_team_name(item.get("teams", {}).get("home", {}).get("name", ""))
            a = fold_team_name(item.get("teams", {}).get("away", {}).get("name", ""))
            if h == home_fold and a == away_fold:
                return item.get("fixture", {}).get("id")

        # Second pass: partial containment to handle abbreviations (e.g. "Atl. Madrid").
        for item in fixtures:
            h = fold_team_name(item.get("teams", {}).get("home", {}).get("name", ""))
            a = fold_team_name(item.get("teams", {}).get("away", {}).get("name", ""))
            if names_loosely_match(home_fold, h) and names_loosely_match(away_fold, a):
                return item.get("fixture", {}).get("id")

        return None

    def get_goals(self, fixture_id: int) -> List[ApiGoal]:
        data = self._request_json("/fixtures/events", {"fixture": fixture_id})
        events = data.get("response", [])
        out: List[ApiGoal] = []
        for event in events:
            if str(event.get("type", "")).lower() != "goal":
                continue

            elapsed = int((event.get("time", {}) or {}).get("elapsed") or 0)
            extra = int((event.get("time", {}) or {}).get("extra") or 0)
            # Keep absolute minute simple: 45+2 => 47.
            abs_minute = elapsed + max(0, extra)
            team = str((event.get("team", {}) or {}).get("name") or "")
            out.append(ApiGoal(elapsed=elapsed, extra=extra, absolute_minute=abs_minute, team=team))

        # Respect API rate limit.
        time.sleep(self.sleep_sec)
        return out


class StatsBombOpenDataClient:
    def __init__(self, open_data_root: Path):
        self.open_data_root = open_data_root
        self.matches_by_date: Dict[str, List[Dict[str, Any]]] = {}
        self.events_dir = self.open_data_root / "data" / "events"
        self._load_match_index()

    def _load_match_index(self) -> None:
        matches_root = self.open_data_root / "data" / "matches"
        if not matches_root.exists():
            raise RuntimeError(f"StatsBomb matches folder not found: {matches_root}")

        for season_file in sorted(matches_root.glob("*/*.json")):
            try:
                rows = json.loads(season_file.read_text(encoding="utf-8"))
            except Exception:
                continue

            for m in rows:
                date = str(m.get("match_date", ""))
                if not date:
                    continue
                self.matches_by_date.setdefault(date, []).append(m)

    def find_fixture(
        self, date_str: str, home: str, away: str, league: str = "", season: str = ""
    ) -> Optional[int]:
        candidates = self.matches_by_date.get(date_str, [])
        if not candidates:
            return None

        home_fold = fold_team_name(home)
        away_fold = fold_team_name(away)
        league_fold = fold_league_name(league)

        # First pass: exact-ish name + optional league hint.
        for m in candidates:
            m_home = fold_team_name((m.get("home_team") or {}).get("home_team_name", ""))
            m_away = fold_team_name((m.get("away_team") or {}).get("away_team_name", ""))
            m_league = fold_league_name(((m.get("competition") or {}).get("competition_name") or ""))

            league_ok = not league_fold or not m_league or league_fold == m_league
            if league_ok and m_home == home_fold and m_away == away_fold:
                return int(m["match_id"])

        # Second pass: loose name match.
        for m in candidates:
            m_home = fold_team_name((m.get("home_team") or {}).get("home_team_name", ""))
            m_away = fold_team_name((m.get("away_team") or {}).get("away_team_name", ""))
            m_league = fold_league_name(((m.get("competition") or {}).get("competition_name") or ""))
            league_ok = not league_fold or not m_league or league_fold == m_league
            if league_ok and names_loosely_match(home_fold, m_home) and names_loosely_match(away_fold, m_away):
                return int(m["match_id"])

        return None

    def get_goals(self, fixture_id: int) -> List[ApiGoal]:
        event_file = self.events_dir / f"{fixture_id}.json"
        if not event_file.exists():
            return []

        try:
            events = json.loads(event_file.read_text(encoding="utf-8"))
        except Exception as exc:
            raise RuntimeError(f"Cannot parse events file: {event_file}") from exc

        out: List[ApiGoal] = []
        for e in events:
            event_type = str(((e.get("type") or {}).get("name")) or "")
            is_goal = False

            if event_type == "Shot":
                outcome = (((e.get("shot") or {}).get("outcome")) or {}).get("name", "")
                if outcome == "Goal":
                    is_goal = True
            elif event_type == "Own Goal For":
                is_goal = True

            if not is_goal:
                continue

            minute = int(e.get("minute") or 0)
            second = int(e.get("second") or 0)
            abs_minute = minute + (1 if second >= 30 else 0)
            team = str(((e.get("team") or {}).get("name")) or "")
            out.append(ApiGoal(elapsed=minute, extra=0, absolute_minute=abs_minute, team=team))

        return out


def fold_team_name(name: str) -> str:
    text = name.lower().strip()
    replacements = {
        "utd": "united",
        "st ": "saint ",
        "st.": "saint",
        "atl.": "atletico",
        "dep.": "deportivo",
        "sg": "saint germain",
        "dyn.": "dynamo",
        "b.": "borussia",
        "fc ": "",
        " cf ": " ",
        " afc ": " ",
        " cfc ": " ",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    text = re.sub(r"[^a-z0-9 ]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def fold_league_name(name: str) -> str:
    text = name.lower().strip()
    replacements = {
        "england_epl": "premier league",
        "spain_laliga": "la liga",
        "germany_bundesliga": "1 bundesliga",
        "italy_serie-a": "serie a",
        "europe_uefa-champions-league": "champions league",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    text = re.sub(r"[^a-z0-9 ]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def names_loosely_match(a: str, b: str) -> bool:
    if not a or not b:
        return False
    if a == b:
        return True
    if a in b or b in a:
        return True
    a_tokens = set(a.split())
    b_tokens = set(b.split())
    return len(a_tokens & b_tokens) >= max(1, min(len(a_tokens), len(b_tokens)) - 1)


def parse_match_dir_name(name: str) -> Tuple[str, str, str]:
    m = MATCH_DIR_RE.match(name)
    if not m:
        raise ValueError(f"Cannot parse match folder name: {name}")
    return m.group("date"), m.group("home"), m.group("away")


def parse_annotation_goal(game_time: str) -> AnnotatedGoal:
    # Example: "2 - 10:35"
    parts = [p.strip() for p in game_time.split("-")]
    if len(parts) != 2:
        raise ValueError(f"Invalid gameTime format: {game_time}")

    half = int(parts[0])
    mm, ss = parts[1].split(":")
    minute = int(mm)
    sec = int(ss)

    # SoccerNet convention: half-relative clock.
    absolute_min = (half - 1) * 45 + minute + sec / 60.0
    return AnnotatedGoal(
        half=half,
        minute_in_half=minute,
        second_in_minute=sec,
        absolute_minute=absolute_min,
        raw_game_time=game_time,
    )


def load_annotation_goals(label_file: Path) -> List[AnnotatedGoal]:
    data = json.loads(label_file.read_text(encoding="utf-8"))
    anns = data.get("annotations", [])
    goals: List[AnnotatedGoal] = []
    for ann in anns:
        if str(ann.get("label", "")).lower() != "goal":
            continue
        gt = str(ann.get("gameTime", "")).strip()
        if not gt:
            continue
        try:
            goals.append(parse_annotation_goal(gt))
        except Exception:
            # Keep processing even if one annotation is malformed.
            continue
    return sorted(goals, key=lambda x: x.absolute_minute)


def greedy_match(
    ann: Sequence[AnnotatedGoal], api: Sequence[ApiGoal], tolerance_min: float
) -> Tuple[int, float, List[int], List[int]]:
    api_minutes = [g.absolute_minute for g in api]
    used_api: set[int] = set()
    matched = 0
    max_diff = 0.0
    unmatched_ann: List[int] = []

    for i, a in enumerate(ann):
        best_j = None
        best_diff = float("inf")
        for j, m in enumerate(api_minutes):
            if j in used_api:
                continue
            d = abs(a.absolute_minute - m)
            if d <= tolerance_min and d < best_diff:
                best_j = j
                best_diff = d

        if best_j is None:
            unmatched_ann.append(i)
            continue

        used_api.add(best_j)
        matched += 1
        if best_diff > max_diff:
            max_diff = best_diff

    unmatched_api = [j for j in range(len(api_minutes)) if j not in used_api]
    return matched, max_diff, unmatched_ann, unmatched_api


def collect_label_files(dataset_root: Path) -> List[Path]:
    return sorted(dataset_root.rglob("Labels-v2.json"))


def check_match(
    label_file: Path,
    client: Any,
    tolerance_min: float,
    league_hint: str = "",
    season_hint: str = "",
) -> MatchCheckResult:
    match_dir = label_file.parent
    date_str, home, away = parse_match_dir_name(match_dir.name)

    ann_goals = load_annotation_goals(label_file)
    try:
        fixture_id = client.find_fixture(date_str, home, away, league_hint, season_hint)
    except TypeError:
        fixture_id = client.find_fixture(date_str, home, away)
    if fixture_id is None:
        return MatchCheckResult(
            match_dir=str(match_dir),
            date=date_str,
            home=home,
            away=away,
            annotation_goals=len(ann_goals),
            api_goals=0,
            matched_goals=0,
            unmatched_annotations=len(ann_goals),
            unmatched_api_goals=0,
            max_abs_minute_diff=0.0,
            status="fixture_not_found",
            notes="Could not match fixture in external source for this date/home/away",
        )

    api_goals = sorted(client.get_goals(fixture_id), key=lambda g: g.absolute_minute)
    matched, max_diff, unmatched_ann, unmatched_api = greedy_match(
        ann_goals, api_goals, tolerance_min=tolerance_min
    )

    status = "ok"
    if unmatched_ann or unmatched_api:
        status = "mismatch"

    note_bits = []
    if unmatched_ann:
        note_bits.append(f"unmatched_annotation_idxs={unmatched_ann}")
    if unmatched_api:
        note_bits.append(f"unmatched_api_goal_idxs={unmatched_api}")

    return MatchCheckResult(
        match_dir=str(match_dir),
        date=date_str,
        home=home,
        away=away,
        annotation_goals=len(ann_goals),
        api_goals=len(api_goals),
        matched_goals=matched,
        unmatched_annotations=len(unmatched_ann),
        unmatched_api_goals=len(unmatched_api),
        max_abs_minute_diff=max_diff,
        status=status,
        notes="; ".join(note_bits),
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Check goal annotations against external goal events")
    p.add_argument("--dataset-root", type=Path, default=Path("dataset"), help="Root dataset directory")
    p.add_argument(
        "--tolerance-minutes",
        type=float,
        default=2.0,
        help="Allowed absolute minute diff when matching annotation goal vs API goal",
    )
    p.add_argument("--max-matches", type=int, default=0, help="Limit number of matches (0 = all)")
    p.add_argument("--out", type=Path, default=Path("goal_annotation_check.csv"), help="Output CSV")
    p.add_argument(
        "--only-with-goal-annotations",
        action="store_true",
        help="Skip matches that have zero local goal annotations",
    )
    p.add_argument(
        "--provider",
        choices=["auto", "statsbomb-open-data", "api-football"],
        default="auto",
        help="Data source for goal events. 'auto' uses statsbomb-open-data if available.",
    )
    p.add_argument(
        "--open-data-root",
        type=Path,
        default=Path("open-data"),
        help="Path to StatsBomb open-data repository root",
    )
    p.add_argument(
        "--rapidapi-host",
        default=os.getenv("RAPIDAPI_HOST", "v3.football.api-sports.io"),
        help="RapidAPI host for API-Football",
    )
    p.add_argument(
        "--rapidapi-key",
        default=os.getenv("RAPIDAPI_KEY", ""),
        help="RapidAPI key (or use RAPIDAPI_KEY env var)",
    )
    return p.parse_args()


def write_csv(path: Path, rows: Sequence[MatchCheckResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "match_dir",
                "date",
                "home",
                "away",
                "annotation_goals",
                "api_goals",
                "matched_goals",
                "unmatched_annotations",
                "unmatched_api_goals",
                "max_abs_minute_diff",
                "status",
                "notes",
            ]
        )
        for r in rows:
            w.writerow(
                [
                    r.match_dir,
                    r.date,
                    r.home,
                    r.away,
                    r.annotation_goals,
                    r.api_goals,
                    r.matched_goals,
                    r.unmatched_annotations,
                    r.unmatched_api_goals,
                    f"{r.max_abs_minute_diff:.2f}",
                    r.status,
                    r.notes,
                ]
            )


def main() -> int:
    args = parse_args()

    if not args.dataset_root.exists():
        print(f"Dataset root not found: {args.dataset_root}", file=sys.stderr)
        return 2

    provider = args.provider
    if provider == "auto":
        provider = "statsbomb-open-data" if args.open_data_root.exists() else "api-football"

    if provider == "statsbomb-open-data":
        client = StatsBombOpenDataClient(args.open_data_root)
    else:
        if not args.rapidapi_key:
            print(
                "Missing API key. Set RAPIDAPI_KEY env var or pass --rapidapi-key.",
                file=sys.stderr,
            )
            return 2
        client = APIFootballClient(api_key=args.rapidapi_key, host=args.rapidapi_host)

    label_files = collect_label_files(args.dataset_root)
    if not label_files:
        print("No Labels-v2.json files found.", file=sys.stderr)
        return 2

    if args.max_matches > 0:
        label_files = label_files[: args.max_matches]

    rows: List[MatchCheckResult] = []
    processed = 0

    for lf in label_files:
        if args.only_with_goal_annotations:
            goals = load_annotation_goals(lf)
            if not goals:
                continue

        processed += 1
        try:
            rel = lf.relative_to(args.dataset_root)
            league_hint = rel.parts[0] if len(rel.parts) > 0 else ""
            season_hint = rel.parts[1] if len(rel.parts) > 1 else ""
            result = check_match(
                lf,
                client,
                tolerance_min=args.tolerance_minutes,
                league_hint=league_hint,
                season_hint=season_hint,
            )
        except Exception as exc:
            result = MatchCheckResult(
                match_dir=str(lf.parent),
                date="",
                home="",
                away="",
                annotation_goals=0,
                api_goals=0,
                matched_goals=0,
                unmatched_annotations=0,
                unmatched_api_goals=0,
                max_abs_minute_diff=0.0,
                status="error",
                notes=str(exc),
            )

        rows.append(result)

        if processed % 20 == 0:
            print(f"Processed {processed} matches...")

    write_csv(args.out, rows)

    mismatches = sum(1 for r in rows if r.status == "mismatch")
    fixture_missing = sum(1 for r in rows if r.status == "fixture_not_found")
    errors = sum(1 for r in rows if r.status == "error")

    print(f"Done. Checked {len(rows)} matches.")
    print(f"Provider: {provider}")
    print(f"Output: {args.out}")
    print(f"Mismatches: {mismatches}")
    print(f"Fixture not found: {fixture_missing}")
    print(f"Errors: {errors}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
