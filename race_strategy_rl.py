from __future__ import annotations
import argparse
import copy
import hashlib
import json
import math
import os
import random
import time
from collections import Counter
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from matplotlib.lines import Line2D

# =============================================================================
# USER CONFIG
# =============================================================================

# start_prob_M:
#   probability that the driver starts on Medium in a given episode.
#   start compound is sampled once per episode and then fixed for the whole race.
DRIVER_CONFIGS: List[Dict] = [
    {"name": "Piastri",    "grid_pos": 5,  "delta": 0.28, "inventory_M": 2, "inventory_H": 1, "is_ego": True,  "rank_weights": (0.70, 0.20, 0.08, 0.02), "start_prob_M": 1.00, "can_retire": False},
    {"name": "Norris",     "grid_pos": 6,  "delta": 0.30, "inventory_M": 2, "inventory_H": 1, "is_ego": False, "rank_weights": (0.72, 0.18, 0.08, 0.02), "start_prob_M": 1.00, "can_retire": False},
    {"name": "Leclerc",    "grid_pos": 1,  "delta": 0.27, "inventory_M": 2, "inventory_H": 1, "is_ego": False, "rank_weights": (0.65, 0.23, 0.08, 0.04), "start_prob_M": 1.00, "can_retire": False},
    {"name": "Hamilton",   "grid_pos": 2,  "delta": 0.36, "inventory_M": 2, "inventory_H": 1, "is_ego": False, "rank_weights": (0.60, 0.25, 0.10, 0.05), "start_prob_M": 1.00, "can_retire": False},
    {"name": "Verstappen", "grid_pos": 7,  "delta": 0.50, "inventory_M": 1, "inventory_H": 2, "is_ego": False, "rank_weights": (0.74, 0.18, 0.06, 0.02), "start_prob_M": 0.80, "can_retire": False},
    {"name": "Russell",    "grid_pos": 3,  "delta": 0.1, "inventory_M": 2, "inventory_H": 1, "is_ego": False, "rank_weights": (0.68, 0.20, 0.08, 0.04), "start_prob_M": 1.00, "can_retire": False},
    {"name": "Antonelli",  "grid_pos": 4,  "delta": 0.00, "inventory_M": 2, "inventory_H": 1, "is_ego": False, "rank_weights": (0.62, 0.24, 0.10, 0.04), "start_prob_M": 1.00, "can_retire": False},
    {"name": "Alonso",     "grid_pos": 18, "delta": 2.55, "inventory_M": 1, "inventory_H": 2, "is_ego": False, "rank_weights": (0.56, 0.26, 0.12, 0.06), "start_prob_M": 0.40, "can_retire": True},
    {"name": "Stroll",     "grid_pos": 19, "delta": 2.95, "inventory_M": 1, "inventory_H": 2, "is_ego": False, "rank_weights": (0.50, 0.28, 0.14, 0.08), "start_prob_M": 0.70, "can_retire": True},
    {"name": "Gasly",      "grid_pos": 10, "delta": 1.05, "inventory_M": 1, "inventory_H": 2, "is_ego": False, "rank_weights": (0.54, 0.26, 0.14, 0.06), "start_prob_M": 0.90, "can_retire": True},
    {"name": "Colapinto",  "grid_pos": 11, "delta": 1.15, "inventory_M": 1, "inventory_H": 2, "is_ego": False, "rank_weights": (0.52, 0.28, 0.14, 0.06), "start_prob_M": 0.60, "can_retire": True},
    {"name": "Albon",      "grid_pos": 17, "delta": 1.90, "inventory_M": 1, "inventory_H": 2, "is_ego": False, "rank_weights": (0.56, 0.24, 0.12, 0.08), "start_prob_M": 0.50, "can_retire": True},
    {"name": "Sainz",      "grid_pos": 16, "delta": 1.58, "inventory_M": 1, "inventory_H": 2, "is_ego": False, "rank_weights": (0.58, 0.22, 0.12, 0.08), "start_prob_M": 0.30, "can_retire": True},
    {"name": "Ocon",       "grid_pos": 20, "delta": 1.35, "inventory_M": 1, "inventory_H": 2, "is_ego": False, "rank_weights": (0.54, 0.25, 0.13, 0.08), "start_prob_M": 0.30, "can_retire": True},
    {"name": "Bearman",    "grid_pos": 9,  "delta": 1.02, "inventory_M": 1, "inventory_H": 2, "is_ego": False, "rank_weights": (0.50, 0.27, 0.15, 0.08), "start_prob_M": 0.90, "can_retire": True},
    {"name": "Hulkenberg", "grid_pos": 12, "delta": 1.22, "inventory_M": 2, "inventory_H": 1, "is_ego": False, "rank_weights": (0.55, 0.25, 0.13, 0.07), "start_prob_M": 0.70, "can_retire": True},
    {"name": "Bortoleto",  "grid_pos": 13, "delta": 1.17, "inventory_M": 2, "inventory_H": 1, "is_ego": False, "rank_weights": (0.50, 0.27, 0.15, 0.08), "start_prob_M": 0.80, "can_retire": True},
    {"name": "Lawson",     "grid_pos": 14, "delta": 1.25, "inventory_M": 2, "inventory_H": 1, "is_ego": False, "rank_weights": (0.54, 0.26, 0.12, 0.08), "start_prob_M": 0.70, "can_retire": True},
    {"name": "Lindblad",   "grid_pos": 15, "delta": 1.28, "inventory_M": 2, "inventory_H": 1, "is_ego": False, "rank_weights": (0.48, 0.28, 0.16, 0.08), "start_prob_M": 0.70, "can_retire": True},
    {"name": "Hadjar",     "grid_pos": 8,  "delta": 0.90, "inventory_M": 1, "inventory_H": 2, "is_ego": False, "rank_weights": (0.50, 0.27, 0.15, 0.08), "start_prob_M": 0.90, "can_retire": True},
    {"name": "Bottas",     "grid_pos": 21, "delta": 2.60, "inventory_M": 2, "inventory_H": 1, "is_ego": False, "rank_weights": (0.52, 0.26, 0.14, 0.08), "start_prob_M": 0.70, "can_retire": True},
    {"name": "Perez",      "grid_pos": 22, "delta": 2.54, "inventory_M": 2, "inventory_H": 1, "is_ego": False, "rank_weights": (0.50, 0.27, 0.15, 0.08), "start_prob_M": 0.70, "can_retire": True},
]

POINTS_BY_POSITION = {1: 25, 2: 18, 3: 15, 4: 12, 5: 10, 6: 8, 7: 6, 8: 4, 9: 2, 10: 1}

TEAM_COLORS = {
    "MCLAREN": "#FF9933",
    "FERRARI": "#FF0000",
    "MERCEDES": "#33CCCC",
    "RED BULL": "#9900CC",
    "RACING BULLS": "#3333FF",
    "ASTON MARTIN": "#008000",
    "ALPINE": "#FF66FF",
    "WILLIAMS": "#0099FF",
    "HAAS": "#808080",
    "SAUBER": "#00FF00",
    "CADILLAC": "#7A4E2D",
}

DRIVER_TO_TEAM = {
    "Piastri": "MCLAREN",
    "Norris": "MCLAREN",
    "Leclerc": "FERRARI",
    "Hamilton": "FERRARI",
    "Verstappen": "RED BULL",
    "Russell": "MERCEDES",
    "Antonelli": "MERCEDES",
    "Hadjar": "RED BULL",
    "Lawson": "RACING BULLS",
    "Lindblad": "RACING BULLS",
    "Alonso": "ASTON MARTIN",
    "Stroll": "ASTON MARTIN",
    "Gasly": "ALPINE",
    "Colapinto": "ALPINE",
    "Albon": "WILLIAMS",
    "Sainz": "WILLIAMS",
    "Ocon": "HAAS",
    "Bearman": "HAAS",
    "Hulkenberg": "SAUBER",
    "Bortoleto": "SAUBER",
    "Bottas": "CADILLAC",
    "Perez": "CADILLAC",
}

TOP_TEAM_DRIVERS = {"Piastri", "Norris", "Leclerc", "Hamilton", "Russell", "Antonelli", "Verstappen"}

# =============================================================================
# BASIC HELPERS
# =============================================================================

def sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def clip(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def normal_pos(rng: np.random.Generator, mu: float, sigma: float, min_val: float = 1.0) -> float:
    return float(max(min_val, float(mu + rng.normal(0.0, sigma))))


def compound_offset(q: str, Delta_HM: float) -> float:
    return 0.0 if q == "M" else float(Delta_HM)


def compound_deg(q: str, dM: float, dH: float) -> float:
    return float(dM) if q == "M" else float(dH)


def points_for_position(pos: int) -> float:
    return float(POINTS_BY_POSITION.get(int(pos), 0.0))


# =============================================================================
# EPISODE SCENARIO
# =============================================================================

@dataclass(frozen=True)
class NeutralWindow:
    kind: str                   # "VSC" or "SC"
    start_time: float
    end_time: float
    pre_bunch_end_time: Optional[float] = None   # for SC only
    retired_driver: Optional[str] = None

    def active_at(self, t: float) -> bool:
        return bool(self.start_time <= t < self.end_time)

    def regime_at(self, t: float) -> str:
        if not self.active_at(t):
            return "GREEN"
        if self.kind == "VSC":
            return "VSC"
        if self.pre_bunch_end_time is None:
            return "SC_BUNCH"
        return "SC_PRE" if t < self.pre_bunch_end_time else "SC_BUNCH"


@dataclass
class EpisodeScenario:
    Delta_HM: float
    dM: float
    dH: float

    # name -> realised episode-specific pace offset delta_i^(e)
    pace_offsets: Dict[str, float]

    # name -> sampled perturbation epsilon_i^(e)
    pace_perturbations: Dict[str, float]

    start_compounds: Dict[str, str]
    neutral_windows: List[NeutralWindow]
    retirements: Dict[str, float]
    competitor_styles: Dict[str, Dict[str, object]]
    episode_inventories: Dict[str, Tuple[int, int]]   # name -> (inventory_M, inventory_H)


@dataclass
class DriverStatic:
    name: str
    grid_pos: int
    delta_i: float
    inventory_M: int
    inventory_H: int
    rank_weights: Tuple[float, ...]
    start_prob_M: float
    can_retire: bool
    is_ego: bool = False


@dataclass
class RankedStrategy:
    total_time: float
    total_deg_time: float
    total_compound_offset: float
    total_pit_loss: float
    stops: int
    stint_lengths: Tuple[int, ...]
    stint_compounds: Tuple[str, ...]
    pit_laps: Tuple[int, ...]
    sets_used_M: int
    sets_used_H: int


@dataclass
class CarState:
    car_id: int
    name: str
    grid_pos: int
    delta_i: float
    is_ego: bool
    inventory_M_total: int
    inventory_H_total: int
    can_retire: bool

    action_by_lap: List[str] = field(default_factory=list)

    active: bool = True
    retired_at: Optional[float] = None

    n: int = 0
    T_last: float = 0.0
    T_next: float = math.inf

    q_end: str = "M"
    a_end: int = 0
    q_in: str = "M"
    a_in: int = 0

    next_kind: str = "CROSS"      # CROSS, PIT_RELEASE, WAIT_EGO_ACTION
    pit_loss_this_lap: float = 0.0
    t_progress_start: float = 0.0
    drive_mean: float = 0.0

    remaining_M: int = 0
    remaining_H: int = 0
    used_M: bool = False
    used_H: bool = False
    pit_stops_done: int = 0
    last_pit_lap: int = 0

    lap_times: List[float] = field(default_factory=list)
    comps_end: List[str] = field(default_factory=list)
    ages_end: List[int] = field(default_factory=list)
    cum_times_after_lap: List[float] = field(default_factory=list)
    green_no_pit_until_lap: int = 0


@dataclass
class RaceConfig:
    L: int = 60
    B: float = 76.0
    seed: int = 7

    Delta_HM_values: Tuple[float, ...] = (0.0, 0.1, 0.2)
    dM_values: Tuple[float, ...] = (0.05, 0.06, 0.07, 0.08, 0.09)
    dH_values: Tuple[float, ...] = (0.02, 0.03, 0.04, 0.05)

    Pbar: float = 20.0
    sigma_P: float = 0.25
    pit_loss_mult_vsc: float = 0.65
    pit_loss_mult_sc: float = 0.50

    sigma_tau: float = 0.16
    sigma_tau_close: float = 0.45
    g_close: float = 1.20

    g_min: float = 0.25
    g_pass: float = 0.20
    Delta_OT: float = 0.50
    g_buf: float = 0.40
    A0: float = 0.30
    beta_pass: float = 7.0
    kappa_dirty: float = 0.35
    h_dirty: float = 1.20
    h_yield: float = 0.80
    beta_blue: float = 10.0
    Delta_blue: float = 0.20
    A_blue: float = 0.10

    lap1_npasses: int = 6
    lap1_pswap: float = 0.30
    lap1_gap_lo: float = 0.30
    lap1_gap_hi: float = 1.00
    lap1_start_loss_mu: float = 2.50
    lap1_start_loss_sd: float = 0.80

    competitor_top_k: int = 7
    max_stops: int = 2
    min_stint: int = 6
    no_pit_last_n: int = 1
    mandatory_both: bool = True

    ego_max_stops: int = 2
    ego_min_stint: int = 5
    ego_no_pit_last_n: int = 5

    invalid_no_both_penalty: float = 50.0
    other_compound_bonus: float = 10.0
    position_reward_scale: float = 2.0
    trace_ref_time: float = 81.0

    vsc_mult: float = 1.18
    sc_pre_mult: float = 1.28
    sc_bunch_mult: float = 1.38

    vsc_prob_per_lap: float = 0.006
    sc_prob_per_lap: float = 0.014
    sc_after_lap1_episode_prob: float = 0.10
    vsc_min_duration: float = 100.0
    vsc_max_duration: float = 300.0
    sc_min_duration: float = 350.0
    sc_max_duration: float = 650.0
    sc_pre_bunch_duration: float = 130.0

    no_neutral_pit_first_n: int = 5
    no_neutral_pit_last_n: int = 5
    no_pit_lap1: bool = True

    deterministic_eval_episodes: int = 500
    deterministic_trace_episodes: int = 10
    ppo_trace_episodes: int = 20

    # Safety Car bunching controls
    sc_target_gap_p2: float = 0.60
    sc_target_gap_step: float = 0.95
    sigma_sc: float = 0.18
    sc_min_lap_time: float = 85.0
    minority_start_on_asymmetric_inventory_prob: float = 0.0

    # Inventory bias from sampled degradation regime
    inventory_bias_medium_deg_gap_high: float = 0.025
    inventory_bias_medium_deg_gap_similar: float = 0.012
    inventory_bias_prob_twoH_high_medium_deg: float = 0.80
    inventory_bias_prob_twoM_similar_deg: float = 0.80

    # -------------------------------------------------------------------------
    # Competitor-strategy variability
    # -------------------------------------------------------------------------
    top_team_strategy_variability_prob: float = 0.65
    non_top_team_strategy_variability_prob: float = 0.10

    top_team_extend_prob: float = 0.30
    top_team_undercut_prob: float = 0.30

    strategy_extend_laps_min: int = 8
    strategy_extend_laps_max: int = 12
    strategy_undercut_laps_min: int = 2
    strategy_undercut_laps_max: int = 4

    strategy_time_slack_extend: float = 8.0
    strategy_time_slack_undercut: float = 5.0

    expected_vsc_replan_laps: int = 2
    expected_sc_replan_laps: int = 5

    # If a car stays out under SC_PRE, require additional green laps
    # before the next pit opportunity.
    sc_post_green_no_pit_laps: int = 10

    # Episode-level car-driver pace perturbation.
    # DRIVER_CONFIGS["delta"] is the central event-calibrated pace offset.
    # In each episode we sample:
    #   delta_i_episode = delta_i_central + epsilon_i
    # where epsilon_i is drawn from {-0.10, -0.09, ..., 0.10}
    # with Gaussian-shaped discrete weights centred at zero.
    pace_delta_perturb_min: float = -0.10
    pace_delta_perturb_max: float = 0.10
    pace_delta_perturb_step: float = 0.01
    pace_delta_perturb_sigma: float = 0.07

    # Observation ablation used in the paper's ablation study.
    # One of: none, no_local_competitor, no_ego_degradation,
    # no_laps_remaining, no_pit_vulnerability, no_neutral_indicators.
    observation_ablation: str = "none"

# =============================================================================
# EPISODE-SCENARIO SAMPLING
# =============================================================================

def _sample_episode_pace_offsets(
    rng: np.random.Generator,
    cfg: RaceConfig,
    driver_static: List[DriverStatic],
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """
    Sample episode-specific car-driver pace offsets.

    DRIVER_CONFIGS["delta"] is interpreted as the central event-calibrated
    pace offset \\bar{delta}_i.

    In each episode:
        delta_i^(e) = \\bar{delta}_i + epsilon_i^(e)

    where epsilon_i^(e) is sampled from the discrete grid
        {-0.10, -0.09, ..., 0.09, 0.10}
    with Gaussian-shaped discrete weights centred at zero.
    """
    lo = float(cfg.pace_delta_perturb_min)
    hi = float(cfg.pace_delta_perturb_max)
    step = float(cfg.pace_delta_perturb_step)
    sigma = float(cfg.pace_delta_perturb_sigma)

    if step <= 0.0:
        raise ValueError("pace_delta_perturb_step must be positive.")
    if sigma <= 0.0:
        raise ValueError("pace_delta_perturb_sigma must be positive.")
    if hi < lo:
        raise ValueError("pace_delta_perturb_max must be >= pace_delta_perturb_min.")

    n_steps = int(round((hi - lo) / step))
    grid = lo + step * np.arange(n_steps + 1, dtype=float)
    grid = np.round(grid, 10)

    weights = np.exp(-0.5 * (grid / sigma) ** 2)
    weights = weights / np.sum(weights)

    pace_offsets: Dict[str, float] = {}
    pace_perturbations: Dict[str, float] = {}

    for d in driver_static:
        eps = float(rng.choice(grid, p=weights))
        pace_perturbations[d.name] = float(eps)
        pace_offsets[d.name] = float(d.delta_i + eps)

    return pace_offsets, pace_perturbations

def sample_episode_scenario(
    rng: np.random.Generator,
    cfg: RaceConfig,
    driver_static: List[DriverStatic],
    allow_ego_variability: bool = False,
) -> EpisodeScenario:
    Delta_HM = float(rng.choice(np.asarray(cfg.Delta_HM_values, dtype=float)))
    dM = float(rng.choice(np.asarray(cfg.dM_values, dtype=float)))
    dH = float(rng.choice(np.asarray(cfg.dH_values, dtype=float)))

    pace_offsets, pace_perturbations = _sample_episode_pace_offsets(
        rng=rng,
        cfg=cfg,
        driver_static=driver_static,
    )

    episode_inventories = _sample_episode_inventories(
        rng=rng,
        cfg=cfg,
        driver_static=driver_static,
        allow_ego_variability=allow_ego_variability,
        dM=dM,
        dH=dH,
    )

    start_compounds = _sample_start_compounds(
        rng=rng,
        cfg=cfg,
        driver_static=driver_static,
        episode_inventories=episode_inventories,
        allow_ego_variability=allow_ego_variability,
    )

    windows, retirements = _sample_neutral_windows(rng, cfg, driver_static)

    competitor_styles = _sample_competitor_styles(
        rng=rng,
        cfg=cfg,
        driver_static=driver_static,
        allow_ego_variability=allow_ego_variability,
    )

    return EpisodeScenario(
        Delta_HM=Delta_HM,
        dM=dM,
        dH=dH,
        pace_offsets=pace_offsets,
        pace_perturbations=pace_perturbations,
        start_compounds=start_compounds,
        neutral_windows=windows,
        retirements=retirements,
        competitor_styles=competitor_styles,
        episode_inventories=episode_inventories,
    )

def _sample_start_compounds(
    rng: np.random.Generator,
    cfg: RaceConfig,
    driver_static: List[DriverStatic],
    episode_inventories: Dict[str, Tuple[int, int]],
    allow_ego_variability: bool,
) -> Dict[str, str]:
    out: Dict[str, str] = {}

    for d in driver_static:
        inv_M, inv_H = episode_inventories[d.name]

        sampled = "M" if float(rng.random()) < float(d.start_prob_M) else "H"

        if inv_M <= 0 and inv_H > 0:
            sampled = "H"
        elif inv_H <= 0 and inv_M > 0:
            sampled = "M"
        else:
            eligible_driver = (
                d.name in TOP_TEAM_DRIVERS
                and ((not d.is_ego) or allow_ego_variability)
            )

            if eligible_driver and (inv_M, inv_H) in ((2, 1), (1, 2)):
                if float(rng.random()) < float(cfg.minority_start_on_asymmetric_inventory_prob):
                    minority = "H" if inv_H < inv_M else "M"
                    sampled = minority

        out[d.name] = sampled

    return out

def _sample_episode_inventories(
    rng: np.random.Generator,
    cfg: RaceConfig,
    driver_static: List[DriverStatic],
    allow_ego_variability: bool,
    dM: float,
    dH: float,
) -> Dict[str, Tuple[int, int]]:
    """
    Episode-level inventory sampling driven by the sampled degradation regime.

    If Medium degradation is much higher than Hard degradation,
    1M / 2H becomes much more likely.

    If Medium and Hard degradation are similar,
    2M / 1H becomes much more likely.
    """
    out: Dict[str, Tuple[int, int]] = {}

    p_two_M = _prob_two_M_inventory_from_deg(float(dM), float(dH), cfg)

    for d in driver_static:
        total_sets = int(d.inventory_M) + int(d.inventory_H)

        # Keep any non-3-set driver unchanged.
        if total_sets != 3:
            out[d.name] = (int(d.inventory_M), int(d.inventory_H))
            continue

        if float(rng.random()) < p_two_M:
            inv_M, inv_H = 2, 1
        else:
            inv_M, inv_H = 1, 2

        out[d.name] = (int(inv_M), int(inv_H))

    return out

def _prob_two_M_inventory_from_deg(
    dM: float,
    dH: float,
    cfg: RaceConfig,
) -> float:
    """
    Probability of assigning 2M / 1H.

    If Medium degradation is much worse than Hard degradation,
    1M / 2H becomes much more likely.

    If the two degradations are similar,
    2M / 1H becomes much more likely.
    """
    gap = float(dM - dH)

    high_gap = float(cfg.inventory_bias_medium_deg_gap_high)
    similar_gap = float(cfg.inventory_bias_medium_deg_gap_similar)

    p_two_M_when_high_medium_deg = float(1.0 - cfg.inventory_bias_prob_twoH_high_medium_deg)
    p_two_M_when_similar_deg = float(cfg.inventory_bias_prob_twoM_similar_deg)

    if gap >= high_gap:
        return float(clip(p_two_M_when_high_medium_deg, 0.0, 1.0))

    if gap <= similar_gap:
        return float(clip(p_two_M_when_similar_deg, 0.0, 1.0))

    w = float((gap - similar_gap) / max(1e-12, high_gap - similar_gap))
    p = float((1.0 - w) * p_two_M_when_similar_deg + w * p_two_M_when_high_medium_deg)
    return float(clip(p, 0.0, 1.0))

def _sample_competitor_styles(
    rng: np.random.Generator,
    cfg: RaceConfig,
    driver_static: List[DriverStatic],
    allow_ego_variability: bool,
) -> Dict[str, Dict[str, object]]:
    styles: Dict[str, Dict[str, object]] = {}

    for d in driver_static:
        style_kind = "normal"
        shift_laps = 0

        eligible_top_team_driver = (
            d.name in TOP_TEAM_DRIVERS
            and ((not d.is_ego) or allow_ego_variability)
        )

        if eligible_top_team_driver:
            p_var = float(cfg.top_team_strategy_variability_prob)
            p_extend = float(cfg.top_team_extend_prob)
            p_undercut = float(cfg.top_team_undercut_prob)
        else:
            p_var = float(cfg.non_top_team_strategy_variability_prob)
            p_extend = 0.5
            p_undercut = 0.5

        if float(rng.random()) < p_var:
            u = float(rng.random())

            if u < p_extend:
                style_kind = "extend"
                shift_laps = int(
                    rng.integers(
                        int(cfg.strategy_extend_laps_min),
                        int(cfg.strategy_extend_laps_max) + 1,
                    )
                )
            elif u < p_extend + p_undercut:
                style_kind = "undercut"
                shift_laps = int(
                    rng.integers(
                        int(cfg.strategy_undercut_laps_min),
                        int(cfg.strategy_undercut_laps_max) + 1,
                    )
                )

        styles[d.name] = {
            "style": style_kind,
            "shift_laps": int(shift_laps),
        }

    return styles
def _choose_retirement_driver(rng: np.random.Generator, driver_static: List[DriverStatic], already_retired: set[str]) -> Optional[str]:
    pool = [d.name for d in driver_static if d.can_retire and d.name not in already_retired and d.name not in TOP_TEAM_DRIVERS]
    if not pool:
        return None
    return str(rng.choice(np.asarray(pool, dtype=object)))


def _sample_neutral_windows(
    rng: np.random.Generator,
    cfg: RaceConfig,
    driver_static: List[DriverStatic],
) -> Tuple[List[NeutralWindow], Dict[str, float]]:
    windows: List[NeutralWindow] = []
    retirements: Dict[str, float] = {}
    retired_names: set[str] = set()

    approx_leader_lap_times = [0.0]
    t = 0.0
    for _ in range(1, cfg.L + 1):
        t += cfg.B
        approx_leader_lap_times.append(float(t))

    def overlaps(t0: float, t1: float) -> bool:
        for w in windows:
            if not (t1 <= w.start_time or t0 >= w.end_time):
                return True
        return False

    # Special SC immediately after lap 1 with 10% episode chance.
    if float(rng.random()) < float(cfg.sc_after_lap1_episode_prob):
        start_time = float(approx_leader_lap_times[1] + 1e-3)
        duration = float(rng.uniform(cfg.sc_min_duration, cfg.sc_max_duration))
        end_time = float(start_time + duration)
        retired = _choose_retirement_driver(rng, driver_static, retired_names)
        if retired is not None:
            retired_names.add(retired)
            retirements[retired] = start_time
        windows.append(
            NeutralWindow(
                kind="SC",
                start_time=start_time,
                end_time=end_time,
                pre_bunch_end_time=float(min(end_time, start_time + cfg.sc_pre_bunch_duration)),
                retired_driver=retired,
            )
        )

    # Per-lap neutral hazards.
    for leader_lap in range(2, cfg.L - cfg.no_neutral_pit_last_n):
        base_time = float(approx_leader_lap_times[leader_lap])

        # No overlapping deployments.
        if any(w.start_time <= base_time < w.end_time for w in windows):
            continue

        u = float(rng.random())
        kind: Optional[str] = None
        if u < float(cfg.sc_prob_per_lap):
            kind = "SC"
        elif u < float(cfg.sc_prob_per_lap + cfg.vsc_prob_per_lap):
            kind = "VSC"

        if kind is None:
            continue

        start_time = float(base_time + rng.uniform(0.0, cfg.B))
        duration = float(
            rng.uniform(cfg.sc_min_duration, cfg.sc_max_duration)
            if kind == "SC"
            else rng.uniform(cfg.vsc_min_duration, cfg.vsc_max_duration)
        )
        end_time = float(start_time + duration)
        if overlaps(start_time, end_time):
            continue

        retired = _choose_retirement_driver(rng, driver_static, retired_names)
        if retired is not None:
            retired_names.add(retired)
            retirements[retired] = start_time

        windows.append(
            NeutralWindow(
                kind=kind,
                start_time=start_time,
                end_time=end_time,
                pre_bunch_end_time=(None if kind == "VSC" else float(min(end_time, start_time + cfg.sc_pre_bunch_duration))),
                retired_driver=retired,
            )
        )

    windows.sort(key=lambda w: (w.start_time, w.kind))
    return windows, retirements


# Duplicate sample_episode_scenario removed: the canonical version above keeps pace_offsets/pace_perturbations.

# =============================================================================
# STRATEGY ORACLE AWARE OF SCENARIO
# =============================================================================

def _compositions_with_last_min(total: int, n_parts: int, min_each: int, last_min: int) -> List[Tuple[int, ...]]:
    out: List[Tuple[int, ...]] = []

    def rec(remaining: int, parts_left: int, prefix: List[int]) -> None:
        if parts_left == 1:
            if remaining >= last_min:
                out.append(tuple(prefix + [remaining]))
            return
        min_needed_after = (parts_left - 2) * min_each + last_min
        lo = min_each
        hi = remaining - min_needed_after
        for x in range(lo, hi + 1):
            rec(remaining - x, parts_left - 1, prefix + [x])

    if n_parts <= 0:
        return out
    if n_parts == 1:
        return [(total,)] if total >= last_min else []
    rec(total, n_parts, [])
    return out


def _pit_laps_from_lengths(lengths: Tuple[int, ...]) -> Tuple[int, ...]:
    pits: List[int] = []
    cum = 0
    for Ls in lengths[:-1]:
        cum += Ls
        pits.append(cum)
    return tuple(pits)


def regime_at_time(t: float, scenario: EpisodeScenario) -> str:
    for w in scenario.neutral_windows:
        if w.active_at(t):
            return w.regime_at(t)
    return "GREEN"


def neutral_window_at_time(t: float, scenario: EpisodeScenario) -> Optional[NeutralWindow]:
    for w in scenario.neutral_windows:
        if w.active_at(t):
            return w
    return None


def effective_pit_loss_from_regime(regime: str, cfg: RaceConfig) -> float:
    if regime == "VSC":
        return float(cfg.Pbar * cfg.pit_loss_mult_vsc)
    if regime == "SC_PRE":
        return float(cfg.Pbar * cfg.pit_loss_mult_sc)
    return float(cfg.Pbar)


def neutral_discount_feasible(
    *,
    lap_decision: int,
    t_decision: float,
    scenario: EpisodeScenario,
    cfg: RaceConfig,
) -> Tuple[bool, str, float]:
    regime = regime_at_time(float(t_decision), scenario)
    if regime == "GREEN":
        return True, regime, float(cfg.Pbar)

    if lap_decision <= cfg.no_neutral_pit_first_n:
        return False, regime, math.inf
    if lap_decision >= cfg.L - cfg.no_neutral_pit_last_n:
        return False, regime, math.inf

    if regime == "SC_BUNCH":
        return False, regime, math.inf

    w = neutral_window_at_time(float(t_decision), scenario)
    if w is None:
        return True, "GREEN", float(cfg.Pbar)

    pit_loss = effective_pit_loss_from_regime(regime, cfg)
    if regime == "VSC":
        if float(t_decision + pit_loss) > float(w.end_time):
            return False, regime, math.inf
        return True, regime, pit_loss

    if regime == "SC_PRE":
        if w.pre_bunch_end_time is None:
            return False, regime, math.inf
        if float(t_decision + pit_loss) > float(w.pre_bunch_end_time):
            return False, regime, math.inf
        return True, regime, pit_loss

    return True, regime, float(cfg.Pbar)


def strategy_time_scenario_aware(
    *,
    lengths: Tuple[int, ...],
    compounds: Tuple[str, ...],
    start_delta: float,
    d_M: float,
    d_H: float,
    Delta_HM: float,
    cfg: RaceConfig,
    scenario: EpisodeScenario,
) -> Tuple[float, float, float, float, bool]:
    """
    GREEN-only baseline evaluator.

    Important:
    this is now used only for the episode-start deterministic baseline strategy.
    Future SC/VSC is NOT priced in here.
    """
    total_deg = 0.0
    total_offset = 0.0
    total_pit = 0.0
    total_time = 0.0

    for stint_idx, (Ls, q) in enumerate(zip(lengths, compounds)):
        d_q = d_M if q == "M" else d_H
        kappa_q = 0.0 if q == "M" else Delta_HM

        for a in range(Ls):
            tau = float(cfg.B + start_delta + kappa_q + d_q * a)
            total_time += tau
            total_deg += float(d_q * a)
            total_offset += float(kappa_q)

        if stint_idx < len(lengths) - 1:
            total_pit += float(cfg.Pbar)
            total_time += float(cfg.Pbar)

    return float(total_time), float(total_deg), float(total_offset), float(total_pit), True

def plot_race_traces_window_with_neutrals(
    traces: Dict[str, List[float]],
    finish_df: pd.DataFrame,
    env: MultiCarRaceEnv,
    out_png: str,
    title_prefix: str,
    lap_start: int,
    lap_end: int,
) -> None:
    plt.figure(figsize=(14, 8))

    finish_order = finish_df["car"].astype(str).tolist()
    ego_name = env.ego_name
    car_to_team = {c.name: DRIVER_TO_TEAM.get(c.name, "UNKNOWN") for c in env.cars}
    car_to_color = {
        car_name: TEAM_COLORS.get(team_name, "#000000")
        for car_name, team_name in car_to_team.items()
    }
    car_to_obj = {c.name: c for c in env.cars}

    rects = _neutral_rectangles_from_scenario(env.scenario, env)

    ymins = []
    ymaxs = []
    plotted_data = {}

    for nm in finish_order:
        if nm not in traces:
            continue

        tr = traces[nm]
        if len(tr) == 0:
            continue

        x_full = np.arange(0, len(tr) + 1)
        y_full = np.zeros(len(tr) + 1, dtype=float)
        y_full[1:] = np.asarray(tr, dtype=float)

        mask = (x_full >= lap_start) & (x_full <= lap_end)
        x = x_full[mask]
        y = y_full[mask]

        if len(x) == 0:
            continue

        plotted_data[nm] = (x, y)
        ymins.append(float(np.min(y)))
        ymaxs.append(float(np.max(y)))

    ymin = min(ymins) if ymins else -10.0
    ymax = max(ymaxs) if ymaxs else 10.0
    yrange = max(1.0, ymax - ymin)
    ymin_plot = ymin - 0.08 * yrange
    ymax_plot = ymax + 0.12 * yrange

    for x0, x1, kind in rects:
        overlap_start = max(lap_start, x0)
        overlap_end = min(lap_end, x1)
        if overlap_start < overlap_end:
            alpha = 0.12 if kind == "VSC" else 0.18
            color = "#FFD966" if kind == "VSC" else "#D9E2F3"
            plt.axvspan(overlap_start, overlap_end, alpha=alpha, color=color, zorder=0)

    for nm in finish_order:
        if nm not in plotted_data:
            continue

        x, y = plotted_data[nm]
        color = car_to_color.get(nm, "#000000")
        lw = 2.8 if nm == ego_name else 1.2
        alpha = 1.0 if nm == ego_name else 0.80
        z = 5 if nm == ego_name else 2

        plt.plot(x, y, color=color, lw=lw, alpha=alpha, zorder=z)

        car_obj = car_to_obj[nm]
        strategy_txt = strategy_string_from_actions_and_start(
            action_by_lap=car_obj.action_by_lap,
            start_compound=str(car_obj.comps_end[0]) if len(car_obj.comps_end) > 0 else str(car_obj.q_end),
            total_laps=env.cfg.L,
        )

        x_lab = float(x[-1] + 0.15)
        y_lab = float(y[-1])
        plt.text(
            x_lab,
            y_lab,
            f"{nm}  [{strategy_txt}]",
            fontsize=8,
            color=color,
            va="center",
            ha="left",
            zorder=20,
        )

        if not car_obj.active and len(x) > 0:
            xr = len(car_obj.lap_times)
            if lap_start <= xr <= lap_end:
                yr = traces[nm][xr - 1]
                plt.scatter([xr], [yr], marker="x", s=60, linewidths=2.0, color=color, zorder=10)

    events_df = build_events_df_from_env(env)
    if events_df is not None and not events_df.empty:
        for _, ev in events_df.iterrows():
            etype = str(ev.get("type", ""))
            if etype not in ("SL_PASS", "BF_YIELD"):
                continue

            follower = str(ev.get("follower", ""))
            lap_k = int(ev.get("lap", -1))

            if follower not in traces:
                continue
            if lap_k < lap_start or lap_k > lap_end:
                continue
            if lap_k < 1 or lap_k > len(traces[follower]):
                continue

            xev = float(lap_k)
            yev = float(traces[follower][lap_k - 1])
            color = car_to_color.get(follower, "#000000")

            if etype == "SL_PASS":
                plt.scatter([xev], [yev], marker="x", s=45, linewidths=1.8, color=color, zorder=10)
            else:
                plt.scatter([xev], [yev], marker="o", s=42, facecolors="none", edgecolors=color, linewidths=1.8, zorder=10)

    neutral_txt = neutral_windows_lap_summary(env)

    plt.ylim(ymin_plot, ymax_plot)
    plt.xlim(lap_start, lap_end + 2.5)
    plt.xlabel("Lap")
    plt.ylabel(f"Cumulative trace sum(z_ref - tau), z_ref={env.cfg.trace_ref_time:.1f}")
    plt.title(
        f"{title_prefix} | laps {lap_start}-{lap_end} | "
        f"Delta_HM={env.scenario.Delta_HM:.3f}, dM={env.scenario.dM:.3f}, dH={env.scenario.dH:.3f}\n"
        f"lap1={env.episode_info.get('lap1_position', 'nan')}, "
        f"finish={env.episode_info.get('finish_position', 'nan')}, "
        f"points={env.episode_info.get('points', 'nan')}\n"
        f"{neutral_txt}"
    )
    plt.grid(True, alpha=0.25)

    teams_in_plot = []
    for nm in finish_order:
        team = car_to_team.get(nm, "UNKNOWN")
        if team not in teams_in_plot:
            teams_in_plot.append(team)

    team_handles = [
        Line2D([0], [0], color=TEAM_COLORS.get(team, "#000000"), lw=3, label=team)
        for team in teams_in_plot
    ]
    rect_handles = [
        Line2D([0], [0], color="#FFD966", lw=6, alpha=0.6, label="VSC window"),
        Line2D([0], [0], color="#D9E2F3", lw=6, alpha=0.8, label="SC window"),
    ]
    leg1 = plt.legend(handles=team_handles + rect_handles, loc="upper left", ncol=2, fontsize=8, frameon=True)
    plt.gca().add_artist(leg1)

    event_handles = [
        Line2D([0], [0], marker="x", color="black", linestyle="None", markersize=8, label="On-track pass"),
        Line2D([0], [0], marker="o", color="black", markerfacecolor="none", linestyle="None", markersize=8, label="Blue-flag yield"),
    ]
    plt.legend(handles=event_handles, loc="lower left", fontsize=9, frameon=True)

    plt.tight_layout()
    plt.savefig(out_png, dpi=180, bbox_inches="tight")
    plt.close()

def rank_two_compound_strategies(
    *,
    K: int,
    d_M: float,
    d_H: float,
    Delta_HM: float,
    total_sets_M: int,
    total_sets_H: int,
    start_compound: str,
    driver_delta: float,
    cfg: RaceConfig,
    scenario: EpisodeScenario,
    must_use_both: bool = True,
    min_stint: int = 1,
    no_pit_last_n: int = 1,
    max_stops: int = 2,
) -> List[RankedStrategy]:
    """
    Initial deterministic baseline ranking.
    This is GREEN-only by construction.
    """
    if start_compound not in ("M", "H"):
        raise ValueError("start_compound must be 'M' or 'H'.")

    ranked: List[RankedStrategy] = []
    total_sets = total_sets_M + total_sets_H
    max_possible_stops = min(max_stops, total_sets - 1)
    last_stint_min = max(min_stint, no_pit_last_n)

    for S in range(1, max_possible_stops + 2):
        suffix_len = S - 1
        suffix_sequences = [()] if suffix_len == 0 else __import__("itertools").product(("M", "H"), repeat=suffix_len)

        for suffix in suffix_sequences:
            compounds = (start_compound,) + tuple(suffix)
            used_M = sum(1 for q in compounds if q == "M")
            used_H = sum(1 for q in compounds if q == "H")

            if used_M > total_sets_M or used_H > total_sets_H:
                continue
            if must_use_both and (used_M == 0 or used_H == 0):
                continue

            lengths_list = _compositions_with_last_min(
                total=K,
                n_parts=S,
                min_each=min_stint,
                last_min=last_stint_min,
            )

            for lengths in lengths_list:
                pit_laps = _pit_laps_from_lengths(lengths)
                total_time, total_deg, total_offset, total_pit, legal = strategy_time_scenario_aware(
                    lengths=lengths,
                    compounds=compounds,
                    start_delta=driver_delta,
                    d_M=d_M,
                    d_H=d_H,
                    Delta_HM=Delta_HM,
                    cfg=cfg,
                    scenario=scenario,
                )
                if not legal:
                    continue

                ranked.append(
                    RankedStrategy(
                        total_time=float(total_time),
                        total_deg_time=float(total_deg),
                        total_compound_offset=float(total_offset),
                        total_pit_loss=float(total_pit),
                        stops=S - 1,
                        stint_lengths=lengths,
                        stint_compounds=compounds,
                        pit_laps=pit_laps,
                        sets_used_M=used_M,
                        sets_used_H=used_H,
                    )
                )

    ranked.sort(key=lambda x: (x.total_time, x.stops, x.pit_laps, x.stint_compounds))
    return ranked


def actions_from_ranked_strategy(K: int, strat: RankedStrategy) -> List[str]:
    action_by_lap = ["0"] * (K + 1)
    for pit_lap, next_comp in zip(strat.pit_laps, strat.stint_compounds[1:]):
        action_by_lap[int(pit_lap)] = str(next_comp)
    return action_by_lap

def strategy_string_from_actions_and_start(
    action_by_lap: List[str],
    start_compound: str,
    total_laps: int,
) -> str:
    parts = [str(start_compound)]
    for lap in range(1, total_laps + 1):
        if lap < len(action_by_lap) and action_by_lap[lap] in ("M", "H"):
            parts.append(f"L{lap}->{action_by_lap[lap]}")
    return " | ".join(parts)


def neutral_windows_lap_summary(env: "MultiCarRaceEnv") -> str:
    if env.scenario is None or not env.scenario.neutral_windows:
        return "No SC/VSC"

    def leader_lap_at_time(t: float) -> int:
        max_lap = 0
        for c in env.cars:
            cum = 0.0
            for k, tau in enumerate(c.lap_times, start=1):
                cum += float(tau)
                if cum <= t + 1e-9:
                    max_lap = max(max_lap, k)
        return int(max_lap)

    chunks: List[str] = []
    for w in env.scenario.neutral_windows:
        ls = max(1, leader_lap_at_time(float(w.start_time)))
        le = max(ls, leader_lap_at_time(float(w.end_time)))
        chunks.append(f"{w.kind} L{ls}-L{le}")
    return " | ".join(chunks)
# =============================================================================
# ORDER HELPERS
# =============================================================================

def current_lap_anchor_time(car: CarState) -> float:
    return float(max(car.T_last, car.t_progress_start))


def compute_progress_p(t: float, car: CarState) -> float:
    if (not car.active) or car.next_kind in ("PIT_RELEASE", "WAIT_EGO_ACTION"):
        return 0.0
    if not math.isfinite(car.T_next):
        return 0.0
    t0 = float(car.t_progress_start)
    if t <= t0:
        return 0.0
    denom = float(car.T_next - t0)
    if denom <= 1e-12:
        return 0.0
    return clip((t - t0) / denom, 0.0, 1.0)


def classification_order_indices(cars: List[CarState], cfg: RaceConfig) -> List[int]:
    active = [i for i, c in enumerate(cars) if c.active and c.n < cfg.L]
    return sorted(active, key=lambda i: (-int(cars[i].n), float(cars[i].T_last), cars[i].name))


def same_lap_line_predecessor_idx(car_idx: int, cars: List[CarState], cfg: RaceConfig) -> Optional[int]:
    target_n = int(cars[car_idx].n)
    same = [i for i, c in enumerate(cars) if i != car_idx and c.active and c.n < cfg.L and int(c.n) == target_n]
    if not same:
        return None
    same_sorted = sorted(same + [car_idx], key=lambda i: (float(cars[i].T_last), cars[i].name))
    pos = same_sorted.index(car_idx)
    return None if pos == 0 else same_sorted[pos - 1]


def same_lap_anchor_predecessor_idx(car_idx: int, cars: List[CarState], cfg: RaceConfig) -> Optional[int]:
    target_n = int(cars[car_idx].n)
    same = [
        i for i, c in enumerate(cars)
        if i != car_idx and c.active and c.n < cfg.L and c.next_kind == "CROSS" and int(c.n) == target_n
    ]
    if not same:
        return None
    same_sorted = sorted(
        same + [car_idx],
        key=lambda i: (current_lap_anchor_time(cars[i]), float(cars[i].T_last), cars[i].name),
    )
    pos = same_sorted.index(car_idx)
    return None if pos == 0 else same_sorted[pos - 1]


def physical_order_and_headways(
    t: float,
    cars: List[CarState],
    cfg: RaceConfig,
) -> Tuple[List[int], Dict[int, int], Dict[int, float]]:
    active = [i for i, c in enumerate(cars) if c.active and c.n < cfg.L]
    pvals = [(i, compute_progress_p(t, cars[i])) for i in active]
    pvals.sort(key=lambda x: (x[1], cars[x[0]].n), reverse=True)
    order = [i for i, _ in pvals]
    succ_idx: Dict[int, int] = {}
    g_ahead: Dict[int, float] = {}
    if not order:
        return [], succ_idx, g_ahead
    for k, idx in enumerate(order):
        ahead = order[k - 1] if k > 0 else order[-1]
        succ_idx[idx] = ahead
        p_i = compute_progress_p(t, cars[idx])
        p_a = compute_progress_p(t, cars[ahead])
        dist_frac = (p_a - p_i) % 1.0
        g_ahead[idx] = float(dist_frac * cfg.B)
    return order, succ_idx, g_ahead


# =============================================================================
# LAP 1
# =============================================================================

def sample_start_order_adjacent_swaps(
    rng: np.random.Generator,
    grid_order: List[int],
    n_passes: int,
    p_swap: float,
) -> List[int]:
    order = list(grid_order)
    for _ in range(n_passes):
        i = 0
        while i < len(order) - 1:
            if float(rng.random()) < p_swap:
                order[i], order[i + 1] = order[i + 1], order[i]
                i += 2
            else:
                i += 1
    return order


def initialise_T1_from_start_order(
    rng: np.random.Generator,
    cars: List[CarState],
    start_order: List[int],
    cfg: RaceConfig,
) -> None:
    leader_idx = start_order[0]
    leader = cars[leader_idx]
    start_loss = max(0.0, float(rng.normal(cfg.lap1_start_loss_mu, cfg.lap1_start_loss_sd)))
    mu_leader_l1 = float(leader.drive_mean + start_loss)
    T1_leader = float(normal_pos(rng, mu_leader_l1, cfg.sigma_tau_close, min_val=1.0))

    leader.T_next = float(T1_leader)
    leader.next_kind = "CROSS"
    leader.pit_loss_this_lap = 0.0
    leader.t_progress_start = 0.0

    for pos in range(1, len(start_order)):
        idx = start_order[pos]
        ahead_idx = start_order[pos - 1]
        gap = float(rng.uniform(cfg.lap1_gap_lo, cfg.lap1_gap_hi))
        cars[idx].T_next = float(cars[ahead_idx].T_next + gap)
        cars[idx].next_kind = "CROSS"
        cars[idx].pit_loss_this_lap = 0.0
        cars[idx].t_progress_start = 0.0


# =============================================================================
# SCHEDULING
# =============================================================================

def project_to_feasible_time(target: float, other_times: List[float], gmin: float, lb: float, ub: float) -> float:
    intervals = [(t - gmin, t + gmin) for t in other_times]
    intervals.sort(key=lambda x: x[0])

    merged: List[Tuple[float, float]] = []
    for a, b in intervals:
        if not merged:
            merged.append((a, b))
        else:
            pa, pb = merged[-1]
            if a <= pb:
                merged[-1] = (pa, max(pb, b))
            else:
                merged.append((a, b))

    allowed: List[Tuple[float, float]] = []
    if not merged:
        allowed = [(lb, ub)]
    else:
        allowed.append((-math.inf, merged[0][0]))
        for (a1, b1), (a2, b2) in zip(merged[:-1], merged[1:]):
            allowed.append((b1, a2))
        allowed.append((merged[-1][1], math.inf))

    clipped_allowed: List[Tuple[float, float]] = []
    for a, b in allowed:
        aa = max(a, lb)
        bb = min(b, ub)
        if aa < bb:
            clipped_allowed.append((aa, bb))

    if not clipped_allowed:
        return float(clip(target, lb, ub))

    for a, b in clipped_allowed:
        if a <= target <= b:
            return float(target)

    best = None
    best_dist = math.inf
    for a, b in clipped_allowed:
        cand = a if target < a else b
        dist = abs(cand - target)
        if dist < best_dist:
            best_dist = dist
            best = cand
    return float(best if best is not None else clip(target, lb, ub))


def regime_multiplier(regime: str, cfg: RaceConfig) -> float:
    if regime == "VSC":
        return float(cfg.vsc_mult)
    if regime == "SC_PRE":
        return float(cfg.sc_pre_mult)
    if regime == "SC_BUNCH":
        return float(cfg.sc_bunch_mult)
    return 1.0


def neutral_queue_order_indices(
    cars: List[CarState],
    neutral_anchor_map: Dict[str, float],
    cfg: RaceConfig,
) -> List[int]:
    active = [
        i for i, c in enumerate(cars)
        if c.active and c.n < cfg.L and c.next_kind == "CROSS" and c.name in neutral_anchor_map
    ]
    return sorted(
        active,
        key=lambda i: (
            -int(cars[i].n),
            float(neutral_anchor_map[cars[i].name]),
            float(cars[i].T_last),
            cars[i].name,
        ),
    )


def build_neutral_frozen_prev_map(
    cars: List[CarState],
    neutral_anchor_map: Dict[str, float],
    neutral_pit_affected: set[str],
    cfg: RaceConfig,
    order: Optional[List[int]] = None,
) -> Tuple[Dict[str, Optional[str]], Dict[str, float]]:
    if order is None:
        order = neutral_queue_order_indices(cars, neutral_anchor_map, cfg)

    prev_name: Dict[str, Optional[str]] = {}
    gap_to_prev: Dict[str, float] = {}

    if not order:
        return prev_name, gap_to_prev

    first = cars[order[0]]
    prev_name[first.name] = None
    gap_to_prev[first.name] = float("nan")

    for k in range(1, len(order)):
        ia = order[k - 1]
        ib = order[k]
        ahead = cars[ia]
        behind = cars[ib]

        # New train head if lap count drops
        if int(behind.n) < int(ahead.n):
            prev_name[behind.name] = None
            gap_to_prev[behind.name] = float("nan")
            continue

        # Break train if either car was pit-affected during neutral
        if ahead.name in neutral_pit_affected or behind.name in neutral_pit_affected:
            prev_name[behind.name] = None
            gap_to_prev[behind.name] = float("nan")
            continue

        prev_name[behind.name] = ahead.name
        gap_to_prev[behind.name] = float(
            max(cfg.g_min, neutral_anchor_map[behind.name] - neutral_anchor_map[ahead.name])
        )

    return prev_name, gap_to_prev


def sc_target_gap_to_leader(rank: int, cfg: RaceConfig) -> float:
    if rank <= 0:
        return 0.0
    return float(cfg.sc_target_gap_p2 + (rank - 1) * cfg.sc_target_gap_step)


def schedule_next_crossing_green(
    rng: np.random.Generator,
    t_start: float,
    car_idx: int,
    cars: List[CarState],
    cfg: RaceConfig,
    succ_idx: Dict[int, int],
    g_ahead: Dict[int, float],
    Delta_HM: float,
    dM: float,
    dH: float,
    event_log: Optional[List[Dict]] = None,
) -> None:
    car = cars[car_idx]

    same_pred_idx = same_lap_anchor_predecessor_idx(car_idx, cars, cfg)
    same_pred = cars[same_pred_idx] if same_pred_idx is not None else None
    h_sched = (
        float(max(0.0, current_lap_anchor_time(car) - current_lap_anchor_time(same_pred)))
        if same_pred is not None else float("nan")
    )
    same_pred_next = (
        float(same_pred.T_next)
        if (same_pred is not None and same_pred.next_kind == "CROSS" and math.isfinite(same_pred.T_next))
        else math.inf
    )

    phys_ahead_idx = succ_idx.get(car_idx, car_idx)
    phys_ahead = cars[phys_ahead_idx]
    raw_g_proxy = float(g_ahead.get(car_idx, cfg.B))
    BF = int(phys_ahead_idx != car_idx and car.n > phys_ahead.n)
    phys_ahead_next = (
        float(phys_ahead.T_next)
        if (phys_ahead_idx != car_idx and phys_ahead.next_kind == "CROSS" and math.isfinite(phys_ahead.T_next))
        else math.inf
    )

    A_sl = float(same_pred.drive_mean - car.drive_mean) if same_pred is not None else float("nan")
    A_bf = float(phys_ahead.drive_mean - car.drive_mean) if phys_ahead_idx != car_idx else float("nan")

    do_pass = False
    do_yield = False

    if same_pred is not None and math.isfinite(same_pred_next):
        eligible = (
            h_sched <= cfg.Delta_OT + cfg.g_buf
            and A_sl >= h_sched
            and A_sl >= cfg.A0
        )
        if eligible:
            do_pass = bool(rng.random() < sigmoid(cfg.beta_pass * (A_sl - cfg.Delta_OT)))

    if not do_pass and BF == 1 and math.isfinite(phys_ahead_next) and phys_ahead_idx != car_idx:
        eligible_bf = (raw_g_proxy <= cfg.h_yield) and (A_bf >= cfg.A_blue)
        if eligible_bf:
            do_yield = bool(rng.random() < sigmoid(cfg.beta_blue * (A_bf - cfg.Delta_blue)))

    dirty = 0.0
    if same_pred is not None and not do_pass and not do_yield and h_sched <= cfg.h_dirty:
        dirty = float(cfg.kappa_dirty * max(0.0, 1.0 - h_sched / cfg.h_dirty))

    mu_drive = float(car.drive_mean + dirty)
    sigma = float(cfg.sigma_tau_close if (same_pred is not None and h_sched <= cfg.g_close) else cfg.sigma_tau)

    other_times = sorted(
        float(c.T_next)
        for j, c in enumerate(cars)
        if j != car_idx and c.active and c.n < cfg.L and c.next_kind == "CROSS" and math.isfinite(c.T_next)
    )

    accepted = False
    final_T = math.inf

    for _ in range(500):
        tau_try = float(normal_pos(rng, mu_drive, sigma, min_val=1.0))
        cand_T = float(t_start + tau_try)

        ok = True
        for t_other in other_times:
            if abs(cand_T - t_other) < cfg.g_min:
                ok = False
                break
        if not ok:
            continue

        if same_pred is not None and math.isfinite(same_pred_next):
            if do_pass:
                if cand_T > same_pred_next - cfg.g_pass:
                    continue
            else:
                if cand_T < same_pred_next + cfg.g_min:
                    continue

        if do_yield and math.isfinite(phys_ahead_next):
            if cand_T > phys_ahead_next - cfg.g_pass:
                continue

        accepted = True
        final_T = cand_T
        break

    if not accepted:
        lb = -math.inf
        ub = math.inf

        if same_pred is not None and math.isfinite(same_pred_next):
            if do_pass:
                ub = min(ub, same_pred_next - cfg.g_pass)
            else:
                lb = max(lb, same_pred_next + cfg.g_min)

        if do_yield and math.isfinite(phys_ahead_next):
            ub = min(ub, phys_ahead_next - cfg.g_pass)

        raw_target = float(t_start + normal_pos(rng, mu_drive, sigma, min_val=1.0))
        final_T = project_to_feasible_time(raw_target, other_times, float(cfg.g_min), float(lb), float(ub))

    car.T_next = float(max(final_T, t_start + 1e-6))

    if event_log is not None:
        lap_k = int(car.n + 1)
        if do_pass and same_pred is not None:
            event_log.append(
                {"time": float(t_start), "lap": lap_k, "type": "SL_PASS", "follower": car.name, "leader": same_pred.name}
            )
        if do_yield and phys_ahead_idx != car_idx:
            event_log.append(
                {"time": float(t_start), "lap": lap_k, "type": "BF_YIELD", "follower": car.name, "leader": phys_ahead.name}
            )

def schedule_next_crossing_neutral(
    rng: np.random.Generator,
    t_start: float,
    current_time: float,
    car_idx: int,
    cars: List[CarState],
    cfg: RaceConfig,
    regime: str,
    neutral_order: List[int],
    neutral_frozen_gap_to_prev: Dict[str, float],
) -> None:
    car = cars[car_idx]
    pos = neutral_order.index(car_idx)

    leader_idx = neutral_order[0]
    leader_car = cars[leader_idx]

    pred_idx = neutral_order[pos - 1] if pos > 0 else None
    pred_car = cars[pred_idx] if pred_idx is not None else None

    if regime in ("VSC", "SC_PRE"):
        mult = regime_multiplier(regime, cfg)
        sigma = float(mult * cfg.sigma_tau)

        if pred_car is None:
            mu_drive = float(mult * car.drive_mean)
            tau_drive = float(normal_pos(rng, mu_drive, sigma, min_val=1.0))
            final_T = float(t_start + tau_drive)
        else:
            frozen_gap = float(neutral_frozen_gap_to_prev.get(car.name, float("nan")))
            if not math.isfinite(frozen_gap):
                frozen_gap = float(max(cfg.g_min, current_lap_anchor_time(car) - current_lap_anchor_time(pred_car)))

            if not math.isfinite(pred_car.T_next):
                mu_drive = float(mult * car.drive_mean)
                tau_drive = float(normal_pos(rng, mu_drive, sigma, min_val=1.0))
                final_T = float(t_start + tau_drive)
            else:
                final_T = float(pred_car.T_next + frozen_gap)

        car.T_next = float(max(final_T, t_start + 1e-6))
        return

    if regime != "SC_BUNCH":
        raise ValueError(f"Unsupported neutral regime {regime}")

    # True SC bunching
    if pos == 0:
        mult = float(cfg.sc_bunch_mult)
        sigma = float(mult * cfg.sigma_sc)
        mu_drive = float(mult * leader_car.drive_mean)
        tau_drive = float(normal_pos(rng, mu_drive, sigma, min_val=1.0))
        tau_drive = float(max(cfg.sc_min_lap_time, tau_drive))
        final_T = float(t_start + tau_drive)
        car.T_next = float(max(final_T, t_start + 1e-6))
        return

    target_gap = float(sc_target_gap_to_leader(pos, cfg))
    desired_T = float(leader_car.T_next + target_gap)

    min_cross_time = float(t_start + cfg.sc_min_lap_time)
    desired_T = float(max(desired_T, min_cross_time))

    if pred_car is None or not math.isfinite(pred_car.T_next):
        lb = float(current_time + 1e-6)
    else:
        lb = float(pred_car.T_next + cfg.g_min)

    final_T = float(max(lb, desired_T))
    car.T_next = float(max(final_T, t_start + 1e-6))

def schedule_next_crossing(
    rng: np.random.Generator,
    t_start: float,
    current_time: float,
    car_idx: int,
    cars: List[CarState],
    cfg: RaceConfig,
    succ_idx: Dict[int, int],
    g_ahead: Dict[int, float],
    Delta_HM: float,
    dM: float,
    dH: float,
    scenario: EpisodeScenario,
    neutral_anchor_map: Dict[str, float],
    neutral_pit_affected: set[str],
    event_log: Optional[List[Dict]] = None,
    regime_override: Optional[str] = None,
) -> None:
    regime = str(regime_override) if regime_override is not None else regime_at_time(float(t_start), scenario)

    if regime == "GREEN":
        schedule_next_crossing_green(
            rng=rng,
            t_start=t_start,
            car_idx=car_idx,
            cars=cars,
            cfg=cfg,
            succ_idx=succ_idx,
            g_ahead=g_ahead,
            Delta_HM=Delta_HM,
            dM=dM,
            dH=dH,
            event_log=event_log,
        )
        return

    neutral_order = neutral_queue_order_indices(cars, neutral_anchor_map, cfg)
    neutral_order = [i for i in neutral_order if cars[i].active and cars[i].n < cfg.L]

    _, frozen_gap_to_prev = build_neutral_frozen_prev_map(
        cars=cars,
        neutral_anchor_map=neutral_anchor_map,
        neutral_pit_affected=neutral_pit_affected,
        cfg=cfg,
        order=neutral_order,
    )

    if car_idx not in neutral_order:
        neutral_order = neutral_order + [car_idx]
        frozen_gap_to_prev.setdefault(cars[car_idx].name, float("nan"))

    schedule_next_crossing_neutral(
        rng=rng,
        t_start=t_start,
        current_time=current_time,
        car_idx=car_idx,
        cars=cars,
        cfg=cfg,
        regime=regime,
        neutral_order=neutral_order,
        neutral_frozen_gap_to_prev=frozen_gap_to_prev,
    )


# =============================================================================
# ENVIRONMENT
# =============================================================================

class MultiCarRaceEnv:
    """
    Two ego modes:
      - deterministic: ego uses same scenario-aware ranked deterministic strategy as competitors
      - ppo: ego acts lap by lap with PPO

    Neutral logic is episode-sampled and changes every episode.
    SC/VSC windows are time windows, not fixed lap seeds.
    """
    def __init__(self, race_cfg: RaceConfig, driver_cfgs: List[Dict], seed: int, ego_mode: str = "ppo"):
        self.cfg = race_cfg
        self.base_seed = int(seed)
        self.ego_mode = str(ego_mode)
        self.driver_static = self._build_driver_static(driver_cfgs)

        ego_names = [d.name for d in self.driver_static if d.is_ego]
        if len(ego_names) != 1:
            raise ValueError("Exactly one ego car is required.")
        self.ego_name = ego_names[0]

        self.episode_index = 0
        self.rng = np.random.default_rng(self.base_seed)
        self.scenario: Optional[EpisodeScenario] = None

        self.cars: List[CarState] = []
        self.ego_idx: int = -1
        self.current_time: float = 0.0
        self.pending_ego_decision: bool = False
        self.done: bool = False

        self.event_log: List[Dict] = []
        self.decision_log: List[Dict] = []
        self.episode_info: Dict[str, float | int | str | List] = {}
        self.last_obs: Optional[np.ndarray] = None
        # Temporary exploration shaping: disabled globally during training once
        # validation legality reaches 100%.
        self.compound_bonus_active: bool = True
        self.retirement_queue: List[Tuple[float, str]] = []
        self._just_triggered_neutral: bool = False

    @staticmethod
    def _build_driver_static(driver_cfgs: List[Dict]) -> List[DriverStatic]:
        out: List[DriverStatic] = []
        for d in driver_cfgs:
            out.append(
                DriverStatic(
                    name=str(d["name"]),
                    grid_pos=int(d["grid_pos"]),
                    delta_i=float(d["delta"]),
                    inventory_M=int(d["inventory_M"]),
                    inventory_H=int(d["inventory_H"]),
                    rank_weights=tuple(float(x) for x in d.get("rank_weights", (1.0,))),
                    start_prob_M=float(d.get("start_prob_M", 0.5)),
                    can_retire=bool(d.get("can_retire", True)),
                    is_ego=bool(d.get("is_ego", False)),
                )
            )
        return sorted(out, key=lambda x: x.grid_pos)

    def _leader_current_lap_marker(self) -> int:
        order = classification_order_indices(self.cars, self.cfg)
        if not order:
            return int(self.cfg.L + 1)
        leader = self.cars[order[0]]
        return int(min(self.cfg.L, leader.n + 1))

    def _pace_regime_for_next_lap(self, car: CarState, t_ref: Optional[float] = None) -> str:
        """
        Official regime by time, plus VSC tail-extension logic:
        if official VSC has ended but the leader had already started/scheduled lap K
        under VSC, then all cars still schedule up to lap K under VSC.
        """
        tt = float(car.t_progress_start if t_ref is None else t_ref)
        official_regime = self._current_regime(tt)

        if official_regime != "GREEN":
            return str(official_regime)

        if self.vsc_tail_apply_until_lap is not None:
            if int(car.n + 1) <= int(self.vsc_tail_apply_until_lap):
                return "VSC"

        return "GREEN"

    def _pace_regime_for_completed_lap(self, car: CarState, lap_completed: int, t_cross: float) -> str:
        """
        Same idea as _pace_regime_for_next_lap, but for the lap that just finished.
        Needed so tyre-age / anchor handling stays consistent on the VSC tail lap.
        """
        official_regime = self._current_regime(float(t_cross))

        if official_regime != "GREEN":
            return str(official_regime)

        if self.vsc_tail_apply_until_lap is not None:
            if int(lap_completed) <= int(self.vsc_tail_apply_until_lap):
                return "VSC"

        return "GREEN"

    def _maybe_release_vsc_tail(self) -> None:
        """
        Once every active car has moved beyond the forced VSC tail lap,
        clear the leftover neutral state.
        """
        if self.vsc_tail_apply_until_lap is None:
            return

        still_need_tail = any(
            c.active and c.n < self.cfg.L and int(c.n + 1) <= int(self.vsc_tail_apply_until_lap)
            for c in self.cars
        )

        if still_need_tail:
            return

        self.vsc_tail_apply_until_lap = None
        self._clear_neutral_state()

    def _normalise_rank_weights(self, weights: Sequence[float], top_k: int) -> np.ndarray:
        w = np.asarray(list(weights[:top_k]), dtype=float)
        if w.size < top_k:
            w = np.concatenate([w, np.zeros(top_k - w.size, dtype=float)], axis=0)
        w = np.clip(w, 0.0, None)
        if w.sum() <= 0.0:
            w = np.ones(top_k, dtype=float)
        return w / w.sum()


    def _sample_competitor_strategy(self, drv: DriverStatic, start_compound: str) -> Tuple[List[str], RankedStrategy, int]:
        """
        Episode-start baseline strategy using the sampled episode inventory
        and the sampled episode-specific pace offset.

        DRIVER_CONFIGS["delta"] is the central pace estimate, while
        self.scenario.pace_offsets[drv.name] is the realised delta_i^(e)
        for this episode.
        """
        assert self.scenario is not None

        inv_M_ep, inv_H_ep = self.scenario.episode_inventories[drv.name]
        driver_delta_ep = float(self.scenario.pace_offsets.get(drv.name, drv.delta_i))

        ranked = rank_two_compound_strategies(
            K=self.cfg.L,
            d_M=self.scenario.dM,
            d_H=self.scenario.dH,
            Delta_HM=self.scenario.Delta_HM,
            total_sets_M=int(inv_M_ep),
            total_sets_H=int(inv_H_ep),
            start_compound=start_compound,
            driver_delta=driver_delta_ep,
            cfg=self.cfg,
            scenario=self.scenario,
            must_use_both=self.cfg.mandatory_both,
            min_stint=self.cfg.min_stint,
            no_pit_last_n=self.cfg.no_pit_last_n,
            max_stops=self.cfg.max_stops,
        )

        if len(ranked) == 0:
            raise RuntimeError(f"No feasible baseline strategy for {drv.name}.")

        chosen, rank_idx = self._choose_style_adjusted_ranked_strategy(drv, ranked)
        return actions_from_ranked_strategy(self.cfg.L, chosen), chosen, rank_idx


    def _build_episode_cars(self) -> None:
        assert self.scenario is not None
        self.cars = []

        for i, drv in enumerate(self.driver_static):
            start_comp = self.scenario.start_compounds[drv.name]
            inv_M_ep, inv_H_ep = self.scenario.episode_inventories[drv.name]
            driver_delta_ep = float(self.scenario.pace_offsets.get(drv.name, drv.delta_i))

            if drv.is_ego and self.ego_mode == "ppo":
                action_by_lap = ["0"] * (self.cfg.L + 1)
            else:
                action_by_lap, _chosen, _rank_idx = self._sample_competitor_strategy(
                    drv=drv,
                    start_compound=start_comp,
                )

            remaining_M = int(inv_M_ep) - (1 if start_comp == "M" else 0)
            remaining_H = int(inv_H_ep) - (1 if start_comp == "H" else 0)

            car = CarState(
                car_id=i + 1,
                name=drv.name,
                grid_pos=drv.grid_pos,
                delta_i=driver_delta_ep,
                is_ego=drv.is_ego,
                inventory_M_total=int(inv_M_ep),
                inventory_H_total=int(inv_H_ep),
                can_retire=drv.can_retire,
                action_by_lap=action_by_lap,
                q_end=start_comp,
                q_in=start_comp,
                remaining_M=int(remaining_M),
                remaining_H=int(remaining_H),
                used_M=(start_comp == "M"),
                used_H=(start_comp == "H"),
            )
            self.cars.append(car)

        self.ego_idx = next(i for i, c in enumerate(self.cars) if c.is_ego)

        for c in self.cars:
            kappa = compound_offset(c.q_in, self.scenario.Delta_HM)
            d_q = compound_deg(c.q_in, self.scenario.dM, self.scenario.dH)
            c.drive_mean = float(self.cfg.B + c.delta_i + kappa + d_q * c.a_in)

    def _build_retirement_queue(self) -> None:
        assert self.scenario is not None
        self.retirement_queue = sorted([(t, nm) for nm, t in self.scenario.retirements.items()], key=lambda x: (x[0], x[1]))

    def _freeze_neutral_state(self) -> None:
        self.neutral_anchor_map = {
            c.name: float(current_lap_anchor_time(c))
            for c in self.cars
            if c.active and c.n < self.cfg.L and c.next_kind == "CROSS"
        }
        self.neutral_pit_affected = set()
        self.vsc_tail_apply_until_lap = None

    def _clear_neutral_state(self) -> None:
        self.neutral_anchor_map = {}
        self.neutral_pit_affected = set()

    def _update_neutral_boundary_state(self, t_event: float) -> None:
        regime_now = self._current_regime(float(t_event))

        if regime_now != self.last_regime:
            if self.last_regime == "GREEN" and regime_now in ("VSC", "SC_PRE", "SC_BUNCH"):
                self._freeze_neutral_state()
                self._just_triggered_neutral = True

            elif self.last_regime in ("VSC", "SC_PRE", "SC_BUNCH") and regime_now == "GREEN":
                # Official neutral ended.
                # For VSC only: keep scheduling the leader's CURRENT lap as VSC
                # for everybody, even if trailing cars schedule it later.
                if self.last_regime == "VSC":
                    self.vsc_tail_apply_until_lap = int(self._leader_current_lap_marker())
                else:
                    self._clear_neutral_state()

            self.last_regime = regime_now

        self._maybe_release_vsc_tail()

    def _initialise_lap1(self) -> None:
        grid_order = list(range(len(self.cars)))
        start_order = sample_start_order_adjacent_swaps(
            self.rng,
            grid_order=grid_order,
            n_passes=self.cfg.lap1_npasses,
            p_swap=self.cfg.lap1_pswap,
        )
        initialise_T1_from_start_order(self.rng, self.cars, start_order, self.cfg)

    def _current_regime(self, t: Optional[float] = None) -> str:
        assert self.scenario is not None
        tt = self.current_time if t is None else float(t)
        return regime_at_time(tt, self.scenario)

    def _discounted_pit_feasible_now(self, lap_decision: int, t_decision: float) -> Tuple[bool, str, float]:
        assert self.scenario is not None
        return neutral_discount_feasible(lap_decision=lap_decision, t_decision=t_decision, scenario=self.scenario, cfg=self.cfg)
    def _current_neutral_window_id(self, t: Optional[float] = None) -> Optional[int]:
        assert self.scenario is not None
        tt = self.current_time if t is None else float(t)
        for idx, w in enumerate(self.scenario.neutral_windows):
            if w.active_at(tt):
                return int(idx)
        return None

    def _choose_reoptimised_plan_with_90_10(self, plans: List[Dict]) -> Dict:
        """
        90%: choose the globally optimal plan.
        10%: choose the best plan whose immediate action differs from the optimal one.
        If no alternative immediate-action family exists, fall back to the optimum.
        """
        if len(plans) == 0:
            raise RuntimeError("No plans available for re-optimisation.")

        best_plan = plans[0]
        best_action = str(best_plan["immediate_action"])

        by_action: Dict[str, Dict] = {}
        for p in plans:
            a = str(p["immediate_action"])
            if a not in by_action:
                by_action[a] = p

        alternative_actions = [a for a in by_action.keys() if a != best_action]

        if len(alternative_actions) == 0:
            return best_plan

        if float(self.rng.random()) < 0.90:
            return best_plan

        alt_action = str(self.rng.choice(np.asarray(alternative_actions, dtype=object)))
        return by_action[alt_action]
    


    def _maybe_replan_deterministic_car(self, car: CarState) -> None:
        """
        One-shot local deterministic re-optimisation per neutral window.

        VSC:
          one re-optimisation allowed during the window.

        SC_PRE:
          one re-optimisation allowed during SC_PRE only.

        SC_BUNCH:
          no re-optimisation.

        Important:
          competitor min-stint and late-pit constraints still apply during
          neutral-phase re-optimisation. A neutral may discount pit loss, but it
          does not waive sporting / strategy feasibility constraints.

        During VSC / SC_PRE, the local continuation re-optimisation is
        independent of the episode-start extend/undercut tactical label.
        """
        if not car.active or car.n >= self.cfg.L:
            return

        if car.is_ego and self.ego_mode == "ppo":
            return

        regime = self._current_regime(float(car.T_last))
        if regime not in ("VSC", "SC_PRE"):
            return

        window_id = self._current_neutral_window_id(float(car.T_last))
        if window_id is None:
            return

        last_done = self.neutral_replan_window_by_car.get(car.name, None)
        if last_done == window_id:
            return

        feasible_now, regime_now, pit_loss_now = self._discounted_pit_feasible_now(
            int(car.n),
            float(car.T_last),
        )

        # Apply the SAME competitor stop-feasibility rules during SC/VSC
        # re-optimisation as in the normal strategy model.
        first_pit_ok = int(car.n) >= int(self.cfg.min_stint)
        stint_ok = (int(car.n) - int(car.last_pit_lap)) >= int(self.cfg.min_stint)
        enough_laps_left = (int(self.cfg.L) - int(car.n)) > int(self.cfg.no_pit_last_n)
        timing_ok = bool(first_pit_ok and stint_ok and enough_laps_left)

        immediate_stop_allowed = bool(
            feasible_now
            and regime_now in ("VSC", "SC_PRE")
            and timing_ok
            and car.pit_stops_done < self.cfg.max_stops
            and (car.remaining_M > 0 or car.remaining_H > 0)
        )

        plans = self.rank_continuation_strategies_from_state(
            current_lap=int(car.n),
            total_laps=int(self.cfg.L),
            current_compound=str(car.q_end),
            current_age=int(car.a_end),
            remaining_M=int(car.remaining_M),
            remaining_H=int(car.remaining_H),
            used_M=bool(car.used_M),
            used_H=bool(car.used_H),
            pit_stops_done=int(car.pit_stops_done),
            driver_delta=float(car.delta_i),
            d_M=float(self.scenario.dM),
            d_H=float(self.scenario.dH),
            Delta_HM=float(self.scenario.Delta_HM),
            cfg=self.cfg,
            immediate_stop_allowed=immediate_stop_allowed,
            immediate_stop_regime=str(regime_now),
            immediate_pit_loss=float(pit_loss_now if immediate_stop_allowed else self.cfg.Pbar),
            must_use_both=bool(self.cfg.mandatory_both),
        )

        if len(plans) == 0:
            self.neutral_replan_window_by_car[car.name] = int(window_id)
            return

        chosen_plan = self._choose_reoptimised_plan_with_style(
            car=car,
            plans=plans,
            regime=str(regime),
        )

        new_actions = list(car.action_by_lap)
        future_actions = chosen_plan["future_action_by_lap"]

        for lap_k in range(int(car.n), int(self.cfg.L) + 1):
            new_actions[lap_k] = future_actions[lap_k]

        car.action_by_lap = new_actions
        self.neutral_replan_window_by_car[car.name] = int(window_id)
    def _first_pit_lap_ranked_strategy(self, strat: RankedStrategy) -> Optional[int]:
        if strat is None or len(strat.pit_laps) == 0:
            return None
        return int(strat.pit_laps[0])

    def _first_future_pit_lap_from_actions(
        self,
        action_by_lap: List[str],
        current_lap: int,
    ) -> Optional[int]:
        for lap in range(int(current_lap), int(self.cfg.L) + 1):
            if lap < len(action_by_lap) and action_by_lap[lap] in ("M", "H"):
                return int(lap)
        return None

    def _competitor_style(self, name: str) -> Dict[str, object]:
        if self.scenario is None:
            return {"style": "normal", "shift_laps": 0}
        return self.scenario.competitor_styles.get(name, {"style": "normal", "shift_laps": 0})

    def _choose_style_adjusted_ranked_strategy(
        self,
        drv: DriverStatic,
        ranked: List[RankedStrategy],
    ) -> Tuple[RankedStrategy, int]:
        """
        Start-of-episode baseline choice with optional style perturbation.
        """
        if len(ranked) == 0:
            raise RuntimeError(f"No ranked strategies available for {drv.name}.")

        k = min(self.cfg.competitor_top_k, len(ranked))
        w = self._normalise_rank_weights(drv.rank_weights, k)
        rank_idx = int(self.rng.choice(np.arange(k), p=w))
        weighted_choice = ranked[rank_idx]

        # In PPO mode, keep ego untouched.
        # In deterministic mode, let ego behave like the other top teams.
        if drv.is_ego and self.ego_mode != "deterministic":
            return weighted_choice, rank_idx

        style_info = self._competitor_style(drv.name)
        style_kind = str(style_info.get("style", "normal"))
        shift_laps = int(style_info.get("shift_laps", 0))

        if style_kind == "normal":
            return weighted_choice, rank_idx

        ref = ranked[0]
        ref_first = self._first_pit_lap_ranked_strategy(ref)

        if ref_first is None:
            return weighted_choice, rank_idx

        if style_kind == "extend":
            target_first = int(ref_first + shift_laps)
            candidates = [
                s for s in ranked
                if (
                    self._first_pit_lap_ranked_strategy(s) is not None
                    and int(self._first_pit_lap_ranked_strategy(s)) >= target_first
                    and float(s.total_time) <= float(ref.total_time + self.cfg.strategy_time_slack_extend)
                )
            ]
            if candidates:
                return candidates[0], 0
            return weighted_choice, rank_idx

        if style_kind == "undercut":
            target_first = int(max(self.cfg.min_stint, ref_first - shift_laps))
            candidates = [
                s for s in ranked
                if (
                    self._first_pit_lap_ranked_strategy(s) is not None
                    and int(self._first_pit_lap_ranked_strategy(s)) <= target_first
                    and float(s.total_time) <= float(ref.total_time + self.cfg.strategy_time_slack_undercut)
                )
            ]
            if candidates:
                return candidates[0], 0
            return weighted_choice, rank_idx

        return weighted_choice, rank_idx



   
    def _choose_reoptimised_plan_with_style(
        self,
        car: CarState,
        plans: List[Dict],
        regime: Optional[str] = None,
    ) -> Dict:
        """
        Re-optimised plan chooser.

        Behaviour:
        - During VSC / SC_PRE:
            ignore extend / undercut style and behave like "normal",
            i.e. use the same stochastic re-optimised selection rule,
            EXCEPT for very old tyres with enough race left:
                if tyre age >= 35 and laps remaining >= 8,
                and a pit-now plan exists,
                then choose a pit-now plan with 100% probability.
        - Outside those neutral replans:
            preserve the original style behaviour.

        This means:
        - initial baseline strategy choice can still be extend / undercut
        - but neutral re-optimisation is style-neutral
        """
        if len(plans) == 0:
            raise RuntimeError("No plans available for re-optimisation.")

        # ---------------------------------------------------------------------
        # PATCH:
        # During neutral re-optimisation, all deterministic competitors
        # behave like "normal" drivers, regardless of extend / undercut style.
        #
        # EXTRA PATCH:
        # If tyres are very old and there is still enough race left, force
        # the driver to box now if any pit-now plan is feasible.
        # ---------------------------------------------------------------------
        if regime in ("VSC", "SC_PRE"):
            current_age = int(car.a_end)
            laps_remaining = int(self.cfg.L - car.n)

            # Force a pit-now choice for very old tyres if there is enough race left
            if current_age >= 35 and laps_remaining >= 8:
                pit_now_candidates = [
                    p for p in plans
                    if str(p["immediate_action"]) in ("M", "H")
                ]
                if pit_now_candidates:
                    # plans are already sorted by total_time,
                    # so the first pit-now candidate is the best pit-now continuation
                    return pit_now_candidates[0]

            # Otherwise behave like a normal driver under neutral replan
            return self._choose_reoptimised_plan_with_90_10(plans)

        best_plan = plans[0]

        # In PPO mode, keep ego special.
        # In deterministic mode, let ego behave like the other top teams.
        if car.is_ego and self.ego_mode != "deterministic":
            return self._choose_reoptimised_plan_with_90_10(plans)

        style_info = self._competitor_style(car.name)
        style_kind = str(style_info.get("style", "normal"))
        shift_laps = int(style_info.get("shift_laps", 0))

        if style_kind == "normal":
            return self._choose_reoptimised_plan_with_90_10(plans)

        current_lap = int(car.n)
        base_first = self._first_future_pit_lap_from_actions(
            best_plan["future_action_by_lap"],
            current_lap,
        )

        if style_kind == "extend":
            if best_plan["immediate_action"] in ("M", "H"):
                target_first = int(current_lap + max(1, shift_laps))
            elif base_first is not None:
                target_first = int(base_first + max(1, shift_laps))
            else:
                target_first = int(current_lap + max(1, shift_laps))

            stay_out_candidates: List[Dict] = []
            for p in plans:
                if str(p["immediate_action"]) != "0":
                    continue
                fp = self._first_future_pit_lap_from_actions(
                    p["future_action_by_lap"],
                    current_lap,
                )
                if fp is None or int(fp) >= target_first:
                    stay_out_candidates.append(p)

            if stay_out_candidates:
                return stay_out_candidates[0]

            return self._choose_reoptimised_plan_with_90_10(plans)

        if style_kind == "undercut":
            pit_now_candidates = [p for p in plans if str(p["immediate_action"]) in ("M", "H")]
            if pit_now_candidates:
                return pit_now_candidates[0]

            if base_first is not None:
                target_first = int(max(current_lap + 1, base_first - max(1, shift_laps)))
                earlier_candidates: List[Dict] = []
                for p in plans:
                    fp = self._first_future_pit_lap_from_actions(
                        p["future_action_by_lap"],
                        current_lap,
                    )
                    if fp is not None and int(fp) <= target_first:
                        earlier_candidates.append(p)
                if earlier_candidates:
                    return earlier_candidates[0]

            return self._choose_reoptimised_plan_with_90_10(plans)

        return self._choose_reoptimised_plan_with_90_10(plans)
    def _new_neutral_triggered_since_last_decision(self) -> bool:
        return bool(self._just_triggered_neutral)
    
    def reset(self) -> np.ndarray:
        self.episode_index += 1
        self.rng = np.random.default_rng(self.base_seed + 10_000 * self.episode_index)

        allow_ego_variability = bool(self.ego_mode == "deterministic")
        self.scenario = sample_episode_scenario(
            rng=self.rng,
            cfg=self.cfg,
            driver_static=self.driver_static,
            allow_ego_variability=allow_ego_variability,
        )

        self._build_episode_cars()
        self._build_retirement_queue()
        self._initialise_lap1()

        self.current_time = 0.0
        self.pending_ego_decision = False
        self.done = False
        self.event_log = []
        self.decision_log = []
        self._just_triggered_neutral = False
        self.last_regime = "GREEN"
        self._clear_neutral_state()
        self.vsc_tail_apply_until_lap: Optional[int] = None

        self.neutral_replan_window_by_car = {}

        self.episode_info = {
            "episode_index": int(self.episode_index),
            "ego_name": self.ego_name,
            "lap1_position": math.nan,
            "finish_position": math.nan,
            "finish_position_raw": math.nan,
            "points": math.nan,
            "legal_both_compounds": False,
            "Delta_HM": float(self.scenario.Delta_HM),
            "dM": float(self.scenario.dM),
            "dH": float(self.scenario.dH),
            "n_vsc": int(sum(w.kind == "VSC" for w in self.scenario.neutral_windows)),
            "n_sc": int(sum(w.kind == "SC" for w in self.scenario.neutral_windows)),
        }

        for name, (inv_M, inv_H) in self.scenario.episode_inventories.items():
            self.episode_info[f"{name}_inv_M"] = int(inv_M)
            self.episode_info[f"{name}_inv_H"] = int(inv_H)

        self._advance_until_ego_decision_or_done()
        self.last_obs = self._get_obs() if not self.done else np.zeros(self.obs_dim(), dtype=np.float32)
        return self.last_obs.copy()
    
    def obs_dim(self) -> int:
        return 17

    def _ego_car(self) -> CarState:
        return self.cars[self.ego_idx]

    def _remaining_laps(self) -> int:
        return int(self.cfg.L - self._ego_car().n)

    def _need_pit_for_both(self, car: Optional[CarState]) -> int:
        if car is None:
            return 0
        return int(not (car.used_M and car.used_H))

    def _decision_state_compound_and_age(self, car: Optional[CarState]) -> Tuple[str, int]:
        if car is None:
            return "M", 0
        if car.is_ego and self.pending_ego_decision:
            return car.q_end, int(car.a_end)
        if car.n >= self.cfg.L:
            return car.q_end, int(car.a_end)
        return car.q_in, int(car.a_in)

    def _decision_state_degradation(self, car: Optional[CarState]) -> float:
        if car is None or self.scenario is None:
            return 0.0
        q, a = self._decision_state_compound_and_age(car)
        d_q = compound_deg(q, self.scenario.dM, self.scenario.dH)
        return float(d_q * a)

    def _decision_state_drive_mean(self, car: Optional[CarState]) -> float:
        if car is None or self.scenario is None:
            return 0.0
        q, a = self._decision_state_compound_and_age(car)
        return float(self.cfg.B + car.delta_i + compound_offset(q, self.scenario.Delta_HM) + compound_deg(q, self.scenario.dM, self.scenario.dH) * a)

    def _classification_snapshot(self) -> Tuple[List[int], Dict[str, int], Dict[str, float], Dict[str, float], Dict[str, Optional[int]], Dict[str, Optional[int]]]:
        order = classification_order_indices(self.cars, self.cfg)
        pos_map: Dict[str, int] = {}
        gap_leader: Dict[str, float] = {}
        gap_ahead: Dict[str, float] = {}
        ahead_idx_map: Dict[str, Optional[int]] = {}
        behind_idx_map: Dict[str, Optional[int]] = {}

        if not order:
            return order, pos_map, gap_leader, gap_ahead, ahead_idx_map, behind_idx_map

        leader_idx = order[0]
        leader_time = float(self.cars[leader_idx].T_last)
        for pos, idx in enumerate(order, start=1):
            c = self.cars[idx]
            pos_map[c.name] = int(pos)
            gap_leader[c.name] = float(c.T_last - leader_time)
            if pos == 1:
                gap_ahead[c.name] = float("nan")
                ahead_idx_map[c.name] = None
            else:
                ahead = order[pos - 2]
                gap_ahead[c.name] = float(c.T_last - self.cars[ahead].T_last)
                ahead_idx_map[c.name] = int(ahead)
            behind_idx_map[c.name] = None if pos == len(order) else int(order[pos])
        return order, pos_map, gap_leader, gap_ahead, ahead_idx_map, behind_idx_map
    

    def _stint_time_from_state(
        self,
        *,
        length: int,
        compound: str,
        start_age: int,
        driver_delta: float,
        d_M: float,
        d_H: float,
        Delta_HM: float,
        cfg: RaceConfig,
        neutral_laps_at_start: int = 0,
        neutral_regime: str = "GREEN",
    ) -> Tuple[float, float, float, int]:
        total_time = 0.0
        total_deg = 0.0
        total_offset = 0.0

        d_q = d_M if compound == "M" else d_H
        kappa_q = 0.0 if compound == "M" else Delta_HM

        age = int(start_age)
        remaining_neutral = int(max(0, neutral_laps_at_start))

        for _ in range(length):
            base_tau = float(cfg.B + driver_delta + kappa_q + d_q * age)

            if remaining_neutral > 0:
                mult = regime_multiplier(neutral_regime, cfg)
                tau = float(mult * base_tau)
                total_time += tau
                total_deg += float(d_q * age)
                total_offset += float(kappa_q)
                remaining_neutral -= 1
                # tyre age frozen during neutral
            else:
                tau = float(base_tau)
                total_time += tau
                total_deg += float(d_q * age)
                total_offset += float(kappa_q)
                age += 1

        return float(total_time), float(total_deg), float(total_offset), int(age)
    

    def rank_continuation_strategies_from_state(
        self,
        *,
        current_lap: int,
        total_laps: int,
        current_compound: str,
        current_age: int,
        remaining_M: int,
        remaining_H: int,
        used_M: bool,
        used_H: bool,
        pit_stops_done: int,
        driver_delta: float,
        d_M: float,
        d_H: float,
        Delta_HM: float,
        cfg: RaceConfig,
        immediate_stop_allowed: bool,
        immediate_stop_regime: str,
        immediate_pit_loss: float,
        must_use_both: bool = True,
    ) -> List[Dict]:
        """
        Local deterministic re-optimisation from the CURRENT state.

        Immediate decision set:
          - "0" = stay out
          - "M" = pit now for Medium
          - "H" = pit now for Hard

        Approximation:
        if re-optimisation happens during VSC / SC_PRE, continuation pricing
        assumes an expected number of future neutral laps, during which lap
        times are neutral-scaled and tyre age is frozen.

        NEW:
        if the car is in SC_PRE and chooses to stay out now, then any future pit
        must be delayed until after:
            expected_sc_replan_laps + sc_post_green_no_pit_laps

        This is an oracle-side approximation of the runtime SC stay-out lock.
        A no-more-stops plan is still allowed.
        """
        laps_remaining = int(total_laps - current_lap)
        if laps_remaining <= 0:
            return []

        if immediate_stop_regime == "VSC":
            expected_neutral_laps = int(min(cfg.expected_vsc_replan_laps, laps_remaining))
        elif immediate_stop_regime == "SC_PRE":
            expected_neutral_laps = int(min(cfg.expected_sc_replan_laps, laps_remaining))
        else:
            expected_neutral_laps = 0

        plans: List[Dict] = []

        immediate_actions: List[str] = ["0"]
        if immediate_stop_allowed:
            if remaining_M > 0:
                immediate_actions.append("M")
            if remaining_H > 0:
                immediate_actions.append("H")

        for immediate_action in immediate_actions:
            comp0 = str(current_compound)
            age0 = int(current_age)
            remM0 = int(remaining_M)
            remH0 = int(remaining_H)
            usedM0 = bool(used_M)
            usedH0 = bool(used_H)
            stops_done0 = int(pit_stops_done)
            immediate_pit_cost = 0.0

            if immediate_action in ("M", "H"):
                comp0 = str(immediate_action)
                age0 = 0
                stops_done0 += 1
                immediate_pit_cost = float(immediate_pit_loss)

                if immediate_action == "M":
                    if remM0 <= 0:
                        continue
                    remM0 -= 1
                    usedM0 = True
                else:
                    if remH0 <= 0:
                        continue
                    remH0 -= 1
                    usedH0 = True

            max_future_stops = min(int(cfg.max_stops) - stops_done0, remM0 + remH0)
            max_future_stops = max(0, max_future_stops)
            last_stint_min = max(int(cfg.min_stint), int(cfg.no_pit_last_n))

            # NEW:
            # If the car stays out under SC_PRE, then the first future pit
            # (if any) must be delayed until after the estimated remaining SC
            # laps plus the post-SC green no-pit lock.
            #
            # Example:
            # current_lap = 26, expected_sc_replan_laps = 4, lock = 10
            # => first future pit must be at lap >= 40
            sc_stay_out_first_pit_min_rel = 0
            if immediate_action == "0" and immediate_stop_regime == "SC_PRE":
                sc_stay_out_first_pit_min_rel = int(
                    expected_neutral_laps + cfg.sc_post_green_no_pit_laps
                )

            for S in range(1, max_future_stops + 2):
                suffix_len = S - 1
                suffix_sequences = [()] if suffix_len == 0 else __import__("itertools").product(("M", "H"), repeat=suffix_len)

                for suffix in suffix_sequences:
                    compounds = (comp0,) + tuple(suffix)

                    used_new_M = sum(1 for q in suffix if q == "M")
                    used_new_H = sum(1 for q in suffix if q == "H")

                    if used_new_M > remM0 or used_new_H > remH0:
                        continue

                    final_used_M = bool(usedM0 or ("M" in compounds))
                    final_used_H = bool(usedH0 or ("H" in compounds))
                    if must_use_both and not (final_used_M and final_used_H):
                        continue

                    lengths_list = _compositions_with_last_min(
                        total=laps_remaining,
                        n_parts=S,
                        min_each=int(cfg.min_stint),
                        last_min=last_stint_min,
                    )

                    for lengths in lengths_list:
                        # NEW:
                        # If we stayed out under SC_PRE and this plan contains
                        # at least one future stop, then the first stint must be
                        # long enough to respect the SC stay-out lock.
                        #
                        # S == 1 means no more future pit stops, which is allowed.
                        if (
                            immediate_action == "0"
                            and immediate_stop_regime == "SC_PRE"
                            and S >= 2
                            and int(lengths[0]) < int(sc_stay_out_first_pit_min_rel)
                        ):
                            continue

                        total_time = float(immediate_pit_cost)
                        total_deg = 0.0
                        total_offset = 0.0
                        total_pit = float(immediate_pit_cost)

                        remaining_neutral = int(expected_neutral_laps)

                        t0, deg0, off0, _end_age0 = self._stint_time_from_state(
                            length=int(lengths[0]),
                            compound=str(compounds[0]),
                            start_age=int(age0),
                            driver_delta=float(driver_delta),
                            d_M=float(d_M),
                            d_H=float(d_H),
                            Delta_HM=float(Delta_HM),
                            cfg=cfg,
                            neutral_laps_at_start=remaining_neutral,
                            neutral_regime=str(immediate_stop_regime),
                        )
                        consumed_neutral_0 = min(int(lengths[0]), remaining_neutral)
                        remaining_neutral -= consumed_neutral_0

                        total_time += t0
                        total_deg += deg0
                        total_offset += off0

                        for j in range(1, len(lengths)):
                            Lj = int(lengths[j])
                            qj = str(compounds[j])

                            total_time += float(cfg.Pbar)
                            total_pit += float(cfg.Pbar)

                            tj, degj, offj, _end_age_j = self._stint_time_from_state(
                                length=Lj,
                                compound=qj,
                                start_age=0,
                                driver_delta=float(driver_delta),
                                d_M=float(d_M),
                                d_H=float(d_H),
                                Delta_HM=float(Delta_HM),
                                cfg=cfg,
                                neutral_laps_at_start=remaining_neutral,
                                neutral_regime=str(immediate_stop_regime),
                            )
                            consumed_neutral_j = min(Lj, remaining_neutral)
                            remaining_neutral -= consumed_neutral_j

                            total_time += tj
                            total_deg += degj
                            total_offset += offj

                        future_action_by_lap = ["0"] * (total_laps + 1)

                        if immediate_action in ("M", "H"):
                            future_action_by_lap[int(current_lap)] = str(immediate_action)

                        future_pit_laps = _pit_laps_from_lengths(tuple(int(x) for x in lengths))
                        for rel_pit_lap, next_comp in zip(future_pit_laps, compounds[1:]):
                            abs_pit_lap = int(current_lap + rel_pit_lap)
                            if 0 <= abs_pit_lap <= total_laps:
                                future_action_by_lap[abs_pit_lap] = str(next_comp)

                        plans.append(
                            {
                                "total_time": float(total_time),
                                "total_deg_time": float(total_deg),
                                "total_compound_offset": float(total_offset),
                                "total_pit_loss": float(total_pit),
                                "immediate_action": str(immediate_action),
                                "immediate_stop_regime": str(immediate_stop_regime),
                                "future_action_by_lap": future_action_by_lap,
                                "stint_lengths": tuple(int(x) for x in lengths),
                                "stint_compounds": tuple(str(x) for x in compounds),
                            }
                        )

        plans.sort(
            key=lambda x: (
                x["total_time"],
                x["total_pit_loss"],
                x["stint_lengths"],
                x["stint_compounds"],
            )
        )
        return plans

    def _count_cars_that_jump_if_ego_pits_now(self, ego_name: str, pos_map: Dict[str, int]) -> int:
        ego = self._ego_car()
        feasible, _regime, pit_loss = self._discounted_pit_feasible_now(int(ego.n), float(ego.T_last))
        if not feasible:
            return len(self.cars)
        ego_pos = int(pos_map.get(ego_name, len(self.cars)))
        ego_after_pit_time = float(ego.T_last + pit_loss)
        count = 0
        for c in self.cars:
            if (not c.active) or c.name == ego_name or c.n >= self.cfg.L:
                continue
            other_pos = int(pos_map.get(c.name, len(self.cars)))
            if other_pos <= ego_pos:
                continue
            if int(c.n) != int(ego.n):
                continue
            if float(c.T_last) <= ego_after_pit_time + 1e-12:
                count += 1
        return int(count)

    def _get_obs(self) -> np.ndarray:
        ego = self._ego_car()
        order, pos_map, gap_leader, gap_ahead, ahead_idx_map, behind_idx_map = self._classification_snapshot()

        ahead_idx = ahead_idx_map.get(ego.name, None)
        behind_idx = behind_idx_map.get(ego.name, None)
        ahead_car = self.cars[ahead_idx] if ahead_idx is not None else None
        behind_car = self.cars[behind_idx] if behind_idx is not None else None

        gap_a = float(gap_ahead.get(ego.name, 0.0))
        gap_b = float("nan") if behind_car is None else float(behind_car.T_last - ego.T_last)

        D_ego = self._decision_state_degradation(ego)
        D_ahead = self._decision_state_degradation(ahead_car)
        D_behind = self._decision_state_degradation(behind_car)

        pace_ego = self._decision_state_drive_mean(ego)
        pace_ahead_diff = self._decision_state_drive_mean(ahead_car) - pace_ego if ahead_car is not None else 0.0
        pace_behind_diff = self._decision_state_drive_mean(behind_car) - pace_ego if behind_car is not None else 0.0

        cars_jump_if_pit_now = self._count_cars_that_jump_if_ego_pits_now(ego_name=ego.name, pos_map=pos_map)

        can_discount_now, regime_now, _pit_loss_now = self._discounted_pit_feasible_now(int(ego.n), float(ego.T_last))
        can_discount = float(can_discount_now and regime_now in ("VSC", "SC_PRE"))
        new_neutral_flag = float(self._new_neutral_triggered_since_last_decision())

        obs = np.array(
            [
                float(self._remaining_laps() / self.cfg.L),
                float(D_ego / max(1e-6, self.scenario.dM * self.cfg.L)),
                float(D_ahead / max(1e-6, self.scenario.dM * self.cfg.L)),
                float(D_behind / max(1e-6, self.scenario.dM * self.cfg.L)),
                float(int(pos_map.get(ego.name, len(order))) / len(self.cars)),
                float(ego.remaining_M / max(1, ego.inventory_M_total)),
                float(ego.remaining_H / max(1, ego.inventory_H_total)),
                float(np.clip(gap_leader.get(ego.name, 0.0), 0.0, 120.0) / 120.0),
                0.0 if not math.isfinite(gap_a) else float(np.clip(gap_a, 0.0, 30.0) / 30.0),
                0.0 if not math.isfinite(gap_b) else float(np.clip(gap_b, 0.0, 30.0) / 30.0),
                float(min(cars_jump_if_pit_now, len(self.cars)) / len(self.cars)),
                float(pace_ahead_diff / 20.0),
                float(pace_behind_diff / 20.0),
                float(self._need_pit_for_both(ahead_car)),
                float(self._need_pit_for_both(behind_car)),
                can_discount,
                new_neutral_flag,
            ],
            dtype=np.float32,
        )

        ablation = str(self.cfg.observation_ablation).strip().lower()
        if ablation == "no_local_competitor":
            # o_3, o_4: competitor degradation
            # o_9, o_10: local gaps
            # o_12, o_13: relative pace
            # o_14, o_15: remaining compound requirement
            for idx in (2, 3, 8, 9, 11, 12, 13, 14):
                obs[idx] = 0.0
        elif ablation == "no_ego_degradation":
            obs[1] = 0.0
        elif ablation == "no_laps_remaining":
            obs[0] = 0.0
        elif ablation == "no_pit_vulnerability":
            obs[10] = 0.0
        elif ablation == "no_neutral_indicators":
            obs[15] = 0.0
            obs[16] = 0.0
        elif ablation not in ("", "none"):
            raise ValueError(f"Unknown observation_ablation: {self.cfg.observation_ablation}")

        return obs

    def action_mask(self) -> np.ndarray:
        ego = self._ego_car()
        if self.done or not self.pending_ego_decision:
            return np.array([1, 0, 0], dtype=np.int64)

        can_stay_out = 1

        if self.cfg.no_pit_lap1 and int(ego.n) <= 1:
            return np.array([1, 0, 0], dtype=np.int64)

        first_pit_ok = int(ego.n) >= int(self.cfg.ego_min_stint)
        stint_ok = (int(ego.n) - int(ego.last_pit_lap)) >= int(self.cfg.ego_min_stint)
        enough_laps_left = (int(self.cfg.L) - int(ego.n)) > int(self.cfg.ego_no_pit_last_n)
        timing_ok = bool(first_pit_ok and stint_ok and enough_laps_left)
        feasible_now, _regime, _pit_loss = self._discounted_pit_feasible_now(int(ego.n), float(ego.T_last))

        # NEW:
        # SC stay-out lock blocks pitting only in GREEN.
        green_locked = self._green_sc_lock_active(ego, float(ego.T_last))

        can_pit_M = int(
            ego.pit_stops_done < self.cfg.ego_max_stops
            and ego.remaining_M > 0
            and ego.n < self.cfg.L
            and timing_ok
            and feasible_now
            and not green_locked
        )
        can_pit_H = int(
            ego.pit_stops_done < self.cfg.ego_max_stops
            and ego.remaining_H > 0
            and ego.n < self.cfg.L
            and timing_ok
            and feasible_now
            and not green_locked
        )
        return np.array([can_stay_out, can_pit_M, can_pit_H], dtype=np.int64)

    def _green_sc_lock_active(self, car: CarState, t_decision: Optional[float] = None) -> bool:
        """
        SC stay-out lock is active only in GREEN.
        During a new VSC / SC, the lock is temporarily waived.
        """
        tt = float(car.T_last) if t_decision is None else float(t_decision)
        return bool(
            self._current_regime(tt) == "GREEN"
            and int(car.n) < int(car.green_no_pit_until_lap)
        )

    def _apply_sc_stay_out_lock(self, car: CarState) -> None:
        """
        If a car stays out during SC_PRE, create / refresh the green-only no-pit lock.

        Example:
        if current lap is 26, expected_sc_replan_laps=4, and extra green lock is 10,
        then the first allowed future pit is lap 40.
        """
        new_lock_until = int(
            car.n
            + self.cfg.expected_sc_replan_laps
            + self.cfg.sc_post_green_no_pit_laps
        )
        car.green_no_pit_until_lap = max(int(car.green_no_pit_until_lap), new_lock_until)

    def _retire_driver(self, name: str, t_ret: float) -> None:
        for c in self.cars:
            if c.name != name or not c.active:
                continue
            c.active = False
            c.retired_at = float(t_ret)
            c.T_next = math.inf
            self.event_log.append({"time": float(t_ret), "lap": int(c.n), "type": "RETIREMENT", "follower": c.name, "leader": ""})
            return

    def _process_retirements_at_current_time(self) -> None:
        while self.retirement_queue and abs(self.retirement_queue[0][0] - self.current_time) <= 1e-9:
            t_ret, nm = self.retirement_queue.pop(0)
            self._retire_driver(nm, t_ret)
            self._just_triggered_neutral = True
            if self._current_regime(float(t_ret)) in ("VSC", "SC_PRE", "SC_BUNCH"):
                self.neutral_pit_affected.add(nm)
                self.neutral_anchor_map.pop(nm, None)
    
  
    def _record_ego_decision(
        self,
        car: CarState,
        action_label: str,
        regime: str,
        new_neutral: bool,
        discounted_pit_available: bool,
    ) -> None:
        if not car.is_ego:
            return
        self.decision_log.append(
            {
                "lap": int(car.n),
                "time": float(car.T_last),
                "action": str(action_label),
                "pit": bool(action_label in ("M", "H")),
                "regime": str(regime),
                "new_neutral": bool(new_neutral),
                "discounted_pit_available": bool(discounted_pit_available),
            }
        )

    def _commit_pit_choice(self, car: CarState, next_comp: str) -> None:
        feasible, regime, pit_loss_mean = self._discounted_pit_feasible_now(int(car.n), float(car.T_last))
        if not feasible:
            raise RuntimeError(f"Illegal pit attempt for {car.name} at lap {car.n} / t={car.T_last:.3f}")

        car.pit_stops_done += 1
        car.last_pit_lap = int(car.n)
        car.q_in = str(next_comp)
        car.a_in = 0

        if next_comp == "M":
            if car.remaining_M <= 0:
                raise RuntimeError(f"{car.name} tried to use unavailable Medium set.")
            car.remaining_M -= 1
            car.used_M = True
        elif next_comp == "H":
            if car.remaining_H <= 0:
                raise RuntimeError(f"{car.name} tried to use unavailable Hard set.")
            car.remaining_H -= 1
            car.used_H = True
        else:
            raise ValueError("next_comp must be M or H.")

        # NEW:
        # Once the car actually pits, any previous SC stay-out lock is cleared.
        car.green_no_pit_until_lap = 0

        P = float(max(5.0, pit_loss_mean + self.rng.normal(0.0, self.cfg.sigma_P)))
        car.pit_loss_this_lap = float(P)
        car.next_kind = "PIT_RELEASE"
        car.t_progress_start = float(car.T_last + P)
        car.T_next = float(car.T_last + P)

        if regime in ("VSC", "SC_PRE", "SC_BUNCH"):
            self.neutral_pit_affected.add(car.name)
            self.neutral_anchor_map.pop(car.name, None)

        assert self.scenario is not None
        kappa = compound_offset(car.q_in, self.scenario.Delta_HM)
        d_q = compound_deg(car.q_in, self.scenario.dM, self.scenario.dH)
        car.drive_mean = float(self.cfg.B + car.delta_i + kappa + d_q * car.a_in)

    def _commit_stay_out(self, car: CarState) -> None:
        car.q_in = car.q_end
        car.a_in = int(car.a_end)
        car.next_kind = "CROSS"
        car.pit_loss_this_lap = 0.0
        car.t_progress_start = float(car.T_last)
        car.T_next = math.inf
        assert self.scenario is not None
        kappa = compound_offset(car.q_in, self.scenario.Delta_HM)
        d_q = compound_deg(car.q_in, self.scenario.dM, self.scenario.dH)
        car.drive_mean = float(self.cfg.B + car.delta_i + kappa + d_q * car.a_in)

    def _choose_competitor_action(self, car: CarState) -> str:
        if car.n < 0 or car.n >= len(car.action_by_lap):
            return "0"

        # If no pit on lap 1 globally, enforce it.
        if self.cfg.no_pit_lap1 and int(car.n) <= 1:
            return "0"

        # NEW:
        # SC stay-out lock only blocks pitting in GREEN.
        # During a new VSC / SC, the lock is waived.
        if self._green_sc_lock_active(car, float(car.T_last)):
            return "0"

        return str(car.action_by_lap[car.n])

    def _post_cross_choose_next_action(self, car: CarState, ego_action: Optional[int]) -> None:
        if (not car.active) or car.n >= self.cfg.L:
            car.T_next = math.inf
            car.next_kind = "CROSS"
            return

        decision_regime = self._current_regime(float(car.T_last))

        if car.is_ego and self.ego_mode == "ppo":
            if ego_action is None:
                car.next_kind = "WAIT_EGO_ACTION"
                car.T_next = math.inf
                return

            mask = self.action_mask()
            if int(ego_action) == 1 and mask[1] == 1:
                self._commit_pit_choice(car, "M")
            elif int(ego_action) == 2 and mask[2] == 1:
                self._commit_pit_choice(car, "H")
            else:
                self._commit_stay_out(car)
                if decision_regime == "SC_PRE":
                    self._apply_sc_stay_out_lock(car)
            return

        # deterministic competitor / deterministic ego:
        # locally re-optimise once if a neutral opportunity has appeared
        self._maybe_replan_deterministic_car(car)

        u = self._choose_competitor_action(car)
        feasible, regime_now, _pit_loss = self._discounted_pit_feasible_now(
            int(car.n),
            float(car.T_last),
        )
        discounted_available = bool(
            feasible and regime_now in ("VSC", "SC_PRE")
        )
        actual_u = str(u) if (u in ("M", "H") and feasible) else "0"

        self._record_ego_decision(
            car=car,
            action_label=actual_u,
            regime=str(decision_regime),
            new_neutral=bool(self._new_neutral_triggered_since_last_decision()),
            discounted_pit_available=discounted_available,
        )

        if actual_u == "0":
            self._commit_stay_out(car)
            if decision_regime == "SC_PRE":
                self._apply_sc_stay_out_lock(car)
        else:
            self._commit_pit_choice(car, actual_u)

    def _schedule_all_unscheduled_crossings(self) -> None:
        assert self.scenario is not None

        self._update_neutral_boundary_state(self.current_time)

        order, succ_idx, g_ahead = physical_order_and_headways(self.current_time, self.cars, self.cfg)
        need_schedule = [
            i for i, c in enumerate(self.cars)
            if c.active and c.n < self.cfg.L and c.next_kind == "CROSS" and not math.isfinite(c.T_next)
        ]
        need_schedule = sorted(
            need_schedule,
            key=lambda i: (-self.cars[i].n, float(self.cars[i].T_last), self.cars[i].name)
        )

        for i in need_schedule:
            c = self.cars[i]
            pace_regime = self._pace_regime_for_next_lap(c, float(c.t_progress_start))

            if pace_regime in ("VSC", "SC_PRE", "SC_BUNCH"):
                if c.name not in self.neutral_anchor_map:
                    self.neutral_anchor_map[c.name] = float(current_lap_anchor_time(c))

            schedule_next_crossing(
                rng=self.rng,
                t_start=float(c.t_progress_start),
                current_time=float(self.current_time),
                car_idx=i,
                cars=self.cars,
                cfg=self.cfg,
                succ_idx=succ_idx,
                g_ahead=g_ahead,
                Delta_HM=self.scenario.Delta_HM,
                dM=self.scenario.dM,
                dH=self.scenario.dH,
                scenario=self.scenario,
                neutral_anchor_map=self.neutral_anchor_map,
                neutral_pit_affected=self.neutral_pit_affected,
                event_log=self.event_log,
                regime_override=str(pace_regime),
            )

        self._maybe_release_vsc_tail()

    def _next_event_time(self) -> float:
        car_times = [c.T_next for c in self.cars if c.active and c.n < self.cfg.L and math.isfinite(c.T_next)]
        retire_time = self.retirement_queue[0][0] if self.retirement_queue else math.inf
        return float(min(car_times + [retire_time])) if (car_times or self.retirement_queue) else math.inf

    def _recompute_positions_after_completed_lap(self, lap_k: int) -> None:
        # positions are derived later from cumulative times; kept here only for lap1 bookkeeping if needed.
        return None
    
    def _process_event_set(self, t_event: float, ego_action: Optional[int]) -> bool:
        active_idx = [i for i, c in enumerate(self.cars) if c.active and c.n < self.cfg.L]
        E = [i for i in active_idx if math.isfinite(self.cars[i].T_next) and abs(self.cars[i].T_next - t_event) <= 1e-9]

        self.current_time = float(t_event)
        self._just_triggered_neutral = False
        self._update_neutral_boundary_state(self.current_time)
        self._process_retirements_at_current_time()

        ego_crossed = False

        # PIT_RELEASE first
        for i in E:
            c = self.cars[i]
            if c.next_kind != "PIT_RELEASE":
                continue

            c.next_kind = "CROSS"
            c.t_progress_start = float(t_event)
            c.T_next = math.inf

            next_pace_regime = self._pace_regime_for_next_lap(c, float(t_event))
            if next_pace_regime in ("VSC", "SC_PRE", "SC_BUNCH"):
                self.neutral_pit_affected.discard(c.name)
                self.neutral_anchor_map[c.name] = float(c.t_progress_start)
            else:
                self.neutral_pit_affected.discard(c.name)

        # CROSS second
        crossers = [
            i for i in E
            if self.cars[i].active
            and self.cars[i].next_kind == "CROSS"
            and not (self.cars[i].is_ego and self.pending_ego_decision and self.ego_mode == "ppo")
        ]

        for i in crossers:
            c = self.cars[i]
            if not math.isfinite(c.T_next):
                continue

            t_cross = float(c.T_next)
            lap_k_completed = int(c.n + 1)
            lap_time = float(t_cross - c.T_last)

            cross_regime = self._pace_regime_for_completed_lap(c, lap_k_completed, t_cross)

            c.lap_times.append(lap_time)
            c.q_end = c.q_in

            if cross_regime == "GREEN":
                c.a_end = int(c.a_in + 1)
            else:
                c.a_end = int(c.a_in)

            c.comps_end.append(c.q_end)
            c.ages_end.append(c.a_end)

            c.n += 1
            c.T_last = t_cross
            c.pit_loss_this_lap = 0.0
            c.cum_times_after_lap.append(float(c.T_last))

            if c.is_ego:
                ego_crossed = True

            if cross_regime in ("VSC", "SC_PRE", "SC_BUNCH"):
                self.neutral_anchor_map[c.name] = float(c.T_last)
            else:
                self.neutral_anchor_map.pop(c.name, None)

            self._post_cross_choose_next_action(
                c,
                ego_action if (c.is_ego and self.ego_mode == "ppo") else None
            )

        self._maybe_release_vsc_tail()
        return bool(ego_crossed)

    
    def _advance_until_ego_decision_or_done(self) -> None:
        self.pending_ego_decision = False

        while True:
            active = [c for c in self.cars if c.active and c.n < self.cfg.L]
            if not active:
                self.done = True
                self.pending_ego_decision = False
                self._finalise_episode()
                return

            next_t = self._next_event_time()
            if not math.isfinite(next_t):
                self.done = True
                self.pending_ego_decision = False
                self._finalise_episode()
                return

            ego_crossed = self._process_event_set(next_t, ego_action=None)

            if (not self._ego_car().active) or self._ego_car().n >= self.cfg.L:
                self.pending_ego_decision = False
                self.done = False
                self._schedule_all_unscheduled_crossings()
                continue

            if ego_crossed and self.ego_mode == "ppo":
                self.pending_ego_decision = True
                self.done = False
                return

            self._schedule_all_unscheduled_crossings()

    def _finalise_episode(self) -> None:
        ego = self._ego_car()

        lap1_rows = [
            (i, float(self.cars[i].lap_times[0]))
            for i in range(len(self.cars))
            if len(self.cars[i].lap_times) >= 1
        ]
        lap1_rows.sort(key=lambda x: (x[1], self.cars[x[0]].name))

        lap1_pos = math.nan
        for pos, (idx, _) in enumerate(lap1_rows, start=1):
            if idx == self.ego_idx:
                lap1_pos = int(pos)
                break

        finish_rows = []
        for i, c in enumerate(self.cars):
            if c.active and len(c.lap_times) >= self.cfg.L:
                finish_key = (0, float(sum(c.lap_times[:self.cfg.L])))
            else:
                laps_done = int(len(c.lap_times))
                retired_time = float(c.retired_at) if c.retired_at is not None else math.inf
                finish_key = (1, -laps_done, retired_time)
            finish_rows.append((i, finish_key))

        finish_rows.sort(key=lambda x: (x[1], self.cars[x[0]].name))

        raw_finish_pos = len(self.cars)
        for pos, (idx, _) in enumerate(finish_rows, start=1):
            if idx == self.ego_idx:
                raw_finish_pos = int(pos)
                break

        legal = bool(ego.used_M and ego.used_H)

        if legal and ego.active and len(ego.lap_times) >= self.cfg.L:
            reported_finish_pos = int(raw_finish_pos)
            reported_points = float(points_for_position(reported_finish_pos))
        else:
            reported_finish_pos = int(len(self.cars))
            reported_points = 0.0

        shaped_position_bonus = float(
            self.cfg.position_reward_scale * (len(self.cars) + 1 - reported_finish_pos)
        )
        if reported_finish_pos <= 3:
            finish_band_bonus = 10.0
        elif reported_finish_pos <= 6:
            finish_band_bonus = 6.0
        elif reported_finish_pos <= 10:
            finish_band_bonus = 2.0
        else:
            finish_band_bonus = 0.0

        training_reward = float(reported_points + shaped_position_bonus + finish_band_bonus)
        if not legal:
            training_reward -= float(self.cfg.invalid_no_both_penalty)

        self.episode_info["lap1_position"] = lap1_pos
        self.episode_info["finish_position_raw"] = int(raw_finish_pos)
        self.episode_info["finish_position"] = int(reported_finish_pos)
        self.episode_info["points"] = float(reported_points)
        self.episode_info["training_reward"] = float(training_reward)
        self.episode_info["legal_both_compounds"] = bool(legal)
        self.episode_info["ego_retired"] = bool(not ego.active)
        self.episode_info["ego_retired_at"] = float(ego.retired_at) if ego.retired_at is not None else math.nan

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        if self.done:
            return np.zeros(self.obs_dim(), dtype=np.float32), 0.0, True, dict(self.episode_info)

        if self.ego_mode != "ppo":
            raise RuntimeError("step(action) only valid in ego_mode='ppo'.")

        if not self.pending_ego_decision:
            raise RuntimeError("step(action) called when no ego decision is pending.")

        ego = self._ego_car()
        mask = self.action_mask().copy()
        legal_before_action = bool(ego.used_M and ego.used_H)
        self.pending_ego_decision = False

        actual_action_label = "0"
        if int(action) == 1 and mask[1] == 1:
            actual_action_label = "M"
            self._commit_pit_choice(ego, "M")
        elif int(action) == 2 and mask[2] == 1:
            actual_action_label = "H"
            self._commit_pit_choice(ego, "H")
        else:
            self._commit_stay_out(ego)

        feasible_now, regime_now, _pit_loss_now = self._discounted_pit_feasible_now(
            int(ego.n),
            float(ego.T_last),
        )
        self._record_ego_decision(
            car=ego,
            action_label=actual_action_label,
            regime=str(regime_now),
            new_neutral=bool(self._new_neutral_triggered_since_last_decision()),
            discounted_pit_available=bool(
                feasible_now and regime_now in ("VSC", "SC_PRE")
            ),
        )

        # Diagnostics only: record the actual PPO decision in the same per-lap
        # strategy container already used by deterministic competitors.
        if 0 <= int(ego.n) < len(ego.action_by_lap):
            ego.action_by_lap[int(ego.n)] = actual_action_label

        reward = 0.0
        if (
            self.compound_bonus_active
            and (not legal_before_action)
            and bool(ego.used_M and ego.used_H)
        ):
            reward += float(self.cfg.other_compound_bonus)

        self._schedule_all_unscheduled_crossings()
        self._advance_until_ego_decision_or_done()

        if self.done:
            reward += float(self.episode_info["training_reward"])
            obs = np.zeros(self.obs_dim(), dtype=np.float32)
        else:
            obs = self._get_obs()

        self.last_obs = obs.copy()
        return obs, float(reward), bool(self.done), dict(self.episode_info)


# =============================================================================
# PPO
# =============================================================================

@dataclass
class PPOConfig:
    seed: int = 0
    device: str = "cpu"
    n_envs: int = 16
    rollout_steps: int = 236
    max_updates: int = 6000
    gamma: float = 0.999
    gae_lambda: float = 0.99
    clip_eps: float = 0.20
    lr: float = 3e-4
    entropy_coef: float = 0.02
    value_coef: float = 0.50
    max_grad_norm: float = 0.50
    ppo_epochs: int = 6
    minibatch_size: int = 512

    eval_every: int = 25
    eval_episodes: int = 150

    convergence_patience: int = 6
    convergence_points_tol: float = 0.02
    convergence_action_agreement: float = 0.995
    min_evals_before_convergence: int = 5

    final_eval_episodes: int = 500
    trace_eval_episodes: int = 20

    # exploration curriculum around newly-triggered neutrals
    # NOTE: curriculum interventions are tracked and, by default, excluded from
    # the policy-gradient term so forced actions do not corrupt PPO log-prob ratios.
    override_fraction_updates: float = 0.25
    override_prob_on_new_neutral: float = 0.35
    override_force_pit_prob: float = 0.50

    # PPO stability / better data usage
    target_kl: float = 0.03
    early_stop_kl_mult: float = 1.50
    value_clip_eps: float = 0.20
    entropy_coef_final: Optional[float] = 0.005
    entropy_anneal_updates: int = 2500
    policy_update_weight_for_overrides: float = 0.0

    # Periodic checkpoints and diagnostics
    checkpoint_every: int = 1000
    diagnostics_every: int = 1
    write_plots_every: int = 25
    write_rollout_npz_every: int = 100
    health_json_every: int = 1

    # Health thresholds used only for warnings / convergence gating.
    max_allowed_same_buffer_streak: int = 0
    min_nonzero_grad_norm: float = 1e-8
    min_policy_param_delta: float = 1e-10
    max_action_collapse_rate: float = 0.985
    min_entropy_after_warmup: float = 0.02
    min_eval_points_improvement_for_convergence: float = 0.05
    stop_early_on_convergence: bool = False

    out_dir: str = "ppo_multicar_neutral_out"


# -----------------------------------------------------------------------------
# PPO diagnostics / health helpers
# -----------------------------------------------------------------------------

def _safe_float(x: object, default: float = math.nan) -> float:
    try:
        y = float(x)
        return y if math.isfinite(y) else default
    except Exception:
        return default


def _json_default(obj: object) -> object:
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        val = float(obj)
        return val if math.isfinite(val) else None
    if isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return str(obj)


def write_json_atomic(path: str | Path, payload: Dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=_json_default)
    os.replace(tmp, path)


def append_health_event(out_dir: str, msg: str) -> None:
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    with open(Path(out_dir) / "health_events.log", "a", encoding="utf-8") as f:
        f.write(msg.rstrip() + "\n")


def current_entropy_coef(ppo_cfg: PPOConfig, update: int) -> float:
    start = float(ppo_cfg.entropy_coef)
    final = start if ppo_cfg.entropy_coef_final is None else float(ppo_cfg.entropy_coef_final)
    horizon = max(1, int(ppo_cfg.entropy_anneal_updates))
    frac = min(1.0, max(0.0, float(update) / float(horizon)))
    return float((1.0 - frac) * start + frac * final)


def explained_variance(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    y_pred = np.asarray(y_pred, dtype=np.float64)
    y_true = np.asarray(y_true, dtype=np.float64)
    var_y = float(np.var(y_true))
    if var_y < 1e-12:
        return float("nan")
    return float(1.0 - np.var(y_true - y_pred) / var_y)


def clone_model_params(model: nn.Module) -> List[torch.Tensor]:
    return [p.detach().clone() for p in model.parameters() if p.requires_grad]


def model_param_delta_l2(model: nn.Module, before: List[torch.Tensor]) -> float:
    total = 0.0
    k = 0
    with torch.no_grad():
        for p in model.parameters():
            if not p.requires_grad:
                continue
            if k >= len(before):
                break
            d = p.detach() - before[k].to(device=p.device)
            total += float(torch.sum(d * d).detach().cpu().item())
            k += 1
    return float(math.sqrt(max(0.0, total)))


def rollout_buffer_fingerprint(*arrays: np.ndarray) -> str:
    h = hashlib.sha256()
    for idx, arr in enumerate(arrays):
        a = np.ascontiguousarray(arr)
        h.update(f"array_{idx}:{a.shape}:{a.dtype}".encode("utf-8"))
        h.update(a.view(np.uint8))
    return h.hexdigest()


def rollout_buffer_diagnostics(
    *,
    update: int,
    obs_buf: np.ndarray,
    act_buf: np.ndarray,
    rew_buf: np.ndarray,
    done_buf: np.ndarray,
    mask_buf: np.ndarray,
    policy_weight_buf: np.ndarray,
    override_buf: np.ndarray,
    val_buf: np.ndarray,
    adv_buf_raw: np.ndarray,
    ret_buf: np.ndarray,
    prev_hash: Optional[str],
    same_buffer_streak: int,
    env_episode_indices: Sequence[int],
    rollout_seconds: float,
) -> Dict[str, object]:
    B = int(act_buf.size)
    flat_actions = act_buf.reshape(-1)
    action_counts = {int(a): int(np.sum(flat_actions == a)) for a in (0, 1, 2)}
    action_rates = {int(a): float(action_counts[a] / max(1, B)) for a in (0, 1, 2)}

    obs_flat = obs_buf.reshape(B, obs_buf.shape[-1])
    if B > 0:
        rounded_obs = np.round(obs_flat, decimals=6)
        unique_obs_frac = float(np.unique(rounded_obs, axis=0).shape[0] / max(1, B))
    else:
        unique_obs_frac = 0.0

    fp = rollout_buffer_fingerprint(obs_buf, act_buf, rew_buf, done_buf, mask_buf)
    changed = bool(prev_hash is None or fp != prev_hash)

    mask_flat = mask_buf.reshape(B, mask_buf.shape[-1])
    legal_rates = {
        int(a): float(np.mean(mask_flat[:, a])) if B > 0 else 0.0
        for a in range(mask_flat.shape[-1])
    }

    return {
        "update": int(update),
        "buffer_hash": fp,
        "buffer_hash_prefix": fp[:16],
        "buffer_changed_vs_prev": bool(changed),
        "same_buffer_streak": int(same_buffer_streak),
        "samples": int(B),
        "rollout_seconds": float(rollout_seconds),
        "env_episode_index_min": int(min(env_episode_indices)) if env_episode_indices else 0,
        "env_episode_index_max": int(max(env_episode_indices)) if env_episode_indices else 0,
        "done_count": int(np.sum(done_buf)),
        "done_rate": float(np.mean(done_buf)) if B > 0 else 0.0,
        "reward_mean": float(np.mean(rew_buf)),
        "reward_std": float(np.std(rew_buf)),
        "reward_min": float(np.min(rew_buf)),
        "reward_max": float(np.max(rew_buf)),
        "reward_nonzero_rate": float(np.mean(np.abs(rew_buf) > 1e-12)),
        "obs_unique_frac_rounded_6dp": float(unique_obs_frac),
        "obs_global_std_mean": float(np.mean(np.std(obs_flat, axis=0))) if B > 1 else 0.0,
        "value_mean": float(np.mean(val_buf)),
        "value_std": float(np.std(val_buf)),
        "adv_raw_mean": float(np.mean(adv_buf_raw)),
        "adv_raw_std": float(np.std(adv_buf_raw)),
        "ret_mean": float(np.mean(ret_buf)),
        "ret_std": float(np.std(ret_buf)),
        "action_0_count": action_counts[0],
        "action_1_M_count": action_counts[1],
        "action_2_H_count": action_counts[2],
        "action_0_rate": action_rates[0],
        "action_1_M_rate": action_rates[1],
        "action_2_H_rate": action_rates[2],
        "max_action_rate": float(max(action_rates.values())),
        "mask_0_legal_rate": legal_rates.get(0, 0.0),
        "mask_1_M_legal_rate": legal_rates.get(1, 0.0),
        "mask_2_H_legal_rate": legal_rates.get(2, 0.0),
        "override_count": int(np.sum(override_buf)),
        "override_rate": float(np.mean(override_buf)) if B > 0 else 0.0,
        "policy_weight_mean": float(np.mean(policy_weight_buf)) if B > 0 else 0.0,
    }


def write_training_monitor_plots(
    out_dir: str,
    eval_rows: List[Dict],
    train_diag_rows: List[Dict],
    buffer_rows: List[Dict],
) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    def plot_one(df: pd.DataFrame, x: str, y: str, name: str, ylabel: str) -> None:
        if df.empty or x not in df.columns or y not in df.columns:
            return
        dd = df[[x, y]].dropna()
        if dd.empty:
            return
        plt.figure(figsize=(9, 5))
        plt.plot(dd[x].to_numpy(), dd[y].to_numpy(), marker="o", linewidth=1.5)
        plt.xlabel(x)
        plt.ylabel(ylabel)
        plt.title(name.replace("_", " "))
        plt.grid(True, alpha=0.25)
        plt.tight_layout()
        plt.savefig(out / name, dpi=150, bbox_inches="tight")
        plt.close()

    eval_df = pd.DataFrame(eval_rows)
    diag_df = pd.DataFrame(train_diag_rows)
    buf_df = pd.DataFrame(buffer_rows)

    plot_one(eval_df, "update", "avg_points", "learning_curve_avg_points.png", "Average points")
    plot_one(eval_df, "update", "avg_finish_position", "learning_curve_avg_finish_position.png", "Average finish position")
    plot_one(eval_df, "update", "legal_rate", "learning_curve_legal_rate.png", "Legal both compounds rate")
    plot_one(eval_df, "update", "top_strategy_share", "strategy_top_share.png", "Most common strategy share")
    plot_one(diag_df, "update", "entropy_mean", "ppo_entropy.png", "Policy entropy")
    plot_one(diag_df, "update", "approx_kl_mean", "ppo_approx_kl.png", "Approx KL")
    plot_one(diag_df, "update", "grad_norm_mean", "ppo_grad_norm.png", "Gradient norm")
    plot_one(diag_df, "update", "policy_param_delta_l2", "ppo_policy_param_delta.png", "Parameter update L2")
    plot_one(buf_df, "update", "obs_unique_frac_rounded_6dp", "buffer_unique_obs_fraction.png", "Unique obs fraction")

    if not buf_df.empty and {"update", "action_0_rate", "action_1_M_rate", "action_2_H_rate"}.issubset(buf_df.columns):
        plt.figure(figsize=(9, 5))
        plt.plot(buf_df["update"], buf_df["action_0_rate"], marker="o", linewidth=1.2, label="stay out")
        plt.plot(buf_df["update"], buf_df["action_1_M_rate"], marker="o", linewidth=1.2, label="pit M")
        plt.plot(buf_df["update"], buf_df["action_2_H_rate"], marker="o", linewidth=1.2, label="pit H")
        plt.xlabel("update")
        plt.ylabel("action rate")
        plt.title("rollout action rates")
        plt.grid(True, alpha=0.25)
        plt.legend()
        plt.tight_layout()
        plt.savefig(out / "rollout_action_rates.png", dpi=150, bbox_inches="tight")
        plt.close()


    # Figure used in the manuscript: median held-out finishing position and IQR.
    required_finish = {"update", "median_finish_position", "finish_q25", "finish_q75"}
    if not eval_df.empty and required_finish.issubset(eval_df.columns):
        dd = eval_df[list(required_finish)].dropna().sort_values("update")
        if not dd.empty:
            plt.figure(figsize=(9, 5))
            x = dd["update"].to_numpy(dtype=float)
            med = dd["median_finish_position"].to_numpy(dtype=float)
            q25 = dd["finish_q25"].to_numpy(dtype=float)
            q75 = dd["finish_q75"].to_numpy(dtype=float)
            plt.plot(x, med, linewidth=1.8)
            plt.fill_between(x, q25, q75, alpha=0.20)
            plt.xlabel("PPO update")
            plt.ylabel("Finishing position")
            plt.title("Median finishing position during PPO training")
            plt.gca().invert_yaxis()
            plt.grid(True, alpha=0.25)
            plt.tight_layout()
            plt.savefig(out / "median_finish_vs_update.png", dpi=180, bbox_inches="tight")
            plt.close()

    # Figure used in the manuscript: policy change between successive
    # validation checkpoints on a fixed set of probe observations.
    required_stability = {
        "update",
        "relative_parameter_change_vs_prev_eval",
        "policy_kl_vs_prev_eval",
    }
    if not eval_df.empty and required_stability.issubset(eval_df.columns):
        dd = (
            eval_df[list(required_stability)]
            .replace([np.inf, -np.inf], np.nan)
            .dropna()
            .sort_values("update")
        )
        if not dd.empty:
            fig, ax1 = plt.subplots(figsize=(9, 5))
            x = dd["update"].to_numpy(dtype=float)
            y1 = dd["relative_parameter_change_vs_prev_eval"].to_numpy(dtype=float)
            y2 = dd["policy_kl_vs_prev_eval"].to_numpy(dtype=float)

            line1 = ax1.plot(
                x,
                y1,
                linewidth=1.8,
                label="Relative parameter change",
            )[0]
            ax1.set_xlabel("PPO update")
            ax1.set_ylabel("Relative parameter change")
            ax1.grid(True, alpha=0.25)

            ax2 = ax1.twinx()
            line2 = ax2.plot(
                x,
                y2,
                linewidth=1.8,
                linestyle="--",
                label="Policy KL divergence",
            )[0]
            ax2.set_ylabel("Mean policy KL divergence")

            ax1.legend(
                [line1, line2],
                [line1.get_label(), line2.get_label()],
                loc="upper right",
            )
            plt.title("Policy stability between validation checkpoints")
            fig.tight_layout()
            fig.savefig(
                out / "convergence_parameters_diagnostics.png",
                dpi=180,
                bbox_inches="tight",
            )
            plt.close(fig)


def assess_policy_health(
    *,
    update: int,
    ppo_cfg: PPOConfig,
    train_row: Optional[Dict],
    buffer_row: Optional[Dict],
    latest_eval_row: Optional[Dict],
    initial_eval_row: Optional[Dict],
) -> Dict[str, object]:
    alerts: List[str] = []
    warmup_update = max(int(ppo_cfg.eval_every), int(0.05 * ppo_cfg.max_updates))

    if buffer_row is not None:
        if not bool(buffer_row.get("buffer_changed_vs_prev", True)):
            alerts.append("ROLLOUT_BUFFER_REUSED_HASH_EQUAL_PREVIOUS_UPDATE")
        if int(buffer_row.get("same_buffer_streak", 0)) > int(ppo_cfg.max_allowed_same_buffer_streak):
            alerts.append("ROLLOUT_BUFFER_SAME_STREAK_TOO_HIGH")
        if float(buffer_row.get("obs_unique_frac_rounded_6dp", 1.0)) < 0.05:
            alerts.append("ROLLOUT_OBSERVATIONS_LOW_DIVERSITY")
        if update >= warmup_update and float(buffer_row.get("max_action_rate", 0.0)) >= float(ppo_cfg.max_action_collapse_rate):
            alerts.append("ROLLOUT_ACTION_COLLAPSE")

    if train_row is not None:
        grad_norm = _safe_float(train_row.get("grad_norm_mean"), 0.0)
        param_delta = _safe_float(train_row.get("policy_param_delta_l2"), 0.0)
        entropy = _safe_float(train_row.get("entropy_mean"), 0.0)
        approx_kl = _safe_float(train_row.get("approx_kl_mean"), 0.0)
        if not math.isfinite(grad_norm) or grad_norm <= float(ppo_cfg.min_nonzero_grad_norm):
            alerts.append("GRADIENTS_DEAD_OR_NONFINITE")
        if not math.isfinite(param_delta) or param_delta <= float(ppo_cfg.min_policy_param_delta):
            alerts.append("POLICY_PARAMETERS_NOT_CHANGING")
        if update >= warmup_update and entropy < float(ppo_cfg.min_entropy_after_warmup):
            alerts.append("POLICY_ENTROPY_TOO_LOW")
        if not math.isfinite(approx_kl):
            alerts.append("APPROX_KL_NONFINITE")

    if latest_eval_row is not None:
        legal_rate = _safe_float(latest_eval_row.get("legal_rate"), 0.0)
        avg_points = _safe_float(latest_eval_row.get("avg_points"), 0.0)
        top_strategy_share = _safe_float(latest_eval_row.get("top_strategy_share"), 0.0)
        avg_pit_stops = _safe_float(latest_eval_row.get("avg_pit_stops"), 0.0)
        action0_rate = _safe_float(latest_eval_row.get("eval_action_0_rate"), 0.0)
        if legal_rate < 0.90 and update >= warmup_update:
            alerts.append("EVAL_LEGAL_RATE_LOW")
        if update >= warmup_update and (avg_pit_stops < 0.5 or action0_rate > 0.98):
            alerts.append("EVAL_STAY_OUT_COLLAPSE")
        if update >= warmup_update and top_strategy_share >= 0.97 and avg_points <= 1.0:
            alerts.append("EVAL_STRATEGY_COLLAPSE")
        if initial_eval_row is not None and update >= warmup_update:
            init_points = _safe_float(initial_eval_row.get("avg_points"), avg_points)
            if avg_points + 1e-9 < init_points - 0.25:
                alerts.append("EVAL_REGRESSED_BELOW_INITIAL_POINTS")

    return {
        "update": int(update),
        "ok": bool(len(alerts) == 0),
        "alerts": alerts,
        "n_alerts": int(len(alerts)),
    }


def save_basic_checkpoint(
    *,
    model: nn.Module,
    optimizer: Optional[optim.Optimizer],
    path_model: str,
    path_full: str,
    update: int,
    obs_dim: int,
    extra: Optional[Dict] = None,
) -> None:
    torch.save(model.state_dict(), path_model)
    payload = {
        "update": int(update),
        "obs_dim": int(obs_dim),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": None if optimizer is None else optimizer.state_dict(),
        "extra": {} if extra is None else dict(extra),
    }
    torch.save(payload, path_full)


class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
        )
        self.pi = nn.Linear(hidden, 3)
        self.v = nn.Linear(hidden, 1)

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.net(obs)
        return self.pi(h), self.v(h).squeeze(-1)


def masked_categorical_sample(logits: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    masked_logits = torch.where(mask.bool(), logits, torch.full_like(logits, -1e9))
    dist = torch.distributions.Categorical(logits=masked_logits)
    a = dist.sample()
    logp = dist.log_prob(a)
    return a, logp


def masked_categorical_logp_entropy(
    logits: torch.Tensor,
    actions: torch.Tensor,
    mask: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    masked_logits = torch.where(mask.bool(), logits, torch.full_like(logits, -1e9))
    dist = torch.distributions.Categorical(logits=masked_logits)
    return dist.log_prob(actions), dist.entropy()


@torch.no_grad()
def greedy_action(model: ActorCritic, obs: np.ndarray, mask: np.ndarray, device: str) -> int:
    obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
    logits_t, _ = model(obs_t)
    logits = logits_t.squeeze(0).detach().cpu().numpy()
    masked_logits = logits.copy()
    masked_logits[mask.astype(bool) == False] = -1e9
    return int(np.argmax(masked_logits))


def action_agreement(seq_a: Sequence[int], seq_b: Sequence[int]) -> float:
    n = min(len(seq_a), len(seq_b))
    if n == 0:
        return 0.0
    same = sum(int(seq_a[i] == seq_b[i]) for i in range(n))
    return float(same / n)



def collect_fixed_probe_observations(
    race_cfg: RaceConfig,
    *,
    n_episodes: int = 20,
    max_observations: int = 1500,
    seed0: int = 880_000,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Construct a fixed set of probe observations and action masks.

    A seeded random legal policy is used only to populate the probe set. The
    same probes are then reused at every validation checkpoint when measuring
    how much the policy distribution has changed.
    """
    obs_rows: List[np.ndarray] = []
    mask_rows: List[np.ndarray] = []
    action_rng = np.random.default_rng(int(seed0 + 12345))

    for ep in range(int(n_episodes)):
        if len(obs_rows) >= int(max_observations):
            break

        env = MultiCarRaceEnv(
            race_cfg=race_cfg,
            driver_cfgs=DRIVER_CONFIGS,
            seed=int(seed0 + ep),
            ego_mode="ppo",
        )
        obs = env.reset()
        done = False

        while not done and len(obs_rows) < int(max_observations):
            mask = env.action_mask().astype(np.int64)
            obs_rows.append(obs.astype(np.float32).copy())
            mask_rows.append(mask.copy())

            legal = np.flatnonzero(mask > 0)
            action = int(action_rng.choice(legal))
            obs, _reward, done, _info = env.step(action)

    if not obs_rows:
        raise RuntimeError("Could not construct fixed policy-probe observations.")

    return (
        np.stack(obs_rows, axis=0).astype(np.float32),
        np.stack(mask_rows, axis=0).astype(np.int64),
    )


@torch.no_grad()
def policy_kl_on_fixed_probes(
    current_model: ActorCritic,
    previous_model: ActorCritic,
    probe_obs: np.ndarray,
    probe_masks: np.ndarray,
    device: str,
) -> float:
    obs_t = torch.tensor(probe_obs, dtype=torch.float32, device=device)
    mask_t = torch.tensor(probe_masks, dtype=torch.bool, device=device)

    current_logits, _ = current_model(obs_t)
    previous_logits, _ = previous_model(obs_t)

    very_negative_current = torch.full_like(current_logits, -1e9)
    very_negative_previous = torch.full_like(previous_logits, -1e9)

    current_logits = torch.where(mask_t, current_logits, very_negative_current)
    previous_logits = torch.where(mask_t, previous_logits, very_negative_previous)

    current_logp = torch.log_softmax(current_logits, dim=-1)
    previous_logp = torch.log_softmax(previous_logits, dim=-1)
    current_p = torch.softmax(current_logits, dim=-1)

    kl = torch.sum(
        current_p * (current_logp - previous_logp),
        dim=-1,
    )
    return float(torch.mean(kl).detach().cpu().item())


@torch.no_grad()
def relative_parameter_change(
    current_model: nn.Module,
    previous_model: nn.Module,
) -> float:
    numerator = 0.0
    denominator = 0.0

    for current_param, previous_param in zip(
        current_model.parameters(),
        previous_model.parameters(),
    ):
        current_tensor = current_param.detach()
        previous_tensor = previous_param.detach().to(current_tensor.device)
        diff = current_tensor - previous_tensor

        numerator += float(torch.sum(diff * diff).cpu().item())
        denominator += float(torch.sum(previous_tensor * previous_tensor).cpu().item())

    return float(
        math.sqrt(max(0.0, numerator))
        / max(1e-12, math.sqrt(max(0.0, denominator)))
    )


def maybe_override_action_for_curriculum(
    env: MultiCarRaceEnv,
    proposed_action: int,
    update_idx: int,
    ppo_cfg: PPOConfig,
) -> Tuple[int, bool]:
    """
    Return (action_used, intervention_used).

    The intervention flag is important: when the curriculum forcibly changes the
    action, train_ppo can keep the transition for value learning while excluding
    it from the policy-gradient term. That prevents PPO from pretending that a
    forced action was sampled by the policy.
    """
    cutoff_update = int(max(1, round(ppo_cfg.override_fraction_updates * ppo_cfg.max_updates)))
    if update_idx > cutoff_update:
        return int(proposed_action), False

    if not env._new_neutral_triggered_since_last_decision():
        return int(proposed_action), False

    if float(env.rng.random()) >= float(ppo_cfg.override_prob_on_new_neutral):
        return int(proposed_action), False

    mask = env.action_mask()
    legal_pits = [a for a in [1, 2] if mask[a] == 1]

    if float(env.rng.random()) < float(ppo_cfg.override_force_pit_prob):
        if legal_pits:
            return int(env.rng.choice(np.asarray(legal_pits, dtype=int))), True
        return 0, True

    return 0, True


def make_envs(race_cfg: RaceConfig, ppo_cfg: PPOConfig) -> List[MultiCarRaceEnv]:
    return [
        MultiCarRaceEnv(race_cfg=race_cfg, driver_cfgs=DRIVER_CONFIGS, seed=ppo_cfg.seed + 1000 + i, ego_mode="ppo")
        for i in range(ppo_cfg.n_envs)
    ]


@torch.no_grad()
def evaluate_policy(
    model: ActorCritic,
    race_cfg: RaceConfig,
    device: str,
    n_episodes: int,
    seed0: int = 50_000,
) -> Tuple[Dict[str, float], pd.DataFrame, List[int]]:
    rows: List[Dict] = []
    action_trace: List[int] = []

    for ep in range(n_episodes):
        env = MultiCarRaceEnv(race_cfg=race_cfg, driver_cfgs=DRIVER_CONFIGS, seed=seed0 + ep, ego_mode="ppo")
        obs = env.reset()
        done = False
        info: Dict = {}
        action_seq: List[int] = []

        while not done:
            mask = env.action_mask()
            a = greedy_action(model, obs, mask, device=device)
            action_trace.append(int(a))
            action_seq.append(int(a))
            obs, _r, done, info = env.step(a)

        ego = env._ego_car()
        start_comp = str(ego.comps_end[0]) if len(ego.comps_end) > 0 else str(ego.q_end)
        ego_strategy = strategy_string_from_actions_and_start(
            action_by_lap=ego.action_by_lap,
            start_compound=start_comp,
            total_laps=env.cfg.L,
        )
        pit_laps = [int(lap) for lap, x in enumerate(ego.action_by_lap) if str(x) in ("M", "H")]
        n_actions = len(action_seq)
        n0 = int(sum(a == 0 for a in action_seq))
        nM = int(sum(a == 1 for a in action_seq))
        nH = int(sum(a == 2 for a in action_seq))

        rows.append(
            {
                "episode": int(ep),
                "lap1_position": info.get("lap1_position", np.nan),
                "finish_position": info.get("finish_position", np.nan),
                "finish_position_raw": info.get("finish_position_raw", np.nan),
                "points": info.get("points", np.nan),
                "training_reward": info.get("training_reward", np.nan),
                "legal_both_compounds": info.get("legal_both_compounds", False),
                "ego_retired": info.get("ego_retired", False),
                "n_vsc": info.get("n_vsc", 0),
                "n_sc": info.get("n_sc", 0),
                "Delta_HM": info.get("Delta_HM", np.nan),
                "dM": info.get("dM", np.nan),
                "dH": info.get("dH", np.nan),
                "ego_pit_stops_done": int(ego.pit_stops_done),
                "ego_used_M": bool(ego.used_M),
                "ego_used_H": bool(ego.used_H),
                "ego_final_compound": str(ego.q_end),
                "ego_strategy": ego_strategy,
                "ego_pit_laps": "|".join(str(x) for x in pit_laps),
                "first_pit_lap": pit_laps[0] if len(pit_laps) >= 1 else np.nan,
                "second_pit_lap": pit_laps[1] if len(pit_laps) >= 2 else np.nan,
                "n_policy_decisions": int(n_actions),
                "n_action_0": int(n0),
                "n_action_1_M": int(nM),
                "n_action_2_H": int(nH),
                "action_0_rate": float(n0 / max(1, n_actions)),
                "action_1_M_rate": float(nM / max(1, n_actions)),
                "action_2_H_rate": float(nH / max(1, n_actions)),
            }
        )

    df = pd.DataFrame(rows)
    if df.empty:
        summary = {
            "avg_points": float("nan"),
            "min_points": float("nan"),
            "max_points": float("nan"),
            "avg_finish_position": float("nan"),
            "median_finish_position": float("nan"),
            "finish_q25": float("nan"),
            "finish_q75": float("nan"),
            "min_finish_position": float("nan"),
            "max_finish_position": float("nan"),
            "avg_lap1_position": float("nan"),
            "podium_probability": float("nan"),
            "win_probability": float("nan"),
            "legal_rate": float("nan"),
            "retire_rate": float("nan"),
            "avg_n_vsc": float("nan"),
            "avg_n_sc": float("nan"),
            "avg_pit_stops": float("nan"),
            "unique_strategy_count": 0,
            "top_strategy_share": float("nan"),
            "eval_action_0_rate": float("nan"),
            "eval_action_1_M_rate": float("nan"),
            "eval_action_2_H_rate": float("nan"),
        }
        return summary, df, action_trace

    strategy_counts = Counter(str(x) for x in df["ego_strategy"].astype(str).tolist())
    top_strategy, top_count = strategy_counts.most_common(1)[0] if strategy_counts else ("", 0)
    total_decisions = int(df["n_policy_decisions"].sum())
    total_n0 = int(df["n_action_0"].sum())
    total_nM = int(df["n_action_1_M"].sum())
    total_nH = int(df["n_action_2_H"].sum())

    summary = {
        "avg_points": float(df["points"].mean()),
        "min_points": float(df["points"].min()),
        "max_points": float(df["points"].max()),
        "avg_finish_position": float(df["finish_position"].mean()),
        "median_finish_position": float(df["finish_position"].median()),
        "finish_q25": float(df["finish_position"].quantile(0.25)),
        "finish_q75": float(df["finish_position"].quantile(0.75)),
        "min_finish_position": float(df["finish_position"].min()),
        "max_finish_position": float(df["finish_position"].max()),
        "avg_lap1_position": float(df["lap1_position"].mean()),
        "podium_probability": float((df["finish_position"] <= 3).mean()),
        "win_probability": float((df["finish_position"] == 1).mean()),
        "legal_rate": float(df["legal_both_compounds"].mean()),
        "retire_rate": float(df["ego_retired"].mean()),
        "avg_n_vsc": float(df["n_vsc"].mean()),
        "avg_n_sc": float(df["n_sc"].mean()),
        "avg_pit_stops": float(df["ego_pit_stops_done"].mean()),
        "unique_strategy_count": int(len(strategy_counts)),
        "top_strategy_share": float(top_count / max(1, len(df))),
        "top_strategy": str(top_strategy),
        "eval_action_0_rate": float(total_n0 / max(1, total_decisions)),
        "eval_action_1_M_rate": float(total_nM / max(1, total_decisions)),
        "eval_action_2_H_rate": float(total_nH / max(1, total_decisions)),
    }
    return summary, df, action_trace


# =============================================================================
# TRACE / DATA EXTRACTION
# =============================================================================

def build_lap_df_from_env(env: MultiCarRaceEnv) -> pd.DataFrame:
    rows: List[Dict] = []

    for c in env.cars:
        n_done = len(c.lap_times)
        for k, tau in enumerate(c.lap_times, start=1):
            rows.append(
                {
                    "car": c.name,
                    "grid_pos": int(c.grid_pos),
                    "lap": int(k),
                    "lap_time": float(tau),
                    "compound": str(c.comps_end[k - 1]) if k - 1 < len(c.comps_end) else "",
                    "tyre_age_end": int(c.ages_end[k - 1]) if k - 1 < len(c.ages_end) else np.nan,
                    "retired": bool(not c.active),
                    "retired_at": float(c.retired_at) if c.retired_at is not None else np.nan,
                    "completed_full_race": bool(c.active and n_done >= env.cfg.L),
                }
            )

    lap_df = pd.DataFrame(rows)
    if lap_df.empty:
        return pd.DataFrame(
            columns=[
                "car", "grid_pos", "lap", "lap_time", "compound", "tyre_age_end",
                "cum_time", "position", "retired", "retired_at", "completed_full_race"
            ]
        )

    lap_df = lap_df.sort_values(["car", "lap"]).reset_index(drop=True)
    lap_df["cum_time"] = lap_df.groupby("car")["lap_time"].cumsum()
    lap_df["position"] = (
        lap_df.groupby("lap")["cum_time"]
        .rank(method="first", ascending=True)
        .astype(int)
    )
    return lap_df


def build_finish_df_from_env(env: MultiCarRaceEnv) -> pd.DataFrame:
    rows: List[Dict] = []
    for c in env.cars:
        if c.active and len(c.lap_times) >= env.cfg.L:
            finish_time = float(sum(c.lap_times[:env.cfg.L]))
            classified = True
        else:
            finish_time = math.inf
            classified = False

        rows.append(
            {
                "car": c.name,
                "grid_pos": int(c.grid_pos),
                "finish_time": float(finish_time),
                "classified": bool(classified),
                "retired": bool(not c.active),
                "retired_at": float(c.retired_at) if c.retired_at is not None else np.nan,
            }
        )

    finish_df = pd.DataFrame(rows).sort_values(["finish_time", "car"]).reset_index(drop=True)
    finish_df["finish_pos"] = np.arange(1, len(finish_df) + 1)
    return finish_df


def build_events_df_from_env(env: MultiCarRaceEnv) -> pd.DataFrame:
    if not env.event_log:
        return pd.DataFrame(columns=["time", "lap", "type", "follower", "leader"])
    return pd.DataFrame(env.event_log).sort_values(["time", "lap", "type"]).reset_index(drop=True)


def build_traces_from_env(env: MultiCarRaceEnv) -> Dict[str, List[float]]:
    traces: Dict[str, List[float]] = {}
    z_ref = float(env.cfg.trace_ref_time)
    for c in env.cars:
        cum = 0.0
        arr: List[float] = []
        for tau in c.lap_times:
            cum += float(z_ref - float(tau))
            arr.append(float(cum))
        traces[c.name] = arr
    return traces


# =============================================================================
# PLOTTING
# =============================================================================

def _neutral_rectangles_from_scenario(
    scenario: EpisodeScenario,
    env: MultiCarRaceEnv,
) -> List[Tuple[int, int, str]]:
    rects: List[Tuple[int, int, str]] = []

    def leader_lap_at_time(t: float) -> int:
        max_lap = 0
        for c in env.cars:
            cum = 0.0
            for k, tau in enumerate(c.lap_times, start=1):
                cum += float(tau)
                if cum <= t + 1e-9:
                    max_lap = max(max_lap, k)
        return int(max_lap)

    for w in scenario.neutral_windows:
        lap_start = max(1, leader_lap_at_time(float(w.start_time)))
        lap_end = max(lap_start, leader_lap_at_time(float(w.end_time)))
        rects.append((lap_start, lap_end + 1, w.kind))
    return rects


def plot_race_traces_with_neutrals(
    traces: Dict[str, List[float]],
    finish_df: pd.DataFrame,
    env: MultiCarRaceEnv,
    out_png: str,
    title_prefix: str,
) -> None:
    plt.figure(figsize=(14, 8))

    finish_order = finish_df["car"].astype(str).tolist()
    ego_name = env.ego_name
    car_to_team = {c.name: DRIVER_TO_TEAM.get(c.name, "UNKNOWN") for c in env.cars}
    car_to_color = {car_name: TEAM_COLORS.get(team_name, "#000000") for car_name, team_name in car_to_team.items()}
    car_to_obj = {c.name: c for c in env.cars}

    rects = _neutral_rectangles_from_scenario(env.scenario, env)

    ymins = []
    ymaxs = []
    for nm, tr in traces.items():
        if len(tr) > 0:
            ymins.append(min([0.0] + tr))
            ymaxs.append(max([0.0] + tr))
    ymin = min(ymins) if ymins else -10.0
    ymax = max(ymaxs) if ymaxs else 10.0
    yrange = max(1.0, ymax - ymin)
    ymin_plot = ymin - 0.08 * yrange
    ymax_plot = ymax + 0.12 * yrange

    for x0, x1, kind in rects:
        alpha = 0.12 if kind == "VSC" else 0.18
        color = "#FFD966" if kind == "VSC" else "#D9E2F3"
        plt.axvspan(x0, x1, alpha=alpha, color=color, zorder=0)

    for nm in finish_order:
        if nm not in traces:
            continue

        tr = traces[nm]
        if len(tr) == 0:
            continue

        x = np.arange(0, len(tr) + 1)
        y = np.zeros(len(tr) + 1, dtype=float)
        y[1:] = np.asarray(tr, dtype=float)

        color = car_to_color.get(nm, "#000000")
        lw = 2.8 if nm == ego_name else 1.2
        alpha = 1.0 if nm == ego_name else 0.80
        z = 5 if nm == ego_name else 2

        plt.plot(x, y, color=color, lw=lw, alpha=alpha, zorder=z)

        car_obj = car_to_obj[nm]
        strategy_txt = strategy_string_from_actions_and_start(
            action_by_lap=car_obj.action_by_lap,
            start_compound=str(car_obj.comps_end[0]) if len(car_obj.comps_end) > 0 else str(car_obj.q_end),
            total_laps=env.cfg.L,
        )

        x_lab = float(len(tr) + 0.35)
        y_lab = float(tr[-1])
        plt.text(
            x_lab,
            y_lab,
            f"{nm}  [{strategy_txt}]",
            fontsize=8,
            color=color,
            va="center",
            ha="left",
            zorder=20,
        )

        if not car_obj.active and len(tr) > 0:
            xr = len(tr)
            yr = tr[-1]
            plt.scatter([xr], [yr], marker="x", s=60, linewidths=2.0, color=color, zorder=10)

    events_df = build_events_df_from_env(env)
    if events_df is not None and not events_df.empty:
        for _, ev in events_df.iterrows():
            etype = str(ev.get("type", ""))
            if etype not in ("SL_PASS", "BF_YIELD"):
                continue

            follower = str(ev.get("follower", ""))
            lap_k = int(ev.get("lap", -1))

            if follower not in traces:
                continue
            if lap_k < 1 or lap_k > len(traces[follower]):
                continue

            x = float(lap_k)
            y = float(traces[follower][lap_k - 1])
            color = car_to_color.get(follower, "#000000")

            if etype == "SL_PASS":
                plt.scatter([x], [y], marker="x", s=45, linewidths=1.8, color=color, zorder=10)
            else:
                plt.scatter([x], [y], marker="o", s=42, facecolors="none", edgecolors=color, linewidths=1.8, zorder=10)

    neutral_txt = neutral_windows_lap_summary(env)

    plt.ylim(ymin_plot, ymax_plot)
    plt.xlim(0, env.cfg.L + 8)
    plt.xlabel("Lap")
    plt.ylabel(f"Cumulative trace sum(z_ref - tau), z_ref={env.cfg.trace_ref_time:.1f}")
    plt.title(
        f"{title_prefix} | Delta_HM={env.scenario.Delta_HM:.3f}, dM={env.scenario.dM:.3f}, dH={env.scenario.dH:.3f}\n"
        f"lap1={env.episode_info.get('lap1_position', 'nan')}, "
        f"finish={env.episode_info.get('finish_position', 'nan')}, "
        f"points={env.episode_info.get('points', 'nan')}\n"
        f"{neutral_txt}"
    )
    plt.grid(True, alpha=0.25)

    teams_in_plot = []
    for nm in finish_order:
        team = car_to_team.get(nm, "UNKNOWN")
        if team not in teams_in_plot:
            teams_in_plot.append(team)

    team_handles = [
        Line2D([0], [0], color=TEAM_COLORS.get(team, "#000000"), lw=3, label=team)
        for team in teams_in_plot
    ]
    rect_handles = [
        Line2D([0], [0], color="#FFD966", lw=6, alpha=0.6, label="VSC window"),
        Line2D([0], [0], color="#D9E2F3", lw=6, alpha=0.8, label="SC window"),
    ]
    leg1 = plt.legend(handles=team_handles + rect_handles, loc="upper left", ncol=2, fontsize=8, frameon=True)
    plt.gca().add_artist(leg1)

    event_handles = [
        Line2D([0], [0], marker="x", color="black", linestyle="None", markersize=8, label="On-track pass"),
        Line2D([0], [0], marker="o", color="black", markerfacecolor="none", linestyle="None", markersize=8, label="Blue-flag yield"),
    ]
    plt.legend(handles=event_handles, loc="lower left", fontsize=9, frameon=True)

    plt.tight_layout()
    plt.savefig(out_png, dpi=180, bbox_inches="tight")
    plt.close()


# =============================================================================
# TRACE BUNDLES
# =============================================================================

def save_trace_bundle_ppo(
    model: ActorCritic,
    race_cfg: RaceConfig,
    device: str,
    out_dir: str,
    n_episodes: int = 20,
    seed0: int = 400_000,
) -> pd.DataFrame:
    trace_dir = Path(out_dir) / "ppo_race_traces"
    trace_dir.mkdir(parents=True, exist_ok=True)

    rows: List[Dict] = []
    for ep in range(n_episodes):
        env = MultiCarRaceEnv(race_cfg=race_cfg, driver_cfgs=DRIVER_CONFIGS, seed=seed0 + ep, ego_mode="ppo")
        obs = env.reset()
        done = False
        info: Dict = {}
        action_seq: List[str] = []

        while not done:
            mask = env.action_mask()
            a = greedy_action(model, obs, mask, device=device)
            action_seq.append({0: "0", 1: "M", 2: "H"}[int(a)])
            obs, _r, done, info = env.step(a)

        lap_df = build_lap_df_from_env(env)
        finish_df = build_finish_df_from_env(env)
        events_df = build_events_df_from_env(env)
        traces = build_traces_from_env(env)

        lap_df.to_csv(trace_dir / f"trace_episode_{ep:02d}_lap_data.csv", index=False)
        finish_df.to_csv(trace_dir / f"trace_episode_{ep:02d}_finish.csv", index=False)
        events_df.to_csv(trace_dir / f"trace_episode_{ep:02d}_events.csv", index=False)

        plot_race_traces_with_neutrals(
            traces=traces,
            finish_df=finish_df,
            env=env,
            out_png=str(trace_dir / f"trace_episode_{ep:02d}.png"),
            title_prefix=f"PPO ego race {ep:02d}",
        )

        # 10-lap zoom plots
        for lap_start in range(1, env.cfg.L + 1, 10):
            lap_end = min(lap_start + 9, env.cfg.L)
            plot_race_traces_window_with_neutrals(
                traces=traces,
                finish_df=finish_df,
                env=env,
                out_png=str(trace_dir / f"trace_episode_{ep:02d}_laps_{lap_start:02d}_{lap_end:02d}.png"),
                title_prefix=f"PPO ego race {ep:02d}",
                lap_start=lap_start,
                lap_end=lap_end,
            )

        print(f"\n--- PPO trace episode {ep:02d} ---")
        print(f"Neutrals: {neutral_windows_lap_summary(env)}")
        print("Finish order and strategies:")

        finish_rows_console = []
        for c in env.cars:
            if c.active and len(c.lap_times) >= env.cfg.L:
                finish_key = (0, float(sum(c.lap_times[:env.cfg.L])))
            else:
                finish_key = (1, -len(c.lap_times), float(c.retired_at) if c.retired_at is not None else math.inf)
            finish_rows_console.append((c.name, finish_key))

        finish_rows_console.sort(key=lambda x: x[1])

        for pos, (name, _) in enumerate(finish_rows_console, start=1):
            c = next(cc for cc in env.cars if cc.name == name)
            start_comp = str(c.comps_end[0]) if len(c.comps_end) > 0 else str(c.q_end)
            strat = strategy_string_from_actions_and_start(c.action_by_lap, start_comp, env.cfg.L)
            retired_txt = " RET" if not c.active else ""
            print(f"P{pos:02d}  {name:<12s}  {strat}{retired_txt}")

        ego = env._ego_car()
        rows.append(
            {
                "episode": int(ep),
                "lap1_position": info.get("lap1_position", np.nan),
                "finish_position": info.get("finish_position", np.nan),
                "points": info.get("points", np.nan),
                "legal_both_compounds": info.get("legal_both_compounds", False),
                "ego_retired": info.get("ego_retired", False),
                "ego_pit_stops_done": int(ego.pit_stops_done),
                "ego_final_compound": str(ego.q_end),
                "ego_used_M": bool(ego.used_M),
                "ego_used_H": bool(ego.used_H),
                "ego_action_sequence": "|".join(action_seq),
                "n_events_logged": int(len(events_df)),
            }
        )

    summary_df = pd.DataFrame(rows)
    summary_df.to_csv(trace_dir / "trace_summary.csv", index=False)
    return summary_df
# =============================================================================
# TRAIN PPO
# =============================================================================

def train_ppo(race_cfg: RaceConfig, ppo_cfg: PPOConfig) -> Tuple[ActorCritic, pd.DataFrame, Dict, pd.DataFrame]:
    os.makedirs(ppo_cfg.out_dir, exist_ok=True)
    checkpoint_dir = os.path.join(ppo_cfg.out_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    torch.manual_seed(ppo_cfg.seed)
    np.random.seed(ppo_cfg.seed)
    random.seed(ppo_cfg.seed)

    device = torch.device(ppo_cfg.device)
    envs = make_envs(race_cfg, ppo_cfg)

    first_obs = envs[0].reset()
    obs_dim = first_obs.shape[0]
    model = ActorCritic(obs_dim=obs_dim, hidden=128).to(device)
    optimizer = optim.Adam(model.parameters(), lr=ppo_cfg.lr)

    probe_obs, probe_masks = collect_fixed_probe_observations(
        race_cfg=race_cfg,
        n_episodes=20,
        max_observations=1500,
        seed0=880_000,
    )
    previous_eval_model = ActorCritic(obs_dim=obs_dim, hidden=128).to(device)
    previous_eval_model.load_state_dict(model.state_dict())
    previous_eval_model.eval()

    latest_ckpt_path = os.path.join(ppo_cfg.out_dir, "ppo_ego_multicar_neutral.pt")
    latest_full_ckpt_path = os.path.join(ppo_cfg.out_dir, "ppo_ego_multicar_neutral_full.pt")
    best_ckpt_path = os.path.join(ppo_cfg.out_dir, "ppo_ego_multicar_neutral_best.pt")
    best_full_ckpt_path = os.path.join(ppo_cfg.out_dir, "ppo_ego_multicar_neutral_best_full.pt")

    T = int(ppo_cfg.rollout_steps)
    N = int(ppo_cfg.n_envs)
    B = T * N

    obs_list = [first_obs.copy()] + [env.reset() for env in envs[1:]]

    eval_rows: List[Dict] = []
    train_diag_rows: List[Dict] = []
    buffer_rows: List[Dict] = []
    prev_eval_actions: Optional[List[int]] = None
    stable_count = 0
    converged_update: Optional[int] = None

    best_eval_key: Optional[Tuple[float, float, float]] = None
    best_update: Optional[int] = None
    prev_buffer_hash: Optional[str] = None
    same_buffer_streak: int = 0
    latest_health: Dict[str, object] = {"update": 0, "ok": True, "alerts": []}
    compound_bonus_active = True
    compound_bonus_disabled_update: Optional[int] = None

    def make_eval_key(summary: Dict[str, float]) -> Tuple[float, float, float]:
        return (
            float(summary["legal_rate"]),
            float(summary["avg_points"]),
            -float(summary["avg_finish_position"]),
        )

    print("\n=== INITIAL PPO EVAL BEFORE TRAINING ===", flush=True)
    initial_summary, initial_eval_df, initial_eval_actions = evaluate_policy(
        model=model,
        race_cfg=race_cfg,
        device=str(device),
        n_episodes=ppo_cfg.eval_episodes,
        seed0=90_000,
    )
    initial_eval_df.to_csv(os.path.join(ppo_cfg.out_dir, "initial_eval_episodes.csv"), index=False)
    initial_eval_row = {
        "update": 0,
        **initial_summary,
        "action_agreement_vs_prev_eval": 0.0,
        "delta_avg_points_vs_prev_eval": float("inf"),
        "is_initial_eval": True,
        "relative_parameter_change_vs_prev_eval": float("nan"),
        "policy_kl_vs_prev_eval": float("nan"),
        "compound_bonus_active_after_eval": True,
        "compound_bonus_value_after_eval": float(race_cfg.other_compound_bonus),
    }
    eval_rows.append(initial_eval_row)
    prev_eval_actions = initial_eval_actions
    best_eval_key = make_eval_key(initial_summary)
    best_update = 0
    save_basic_checkpoint(
        model=model,
        optimizer=optimizer,
        path_model=best_ckpt_path,
        path_full=best_full_ckpt_path,
        update=0,
        obs_dim=obs_dim,
        extra={"initial_summary": initial_summary},
    )
    print(
        f"[update {0:4d}] "
        f"avg_points={initial_summary['avg_points']:.3f} "
        f"avg_finish={initial_summary['avg_finish_position']:.3f} "
        f"legal_rate={initial_summary['legal_rate']:.3f} "
        f"top_strategy_share={initial_summary.get('top_strategy_share', float('nan')):.3f} "
        f"unique_strategies={initial_summary.get('unique_strategy_count', 0)}",
        flush=True,
    )

    pd.DataFrame(eval_rows).to_csv(os.path.join(ppo_cfg.out_dir, "training_eval_history.csv"), index=False)

    for update in range(1, int(ppo_cfg.max_updates) + 1):
        rollout_t0 = time.time()
        obs_buf = np.zeros((T, N, obs_dim), dtype=np.float32)
        act_buf = np.zeros((T, N), dtype=np.int64)
        logp_buf = np.zeros((T, N), dtype=np.float32)
        rew_buf = np.zeros((T, N), dtype=np.float32)
        done_buf = np.zeros((T, N), dtype=np.float32)
        val_buf = np.zeros((T, N), dtype=np.float32)
        mask_buf = np.zeros((T, N, 3), dtype=np.float32)
        policy_weight_buf = np.ones((T, N), dtype=np.float32)
        override_buf = np.zeros((T, N), dtype=np.float32)

        for t in range(T):
            for i, env in enumerate(envs):
                obs_buf[t, i] = obs_list[i]
                mask_buf[t, i] = env.action_mask().astype(np.float32)

            obs_t = torch.tensor(obs_buf[t], dtype=torch.float32, device=device)
            mask_t = torch.tensor(mask_buf[t], dtype=torch.float32, device=device)

            with torch.no_grad():
                logits, values = model(obs_t)
                sampled_actions, _sampled_logps = masked_categorical_sample(logits, mask_t)

            sampled_np = sampled_actions.cpu().numpy()
            val_np = values.cpu().numpy()
            used_actions: List[int] = []
            intervention_flags: List[bool] = []

            for i, env in enumerate(envs):
                a_used, intervened = maybe_override_action_for_curriculum(
                    env=env,
                    proposed_action=int(sampled_np[i]),
                    update_idx=update,
                    ppo_cfg=ppo_cfg,
                )
                used_actions.append(int(a_used))
                intervention_flags.append(bool(intervened))

            used_actions_t = torch.tensor(used_actions, dtype=torch.int64, device=device)
            with torch.no_grad():
                used_logps_t, _used_entropy_t = masked_categorical_logp_entropy(logits, used_actions_t, mask_t)
            used_logp_np = used_logps_t.cpu().numpy()

            for i, env in enumerate(envs):
                obs2, r, done, _info = env.step(int(used_actions[i]))
                rew_buf[t, i] = float(r)
                done_buf[t, i] = 1.0 if done else 0.0
                act_buf[t, i] = int(used_actions[i])
                logp_buf[t, i] = float(used_logp_np[i])
                val_buf[t, i] = float(val_np[i])
                override_buf[t, i] = 1.0 if intervention_flags[i] else 0.0
                policy_weight_buf[t, i] = (
                    float(ppo_cfg.policy_update_weight_for_overrides)
                    if intervention_flags[i]
                    else 1.0
                )

                if done:
                    obs_list[i] = env.reset()
                else:
                    obs_list[i] = obs2

        rollout_seconds = float(time.time() - rollout_t0)

        with torch.no_grad():
            last_obs = torch.tensor(np.stack(obs_list, axis=0), dtype=torch.float32, device=device)
            _, last_values = model(last_obs)
            last_values_np = last_values.cpu().numpy()

        adv_buf = np.zeros((T, N), dtype=np.float32)
        ret_buf = np.zeros((T, N), dtype=np.float32)
        for i in range(N):
            gae = 0.0
            for t in reversed(range(T)):
                next_value = last_values_np[i] if t == T - 1 else val_buf[t + 1, i]
                next_nonterminal = 1.0 - done_buf[t, i]
                delta = rew_buf[t, i] + ppo_cfg.gamma * next_value * next_nonterminal - val_buf[t, i]
                gae = delta + ppo_cfg.gamma * ppo_cfg.gae_lambda * next_nonterminal * gae
                adv_buf[t, i] = gae
                ret_buf[t, i] = adv_buf[t, i] + val_buf[t, i]

        obs_flat = obs_buf.reshape(B, obs_dim)
        act_flat = act_buf.reshape(B)
        logp_old_flat = logp_buf.reshape(B)
        adv_raw_flat = adv_buf.reshape(B).copy()
        ret_flat = ret_buf.reshape(B)
        val_old_flat = val_buf.reshape(B)
        mask_flat = mask_buf.reshape(B, 3)
        policy_weight_flat = policy_weight_buf.reshape(B)

        adv_mean = float(adv_raw_flat.mean())
        adv_std = float(adv_raw_flat.std())
        adv_flat = (adv_raw_flat - adv_mean) / (adv_std + 1e-8)

        buffer_hash = rollout_buffer_fingerprint(obs_buf, act_buf, rew_buf, done_buf, mask_buf)
        if prev_buffer_hash is not None and buffer_hash == prev_buffer_hash:
            same_buffer_streak += 1
        else:
            same_buffer_streak = 0
        env_episode_indices = [int(env.episode_index) for env in envs]
        buffer_row = rollout_buffer_diagnostics(
            update=update,
            obs_buf=obs_buf,
            act_buf=act_buf,
            rew_buf=rew_buf,
            done_buf=done_buf,
            mask_buf=mask_buf,
            policy_weight_buf=policy_weight_buf,
            override_buf=override_buf,
            val_buf=val_buf,
            adv_buf_raw=adv_buf,
            ret_buf=ret_buf,
            prev_hash=prev_buffer_hash,
            same_buffer_streak=same_buffer_streak,
            env_episode_indices=env_episode_indices,
            rollout_seconds=rollout_seconds,
        )
        buffer_rows.append(buffer_row)
        prev_buffer_hash = buffer_hash

        idx = np.arange(B)
        params_before = clone_model_params(model)
        entropy_coef_now = current_entropy_coef(ppo_cfg, update)
        mb_stats: List[Dict[str, float]] = []
        early_stop_epoch: Optional[int] = None

        for epoch in range(int(ppo_cfg.ppo_epochs)):
            np.random.shuffle(idx)
            epoch_kls: List[float] = []
            for start_mb in range(0, B, int(ppo_cfg.minibatch_size)):
                mb = idx[start_mb:start_mb + int(ppo_cfg.minibatch_size)]
                if len(mb) == 0:
                    continue

                obs_mb = torch.tensor(obs_flat[mb], dtype=torch.float32, device=device)
                act_mb = torch.tensor(act_flat[mb], dtype=torch.int64, device=device)
                logp_old_mb = torch.tensor(logp_old_flat[mb], dtype=torch.float32, device=device)
                adv_mb = torch.tensor(adv_flat[mb], dtype=torch.float32, device=device)
                ret_mb = torch.tensor(ret_flat[mb], dtype=torch.float32, device=device)
                val_old_mb = torch.tensor(val_old_flat[mb], dtype=torch.float32, device=device)
                mask_mb = torch.tensor(mask_flat[mb], dtype=torch.float32, device=device)
                policy_weight_mb = torch.tensor(policy_weight_flat[mb], dtype=torch.float32, device=device)

                logits, values = model(obs_mb)
                logp, entropy = masked_categorical_logp_entropy(logits, act_mb, mask_mb)

                logratio = logp - logp_old_mb
                ratio = torch.exp(logratio)
                surr1 = ratio * adv_mb
                surr2 = torch.clamp(ratio, 1.0 - ppo_cfg.clip_eps, 1.0 + ppo_cfg.clip_eps) * adv_mb
                pg_terms = -torch.min(surr1, surr2)

                policy_weight_sum = torch.clamp(policy_weight_mb.sum(), min=1e-8)
                policy_loss = torch.sum(pg_terms * policy_weight_mb) / policy_weight_sum

                if float(ppo_cfg.value_clip_eps) > 0.0:
                    values_clipped = val_old_mb + torch.clamp(
                        values - val_old_mb,
                        -float(ppo_cfg.value_clip_eps),
                        float(ppo_cfg.value_clip_eps),
                    )
                    value_loss_unclipped = (ret_mb - values) ** 2
                    value_loss_clipped = (ret_mb - values_clipped) ** 2
                    value_loss = torch.mean(torch.max(value_loss_unclipped, value_loss_clipped))
                else:
                    value_loss = torch.mean((ret_mb - values) ** 2)

                entropy_loss = -torch.mean(entropy)
                loss = policy_loss + ppo_cfg.value_coef * value_loss + entropy_coef_now * entropy_loss

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                grad_norm_before_clip = nn.utils.clip_grad_norm_(model.parameters(), ppo_cfg.max_grad_norm)
                optimizer.step()

                with torch.no_grad():
                    approx_kl_t = ((ratio - 1.0) - logratio) * policy_weight_mb
                    approx_kl = torch.sum(approx_kl_t) / policy_weight_sum
                    clip_frac = torch.sum(((torch.abs(ratio - 1.0) > ppo_cfg.clip_eps).float()) * policy_weight_mb) / policy_weight_sum

                stat = {
                    "policy_loss": float(policy_loss.detach().cpu().item()),
                    "value_loss": float(value_loss.detach().cpu().item()),
                    "entropy": float(torch.mean(entropy).detach().cpu().item()),
                    "approx_kl": float(approx_kl.detach().cpu().item()),
                    "clip_frac": float(clip_frac.detach().cpu().item()),
                    "ratio_mean": float(torch.mean(ratio).detach().cpu().item()),
                    "ratio_std": float(torch.std(ratio).detach().cpu().item()) if ratio.numel() > 1 else 0.0,
                    "grad_norm": float(grad_norm_before_clip.detach().cpu().item() if torch.is_tensor(grad_norm_before_clip) else grad_norm_before_clip),
                    "policy_weight_mean": float(torch.mean(policy_weight_mb).detach().cpu().item()),
                }
                mb_stats.append(stat)
                epoch_kls.append(stat["approx_kl"])

            if epoch_kls and float(np.nanmean(epoch_kls)) > float(ppo_cfg.early_stop_kl_mult) * float(ppo_cfg.target_kl):
                early_stop_epoch = int(epoch + 1)
                break

        param_delta = model_param_delta_l2(model, params_before)

        def stat_mean(name: str, default: float = 0.0) -> float:
            vals = [float(s[name]) for s in mb_stats if name in s and math.isfinite(float(s[name]))]
            return float(np.mean(vals)) if vals else float(default)

        def stat_max(name: str, default: float = 0.0) -> float:
            vals = [float(s[name]) for s in mb_stats if name in s and math.isfinite(float(s[name]))]
            return float(np.max(vals)) if vals else float(default)

        train_row = {
            "update": int(update),
            "optimizer_steps": int(len(mb_stats)),
            "early_stop_epoch": -1 if early_stop_epoch is None else int(early_stop_epoch),
            "entropy_coef": float(entropy_coef_now),
            "policy_loss_mean": stat_mean("policy_loss"),
            "value_loss_mean": stat_mean("value_loss"),
            "entropy_mean": stat_mean("entropy"),
            "approx_kl_mean": stat_mean("approx_kl"),
            "approx_kl_max": stat_max("approx_kl"),
            "clip_frac_mean": stat_mean("clip_frac"),
            "ratio_mean": stat_mean("ratio_mean", 1.0),
            "ratio_std_mean": stat_mean("ratio_std"),
            "grad_norm_mean": stat_mean("grad_norm"),
            "grad_norm_max": stat_max("grad_norm"),
            "policy_param_delta_l2": float(param_delta),
            "adv_raw_mean": float(adv_mean),
            "adv_raw_std": float(adv_std),
            "explained_variance": float(explained_variance(val_old_flat, ret_flat)),
            "rollout_seconds": float(rollout_seconds),
            "total_update_seconds": float(time.time() - rollout_t0),
        }
        train_diag_rows.append(train_row)

        save_basic_checkpoint(
            model=model,
            optimizer=optimizer,
            path_model=latest_ckpt_path,
            path_full=latest_full_ckpt_path,
            update=update,
            obs_dim=obs_dim,
            extra={"train_row": train_row, "buffer_row": buffer_row},
        )

        if ppo_cfg.checkpoint_every > 0 and update % ppo_cfg.checkpoint_every == 0:
            periodic_ckpt_path = os.path.join(
                checkpoint_dir,
                f"ppo_ego_multicar_neutral_update_{update:05d}.pt",
            )
            periodic_full_ckpt_path = os.path.join(
                checkpoint_dir,
                f"ppo_ego_multicar_neutral_update_{update:05d}_full.pt",
            )
            save_basic_checkpoint(
                model=model,
                optimizer=optimizer,
                path_model=periodic_ckpt_path,
                path_full=periodic_full_ckpt_path,
                update=update,
                obs_dim=obs_dim,
                extra={"train_row": train_row, "buffer_row": buffer_row},
            )

        latest_eval_row: Optional[Dict] = None
        if update == 1 or update % ppo_cfg.eval_every == 0:
            summary, eval_episode_df, eval_actions = evaluate_policy(
                model=model,
                race_cfg=race_cfg,
                device=str(device),
                n_episodes=ppo_cfg.eval_episodes,
                seed0=100_000,
            )
            eval_episode_df.to_csv(
                os.path.join(ppo_cfg.out_dir, f"eval_episodes_update_{update:05d}.csv"),
                index=False,
            )
            agree = action_agreement(prev_eval_actions, eval_actions) if prev_eval_actions is not None else 0.0
            prev_points = eval_rows[-1]["avg_points"] if len(eval_rows) > 0 else None
            delta_points = abs(summary["avg_points"] - prev_points) if prev_points is not None else float("inf")

            if (
                compound_bonus_active
                and float(summary["legal_rate"]) >= 1.0 - 1e-12
            ):
                compound_bonus_active = False
                compound_bonus_disabled_update = int(update)
                for train_env in envs:
                    train_env.compound_bonus_active = False
                print(
                    f"[update {update:4d}] validation legality reached 1.000; "
                    "disabling the one-off other-compound exploration bonus for subsequent training.",
                    flush=True,
                )

            relative_change_eval = relative_parameter_change(
                current_model=model,
                previous_model=previous_eval_model,
            )
            policy_kl_eval = policy_kl_on_fixed_probes(
                current_model=model,
                previous_model=previous_eval_model,
                probe_obs=probe_obs,
                probe_masks=probe_masks,
                device=str(device),
            )

            latest_eval_row = {
                "update": int(update),
                **summary,
                "relative_parameter_change_vs_prev_eval": float(relative_change_eval),
                "policy_kl_vs_prev_eval": float(policy_kl_eval),
                "action_agreement_vs_prev_eval": float(agree),
                "delta_avg_points_vs_prev_eval": float(delta_points),
                "is_initial_eval": False,
                "compound_bonus_active_after_eval": bool(compound_bonus_active),
                "compound_bonus_value_after_eval": (
                    float(race_cfg.other_compound_bonus) if compound_bonus_active else 0.0
                ),
            }
            eval_rows.append(latest_eval_row)
            prev_eval_actions = eval_actions
            previous_eval_model.load_state_dict(model.state_dict())
            previous_eval_model.eval()

            current_key = make_eval_key(summary)
            if best_eval_key is None or current_key > best_eval_key:
                best_eval_key = current_key
                best_update = int(update)
                save_basic_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    path_model=best_ckpt_path,
                    path_full=best_full_ckpt_path,
                    update=update,
                    obs_dim=obs_dim,
                    extra={"summary": summary, "train_row": train_row, "buffer_row": buffer_row},
                )

            print(
                f"[update {update:4d}] "
                f"avg_points={summary['avg_points']:.3f} "
                f"avg_finish={summary['avg_finish_position']:.3f} "
                f"avg_lap1={summary['avg_lap1_position']:.3f} "
                f"legal_rate={summary['legal_rate']:.3f} "
                f"retire_rate={summary['retire_rate']:.3f} "
                f"pit_stops={summary.get('avg_pit_stops', float('nan')):.2f} "
                f"top_strategy_share={summary.get('top_strategy_share', float('nan')):.3f} "
                f"action_agreement={agree:.4f} "
                f"delta_points={delta_points:.4f} "
                f"KL={train_row['approx_kl_mean']:.5f} "
                f"H={train_row['entropy_mean']:.4f} "
                f"grad={train_row['grad_norm_mean']:.3e} "
                f"buf_changed={buffer_row['buffer_changed_vs_prev']} "
                f"| best_update={best_update}",
                flush=True,
            )

            latest_health = assess_policy_health(
                update=update,
                ppo_cfg=ppo_cfg,
                train_row=train_row,
                buffer_row=buffer_row,
                latest_eval_row=latest_eval_row,
                initial_eval_row=initial_eval_row,
            )
            if latest_health["alerts"]:
                msg = f"[health update {update}] " + "; ".join(str(x) for x in latest_health["alerts"])
                print(msg, flush=True)
                append_health_event(ppo_cfg.out_dir, msg)

            eval_points_improvement = float(summary["avg_points"] - initial_eval_row["avg_points"])
            collapsed_and_bad = bool(
                float(summary.get("top_strategy_share", 0.0)) >= 0.98
                and float(summary["avg_points"]) <= float(initial_eval_row["avg_points"]) + 0.25
            )
            strategy_quality_ok = bool(
                float(summary["legal_rate"]) >= 0.99
                and float(summary.get("avg_pit_stops", 0.0)) >= 1.0
                and not collapsed_and_bad
            )
            quality_gate = (
                float(summary["legal_rate"]) >= 0.99
                and float(summary["avg_finish_position"]) <= 6.5
                and eval_points_improvement >= float(ppo_cfg.min_eval_points_improvement_for_convergence)
                and latest_health.get("ok", False)
                and strategy_quality_ok
            )

            if (
                quality_gate
                and len(eval_rows) >= ppo_cfg.min_evals_before_convergence
                and delta_points <= ppo_cfg.convergence_points_tol
                and agree >= ppo_cfg.convergence_action_agreement
            ):
                stable_count += 1
            else:
                stable_count = 0

            if stable_count >= ppo_cfg.convergence_patience:
                converged_update = int(update)
                if ppo_cfg.stop_early_on_convergence:
                    print(
                        f"\nConverged at update {update} with quality + health gates satisfied.\n",
                        flush=True,
                    )
                    break

        if latest_eval_row is None:
            latest_health = assess_policy_health(
                update=update,
                ppo_cfg=ppo_cfg,
                train_row=train_row,
                buffer_row=buffer_row,
                latest_eval_row=None,
                initial_eval_row=initial_eval_row,
            )
            if update == 1 or update % max(1, int(ppo_cfg.diagnostics_every)) == 0:
                print(
                    f"[diag {update:4d}] "
                    f"buf={buffer_row['buffer_hash_prefix']} changed={buffer_row['buffer_changed_vs_prev']} "
                    f"same_streak={buffer_row['same_buffer_streak']} "
                    f"done={buffer_row['done_count']} reward_mean={buffer_row['reward_mean']:.3f} "
                    f"a0={buffer_row['action_0_rate']:.3f} aM={buffer_row['action_1_M_rate']:.3f} aH={buffer_row['action_2_H_rate']:.3f} "
                    f"KL={train_row['approx_kl_mean']:.5f} H={train_row['entropy_mean']:.4f} "
                    f"grad={train_row['grad_norm_mean']:.3e} dtheta={train_row['policy_param_delta_l2']:.3e}",
                    flush=True,
                )
            if latest_health["alerts"] and update % max(1, int(ppo_cfg.diagnostics_every)) == 0:
                msg = f"[health update {update}] " + "; ".join(str(x) for x in latest_health["alerts"])
                print(msg, flush=True)
                append_health_event(ppo_cfg.out_dir, msg)

        if update % max(1, int(ppo_cfg.diagnostics_every)) == 0:
            pd.DataFrame(train_diag_rows).to_csv(os.path.join(ppo_cfg.out_dir, "training_update_diagnostics.csv"), index=False)
            pd.DataFrame(buffer_rows).to_csv(os.path.join(ppo_cfg.out_dir, "rollout_buffer_diagnostics.csv"), index=False)
            pd.DataFrame(eval_rows).to_csv(os.path.join(ppo_cfg.out_dir, "training_eval_history.csv"), index=False)
            write_json_atomic(os.path.join(ppo_cfg.out_dir, "policy_health_latest.json"), latest_health)
            write_json_atomic(os.path.join(ppo_cfg.out_dir, "rollout_fingerprint_latest.json"), buffer_row)

        if int(ppo_cfg.write_rollout_npz_every) > 0 and update % int(ppo_cfg.write_rollout_npz_every) == 0:
            np.savez_compressed(
                os.path.join(ppo_cfg.out_dir, f"rollout_buffer_update_{update:05d}.npz"),
                obs=obs_buf,
                actions=act_buf,
                rewards=rew_buf,
                dones=done_buf,
                values=val_buf,
                masks=mask_buf,
                policy_weights=policy_weight_buf,
                overrides=override_buf,
            )

        if int(ppo_cfg.write_plots_every) > 0 and update % int(ppo_cfg.write_plots_every) == 0:
            write_training_monitor_plots(ppo_cfg.out_dir, eval_rows, train_diag_rows, buffer_rows)

    if not ppo_cfg.stop_early_on_convergence:
        converged_update = int(ppo_cfg.max_updates)
    elif converged_update is None:
        converged_update = int(ppo_cfg.max_updates)

    eval_df = pd.DataFrame(eval_rows)
    train_diag_df = pd.DataFrame(train_diag_rows)
    buffer_diag_df = pd.DataFrame(buffer_rows)

    if os.path.exists(best_ckpt_path):
        model.load_state_dict(torch.load(best_ckpt_path, map_location=device))
        print(f"\nReloaded best checkpoint from update {best_update} for final evaluation.\n", flush=True)
    else:
        print("\nBest checkpoint not found; using latest model for final evaluation.\n", flush=True)

    final_summary, final_eval_df, _ = evaluate_policy(
        model=model,
        race_cfg=race_cfg,
        device=str(device),
        n_episodes=ppo_cfg.final_eval_episodes,
        seed0=200_000,
    )

    trace_summary_df = save_trace_bundle_ppo(
        model=model,
        race_cfg=race_cfg,
        device=str(device),
        out_dir=ppo_cfg.out_dir,
        n_episodes=ppo_cfg.trace_eval_episodes,
        seed0=300_000,
    )

    save_basic_checkpoint(
        model=model,
        optimizer=optimizer,
        path_model=latest_ckpt_path,
        path_full=latest_full_ckpt_path,
        update=int(converged_update),
        obs_dim=obs_dim,
        extra={"final_summary": final_summary},
    )

    eval_df.to_csv(os.path.join(ppo_cfg.out_dir, "training_eval_history.csv"), index=False)
    train_diag_df.to_csv(os.path.join(ppo_cfg.out_dir, "training_update_diagnostics.csv"), index=False)
    buffer_diag_df.to_csv(os.path.join(ppo_cfg.out_dir, "rollout_buffer_diagnostics.csv"), index=False)
    final_eval_df.to_csv(os.path.join(ppo_cfg.out_dir, "final_eval_episodes.csv"), index=False)
    trace_summary_df.to_csv(os.path.join(ppo_cfg.out_dir, "ppo_trace_summary.csv"), index=False)
    write_training_monitor_plots(ppo_cfg.out_dir, eval_rows, train_diag_rows, buffer_rows)

    convergence_summary = {
        "converged_update": int(converged_update),
        "best_update": None if best_update is None else int(best_update),
        "obs_dim": int(obs_dim),
        "state_space_unchanged_obs_dim": int(obs_dim),
        "terminal_reward_unchanged": True,
        "compound_bonus_disabled_update": (
            None if compound_bonus_disabled_update is None else int(compound_bonus_disabled_update)
        ),
        "compound_bonus_active_at_end": bool(compound_bonus_active),
        "final_eval_episodes": int(ppo_cfg.final_eval_episodes),
        "initial_avg_points": float(initial_eval_row["avg_points"]),
        "final_minus_initial_avg_points": float(final_summary["avg_points"] - initial_eval_row["avg_points"]),
        "latest_health_ok": bool(latest_health.get("ok", False)),
        "latest_health_alerts": latest_health.get("alerts", []),
        **final_summary,
    }

    with open(os.path.join(ppo_cfg.out_dir, "convergence_summary.json"), "w", encoding="utf-8") as f:
        json.dump(convergence_summary, f, indent=2, default=_json_default)

    print("\n=== FINAL PPO SUMMARY ===", flush=True)
    print(pd.Series(convergence_summary).to_string(), flush=True)

    return model, eval_df, convergence_summary, final_eval_df



# =============================================================================
# PAPER-LEVEL EVALUATION, ABLATIONS AND ROBUSTNESS
# =============================================================================

def make_paper_race_config() -> RaceConfig:
    """
    Race configuration used for the controller experiments reported in the paper.
    """
    return RaceConfig(
        L=60,
        B=76.0,
        seed=7,
        Delta_HM_values=(0.0, 0.1, 0.2),
        dM_values=(0.05, 0.06, 0.06, 0.06, 0.07, 0.08, 0.09, 0.10),
        dH_values=(0.02, 0.03, 0.03, 0.04, 0.04, 0.05),
        pace_delta_perturb_min=-0.10,
        pace_delta_perturb_max=0.10,
        pace_delta_perturb_step=0.01,
        pace_delta_perturb_sigma=0.07,
        Pbar=20.0,
        sigma_P=0.25,
        pit_loss_mult_vsc=0.65,
        pit_loss_mult_sc=0.50,
        sigma_tau=0.16,
        sigma_tau_close=0.45,
        g_close=1.20,
        g_min=0.25,
        g_pass=0.20,
        Delta_OT=0.50,
        g_buf=0.40,
        A0=0.30,
        beta_pass=7.0,
        kappa_dirty=0.35,
        h_dirty=1.20,
        h_yield=0.80,
        beta_blue=10.0,
        Delta_blue=0.20,
        A_blue=0.10,
        lap1_npasses=6,
        lap1_pswap=0.30,
        lap1_gap_lo=0.30,
        lap1_gap_hi=1.00,
        lap1_start_loss_mu=2.50,
        lap1_start_loss_sd=0.80,
        competitor_top_k=7,
        max_stops=2,
        min_stint=6,
        no_pit_last_n=1,
        mandatory_both=True,
        ego_max_stops=2,
        ego_min_stint=5,
        ego_no_pit_last_n=5,
        invalid_no_both_penalty=50.0,
        other_compound_bonus=10.0,
        trace_ref_time=81.0,
        position_reward_scale=2.0,
        vsc_mult=1.18,
        sc_pre_mult=1.28,
        sc_bunch_mult=1.38,
        vsc_prob_per_lap=0.006,
        sc_prob_per_lap=0.014,
        vsc_min_duration=100.0,
        vsc_max_duration=300.0,
        sc_min_duration=350.0,
        sc_max_duration=650.0,
        sc_pre_bunch_duration=130.0,
        no_neutral_pit_first_n=5,
        no_neutral_pit_last_n=5,
        no_pit_lap1=True,
        deterministic_eval_episodes=500,
        deterministic_trace_episodes=10,
        ppo_trace_episodes=20,
        expected_sc_replan_laps=5,
        observation_ablation="none",
    )


def make_paper_ppo_config(
    *,
    seed: int,
    out_dir: str,
    device: str = "cpu",
    neutral_curriculum: bool = True,
) -> PPOConfig:
    """
    PPO configuration reported in the paper.
    """
    return PPOConfig(
        seed=int(seed),
        device=str(device),
        n_envs=16,
        rollout_steps=4 * (60 - 1),  # 236
        max_updates=6000,
        gamma=0.999,
        gae_lambda=0.99,
        clip_eps=0.20,
        lr=3e-4,
        entropy_coef=0.02,
        entropy_coef_final=0.005,
        entropy_anneal_updates=2500,
        value_coef=0.50,
        value_clip_eps=0.20,
        target_kl=0.03,
        early_stop_kl_mult=1.50,
        max_grad_norm=0.50,
        ppo_epochs=6,
        minibatch_size=512,
        eval_every=25,
        eval_episodes=150,
        final_eval_episodes=500,
        trace_eval_episodes=20,
        override_fraction_updates=(0.25 if neutral_curriculum else 0.0),
        override_prob_on_new_neutral=0.35,
        override_force_pit_prob=0.50,
        policy_update_weight_for_overrides=0.0,
        checkpoint_every=1000,
        diagnostics_every=1,
        write_plots_every=25,
        write_rollout_npz_every=100,
        health_json_every=1,
        stop_early_on_convergence=False,
        out_dir=str(out_dir),
    )


def summarize_outcomes(df: pd.DataFrame) -> Dict[str, float]:
    if df.empty:
        return {
            "mean_finish": float("nan"),
            "median_finish": float("nan"),
            "mean_points": float("nan"),
            "median_points": float("nan"),
            "podium_probability": float("nan"),
            "win_probability": float("nan"),
            "legal_rate": float("nan"),
        }

    return {
        "mean_finish": float(df["finish_position"].mean()),
        "median_finish": float(df["finish_position"].median()),
        "mean_points": float(df["points"].mean()),
        "median_points": float(df["points"].median()),
        "podium_probability": float((df["finish_position"] <= 3).mean()),
        "win_probability": float((df["finish_position"] == 1).mean()),
        "legal_rate": float(df["legal_both_compounds"].mean()),
    }


def episode_summary_from_env(env: MultiCarRaceEnv) -> Dict[str, object]:
    info = dict(env.episode_info)
    decisions = list(env.decision_log)
    pit_decisions = [d for d in decisions if bool(d.get("pit", False))]

    new_discount_opportunities = [
        d for d in decisions
        if bool(d.get("new_neutral", False))
        and bool(d.get("discounted_pit_available", False))
    ]
    new_discount_pits = [
        d for d in new_discount_opportunities
        if bool(d.get("pit", False))
    ]
    discounted_pits = [
        d for d in pit_decisions
        if str(d.get("regime", "")) in ("VSC", "SC_PRE")
    ]

    first_pit_lap = (
        float(pit_decisions[0]["lap"])
        if pit_decisions
        else float("nan")
    )

    return {
        "finish_position": int(info.get("finish_position", len(env.cars))),
        "points": float(info.get("points", 0.0)),
        "legal_both_compounds": bool(info.get("legal_both_compounds", False)),
        "first_pit_lap": first_pit_lap,
        "pit_stops": int(len(pit_decisions)),
        "new_discount_opportunities": int(len(new_discount_opportunities)),
        "new_discount_pits": int(len(new_discount_pits)),
        "discounted_pits": int(len(discounted_pits)),
    }


def _sample_benchmark_ego_style(
    env: MultiCarRaceEnv,
    seed: int,
) -> None:
    """
    Give the benchmark ego the same top-team tactical-style distribution used
    for front-running non-ego competitors. This is sampled only on the benchmark
    branch and therefore does not alter the shared initial race state.
    """
    assert env.scenario is not None
    rng = np.random.default_rng(int(seed))
    style_kind = "normal"
    shift_laps = 0

    if float(rng.random()) < float(env.cfg.top_team_strategy_variability_prob):
        u = float(rng.random())
        if u < float(env.cfg.top_team_extend_prob):
            style_kind = "extend"
            shift_laps = int(
                rng.integers(
                    int(env.cfg.strategy_extend_laps_min),
                    int(env.cfg.strategy_extend_laps_max) + 1,
                )
            )
        elif u < float(env.cfg.top_team_extend_prob + env.cfg.top_team_undercut_prob):
            style_kind = "undercut"
            shift_laps = int(
                rng.integers(
                    int(env.cfg.strategy_undercut_laps_min),
                    int(env.cfg.strategy_undercut_laps_max) + 1,
                )
            )

    env.scenario.competitor_styles[env.ego_name] = {
        "style": style_kind,
        "shift_laps": int(shift_laps),
    }


def run_benchmark_from_shared_state(
    shared_env: MultiCarRaceEnv,
    benchmark_seed: int,
) -> MultiCarRaceEnv:
    """
    Continue a copy of the common initial state using the domain-informed
    benchmark for the ego car.
    """
    env = copy.deepcopy(shared_env)
    env.ego_mode = "deterministic"

    _sample_benchmark_ego_style(env, benchmark_seed)

    ego_driver = next(d for d in env.driver_static if d.is_ego)
    start_compound = str(env.scenario.start_compounds[ego_driver.name])

    actions, _chosen, _rank_idx = env._sample_competitor_strategy(
        drv=ego_driver,
        start_compound=start_compound,
    )
    ego = env._ego_car()
    ego.action_by_lap = list(actions)

    # The shared state is paused at the first ego decision.
    env.pending_ego_decision = False
    env._post_cross_choose_next_action(ego, ego_action=None)
    env._schedule_all_unscheduled_crossings()
    env._advance_until_ego_decision_or_done()

    while not env.done:
        env._schedule_all_unscheduled_crossings()
        env._advance_until_ego_decision_or_done()

    return env


@torch.no_grad()
def run_ppo_from_shared_state(
    model: ActorCritic,
    shared_env: MultiCarRaceEnv,
    device: str,
) -> MultiCarRaceEnv:
    """
    Continue a copy of the common initial state using greedy PPO actions.
    """
    env = copy.deepcopy(shared_env)
    env.ego_mode = "ppo"
    env.compound_bonus_active = False

    obs = env._get_obs()
    done = bool(env.done)

    while not done:
        mask = env.action_mask()
        action = greedy_action(model, obs, mask, device=device)
        obs, _reward, done, _info = env.step(action)

    return env


def evaluate_paired_models(
    models: Sequence[Tuple[str, ActorCritic]],
    race_cfg: RaceConfig,
    *,
    device: str,
    n_scenarios: int = 500,
    seed0: int = 600_000,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Paired evaluation used for the main PPO-versus-benchmark comparison.

    Each scenario is simulated to the first ego decision once. That complete
    state is then copied. The benchmark and every PPO training run therefore
    begin from the same realised race state. After the first decision, each
    branch evolves independently.
    """
    ppo_rows: List[Dict[str, object]] = []
    benchmark_rows: List[Dict[str, object]] = []

    for ep in range(int(n_scenarios)):
        scenario_seed = int(seed0 + ep)

        base_env = MultiCarRaceEnv(
            race_cfg=race_cfg,
            driver_cfgs=DRIVER_CONFIGS,
            seed=scenario_seed,
            ego_mode="ppo",
        )
        base_env.reset()

        benchmark_env = run_benchmark_from_shared_state(
            shared_env=base_env,
            benchmark_seed=9_000_000 + scenario_seed,
        )
        benchmark_row = {
            "scenario": int(ep),
            **episode_summary_from_env(benchmark_env),
        }
        benchmark_rows.append(benchmark_row)

        for run_label, model in models:
            ppo_env = run_ppo_from_shared_state(
                model=model,
                shared_env=base_env,
                device=device,
            )
            ppo_rows.append(
                {
                    "training_run": str(run_label),
                    "scenario": int(ep),
                    **episode_summary_from_env(ppo_env),
                }
            )

    ppo_df = pd.DataFrame(ppo_rows)
    benchmark_df = pd.DataFrame(benchmark_rows)

    merged = ppo_df.merge(
        benchmark_df,
        on="scenario",
        how="left",
        suffixes=("_ppo", "_benchmark"),
        validate="many_to_one",
    )

    merged["finish_improvement"] = (
        merged["finish_position_benchmark"]
        - merged["finish_position_ppo"]
    )
    merged["points_improvement"] = (
        merged["points_ppo"]
        - merged["points_benchmark"]
    )
    merged["podium_difference"] = (
        (merged["finish_position_ppo"] <= 3).astype(float)
        - (merged["finish_position_benchmark"] <= 3).astype(float)
    )
    merged["win_difference"] = (
        (merged["finish_position_ppo"] == 1).astype(float)
        - (merged["finish_position_benchmark"] == 1).astype(float)
    )

    return ppo_df, benchmark_df, merged


def paired_two_level_bootstrap(
    paired_df: pd.DataFrame,
    *,
    n_bootstrap: int = 20_000,
    seed: int = 2026,
) -> pd.DataFrame:
    """
    Resample PPO training runs and race scenarios, matching the uncertainty
    analysis described in the paper.
    """
    run_labels = list(pd.unique(paired_df["training_run"]))
    scenarios = sorted(int(x) for x in pd.unique(paired_df["scenario"]))

    metrics = [
        "finish_improvement",
        "points_improvement",
        "podium_difference",
        "win_difference",
    ]

    matrices: Dict[str, np.ndarray] = {}
    for metric in metrics:
        pivot = (
            paired_df
            .pivot(index="training_run", columns="scenario", values=metric)
            .reindex(index=run_labels, columns=scenarios)
        )
        if pivot.isna().any().any():
            raise RuntimeError(
                f"Missing run/scenario combinations in paired bootstrap for {metric}."
            )
        matrices[metric] = pivot.to_numpy(dtype=float)

    rng = np.random.default_rng(int(seed))
    n_runs = len(run_labels)
    n_scenarios = len(scenarios)

    boot = {
        metric: np.empty(int(n_bootstrap), dtype=float)
        for metric in metrics
    }

    for b in range(int(n_bootstrap)):
        run_idx = rng.integers(0, n_runs, size=n_runs)
        scenario_idx = rng.integers(0, n_scenarios, size=n_scenarios)

        for metric in metrics:
            sample = matrices[metric][np.ix_(run_idx, scenario_idx)]
            boot[metric][b] = float(sample.mean())

    rows: List[Dict[str, float | str]] = []
    display_names = {
        "finish_improvement": "Mean finishing-position improvement",
        "points_improvement": "Mean championship-points improvement",
        "podium_difference": "Podium-probability difference",
        "win_difference": "Win-probability difference",
    }

    for metric in metrics:
        observed = float(matrices[metric].mean())
        lo, hi = np.percentile(boot[metric], [2.5, 97.5])
        rows.append(
            {
                "outcome": display_names[metric],
                "observed_difference": observed,
                "ci_2.5_percent": float(lo),
                "ci_97.5_percent": float(hi),
            }
        )

    return pd.DataFrame(rows)


def behavioural_summary(df: pd.DataFrame) -> Dict[str, float]:
    pit_stops_total = float(df["pit_stops"].sum())
    new_discount_opportunities = float(df["new_discount_opportunities"].sum())

    return {
        "mean_first_pit_lap": float(df["first_pit_lap"].mean()),
        "probability_pit_at_new_discounted_neutralisation": (
            float(df["new_discount_pits"].sum()) / new_discount_opportunities
            if new_discount_opportunities > 0
            else float("nan")
        ),
        "fraction_stops_under_discount": (
            float(df["discounted_pits"].sum()) / pit_stops_total
            if pit_stops_total > 0
            else float("nan")
        ),
        "mean_number_of_stops": float(df["pit_stops"].mean()),
        "legal_rate": float(df["legal_both_compounds"].mean()),
    }


def save_main_paired_outputs(
    *,
    models: Sequence[Tuple[str, ActorCritic]],
    race_cfg: RaceConfig,
    out_dir: str,
    device: str,
    n_scenarios: int = 500,
    bootstrap_reps: int = 20_000,
) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    ppo_df, benchmark_df, paired_df = evaluate_paired_models(
        models=models,
        race_cfg=race_cfg,
        device=device,
        n_scenarios=n_scenarios,
        seed0=600_000,
    )

    ppo_df.to_csv(out / "paired_ppo_episodes.csv", index=False)
    benchmark_df.to_csv(out / "paired_benchmark_episodes.csv", index=False)
    paired_df.to_csv(out / "paired_differences.csv", index=False)

    comparison = pd.DataFrame(
        [
            {"controller": "PPO", **summarize_outcomes(ppo_df)},
            {"controller": "Domain-informed benchmark", **summarize_outcomes(benchmark_df)},
        ]
    )
    comparison.to_csv(out / "main_policy_comparison.csv", index=False)

    bootstrap_df = paired_two_level_bootstrap(
        paired_df,
        n_bootstrap=bootstrap_reps,
        seed=2026,
    )
    bootstrap_df.to_csv(out / f"paired_bootstrap_{int(bootstrap_reps)}.csv", index=False)

    behaviour = pd.DataFrame(
        [
            {"controller": "PPO", **behavioural_summary(ppo_df)},
            {
                "controller": "Domain-informed benchmark",
                **behavioural_summary(benchmark_df),
            },
        ]
    )
    behaviour.to_csv(out / "policy_behaviour_summary.csv", index=False)

    print("\n=== MAIN PAIRED COMPARISON ===")
    print(comparison.to_string(index=False))
    print("\n=== PAIRED BOOTSTRAP ===")
    print(bootstrap_df.to_string(index=False))
    print("\n=== POLICY BEHAVIOUR ===")
    print(behaviour.to_string(index=False))


def train_main_seeds(
    *,
    race_cfg: RaceConfig,
    output_root: str,
    device: str,
    seeds: Sequence[int] = (1, 2, 3, 4, 5),
) -> List[Tuple[str, ActorCritic]]:
    root = Path(output_root) / "main_training"
    root.mkdir(parents=True, exist_ok=True)

    models: List[Tuple[str, ActorCritic]] = []
    rows: List[Dict[str, object]] = []

    for seed in seeds:
        run_dir = root / f"seed_{int(seed)}"
        ppo_cfg = make_paper_ppo_config(
            seed=int(seed),
            out_dir=str(run_dir),
            device=device,
            neutral_curriculum=True,
        )

        print(f"\n{'=' * 80}")
        print(f"TRAINING MAIN PPO SEED {seed}")
        print(f"{'=' * 80}\n")

        model, _eval_df, summary, _final_eval_df = train_ppo(
            race_cfg=race_cfg,
            ppo_cfg=ppo_cfg,
        )

        models.append((f"seed_{int(seed)}", model))
        rows.append(
            {
                "seed": int(seed),
                "mean_finish": float(summary["avg_finish_position"]),
                "mean_points": float(summary["avg_points"]),
                "podium_probability": float(summary["podium_probability"]),
                "win_probability": float(summary["win_probability"]),
                "legal_rate": float(summary["legal_rate"]),
            }
        )

    seed_df = pd.DataFrame(rows)
    seed_df.to_csv(root / "ppo_seed_stability.csv", index=False)

    numeric_cols = [
        "mean_finish",
        "mean_points",
        "podium_probability",
        "win_probability",
        "legal_rate",
    ]
    stats_rows = []
    for stat_name, fn in (
        ("mean", lambda x: x.mean()),
        ("sd", lambda x: x.std(ddof=1)),
    ):
        row: Dict[str, object] = {"statistic": stat_name}
        for col in numeric_cols:
            row[col] = float(fn(seed_df[col]))
        stats_rows.append(row)

    pd.DataFrame(stats_rows).to_csv(
        root / "ppo_seed_stability_summary.csv",
        index=False,
    )

    return models


def load_actor_critic_checkpoint(
    checkpoint_path: str | Path,
    *,
    device: str,
    obs_dim: int = 17,
) -> ActorCritic:
    path = Path(checkpoint_path)
    if not path.exists():
        raise FileNotFoundError(path)

    payload = torch.load(path, map_location=torch.device(device))
    if isinstance(payload, dict) and "model_state_dict" in payload:
        state_dict = payload["model_state_dict"]
    else:
        state_dict = payload

    model = ActorCritic(obs_dim=obs_dim, hidden=128).to(torch.device(device))
    model.load_state_dict(state_dict)
    model.eval()
    return model


def load_main_models(
    *,
    output_root: str,
    device: str,
    seeds: Sequence[int] = (1, 2, 3, 4, 5),
) -> List[Tuple[str, ActorCritic]]:
    models: List[Tuple[str, ActorCritic]] = []
    root = Path(output_root) / "main_training"

    for seed in seeds:
        path = root / f"seed_{int(seed)}" / "ppo_ego_multicar_neutral_best.pt"
        model = load_actor_critic_checkpoint(
            path,
            device=device,
            obs_dim=17,
        )
        models.append((f"seed_{int(seed)}", model))

    return models


def robustness_configurations(
    base_cfg: RaceConfig,
) -> Dict[str, RaceConfig]:
    return {
        "nominal": replace(base_cfg),
        "higher_degradation": replace(
            base_cfg,
            dM_values=tuple(1.25 * float(x) for x in base_cfg.dM_values),
            dH_values=tuple(1.25 * float(x) for x in base_cfg.dH_values),
        ),
        "higher_pit_loss": replace(
            base_cfg,
            Pbar=23.0,
        ),
        "more_frequent_neutralisations": replace(
            base_cfg,
            sc_prob_per_lap=1.2 * float(base_cfg.sc_prob_per_lap),
            vsc_prob_per_lap=1.2 * float(base_cfg.vsc_prob_per_lap),
        ),
        "more_difficult_overtaking": replace(
            base_cfg,
            Delta_OT=0.65,
        ),
    }


def run_robustness_suite(
    *,
    models: Sequence[Tuple[str, ActorCritic]],
    base_race_cfg: RaceConfig,
    output_root: str,
    device: str,
    n_scenarios: int = 500,
) -> pd.DataFrame:
    root = Path(output_root) / "robustness"
    root.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, object]] = []

    for name, cfg in robustness_configurations(base_race_cfg).items():
        print(f"\n{'=' * 80}")
        print(f"ROBUSTNESS: {name}")
        print(f"{'=' * 80}\n")

        ppo_df, benchmark_df, paired_df = evaluate_paired_models(
            models=models,
            race_cfg=cfg,
            device=device,
            n_scenarios=n_scenarios,
            seed0=700_000,
        )

        ppo_df.to_csv(root / f"{name}_ppo.csv", index=False)
        benchmark_df.to_csv(root / f"{name}_benchmark.csv", index=False)
        paired_df.to_csv(root / f"{name}_paired.csv", index=False)

        rows.append(
            {
                "race_distribution": name,
                "finish_improvement": float(paired_df["finish_improvement"].mean()),
                "points_improvement": float(paired_df["points_improvement"].mean()),
                "sd_finish_difference": float(
                    paired_df["finish_improvement"].std(ddof=1)
                ),
                "ppo_legal_rate": float(
                    ppo_df["legal_both_compounds"].mean()
                ),
            }
        )

    out_df = pd.DataFrame(rows)
    out_df.to_csv(root / "robustness_summary.csv", index=False)

    print("\n=== ROBUSTNESS SUMMARY ===")
    print(out_df.to_string(index=False))
    return out_df


def run_ablation_suite(
    *,
    base_race_cfg: RaceConfig,
    output_root: str,
    device: str,
    seeds: Sequence[int] = (1,),
) -> pd.DataFrame:
    """
    Retrain each ablated controller from scratch.

    The six variants correspond to the ablation study in the paper. The
    observation ablations retain the 17-dimensional tensor interface but set
    the removed components identically to zero throughout training and
    evaluation, which is equivalent to withholding those quantities from the
    controller.
    """
    root = Path(output_root) / "ablations"
    root.mkdir(parents=True, exist_ok=True)

    variants = [
        ("no_local_competitor_information", "no_local_competitor", True),
        ("no_ego_cumulative_degradation", "no_ego_degradation", True),
        ("no_laps_remaining", "no_laps_remaining", True),
        ("no_pit_vulnerability_estimate", "no_pit_vulnerability", True),
        ("no_neutralisation_indicators", "no_neutral_indicators", True),
        ("no_neutralisation_curriculum", "none", False),
    ]

    rows: List[Dict[str, object]] = []

    for variant_name, observation_ablation, use_curriculum in variants:
        for seed in seeds:
            print(f"\n{'=' * 80}")
            print(f"ABLATION: {variant_name} | seed={seed}")
            print(f"{'=' * 80}\n")

            race_cfg = replace(
                base_race_cfg,
                observation_ablation=str(observation_ablation),
            )
            run_dir = root / variant_name / f"seed_{int(seed)}"
            ppo_cfg = make_paper_ppo_config(
                seed=int(seed),
                out_dir=str(run_dir),
                device=device,
                neutral_curriculum=bool(use_curriculum),
            )

            _model, _eval_df, summary, _final_eval_df = train_ppo(
                race_cfg=race_cfg,
                ppo_cfg=ppo_cfg,
            )

            rows.append(
                {
                    "variant": variant_name,
                    "seed": int(seed),
                    "mean_finish": float(summary["avg_finish_position"]),
                    "mean_points": float(summary["avg_points"]),
                    "podium_probability": float(summary["podium_probability"]),
                    "legal_rate": float(summary["legal_rate"]),
                }
            )

    run_df = pd.DataFrame(rows)
    run_df.to_csv(root / "ablation_runs.csv", index=False)

    summary_df = (
        run_df
        .groupby("variant", as_index=False)
        .agg(
            mean_finish=("mean_finish", "mean"),
            mean_points=("mean_points", "mean"),
            podium_probability=("podium_probability", "mean"),
            legal_rate=("legal_rate", "mean"),
        )
    )
    summary_df.to_csv(root / "ablation_summary.csv", index=False)

    print("\n=== ABLATION SUMMARY ===")
    print(summary_df.to_string(index=False))
    return summary_df


def parse_seed_list(value: str) -> Tuple[int, ...]:
    out = tuple(
        int(x.strip())
        for x in str(value).split(",")
        if x.strip()
    )
    if not out:
        raise ValueError("At least one seed is required.")
    return out



# =============================================================================
# COMMAND-LINE ENTRY POINT
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Formula 1 stochastic race simulator and PPO experiments "
            "corresponding to the paper."
        )
    )
    parser.add_argument(
        "--mode",
        choices=("main", "ablations", "robustness", "all"),
        default="main",
        help=(
            "main: five PPO seeds + paired benchmark + bootstrap; "
            "ablations: six retrained ablations; "
            "robustness: evaluate saved main PPO models under four shifts; "
            "all: run all of the above."
        ),
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help='PyTorch device, e.g. "cpu" or "cuda".',
    )
    parser.add_argument(
        "--output-root",
        default="paper_outputs",
        help="Directory in which experiment outputs are written.",
    )
    parser.add_argument(
        "--ablation-seeds",
        default="1",
        help=(
            "Comma-separated training seeds for each ablation. "
            'Default: "1".'
        ),
    )
    parser.add_argument(
        "--bootstrap-reps",
        type=int,
        default=20_000,
        help="Number of paired two-level bootstrap replicates.",
    )
    parser.add_argument(
        "--scenarios",
        type=int,
        default=500,
        help="Number of paired held-out race scenarios.",
    )

    args = parser.parse_args()

    race_cfg = make_paper_race_config()
    output_root = str(args.output_root)
    device = str(args.device)
    main_seeds = (1, 2, 3, 4, 5)

    models: Optional[List[Tuple[str, ActorCritic]]] = None

    if args.mode in ("main", "all"):
        models = train_main_seeds(
            race_cfg=race_cfg,
            output_root=output_root,
            device=device,
            seeds=main_seeds,
        )
        save_main_paired_outputs(
            models=models,
            race_cfg=race_cfg,
            out_dir=str(Path(output_root) / "main_evaluation"),
            device=device,
            n_scenarios=int(args.scenarios),
            bootstrap_reps=int(args.bootstrap_reps),
        )

    if args.mode in ("robustness", "all"):
        if models is None:
            models = load_main_models(
                output_root=output_root,
                device=device,
                seeds=main_seeds,
            )

        run_robustness_suite(
            models=models,
            base_race_cfg=race_cfg,
            output_root=output_root,
            device=device,
            n_scenarios=int(args.scenarios),
        )

    if args.mode in ("ablations", "all"):
        ablation_seeds = parse_seed_list(args.ablation_seeds)
        run_ablation_suite(
            base_race_cfg=race_cfg,
            output_root=output_root,
            device=device,
            seeds=ablation_seeds,
        )


if __name__ == "__main__":
    main()
