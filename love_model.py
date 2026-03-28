"""
Shared two-agent love / control dynamics (numpy only).

Used by the Matplotlib desktop GUI (`love_control_gui.py`) and the Streamlit app
(`streamlit_app.py`).
"""

from __future__ import annotations

import textwrap
from dataclasses import dataclass

import numpy as np


@dataclass
class LoveSystemParams:
    T: float = 80.0
    dt: float = 0.05
    seed: int = 7

    x1_0: float = 0.2
    x2_0: float = -0.1

    a1: float = 0.8
    a2: float = 0.7

    b1: float = 1.4
    b2: float = 1.2

    k1: float = 0.9
    k2: float = 1.0

    x1_star: float = 0.6
    x2_star: float = 0.4

    K1: float = 0.5
    K2: float = 0.4

    tau1: float = 1.5
    tau2: float = 2.0

    sigma1: float = 0.04
    sigma2: float = 0.04

    shock_time_1: float = 20.0
    shock_mag_1: float = 0.4
    shock_time_2: float = 45.0
    shock_mag_2: float = -0.5


def saturation(x):
    return np.tanh(x)


def external_shock(t, shock_time, shock_mag, width: float = 0.8):
    return shock_mag * np.exp(-0.5 * ((t - shock_time) / width) ** 2)


def simulate(params: LoveSystemParams):
    rng = np.random.default_rng(params.seed)

    n_steps = int(params.T / params.dt) + 1
    t = np.linspace(0.0, params.T, n_steps)

    x1 = np.zeros(n_steps)
    x2 = np.zeros(n_steps)
    r1 = np.zeros(n_steps)
    r2 = np.zeros(n_steps)
    u1 = np.zeros(n_steps)
    u2 = np.zeros(n_steps)

    x1[0] = params.x1_0
    x2[0] = params.x2_0

    d1 = int(params.tau1 / params.dt)
    d2 = int(params.tau2 / params.dt)

    for i in range(1, n_steps):
        x2_delayed = x2[max(0, i - d1)]
        x1_delayed = x1[max(0, i - d2)]

        r1[i] = params.k1 * saturation(x2_delayed)
        r2[i] = params.k2 * saturation(x1_delayed)

        u1[i] = params.K1 * (x2_delayed - params.x1_star)
        u2[i] = params.K2 * (x1_delayed - params.x2_star)

        shock1 = external_shock(t[i], params.shock_time_1, params.shock_mag_1)
        shock2 = external_shock(t[i], params.shock_time_2, params.shock_mag_2)

        noise1 = params.sigma1 * rng.normal()
        noise2 = params.sigma2 * rng.normal()

        dx1 = (
            -params.a1 * x1[i - 1]
            + params.b1 * saturation(x2_delayed)
            + r1[i]
            + u1[i]
            + shock1
            + noise1
        )

        dx2 = (
            -params.a2 * x2[i - 1]
            + params.b2 * saturation(x1_delayed)
            + r2[i]
            + u2[i]
            + shock2
            + noise2
        )

        x1[i] = x1[i - 1] + params.dt * dx1
        x2[i] = x2[i - 1] + params.dt * dx2

    return t, x1, x2, r1, r2, u1, u2


def classify_regime(x1, x2):
    max_abs = max(np.max(np.abs(x1)), np.max(np.abs(x2)))
    if max_abs > 10:
        return "Divergent / unstable"

    tail1 = x1[-200:]
    tail2 = x2[-200:]
    amp1 = np.max(tail1) - np.min(tail1)
    amp2 = np.max(tail2) - np.min(tail2)
    slope1 = np.abs(np.mean(np.diff(tail1)))
    slope2 = np.abs(np.mean(np.diff(tail2)))

    if amp1 < 0.15 and amp2 < 0.15 and slope1 < 0.01 and slope2 < 0.01:
        return "Convergent / stable"
    return "Oscillatory / cycling"


# (name, mapping): keys matching dashboard sliders; other keys are simulation-only overrides.
STABLE_PRESETS: list[tuple[str, dict[str, float | int]]] = [
    (
        "symmetric damped",
        {
            "a1": 1.1,
            "a2": 1.1,
            "b1": 0.32,
            "b2": 0.32,
            "k1": 0.28,
            "k2": 0.28,
            "K1": 0.18,
            "K2": 0.18,
            "tau1": 0.7,
            "tau2": 0.8,
            "sigma1": 0.03,
            "sigma2": 0.03,
            "x1*": 0.55,
            "x2*": 0.42,
            "seed": 11,
        },
    ),
    (
        "asymmetric mild",
        {
            "a1": 1.25,
            "a2": 0.95,
            "b1": 0.38,
            "b2": 0.32,
            "k1": 0.26,
            "k2": 0.22,
            "K1": 0.2,
            "K2": 0.16,
            "tau1": 0.55,
            "tau2": 1.1,
            "sigma1": 0.035,
            "sigma2": 0.04,
            "x1*": 0.6,
            "x2*": 0.38,
            "seed": 19,
        },
    ),
    (
        "high baseline recovery",
        {
            "a1": 1.65,
            "a2": 1.55,
            "b1": 0.55,
            "b2": 0.5,
            "k1": 0.42,
            "k2": 0.38,
            "K1": 0.28,
            "K2": 0.25,
            "tau1": 0.45,
            "tau2": 0.5,
            "sigma1": 0.025,
            "sigma2": 0.025,
            "x1*": 0.45,
            "x2*": 0.48,
            "seed": 23,
        },
    ),
    (
        "low delay calm",
        {
            "a1": 1.0,
            "a2": 1.0,
            "b1": 0.42,
            "b2": 0.4,
            "k1": 0.32,
            "k2": 0.3,
            "K1": 0.18,
            "K2": 0.17,
            "tau1": 0.25,
            "tau2": 0.3,
            "sigma1": 0.02,
            "sigma2": 0.02,
            "x1*": 0.5,
            "x2*": 0.5,
            "seed": 31,
        },
    ),
    (
        "setpoint hold (symmetric)",
        {
            "a1": 1.15,
            "a2": 1.15,
            "b1": 0.65,
            "b2": 0.65,
            "k1": 0.6,
            "k2": 0.6,
            "K1": 0.08,
            "K2": 0.08,
            "tau1": 0.25,
            "tau2": 0.25,
            "sigma1": 0.012,
            "sigma2": 0.012,
            "x1*": 0.5,
            "x2*": 0.5,
            "seed": 11,
            "x1_0": 0.5,
            "x2_0": 0.5,
            "shock_mag_1": 0.0,
            "shock_mag_2": 0.0,
        },
    ),
    (
        "setpoint hold (offset goals)",
        {
            "a1": 1.18,
            "a2": 1.12,
            "b1": 0.62,
            "b2": 0.68,
            "k1": 0.58,
            "k2": 0.63,
            "K1": 0.12,
            "K2": 0.14,
            "tau1": 0.22,
            "tau2": 0.26,
            "sigma1": 0.012,
            "sigma2": 0.014,
            "x1*": 0.48,
            "x2*": 0.58,
            "seed": 17,
            "x1_0": 0.46,
            "x2_0": 0.56,
            "shock_mag_1": 0.0,
            "shock_mag_2": 0.0,
        },
    ),
    (
        "setpoint hold (mild mismatch)",
        {
            "a1": 1.14,
            "a2": 1.16,
            "b1": 0.64,
            "b2": 0.66,
            "k1": 0.59,
            "k2": 0.61,
            "K1": 0.1,
            "K2": 0.1,
            "tau1": 0.24,
            "tau2": 0.24,
            "sigma1": 0.011,
            "sigma2": 0.011,
            "x1*": 0.52,
            "x2*": 0.48,
            "seed": 29,
            "x1_0": 0.51,
            "x2_0": 0.47,
            "shock_mag_1": 0.0,
            "shock_mag_2": 0.0,
        },
    ),
]

UNSTABLE_PRESETS: list[tuple[str, dict[str, float | int]]] = [
    (
        "symmetric gain runaway",
        {
            "a1": 0.42,
            "a2": 0.42,
            "b1": 2.15,
            "b2": 2.15,
            "k1": 1.55,
            "k2": 1.55,
            "K1": 0.85,
            "K2": 0.85,
            "tau1": 0.35,
            "tau2": 0.35,
            "sigma1": 0.02,
            "sigma2": 0.02,
            "x1*": 0.5,
            "x2*": 0.5,
            "seed": 7,
        },
    ),
    (
        "weak baseline recovery, strong coupling",
        {
            "a1": 0.35,
            "a2": 0.38,
            "b1": 1.9,
            "b2": 1.85,
            "k1": 1.65,
            "k2": 1.6,
            "K1": 0.95,
            "K2": 0.9,
            "tau1": 0.4,
            "tau2": 0.45,
            "sigma1": 0.025,
            "sigma2": 0.025,
            "x1*": 0.55,
            "x2*": 0.45,
            "seed": 41,
        },
    ),
    (
        "asymmetric amplifier",
        {
            "a1": 0.48,
            "a2": 0.4,
            "b1": 2.4,
            "b2": 1.65,
            "k1": 1.4,
            "k2": 1.8,
            "K1": 0.75,
            "K2": 1.0,
            "tau1": 0.5,
            "tau2": 2.8,
            "sigma1": 0.03,
            "sigma2": 0.04,
            "x1*": 0.65,
            "x2*": 0.35,
            "seed": 53,
        },
    ),
    (
        "high delay + gain",
        {
            "a1": 0.55,
            "a2": 0.52,
            "b1": 1.75,
            "b2": 1.7,
            "k1": 1.35,
            "k2": 1.3,
            "K1": 0.7,
            "K2": 0.68,
            "tau1": 3.2,
            "tau2": 3.5,
            "sigma1": 0.06,
            "sigma2": 0.06,
            "x1*": 0.5,
            "x2*": 0.5,
            "seed": 67,
        },
    ),
]

SLIDER_DISPLAY_LABELS: dict[str, str] = {
    "a1": "A: baseline recovery rate",
    "a2": "B: baseline recovery rate",
    "b1": "A: partner influence",
    "b2": "B: partner influence",
    "k1": "A: reward sensitivity",
    "k2": "B: reward sensitivity",
    "K1": "A: relationship effort",
    "K2": "B: relationship effort",
    "tau1": "A: reaction delay",
    "tau2": "B: reaction delay",
    "sigma1": "A: misinterpretation",
    "sigma2": "B: misinterpretation",
    "x1*": "A: affection need",
    "x2*": "B: affection need",
    "seed": "random seed",
}

# Slider key -> (min, max); defaults come from LoveSystemParams.
SLIDER_BOUNDS: dict[str, tuple[float, float]] = {
    "a1": (0.0, 2.5),
    "a2": (0.0, 2.5),
    "b1": (0.0, 3.0),
    "b2": (0.0, 3.0),
    "k1": (0.0, 3.0),
    "k2": (0.0, 3.0),
    "K1": (0.0, 2.0),
    "K2": (0.0, 2.0),
    "tau1": (0.0, 8.0),
    "tau2": (0.0, 8.0),
    "sigma1": (0.0, 0.5),
    "sigma2": (0.0, 0.5),
    "x1*": (-2.0, 2.0),
    "x2*": (-2.0, 2.0),
    "seed": (0.0, 100.0),
}

SLIDER_ATTR: dict[str, str] = {
    "a1": "a1",
    "a2": "a2",
    "b1": "b1",
    "b2": "b2",
    "k1": "k1",
    "k2": "k2",
    "K1": "K1",
    "K2": "K2",
    "tau1": "tau1",
    "tau2": "tau2",
    "sigma1": "sigma1",
    "sigma2": "sigma2",
    "x1*": "x1_star",
    "x2*": "x2_star",
    "seed": "seed",
}


def wrap_slider_caption(text: str, width: int = 22) -> str:
    return "\n".join(textwrap.wrap(text, width=width, break_long_words=False, break_on_hyphens=False))


def bowl_surface_mesh(r_max: float = 2.8, n: int = 55, k_parabola: float = 0.18):
    u = np.linspace(-r_max, r_max, n)
    v = np.linspace(-r_max, r_max, n)
    X, Y = np.meshgrid(u, v)
    R2 = X**2 + Y**2
    mask = R2 <= r_max**2
    Z = np.where(mask, k_parabola * R2, np.nan)
    return X, Y, Z, k_parabola


def coupled_bowl_paths(
    x1,
    x2,
    r_base: float = 1.15,
    r_scale: float = 0.55,
    ang_scale: float = 1.05,
    cross: float = 0.42,
):
    phi1 = ang_scale * x1 + cross * np.tanh(x2)
    phi2 = ang_scale * x2 + cross * np.tanh(x1) + 2 * np.pi / 3
    r1 = r_base + r_scale * np.tanh(x2)
    r2 = r_base + r_scale * np.tanh(x1)
    xa = r1 * np.cos(phi1)
    ya = r1 * np.sin(phi1)
    xb = r2 * np.cos(phi2)
    yb = r2 * np.sin(phi2)
    return xa, ya, xb, yb
