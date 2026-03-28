"""
Love control system — Streamlit UI (local or Streamlit Community Cloud).

Run locally:
  uv sync --extra streamlit
  uv run streamlit run streamlit_app.py

Deploy: connect this repo to https://streamlit.io/cloud and set main file to streamlit_app.py.
"""

from __future__ import annotations

import os

# Headless matplotlib before pyplot import (required on servers and Streamlit Cloud).
os.environ.setdefault("MPLBACKEND", "Agg")

import random

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from love_model import (
    SLIDER_ATTR,
    SLIDER_BOUNDS,
    SLIDER_DISPLAY_LABELS,
    STABLE_PRESETS,
    UNSTABLE_PRESETS,
    LoveSystemParams,
    bowl_surface_mesh,
    classify_regime,
    coupled_bowl_paths,
    simulate,
)

st.set_page_config(
    page_title="Love control model",
    page_icon="💞",
    layout="wide",
    initial_sidebar_state="expanded",
)


def _init_session() -> None:
    if st.session_state.get("_love_inited"):
        return
    st.session_state._love_inited = True
    base = LoveSystemParams()
    st.session_state.sim_extras = {}
    for key, attr in SLIDER_ATTR.items():
        v = getattr(base, attr)
        st.session_state[key] = int(v) if key == "seed" else float(v)


def _clear_sim_extras() -> None:
    st.session_state.sim_extras = {}


def _apply_preset(mapping: dict[str, float | int]) -> None:
    st.session_state.sim_extras = {}
    for key, val in mapping.items():
        if key in SLIDER_BOUNDS:
            st.session_state[key] = int(val) if key == "seed" else float(val)
        else:
            st.session_state.sim_extras[key] = val


def _reset_defaults() -> None:
    base = LoveSystemParams()
    st.session_state.sim_extras = {}
    for key, attr in SLIDER_ATTR.items():
        v = getattr(base, attr)
        st.session_state[key] = int(v) if key == "seed" else float(v)


def _params_from_ui() -> LoveSystemParams:
    s = st.session_state
    ex = s.get("sim_extras") or {}
    b = LoveSystemParams()
    kw: dict = {
        "x1_0": float(ex.get("x1_0", b.x1_0)),
        "x2_0": float(ex.get("x2_0", b.x2_0)),
        "shock_time_1": float(ex.get("shock_time_1", b.shock_time_1)),
        "shock_mag_1": float(ex.get("shock_mag_1", b.shock_mag_1)),
        "shock_time_2": float(ex.get("shock_time_2", b.shock_time_2)),
        "shock_mag_2": float(ex.get("shock_mag_2", b.shock_mag_2)),
        "a1": float(s["a1"]),
        "a2": float(s["a2"]),
        "b1": float(s["b1"]),
        "b2": float(s["b2"]),
        "k1": float(s["k1"]),
        "k2": float(s["k2"]),
        "K1": float(s["K1"]),
        "K2": float(s["K2"]),
        "tau1": float(s["tau1"]),
        "tau2": float(s["tau2"]),
        "sigma1": float(s["sigma1"]),
        "sigma2": float(s["sigma2"]),
        "x1_star": float(s["x1*"]),
        "x2_star": float(s["x2*"]),
        "sat_alpha": float(s["sat"]),
        "seed": int(s["seed"]),
    }
    return LoveSystemParams(**kw)


def _slider_pct(key: str, value: float) -> str:
    lo, hi = SLIDER_BOUNDS[key]
    if hi == lo:
        return "—"
    return f"{100.0 * (value - lo) / (hi - lo):.0f}%"


def _render_slider(key: str) -> None:
    lo, hi = SLIDER_BOUNDS[key]
    label = SLIDER_DISPLAY_LABELS.get(key, key)
    if key == "seed":
        st.slider(
            f"{label} · {_slider_pct(key, float(st.session_state[key]))}",
            int(lo),
            int(hi),
            key=key,
            on_change=_clear_sim_extras,
        )
    else:
        st.slider(
            f"{label} · {_slider_pct(key, float(st.session_state[key]))}",
            float(lo),
            float(hi),
            key=key,
            on_change=_clear_sim_extras,
        )


def main() -> None:
    _init_session()

    st.title("Two-agent love control model")
    st.markdown(
        "Coupled emotional states with decay, partner coupling, rewards, setpoint tracking, "
        "delay, and noise. Sliders show **% of each control’s range** in the label. "
        "Some **stable presets** also set hidden simulation fields (initial state, shocks) "
        "so trajectories can sit near the dashed setpoints; adjusting any slider clears those overrides."
    )

    with st.sidebar:
        st.header("Presets & reset")
        if st.button("Reset to defaults", use_container_width=True):
            _reset_defaults()
            st.session_state.pop("last_preset", None)
            st.rerun()
        if st.button("Stable (random)", use_container_width=True):
            name, m = random.choice(STABLE_PRESETS)
            _apply_preset(m)
            st.session_state.last_preset = f"Stable: {name}"
            st.rerun()
        if st.button("Unstable (random)", use_container_width=True):
            name, m = random.choice(UNSTABLE_PRESETS)
            _apply_preset(m)
            st.session_state.last_preset = f"Unstable: {name}"
            st.rerun()
        if st.session_state.get("last_preset"):
            st.caption(st.session_state.last_preset)
        if st.session_state.get("sim_extras"):
            st.caption("Simulation overrides active (from last preset); change any slider to clear.")

        st.divider()
        st.header("Person 1 (A)")
        for key in ("a1", "b1", "k1", "K1", "tau1", "sigma1", "x1*"):
            _render_slider(key)

        st.divider()
        st.header("Person 2 (B)")
        for key in ("a2", "b2", "k2", "K2", "tau2", "sigma2", "x2*"):
            _render_slider(key)

        st.divider()
        st.header("Shared")
        _render_slider("sat")
        _render_slider("seed")

    p = _params_from_ui()
    t, x1, x2, *_ = simulate(p)
    regime = classify_regime(x1, x2)

    st.subheader(f"Emotional states — {regime}")
    fig1, ax1 = plt.subplots(figsize=(11, 3.2))
    ax1.plot(t, x1, label="x₁ Person A", color="C0")
    ax1.plot(t, x2, label="x₂ Person B", color="C1")
    ax1.axhline(p.x1_star, color="C0", linestyle="--", alpha=0.7, label="A setpoint")
    ax1.axhline(p.x2_star, color="C1", linestyle=":", alpha=0.7, label="B setpoint")
    ax1.set_xlabel("Time")
    ax1.set_ylabel("State")
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="lower left", fontsize=8)
    fig1.tight_layout()
    st.pyplot(fig1, clear_figure=True)
    plt.close(fig1)

    r_max = 2.8
    Xb, Yb, Zb, k_parabola = bowl_surface_mesh(r_max=r_max)
    xa, ya, xb, yb = coupled_bowl_paths(x1, x2)
    za = k_parabola * (xa**2 + ya**2)
    zb = k_parabola * (xb**2 + yb**2)

    c1, c2 = st.columns(2)
    with c1:
        st.caption("Bowl — 3D (end of run)")
        fig2 = plt.figure(figsize=(5.5, 4.5))
        ax3 = fig2.add_subplot(111, projection="3d")
        ax3.plot_surface(Xb, Yb, Zb, color="0.88", edgecolor="none", alpha=0.45, rstride=2, cstride=2)
        ax3.plot(xa, ya, za, color="C0", lw=1.0, alpha=0.85, label="A")
        ax3.plot(xb, yb, zb, color="C1", lw=1.0, alpha=0.85, label="B")
        ax3.scatter([xa[-1]], [ya[-1]], [za[-1]], color="C0", s=50)
        ax3.scatter([xb[-1]], [yb[-1]], [zb[-1]], color="C1", s=50)
        ax3.set_xlim(-r_max, r_max)
        ax3.set_ylim(-r_max, r_max)
        z_hi = float(np.nanmax(Zb)) * 1.05
        ax3.set_zlim(0, max(z_hi, 0.5))
        ax3.set_title("Bowl (3D)")
        ax3.legend(fontsize=7)
        fig2.tight_layout()
        st.pyplot(fig2, clear_figure=True)
        plt.close(fig2)

    with c2:
        st.caption("Bowl — plan view")
        fig3, axp = plt.subplots(figsize=(5.5, 5))
        th = np.linspace(0, 2 * np.pi, 200)
        axp.plot(r_max * np.cos(th), r_max * np.sin(th), "k--", alpha=0.25, lw=1)
        axp.plot(xa, ya, color="C0", lw=1.0, alpha=0.85, label="A")
        axp.plot(xb, yb, color="C1", lw=1.0, alpha=0.85, label="B")
        axp.scatter([xa[-1]], [ya[-1]], color="C0", s=45, zorder=5)
        axp.scatter([xb[-1]], [yb[-1]], color="C1", s=45, zorder=5)
        axp.set_aspect("equal")
        axp.set_xlim(-r_max, r_max)
        axp.set_ylim(-r_max, r_max)
        axp.set_title("Bowl plan (2D)")
        axp.grid(True, alpha=0.3)
        axp.legend(fontsize=8)
        fig3.tight_layout()
        st.pyplot(fig3, clear_figure=True)
        plt.close(fig3)


if __name__ == "__main__":
    main()
