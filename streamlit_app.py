"""
Love control system — Streamlit UI (local or Streamlit Community Cloud).

Run locally:
  uv sync
  uv run streamlit run streamlit_app.py

Deploy: connect this repo to https://streamlit.io/cloud and set main file to streamlit_app.py.

Sliders sit above the plots (like the desktop Matplotlib layout). Bowl 3D + plan animation is
embedded with ``st.components.v1.html`` (Plotly ``to_html`` + ``auto_play`` + loop hook) so
Play/autoplay and sliders render correctly—Streamlit’s native ``st.plotly_chart`` clips embedded
controls, and ``st.fragment(run_every=...)`` risks websocket cache glitches.
"""

from __future__ import annotations

import math
import os
import random
from pathlib import Path

# Headless matplotlib before pyplot import (required on servers and Streamlit Cloud).
os.environ.setdefault("MPLBACKEND", "Agg")

import json

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots
import streamlit as st
import streamlit.components.v1 as components
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
    page_title="Relationship control theory",
    layout="wide",
    initial_sidebar_state="collapsed",
)

_COUPLED_SYSTEM_DIAGRAM_SVG = Path(__file__).resolve().parent / "simple_coupled_system_centered_title.svg"


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


def _orbit_camera(jj: float, *, r: float = 2.15, el_deg: float = 22.0) -> dict:
    """Match desktop GUI orbit: azim ≈ 45 + 0.28 * frame_index."""
    az = math.radians(45.0 + 0.28 * jj)
    el = math.radians(el_deg)
    x = r * math.cos(el) * math.cos(az)
    y = r * math.cos(el) * math.sin(az)
    z = r * math.sin(el) + 0.25
    return dict(eye=dict(x=x, y=y, z=z), center=dict(x=0.0, y=0.0, z=0.12), up=dict(x=0, y=0, z=1))


def _animation_j_indices(n: int, max_frames: int = 120) -> list[int]:
    if n <= 0:
        return [0]
    frame_step = max(1, n // max_frames)
    j_list = list(range(0, max(n - 1, 1), frame_step))
    if j_list[-1] != n - 1:
        j_list.append(n - 1)
    return j_list


def _trail_line_indices(j: int, max_points: int) -> np.ndarray:
    """Uniform subsample of indices 0..j so line traces stay cheap for long simulations."""
    j = int(j)
    if j < 0:
        return np.array([0], dtype=int)
    n = j + 1
    if n <= max_points:
        return np.arange(n, dtype=int)
    return np.unique(np.linspace(0, j, num=max_points, dtype=int))


def _bowl_surface_grid_n() -> int:
    n = int(os.environ.get("LOVE_BOWL_MESH_N", "36"))
    return max(24, min(n, 72))


def _bowl_trail_max_points() -> int:
    n = int(os.environ.get("LOVE_BOWL_TRAIL_MAX", "420"))
    return max(32, min(n, 4000))


def _bowl_plotly_max_frames() -> int:
    n = int(os.environ.get("LOVE_BOWL_PLOTLY_FRAMES", "48"))
    return max(8, min(n, 200))


def _bowl_frame_ms() -> int:
    return int(os.environ.get("LOVE_BOWL_FRAME_MS", "85"))


def _subsample_j_list(j_list: list[int], cap: int) -> list[int]:
    if len(j_list) <= cap:
        return j_list
    idx = np.unique(np.linspace(0, len(j_list) - 1, num=cap, dtype=int))
    out = [j_list[int(i)] for i in idx]
    if out[-1] != j_list[-1]:
        out.append(j_list[-1])
    return out


def _bowl_3d_traces_at_j(pl: dict, j: int) -> tuple[list, int]:
    """Return four 3D traces (A/B trails + marbles) and clamped simulation index j."""
    xa = pl["xa"]
    ya = pl["ya"]
    za = pl["za"]
    xb = pl["xb"]
    yb = pl["yb"]
    zb = pl["zb"]
    trail_max = int(pl.get("trail_max_points", _bowl_trail_max_points()))
    n = len(xa)
    j = min(max(int(j), 0), max(n - 1, 0))
    li = _trail_line_indices(j, trail_max)
    return (
        [
            go.Scatter3d(
                x=xa[li],
                y=ya[li],
                z=za[li],
                mode="lines",
                line=dict(color="#1f77b4", width=4),
                showlegend=False,
                hoverinfo="skip",
            ),
            go.Scatter3d(
                x=xb[li],
                y=yb[li],
                z=zb[li],
                mode="lines",
                line=dict(color="#ff7f0e", width=4),
                showlegend=False,
                hoverinfo="skip",
            ),
            go.Scatter3d(
                x=[xa[j]],
                y=[ya[j]],
                z=[za[j]],
                mode="markers",
                marker=dict(size=8, color="#1f77b4"),
                showlegend=False,
                hoverinfo="skip",
            ),
            go.Scatter3d(
                x=[xb[j]],
                y=[yb[j]],
                z=[zb[j]],
                mode="markers",
                marker=dict(size=8, color="#ff7f0e"),
                showlegend=False,
                hoverinfo="skip",
            ),
        ],
        j,
    )


def _bowl_plan_traces_at_j(pl: dict, j: int) -> tuple[list, int]:
    """Return four plan-view traces (A/B lines + markers) and clamped index j."""
    xa = pl["xa"]
    ya = pl["ya"]
    xb = pl["xb"]
    yb = pl["yb"]
    trail_max = int(pl.get("trail_max_points", _bowl_trail_max_points()))
    n = len(xa)
    j = min(max(int(j), 0), max(n - 1, 0))
    li = _trail_line_indices(j, trail_max)
    return (
        [
            go.Scattergl(
                x=xa[li],
                y=ya[li],
                mode="lines",
                line=dict(color="#1f77b4", width=3),
                showlegend=False,
                hoverinfo="skip",
            ),
            go.Scattergl(
                x=xb[li],
                y=yb[li],
                mode="lines",
                line=dict(color="#ff7f0e", width=3),
                showlegend=False,
                hoverinfo="skip",
            ),
            go.Scatter(
                x=[xa[j]],
                y=[ya[j]],
                mode="markers",
                marker=dict(size=11, color="#1f77b4"),
                showlegend=False,
                hoverinfo="skip",
            ),
            go.Scatter(
                x=[xb[j]],
                y=[yb[j]],
                mode="markers",
                marker=dict(size=11, color="#ff7f0e"),
                showlegend=False,
                hoverinfo="skip",
            ),
        ],
        j,
    )


def _bowl_combined_animation_figure(pl: dict) -> go.Figure:
    """3D bowl + 2D plan with frames (surface + rim static; trails + marbles + camera animate)."""
    Xb = pl["Xb"]
    Yb = pl["Yb"]
    Zb = pl["Zb"]
    r_max = float(pl["r_max"])
    z_hi = float(pl["z_hi"])
    j_list = pl["j_list"] or [0]
    frame_js = _subsample_j_list(j_list, _bowl_plotly_max_frames())
    dur = _bowl_frame_ms()

    th = np.linspace(0, 2 * np.pi, 96)
    boundary = go.Scatter(
        x=r_max * np.cos(th),
        y=r_max * np.sin(th),
        mode="lines",
        line=dict(color="gray", dash="dash", width=1),
        showlegend=False,
        hoverinfo="skip",
    )
    j0 = frame_js[0]
    d3_0, _ = _bowl_3d_traces_at_j(pl, j0)
    plan_0, _ = _bowl_plan_traces_at_j(pl, j0)

    surface = go.Surface(
        x=Xb,
        y=Yb,
        z=Zb,
        opacity=0.42,
        showscale=False,
        colorscale=[[0, "#d8d8d8"], [1, "#c0c0c0"]],
        hoverinfo="skip",
        lighting=dict(ambient=0.85, diffuse=0.35, specular=0.2),
    )

    fig = make_subplots(
        rows=1,
        cols=2,
        specs=[[{"type": "scatter3d"}, {"type": "scatter"}]],
        column_widths=[0.52, 0.48],
        horizontal_spacing=0.05,
    )
    fig.add_trace(surface, row=1, col=1)
    for tr in d3_0:
        fig.add_trace(tr, row=1, col=1)
    fig.add_trace(boundary, row=1, col=2)
    for tr in plan_0:
        fig.add_trace(tr, row=1, col=2)

    anim_traces = [1, 2, 3, 4, 6, 7, 8, 9]
    frames: list[go.Frame] = []
    for k, j in enumerate(frame_js):
        d3, _ = _bowl_3d_traces_at_j(pl, j)
        plan, _ = _bowl_plan_traces_at_j(pl, j)
        frames.append(
            go.Frame(
                data=[*d3, *plan],
                traces=anim_traces,
                layout=dict(scene=dict(camera=_orbit_camera(float(k)))),
                name=str(k),
            )
        )
    fig.frames = frames

    anim_args = dict(
        frame=dict(duration=dur, redraw=True),
        fromcurrent=False,
        transition=dict(duration=0),
        mode="immediate",
    )
    fig.update_layout(
        height=500,
        margin=dict(l=4, r=8, t=72, b=28),
        paper_bgcolor="#ffffff",
        showlegend=False,
        updatemenus=[
            dict(
                type="buttons",
                showactive=False,
                direction="left",
                pad=dict(l=10, r=10, t=6, b=6),
                x=0.5,
                xanchor="center",
                y=1.08,
                yanchor="bottom",
                bgcolor="rgba(255,255,255,0.97)",
                bordercolor="rgba(0,0,0,0.14)",
                borderwidth=1,
                font=dict(size=13),
                buttons=[
                    dict(
                        label="▶ Play",
                        method="animate",
                        args=[
                            [str(i) for i in range(len(frames))],
                            anim_args,
                        ],
                    ),
                ],
            )
        ],
        scene=dict(
            xaxis=dict(range=[-r_max, r_max], backgroundcolor="#fafafa", showticklabels=False),
            yaxis=dict(range=[-r_max, r_max], backgroundcolor="#fafafa", showticklabels=False),
            zaxis=dict(range=[0, z_hi], backgroundcolor="#fafafa", showticklabels=False),
            aspectmode="cube",
            camera=_orbit_camera(0.0),
        ),
    )
    fig.update_xaxes(
        range=[-r_max, r_max],
        scaleanchor="y",
        scaleratio=1,
        showticklabels=False,
        showgrid=True,
        gridcolor="rgba(0,0,0,0.12)",
        row=1,
        col=2,
    )
    fig.update_yaxes(
        range=[-r_max, r_max],
        showticklabels=False,
        showgrid=True,
        gridcolor="rgba(0,0,0,0.12)",
        row=1,
        col=2,
    )
    return fig


def _bowl_animation_html(pl: dict) -> str:
    """Serialize animated bowl figure for ``components.html`` (autoplay + infinite loop in browser)."""
    fig = _bowl_combined_animation_figure(pl)
    dur = _bowl_frame_ms()
    anim_opts = dict(
        frame=dict(duration=dur, redraw=True),
        transition=dict(duration=0),
        mode="immediate",
    )
    opts_json = json.dumps(anim_opts)
    loop = os.environ.get("LOVE_BOWL_LOOP", "1").strip().lower() not in ("0", "false", "no")
    post_script = None
    if loop:
        post_script = (
            "var gd = document.getElementById('{plot_id}');\n"
            "if (gd) {\n"
            "  gd.on('plotly_animated', function () {\n"
            "    Plotly.animate('{plot_id}', null, "
            + opts_json
            + ");\n"
            "  });\n"
            "}\n"
        )
    cfg = dict(
        displayModeBar=False,
        plotGlPixelRatio=float(os.environ.get("LOVE_BOWL_GL_PIXEL_RATIO", "1")),
        responsive=True,
    )
    return pio.to_html(
        fig,
        config=cfg,
        auto_play=True,
        animation_opts=anim_opts,
        include_plotlyjs="cdn",
        full_html=True,
        div_id="love-bowl-animation",
        post_script=post_script,
    )


def _onboarding_card() -> None:
    st.markdown(
        """
<div style="
  background: linear-gradient(145deg, #fffbf5 0%, #fff4e6 55%, #ffe8f0 100%);
  border: 1px solid rgba(230, 180, 120, 0.35);
  border-radius: 16px;
  padding: 1.35rem 1.6rem 1.45rem 1.6rem;
  margin-bottom: 1.25rem;
  box-shadow: 0 2px 12px rgba(80, 40, 40, 0.06);
">
  <p style="margin: 0 0 0.5rem 0; font-size: 1.35rem; font-weight: 700; color: #3d2a20; letter-spacing: -0.02em;">
    💛 Relationships as a Coupled System
  </p>
  <p style="margin: 0 0 1rem 0; font-size: 1.05rem; color: #5c4338; line-height: 1.5;">
    Two people continuously shaping each other.
  </p>
  <p style="margin: 0 0 1rem 0; font-size: 1rem; color: #4a362c; line-height: 1.65;">
    <span style="font-weight: 600;">You react</span><br/>
    <span style="font-weight: 600;">You reinforce</span><br/>
    <span style="font-weight: 600;">You drift</span>
  </p>
  <p style="margin: 0; font-size: 1.02rem; color: #5c4338; line-height: 1.55;">
    Turn the knobs below to see how the system behaves.
  </p>
</div>
""",
        unsafe_allow_html=True,
    )


def _coupled_system_diagram() -> None:
    if _COUPLED_SYSTEM_DIAGRAM_SVG.is_file():
        st.image(str(_COUPLED_SYSTEM_DIAGRAM_SVG), width="stretch")


def _attachment_theory_section() -> None:
    st.divider()
    st.markdown(
        """
<div style="
  background: linear-gradient(160deg, #f8fafc 0%, #f1f5f9 45%, #fefce8 100%);
  border: 1px solid rgba(100, 120, 140, 0.2);
  border-radius: 14px;
  padding: 1.25rem 1.4rem 1.3rem 1.4rem;
  margin-top: 0.5rem;
  box-shadow: 0 2px 10px rgba(30, 40, 60, 0.05);
">
  <p style="margin: 0 0 0.85rem 0; font-size: 1.2rem; font-weight: 700; color: #1e293b;">
    Attachment theory analysis
  </p>
  <p style="margin: 0 0 1rem 0; font-size: 1.05rem; color: #334155; line-height: 1.55;">
    💡 <strong>Healthy relationships are not zero-feedback — they are well-tuned feedback systems</strong>
  </p>
  <p style="margin: 0 0 1.1rem 0; font-size: 0.98rem; color: #475569; line-height: 1.55;">
    What you explored with sliders maps to attachment styles as regions in the system:
  </p>

  <div style="margin-bottom: 1rem;">
    <p style="margin: 0 0 0.35rem 0; font-weight: 700; color: #0f172a;">❤️ Secure</p>
    <p style="margin: 0 0 0.25rem 0; font-size: 0.95rem; color: #475569; line-height: 1.5;">
      Moderate influence + reward, strong baseline recovery<br/>
      Low delay, low noise<br/>
      System converges smoothly
    </p>
    <p style="margin: 0; font-size: 0.92rem; color: #64748b;">👉 Stable, responsive, easy repair</p>
  </div>

  <div style="margin-bottom: 1rem;">
    <p style="margin: 0 0 0.35rem 0; font-weight: 700; color: #0f172a;">🔥 Anxious</p>
    <p style="margin: 0 0 0.25rem 0; font-size: 0.95rem; color: #475569; line-height: 1.5;">
      High influence + high reward<br/>
      Weak baseline recovery (needs constant input)<br/>
      System overshoots / oscillates
    </p>
    <p style="margin: 0; font-size: 0.92rem; color: #64748b;">👉 Intense, reactive, amplification loops</p>
  </div>

  <div style="margin-bottom: 1rem;">
    <p style="margin: 0 0 0.35rem 0; font-weight: 700; color: #0f172a;">🧊 Avoidant</p>
    <p style="margin: 0 0 0.25rem 0; font-size: 0.95rem; color: #475569; line-height: 1.5;">
      Low influence + low reward<br/>
      Strong baseline recovery, little partner input<br/>
      Weak coupling
    </p>
    <p style="margin: 0; font-size: 0.92rem; color: #64748b;">👉 Calm but distant, low emotional exchange</p>
  </div>

  <div style="margin-bottom: 0;">
    <p style="margin: 0 0 0.35rem 0; font-weight: 700; color: #0f172a;">🎢 Anxious–Avoidant</p>
    <p style="margin: 0 0 0.25rem 0; font-size: 0.95rem; color: #475569; line-height: 1.5;">
      Asymmetric gains + delays<br/>
      One pushes, one withdraws<br/>
      System cycles instead of settling
    </p>
    <p style="margin: 0; font-size: 0.92rem; color: #64748b;">👉 Push–pull, unstable, repeating patterns</p>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )


def _stability_eigenview_section() -> None:
    st.markdown(
        """
<div style="
  background: linear-gradient(165deg, #f0f9ff 0%, #ecfeff 40%, #f5f3ff 100%);
  border: 1px solid rgba(59, 130, 246, 0.22);
  border-radius: 14px;
  padding: 1.25rem 1.4rem 1.3rem 1.4rem;
  margin-top: 1rem;
  box-shadow: 0 2px 10px rgba(30, 58, 90, 0.06);
">
  <p style="margin: 0 0 0.85rem 0; font-size: 1.2rem; font-weight: 700; color: #0c4a6e;">
    📐 Stability (Eigenview)
  </p>
  <p style="margin: 0 0 0.75rem 0; font-size: 0.98rem; color: #334155; line-height: 1.55;">
    The system’s behavior is determined by its <strong>eigenvalues</strong>.
  </p>
  <p style="margin: 0 0 1rem 0; font-size: 0.95rem; color: #475569; line-height: 1.55;">
    <strong>Negative</strong> → states decay → stable<br/>
    <strong>Positive</strong> → states grow → unstable<br/>
    <strong>Complex</strong> → oscillations (cycles)
  </p>
  <p style="margin: 0 0 0.65rem 0; font-size: 0.98rem; color: #334155; line-height: 1.5;">
    The key balance is:
  </p>
  <p style="margin: 0 0 1rem 0; font-size: 0.95rem; color: #475569; line-height: 1.55;">
    <strong>mutual coupling</strong> vs <strong>baseline recovery rate</strong> vs <strong>adjustment</strong>
  </p>
  <p style="margin: 0 0 0.5rem 0; font-size: 0.95rem; color: #475569; line-height: 1.55;">
    <strong>Coupling</strong> = influence + reward<br/>
    <strong>Adjustment</strong> = how actively you correct toward your needs<br/>
    <strong>Baseline recovery rate</strong> = how fast you settle back toward your emotional baseline
  </p>
  <p style="margin: 0 0 0.5rem 0; font-size: 0.92rem; color: #0369a1; line-height: 1.5;">
    👉 <strong>Stability condition (core):</strong><br/>
    (coupling + adjustment)² &lt; (baseline recovery rate)²
    <span style="color: #64748b; font-weight: 400;"> — roughly: total feedback &lt; baseline recovery strength</span>
  </p>
  <p style="margin: 0; font-size: 0.92rem; color: #64748b; line-height: 1.5;">
    👉 <strong>In simple terms:</strong> you affect each other, but not more than you can recover to baseline and adjust
  </p>
</div>
""",
        unsafe_allow_html=True,
    )


def _limitations_section() -> None:
    st.markdown(
        """
<div style="
  background: linear-gradient(165deg, #fffbeb 0%, #fef3c7 35%, #ffedd5 100%);
  border: 1px solid rgba(245, 158, 11, 0.35);
  border-radius: 14px;
  padding: 1.25rem 1.4rem 1.3rem 1.4rem;
  margin-top: 1rem;
  margin-bottom: 0.5rem;
  box-shadow: 0 2px 10px rgba(120, 80, 20, 0.06);
">
  <p style="margin: 0 0 0.85rem 0; font-size: 1.2rem; font-weight: 700; color: #78350f;">
    ⚠️ Limitations
  </p>
  <p style="margin: 0 0 1rem 0; font-size: 0.98rem; color: #44403c; line-height: 1.55;">
    Humans are <strong>far more complex systems</strong> than this model.
  </p>
  <p style="margin: 0 0 0.45rem 0; font-size: 0.95rem; color: #57534e; line-height: 1.55;">
    We have <strong>memory</strong> — history shapes present behavior<br/>
    We carry <strong>meaning and context</strong> — the same signal can feel different<br/>
    We have <strong>intentions and values</strong> — not everything is reactive<br/>
    We are <strong>nonlinear and discontinuous</strong> — not smooth signals<br/>
    We <strong>change over time</strong> — our “parameters” evolve
  </p>
  <p style="margin: 0.85rem 0 0.4rem 0; font-size: 0.98rem; color: #44403c; line-height: 1.5;">
    We are also <strong>not isolated</strong> — we are coupled to many systems:
  </p>
  <p style="margin: 0; font-size: 0.95rem; color: #57534e; line-height: 1.55;">
    our past selves<br/>
    our environments<br/>
    other relationships
  </p>
</div>
""",
        unsafe_allow_html=True,
    )


def main() -> None:
    _init_session()

    _onboarding_card()
    _coupled_system_diagram()

    st.subheader("Presets")
    b1, b2, b3, b4 = st.columns([1, 1, 1, 2])
    with b1:
        if st.button("Reset to defaults", width="stretch"):
            _reset_defaults()
            st.session_state.pop("last_preset", None)
            st.rerun()
    with b2:
        if st.button("Stable (random)", width="stretch"):
            name, m = random.choice(STABLE_PRESETS)
            _apply_preset(m)
            st.session_state.last_preset = f"Stable: {name}"
            st.rerun()
    with b3:
        if st.button("Unstable (random)", width="stretch"):
            name, m = random.choice(UNSTABLE_PRESETS)
            _apply_preset(m)
            st.session_state.last_preset = f"Unstable: {name}"
            st.rerun()
    with b4:
        if st.session_state.get("last_preset"):
            st.caption(st.session_state.last_preset)
        if st.session_state.get("sim_extras"):
            st.caption("Preset simulation overrides active; change any slider to clear.")

    st.subheader("Parameters")
    ca, cb = st.columns(2, gap="large")
    with ca:
        st.markdown("**Person 1 (A)**")
        for key in ("a1", "b1", "k1", "K1", "tau1", "sigma1", "x1*"):
            _render_slider(key)
    with cb:
        st.markdown("**Person 2 (B)**")
        for key in ("a2", "b2", "k2", "K2", "tau2", "sigma2", "x2*"):
            _render_slider(key)

    st.markdown("**Shared**")
    _render_slider("seed")

    st.divider()

    p = _params_from_ui()
    t, x1, x2, *_ = simulate(p)
    regime = classify_regime(x1, x2)

    r_max = 2.8
    mesh_n = _bowl_surface_grid_n()
    Xb, Yb, Zb, k_parabola = bowl_surface_mesh(r_max=r_max, n=mesh_n)
    xa, ya, xb, yb = coupled_bowl_paths(x1, x2)
    za = k_parabola * (xa**2 + ya**2)
    zb = k_parabola * (xb**2 + yb**2)
    z_hi = float(np.nanmax(Zb)) * 1.05
    z_hi = max(z_hi, 0.5)

    j_list = _animation_j_indices(len(xa))
    trail_cap = _bowl_trail_max_points()
    st.session_state.bowl_animation_payload = {
        "xa": xa,
        "ya": ya,
        "za": za,
        "xb": xb,
        "yb": yb,
        "zb": zb,
        "Xb": Xb,
        "Yb": Yb,
        "Zb": Zb,
        "r_max": r_max,
        "z_hi": float(z_hi),
        "j_list": j_list,
        "trail_max_points": trail_cap,
    }
    st.subheader("Bowl trajectories")
    st.caption(
        "Each marble is one person; its path on the bowl follows the same simulated emotional "
        "states as the time series below (blue = Person A, x₁ · orange = Person B, x₂)."
    )
    pl = st.session_state.bowl_animation_payload
    components.html(
        _bowl_animation_html(pl),
        height=620,
        scrolling=True,
    )

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

    _attachment_theory_section()
    _stability_eigenview_section()
    _limitations_section()


if __name__ == "__main__":
    main()
