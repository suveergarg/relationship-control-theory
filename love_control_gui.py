#!/usr/bin/env python3
"""
Desktop GUI for the two-agent love control model (native Matplotlib window).

For a browser UI (local or Streamlit Cloud), run: ``streamlit run streamlit_app.py``
(see ``pyproject.toml`` optional dependency ``streamlit``).

Uses a non-inline backend so sliders and animation share a real GUI event loop
(typically smoother than Jupyter + ipympl). Tries Qt first, then Tk:

  pip install PySide6   # optional, for QtAgg / Qt5Agg

Stable (random) / Unstable (random) pick one of several named parameter
sets; the chosen name is printed to stderr. Some stable presets include
simulation-only fields (initial state, shocks off) so trajectories settle
near the dashed setpoint lines.
"""

from __future__ import annotations

import importlib
import random
import sys


def _configure_matplotlib_backend() -> str:
    import matplotlib

    candidates = (
        ("QtAgg", "matplotlib.backends.backend_qtagg"),
        ("Qt5Agg", "matplotlib.backends.backend_qt5agg"),
        ("TkAgg", "matplotlib.backends.backend_tkagg"),
    )
    for backend, module in candidates:
        try:
            importlib.import_module(module)
            matplotlib.use(backend, force=True)
            return backend
        except Exception:
            continue
    matplotlib.use("TkAgg", force=True)
    return "TkAgg"


_BACKEND = _configure_matplotlib_backend()

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation as mpl_animation
from matplotlib.widgets import Button, Slider
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from love_model import (
    SLIDER_DISPLAY_LABELS,
    STABLE_PRESETS,
    UNSTABLE_PRESETS,
    LoveSystemParams,
    bowl_surface_mesh,
    classify_regime,
    coupled_bowl_paths,
    simulate,
    wrap_slider_caption,
)

_wrap_slider_caption = wrap_slider_caption


def attach_slider_percent_display(slider: Slider) -> None:
    """Show only the knob position along each slider’s range (0–100%)."""
    lo = float(slider.valmin)
    hi = float(slider.valmax)
    span = hi - lo

    def refresh(_: float | None = None) -> None:
        val = slider.val
        if span == 0:
            slider.valtext.set_text("—")
        else:
            pct = 100.0 * (val - lo) / span
            slider.valtext.set_text(f"{pct:.0f}%")

    slider.on_changed(refresh)
    refresh()


def interactive_dashboard():
    """Single window: sliders (Person 1 / Person 2), emotional states, bowl 3D + plan."""
    params = LoveSystemParams()
    sim_extras: dict[str, float | int] = {}

    t, x1, x2, *_ = simulate(params)
    r_max = 2.8

    fig = plt.figure(figsize=(12, 12.0))
    fig.patch.set_facecolor("0.97")
    try:
        fig.canvas.manager.set_window_title("Love control — desktop GUI")
    except Exception:
        pass

    # Sliders: separate label axes (wrapped text) + track-only Slider — avoids label/value overlap.
    h_row = 0.024
    v_step = 0.038
    y0 = 0.952
    wl = 0.195
    ws = 0.265
    xl_a, xs_a = 0.035, 0.235
    xl_b, xs_b = 0.508, 0.708
    gap_before_shared = 0.018
    y_shared = y0 - 7 * v_step - gap_before_shared
    label_fs, val_fs = 7.5, 7.5

    p1 = [
        ("a1", 0.0, 2.5, params.a1),
        ("b1", 0.0, 3.0, params.b1),
        ("k1", 0.0, 3.0, params.k1),
        ("K1", 0.0, 2.0, params.K1),
        ("tau1", 0.0, 8.0, params.tau1),
        ("sigma1", 0.0, 0.5, params.sigma1),
        ("x1*", -2.0, 2.0, params.x1_star),
    ]
    p2 = [
        ("a2", 0.0, 2.5, params.a2),
        ("b2", 0.0, 3.0, params.b2),
        ("k2", 0.0, 3.0, params.k2),
        ("K2", 0.0, 2.0, params.K2),
        ("tau2", 0.0, 8.0, params.tau2),
        ("sigma2", 0.0, 0.5, params.sigma2),
        ("x2*", -2.0, 2.0, params.x2_star),
    ]

    fig.text(
        0.25,
        0.978,
        "Person 1 (A)",
        ha="center",
        fontsize=11,
        fontweight="bold",
        color="0.1",
    )
    fig.text(
        0.74,
        0.978,
        "Person 2 (B)",
        ha="center",
        fontsize=11,
        fontweight="bold",
        color="0.1",
    )
    y_shared_lbl = y_shared + h_row + 0.016
    fig.text(
        0.50,
        y_shared_lbl,
        "Shared",
        ha="center",
        fontsize=9,
        style="italic",
        color="0.25",
    )

    bowl_bottom = 0.06
    bowl_h = 0.30
    bowl_top = bowl_bottom + bowl_h

    gap_mid = 0.028
    state_h = 0.21
    state_bottom = bowl_top + gap_mid
    state_top = state_bottom + state_h

    ax_state = fig.add_axes([0.10, state_bottom, 0.82, state_h])

    line_x1, = ax_state.plot(t, x1, label="x1: Person A")
    line_x2, = ax_state.plot(t, x2, label="x2: Person B")
    line_x1s = ax_state.axhline(params.x1_star, linestyle="--", alpha=0.7, label="A setpoint")
    line_x2s = ax_state.axhline(params.x2_star, linestyle=":", alpha=0.7, label="B setpoint")
    title_state = ax_state.set_title(
        "Emotional states",
        fontsize=10,
        pad=8,
    )
    ax_state.set_xlabel("Time")
    ax_state.set_ylabel("State")
    ax_state.tick_params(axis="both", labelsize=8)
    ax_state.grid(True, alpha=0.3)
    ax_state.legend(loc="lower left", fontsize=7, framealpha=0.92)

    hint_y = (state_top + y_shared + h_row) / 2
    fig.text(
        0.50,
        hint_y,
        "Sliders: readout is % along each control’s range (not raw parameter units)",
        ha="center",
        va="center",
        fontsize=8,
        color="0.35",
        linespacing=1.2,
    )

    Xb, Yb, Zb, k_parabola = bowl_surface_mesh(r_max=r_max)
    ax_b3d = fig.add_axes([0.06, bowl_bottom, 0.40, bowl_h], projection="3d")
    ax_b3d.plot_surface(Xb, Yb, Zb, color="0.88", edgecolor="none", alpha=0.45, rstride=2, cstride=2)
    bowl_trail_a, = ax_b3d.plot([], [], [], color="C0", lw=1.0, alpha=0.75, label="A")
    bowl_trail_b, = ax_b3d.plot([], [], [], color="C1", lw=1.0, alpha=0.75, label="B")
    (bowl_ball_a,) = ax_b3d.plot([], [], [], "o", color="C0", markersize=8, label="Person A")
    (bowl_ball_b,) = ax_b3d.plot([], [], [], "o", color="C1", markersize=8, label="Person B")
    ax_b3d.set_xlim(-r_max, r_max)
    ax_b3d.set_ylim(-r_max, r_max)
    z_hi = float(np.nanmax(Zb)) * 1.05
    ax_b3d.set_zlim(0, max(z_hi, 0.5))
    ax_b3d.set_title("Bowl (3D)", fontsize=9, pad=2)
    ax_b3d.set_xlabel("x", fontsize=8, labelpad=0)
    ax_b3d.set_ylabel("y", fontsize=8, labelpad=0)
    ax_b3d.set_zlabel("z", fontsize=8, labelpad=2)
    ax_b3d.tick_params(labelsize=7)
    ax_b3d.legend(loc="lower left", fontsize=6, framealpha=0.9)

    ax_bplan = fig.add_axes([0.54, bowl_bottom, 0.40, bowl_h])
    th = np.linspace(0, 2 * np.pi, 200)
    ax_bplan.plot(r_max * np.cos(th), r_max * np.sin(th), "k--", alpha=0.25, lw=1)
    plan_trail_a, = ax_bplan.plot([], [], color="C0", lw=1.0, alpha=0.85, label="A (plan)")
    plan_trail_b, = ax_bplan.plot([], [], color="C1", lw=1.0, alpha=0.85, label="B (plan)")
    plan_ball_a = ax_bplan.scatter([], [], color="C0", s=40, zorder=5)
    plan_ball_b = ax_bplan.scatter([], [], color="C1", s=40, zorder=5)
    ax_bplan.set_aspect("equal")
    ax_bplan.set_xlim(-r_max, r_max)
    ax_bplan.set_ylim(-r_max, r_max)
    ax_bplan.set_title("Bowl plan (2D)", fontsize=9, pad=6)
    ax_bplan.set_xlabel("x", fontsize=8)
    ax_bplan.set_ylabel("y", fontsize=8)
    ax_bplan.tick_params(labelsize=7)
    ax_bplan.grid(True, alpha=0.3)
    ax_bplan.legend(loc="lower left", fontsize=6, framealpha=0.9)

    bowl_anim_holder: dict = {"anim": None}

    def stop_bowl_animation():
        old = bowl_anim_holder["anim"]
        if old is not None:
            try:
                old.event_source.stop()
            except Exception:
                pass
            bowl_anim_holder["anim"] = None

    def refresh_bowl(x1n, x2n, *, rotate_view: bool = True):
        xa, ya, xb, yb = coupled_bowl_paths(x1n, x2n)
        za = k_parabola * (xa**2 + ya**2)
        zb = k_parabola * (xb**2 + yb**2)
        nlen = len(x1n)
        frame_step = max(1, nlen // 300)
        frames = np.arange(0, nlen, frame_step)

        stop_bowl_animation()

        def anim_update(jj):
            j = int(frames[jj])
            bowl_trail_a.set_data(xa[: j + 1], ya[: j + 1])
            bowl_trail_a.set_3d_properties(za[: j + 1])
            bowl_trail_b.set_data(xb[: j + 1], yb[: j + 1])
            bowl_trail_b.set_3d_properties(zb[: j + 1])
            bowl_ball_a.set_data([xa[j]], [ya[j]])
            bowl_ball_a.set_3d_properties([za[j]])
            bowl_ball_b.set_data([xb[j]], [yb[j]])
            bowl_ball_b.set_3d_properties([zb[j]])
            plan_trail_a.set_data(xa[: j + 1], ya[: j + 1])
            plan_trail_b.set_data(xb[: j + 1], yb[: j + 1])
            plan_ball_a.set_offsets(np.array([[xa[j], ya[j]]]))
            plan_ball_b.set_offsets(np.array([[xb[j], yb[j]]]))
            if rotate_view:
                ax_b3d.view_init(elev=22, azim=45 + 0.28 * jj)
            return (bowl_trail_a, bowl_trail_b, bowl_ball_a, bowl_ball_b)

        bowl_anim_holder["anim"] = mpl_animation.FuncAnimation(
            fig,
            anim_update,
            frames=len(frames),
            interval=120,
            blit=False,
            repeat=True,
            cache_frame_data=False,
        )
        fig.canvas.draw_idle()

    sliders: dict[str, Slider] = {}

    def add_slider_row(name: str, vmin: float, vmax: float, vinit: float, xl: float, x_track: float, yb: float):
        lax = fig.add_axes([xl, yb, wl, h_row])
        lax.set_axis_off()
        lax.text(
            1.0,
            0.5,
            _wrap_slider_caption(SLIDER_DISPLAY_LABELS.get(name, name)),
            transform=lax.transAxes,
            ha="right",
            va="center",
            fontsize=label_fs,
            color="0.2",
            linespacing=1.12,
        )
        sax = fig.add_axes([x_track, yb, ws, h_row])
        valfmt = "%0.0f" if name == "seed" else "%1.2f"
        s = Slider(ax=sax, label="", valmin=vmin, valmax=vmax, valinit=vinit, valfmt=valfmt)
        s.valtext.set_fontsize(val_fs)
        sliders[name] = s

    for i, (name, lo, hi, v0) in enumerate(p1):
        add_slider_row(name, lo, hi, v0, xl_a, xs_a, y0 - i * v_step)
    for i, (name, lo, hi, v0) in enumerate(p2):
        add_slider_row(name, lo, hi, v0, xl_b, xs_b, y0 - i * v_step)

    add_slider_row("sat", 0.1, 3.0, params.sat_alpha, xl_a, xs_a, y_shared)
    add_slider_row("seed", 0.0, 100.0, float(params.seed), xl_b, xs_b, y_shared)

    for _s in sliders.values():
        attach_slider_percent_display(_s)

    btn_h, btn_y = 0.034, 0.012
    reset_ax = fig.add_axes([0.05, btn_y, 0.15, btn_h])
    reset_button = Button(reset_ax, "Reset")
    stable_ax = fig.add_axes([0.22, btn_y, 0.24, btn_h])
    stable_button = Button(stable_ax, "Stable (random)")
    unstable_ax = fig.add_axes([0.48, btn_y, 0.30, btn_h])
    unstable_button = Button(unstable_ax, "Unstable (random)")

    def apply_params_to_sliders(mapping: dict[str, float | int]) -> None:
        sim_extras.clear()
        for key, val in mapping.items():
            if key not in sliders:
                sim_extras[key] = val
                continue
            s = sliders[key]
            lo, hi = s.valmin, s.valmax
            v = float(val) if key != "seed" else int(val)
            v = min(hi, max(lo, v))
            s.eventson = False
            s.set_val(v)
            s.eventson = True

    def update(_=None):
        stop_bowl_animation()

        p = LoveSystemParams(
            T=params.T,
            dt=params.dt,
            x1_0=float(sim_extras.get("x1_0", params.x1_0)),
            x2_0=float(sim_extras.get("x2_0", params.x2_0)),
            shock_time_1=float(sim_extras.get("shock_time_1", params.shock_time_1)),
            shock_mag_1=float(sim_extras.get("shock_mag_1", params.shock_mag_1)),
            shock_time_2=float(sim_extras.get("shock_time_2", params.shock_time_2)),
            shock_mag_2=float(sim_extras.get("shock_mag_2", params.shock_mag_2)),
            a1=sliders["a1"].val,
            a2=sliders["a2"].val,
            b1=sliders["b1"].val,
            b2=sliders["b2"].val,
            k1=sliders["k1"].val,
            k2=sliders["k2"].val,
            K1=sliders["K1"].val,
            K2=sliders["K2"].val,
            tau1=sliders["tau1"].val,
            tau2=sliders["tau2"].val,
            sigma1=sliders["sigma1"].val,
            sigma2=sliders["sigma2"].val,
            x1_star=sliders["x1*"].val,
            x2_star=sliders["x2*"].val,
            sat_alpha=sliders["sat"].val,
            seed=int(sliders["seed"].val),
        )

        t2, x1n, x2n, *_ = simulate(p)

        line_x1.set_xdata(t2)
        line_x1.set_ydata(x1n)
        line_x2.set_xdata(t2)
        line_x2.set_ydata(x2n)

        line_x1s.set_ydata([p.x1_star, p.x1_star])
        line_x2s.set_ydata([p.x2_star, p.x2_star])

        ax_state.relim()
        ax_state.autoscale_view()

        title_state.set_text(f"Emotional states — {classify_regime(x1n, x2n)}")
        fig.canvas.draw_idle()
        fig.canvas.flush_events()

        refresh_bowl(x1n, x2n, rotate_view=True)

    debounce_ms = 90
    try:
        deb_timer = fig.canvas.new_timer(interval=debounce_ms)
        deb_timer.single_shot = True
        deb_timer.add_callback(update)

        def on_slider_changed(_val):
            sim_extras.clear()
            stop_bowl_animation()
            deb_timer.stop()
            deb_timer.start()

        def stop_debounce():
            deb_timer.stop()

    except Exception:
        def on_slider_changed(_val):
            sim_extras.clear()
            stop_bowl_animation()
            update()

        def stop_debounce():
            pass

    def reset(_):
        sim_extras.clear()
        for s in sliders.values():
            s.eventson = False
        try:
            for s in sliders.values():
                s.reset()
        finally:
            for s in sliders.values():
                s.eventson = True
        stop_debounce()
        update()

    def stable_preset(_):
        label, mapping = random.choice(STABLE_PRESETS)
        apply_params_to_sliders(mapping)
        stop_debounce()
        update()
        print(f"Stable preset: {label}", file=sys.stderr)

    def unstable_preset(_):
        label, mapping = random.choice(UNSTABLE_PRESETS)
        apply_params_to_sliders(mapping)
        stop_debounce()
        update()
        print(f"Unstable preset: {label}", file=sys.stderr)

    for s in sliders.values():
        s.on_changed(on_slider_changed)

    reset_button.on_clicked(reset)
    stable_button.on_clicked(stable_preset)
    unstable_button.on_clicked(unstable_preset)

    refresh_bowl(x1, x2, rotate_view=True)
    plt.show()


def main(argv: list[str] | None = None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    if argv and argv[0] in ("-h", "--help"):
        print(__doc__)
        print(f"Selected Matplotlib backend: {_BACKEND}")
        return 0
    print(f"Love control GUI — Matplotlib backend: {_BACKEND}", file=sys.stderr)
    interactive_dashboard()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
