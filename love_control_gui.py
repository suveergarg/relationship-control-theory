#!/usr/bin/env python3
"""
Desktop GUI for the two-agent love control model (native Matplotlib window).

Uses a non-inline backend so sliders and animation share a real GUI event loop
(typically smoother than Jupyter + ipympl). Tries Qt first, then Tk:

  pip install PySide6   # optional, for QtAgg / Qt5Agg
"""

from __future__ import annotations

import importlib
import sys
from dataclasses import dataclass


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

    sat_alpha: float = 1.0

    shock_time_1: float = 20.0
    shock_mag_1: float = 0.4
    shock_time_2: float = 45.0
    shock_mag_2: float = -0.5


def saturation(x, alpha: float = 1.0):
    return np.tanh(alpha * x)


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

        r1[i] = params.k1 * saturation(x2_delayed, params.sat_alpha)
        r2[i] = params.k2 * saturation(x1_delayed, params.sat_alpha)

        u1[i] = params.K1 * (x2_delayed - params.x1_star)
        u2[i] = params.K2 * (x1_delayed - params.x2_star)

        shock1 = external_shock(t[i], params.shock_time_1, params.shock_mag_1)
        shock2 = external_shock(t[i], params.shock_time_2, params.shock_mag_2)

        noise1 = params.sigma1 * rng.normal()
        noise2 = params.sigma2 * rng.normal()

        dx1 = (
            -params.a1 * x1[i - 1]
            + params.b1 * saturation(x2_delayed, params.sat_alpha)
            + r1[i]
            + u1[i]
            + shock1
            + noise1
        )

        dx2 = (
            -params.a2 * x2[i - 1]
            + params.b2 * saturation(x1_delayed, params.sat_alpha)
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


def interactive_dashboard():
    """Single window: sliders (Person 1 / Person 2), emotional states, bowl 3D + plan."""
    params = LoveSystemParams()

    t, x1, x2, *_ = simulate(params)
    r_max = 2.8

    fig = plt.figure(figsize=(12, 11.0))
    try:
        fig.canvas.manager.set_window_title("Love control — desktop GUI")
    except Exception:
        pass

    h_sl = 0.016
    v_step = 0.026
    y0 = 0.962
    x_a, w_a = 0.06, 0.40
    x_b, w_b = 0.54, 0.40

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

    slider_specs = []
    for i, (name, lo, hi, v0) in enumerate(p1):
        slider_specs.append((name, lo, hi, v0, [x_a, y0 - i * v_step, w_a, h_sl]))
    for i, (name, lo, hi, v0) in enumerate(p2):
        slider_specs.append((name, lo, hi, v0, [x_b, y0 - i * v_step, w_b, h_sl]))

    y_shared = y0 - 7 * v_step - 0.010
    slider_specs.append(("sat", 0.1, 3.0, params.sat_alpha, [0.10, y_shared, 0.35, h_sl]))
    slider_specs.append(("seed", 0, 100, params.seed, [0.55, y_shared, 0.35, h_sl]))

    fig.text(0.26, 0.993, "Person 1 (A)", ha="center", fontsize=11, fontweight="bold")
    fig.text(0.74, 0.993, "Person 2 (B)", ha="center", fontsize=11, fontweight="bold")
    fig.text(0.50, y_shared + 0.024, "Shared", ha="center", fontsize=9, style="italic", alpha=0.85)

    bowl_bottom = 0.07
    bowl_h = 0.33
    bowl_top = bowl_bottom + bowl_h

    gap = 0.018
    state_bottom = bowl_top + gap
    state_h = 0.24
    state_top = state_bottom + state_h

    ax_state = fig.add_axes([0.10, state_bottom, 0.82, state_h])

    line_x1, = ax_state.plot(t, x1, label="x1: Person A")
    line_x2, = ax_state.plot(t, x2, label="x2: Person B")
    line_x1s = ax_state.axhline(params.x1_star, linestyle="--", alpha=0.7, label="A setpoint")
    line_x2s = ax_state.axhline(params.x2_star, linestyle=":", alpha=0.7, label="B setpoint")
    title_state = ax_state.set_title(f"Emotional states ({classify_regime(x1, x2)})")
    ax_state.set_xlabel("Time")
    ax_state.set_ylabel("State")
    ax_state.grid(True, alpha=0.3)
    ax_state.legend(loc="upper right", fontsize=8)

    fig.text(
        0.50,
        state_top + 0.012,
        "Sliders control both emotional states and bowl",
        ha="center",
        fontsize=9,
        alpha=0.75,
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
    ax_b3d.set_title("Bowl — 3D")
    ax_b3d.set_xlabel("x")
    ax_b3d.set_ylabel("y")
    ax_b3d.set_zlabel("z")
    ax_b3d.legend(loc="upper left", fontsize=7)

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
    ax_bplan.set_title("Bowl — plan (2D)")
    ax_bplan.set_xlabel("x")
    ax_bplan.set_ylabel("y")
    ax_bplan.grid(True, alpha=0.3)
    ax_bplan.legend(fontsize=7)

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
    for name, vmin, vmax, vinit, rect in slider_specs:
        sax = fig.add_axes(rect)
        valfmt = "%0.0f" if name == "seed" else "%1.2f"
        sliders[name] = Slider(ax=sax, label=name, valmin=vmin, valmax=vmax, valinit=vinit, valfmt=valfmt)

    reset_ax = fig.add_axes([0.62, state_bottom - 0.002, 0.10, 0.034])
    reset_button = Button(reset_ax, "Reset")
    unstable_ax = fig.add_axes([0.74, state_bottom - 0.002, 0.14, 0.034])
    unstable_button = Button(unstable_ax, "Unstable preset")

    UNSTABLE_PRESET = {
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
        "sat": 1.0,
        "seed": 7,
    }

    def apply_params_to_sliders(mapping):
        for key, val in mapping.items():
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
            x1_0=params.x1_0,
            x2_0=params.x2_0,
            shock_time_1=params.shock_time_1,
            shock_mag_1=params.shock_mag_1,
            shock_time_2=params.shock_time_2,
            shock_mag_2=params.shock_mag_2,
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

        title_state.set_text(f"Emotional states ({classify_regime(x1n, x2n)})")
        fig.canvas.draw_idle()
        fig.canvas.flush_events()

        refresh_bowl(x1n, x2n, rotate_view=True)

    debounce_ms = 90
    try:
        deb_timer = fig.canvas.new_timer(interval=debounce_ms)
        deb_timer.single_shot = True
        deb_timer.add_callback(update)

        def on_slider_changed(_val):
            stop_bowl_animation()
            deb_timer.stop()
            deb_timer.start()

        def stop_debounce():
            deb_timer.stop()

    except Exception:
        def on_slider_changed(_val):
            stop_bowl_animation()
            update()

        def stop_debounce():
            pass

    def reset(_):
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

    def unstable_preset(_):
        apply_params_to_sliders(UNSTABLE_PRESET)
        stop_debounce()
        update()

    for s in sliders.values():
        s.on_changed(on_slider_changed)

    reset_button.on_clicked(reset)
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
