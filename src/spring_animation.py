"""
spring_animation.py
Three-panel animated comparison of a damped harmonic oscillator:
  Left   – exact physical spring (mass-spring diagram)
  Centre – standard NN prediction evolving epoch by epoch
  Right  – PINN prediction evolving epoch by epoch

Usage in a Jupyter notebook:
    from src.spring_animation import make_animation
    from IPython.display import HTML
    HTML(make_animation().to_jshtml())

Run as a script to save a GIF:
    python src/spring_animation.py          -> spring_comparison.gif
    python src/spring_animation.py out.gif  -> out.gif
"""

import sys
import numpy as np
import torch
import torch.nn as nn
import matplotlib
# Only force the Agg backend when running as a standalone script.
# When imported as a module (e.g., from Jupyter) the caller manages its own
# backend; overriding it here would silently break inline plot display.
if __name__ == "__main__":
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.animation as animation


# ── Physical parameters ───────────────────────────────────────────────────────
OMEGA0 = 2.0 * np.pi          # natural frequency  (rad/s)
ZETA   = 0.15                 # damping ratio
OMEGAD = OMEGA0 * np.sqrt(1.0 - ZETA**2)
T_END  = 3.0                  # simulation domain  (s)
T_DATA = 1.2                  # data available only in [0, T_DATA]
X_SCALE = 0.75                # visual scale for spring displacement


def _exact(t: np.ndarray) -> np.ndarray:
    """Exact solution x(t) for x(0)=1, x'(0)=0."""
    return np.exp(-ZETA * OMEGA0 * t) * (
        np.cos(OMEGAD * t) + (ZETA * OMEGA0 / OMEGAD) * np.sin(OMEGAD * t)
    )


# ── Network definition ────────────────────────────────────────────────────────
class _MLP(nn.Module):
    def __init__(self, width: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, width), nn.Tanh(),
            nn.Linear(width, width), nn.Tanh(),
            nn.Linear(width, width), nn.Tanh(),
            nn.Linear(width, 1),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return self.net(t)


# ── Training data ─────────────────────────────────────────────────────────────
def _make_data(n: int = 30, noise: float = 0.05, seed: int = 0):
    rng = np.random.default_rng(seed)
    t_np = np.sort(rng.uniform(0.0, T_DATA, n))
    x_np = _exact(t_np) + noise * rng.standard_normal(n)
    t_t  = torch.tensor(t_np, dtype=torch.float32).reshape(-1, 1)
    x_t  = torch.tensor(x_np, dtype=torch.float32).reshape(-1, 1)
    return t_np, x_np, t_t, x_t


# ── Joint training with snapshot saving ───────────────────────────────────────
def _train(
    n_epochs:    int = 5000,
    save_every:  int = 50,
    n_col:       int = 1500,
    nn_seed:     int = 1,
    pinn_seed:   int = 2,
):
    t_data_np, x_data_np, t_data, x_data = _make_data()

    torch.manual_seed(nn_seed)
    nn_model = _MLP()
    nn_opt   = torch.optim.Adam(nn_model.parameters(), lr=1e-3)

    torch.manual_seed(pinn_seed)
    pinn_model = _MLP()
    pinn_opt   = torch.optim.Adam(pinn_model.parameters(), lr=1e-3)

    # PINN collocation and IC tensors (fixed throughout training)
    t_col = (T_END * torch.rand(n_col, 1)).requires_grad_(True)
    t0    = torch.zeros(1, 1)

    # Evaluation grid
    t_eval_np = np.linspace(0.0, T_END, 400)
    t_eval    = torch.tensor(t_eval_np, dtype=torch.float32).reshape(-1, 1)

    n_snaps    = n_epochs // save_every + 1
    nn_snaps   = np.empty((n_snaps, len(t_eval_np)), dtype=np.float32)
    pinn_snaps = np.empty((n_snaps, len(t_eval_np)), dtype=np.float32)
    nn_losses  = []
    pinn_losses = []
    epoch_list  = []
    snap_idx    = 0

    for epoch in range(n_epochs + 1):
        # ── Standard NN step ────────────────────────────────────────────────
        nn_opt.zero_grad()
        l_nn = torch.mean((nn_model(t_data) - x_data) ** 2)
        l_nn.backward()
        nn_opt.step()

        # ── PINN step ────────────────────────────────────────────────────────
        pinn_opt.zero_grad()

        x_c = pinn_model(t_col)
        dx  = torch.autograd.grad(
            x_c, t_col,
            grad_outputs=torch.ones_like(x_c),
            create_graph=True,
        )[0]
        ddx = torch.autograd.grad(
            dx, t_col,
            grad_outputs=torch.ones_like(dx),
            create_graph=True,
        )[0]
        res   = ddx + 2.0 * ZETA * OMEGA0 * dx + OMEGA0**2 * x_c
        l_ode = torch.mean(res ** 2)

        # IC: x(0) = 1
        l_x0 = (pinn_model(t0) - 1.0) ** 2

        # IC: x'(0) = 0
        t0g  = t0.clone().requires_grad_(True)
        dx0  = torch.autograd.grad(
            pinn_model(t0g), t0g,
            grad_outputs=torch.ones(1, 1),
            create_graph=True,
        )[0]
        l_v0 = dx0 ** 2

        l_pinn = l_ode + 10.0 * (l_x0 + l_v0).mean()
        l_pinn.backward()
        pinn_opt.step()

        # ── Save snapshot ────────────────────────────────────────────────────
        if epoch % save_every == 0:
            with torch.no_grad():
                nn_snaps[snap_idx]   = nn_model(t_eval).numpy().flatten()
                pinn_snaps[snap_idx] = pinn_model(t_eval).numpy().flatten()
            nn_losses.append(l_nn.item())
            pinn_losses.append(l_pinn.item())
            epoch_list.append(epoch)
            snap_idx += 1

    return (
        t_eval_np,
        _exact(t_eval_np),
        t_data_np, x_data_np,
        nn_snaps[:snap_idx],
        pinn_snaps[:snap_idx],
        nn_losses,
        pinn_losses,
        epoch_list,
    )


# ── Spring geometry helper ────────────────────────────────────────────────────
def _spring_xy(y_top: float, y_bottom: float, n_coils: int = 7, w: float = 0.25):
    """Return (x_pts, y_pts) for a vertical zigzag coil spring."""
    margin   = 0.06
    y_coil_t = y_top    - margin
    y_coil_b = y_bottom + margin
    n_pts    = n_coils * 2
    y_coil   = np.linspace(y_coil_t, y_coil_b, n_pts)
    x_coil   = np.array([w * (1.0 if i % 2 == 0 else -1.0) for i in range(n_pts)])
    x = np.concatenate([[0.0], x_coil, [0.0]])
    y = np.concatenate([[y_top], y_coil, [y_bottom]])
    return x, y


# ── Main factory ──────────────────────────────────────────────────────────────
def make_animation(
    n_epochs:   int   = 5000,
    save_every: int   = 50,
    interval:   int   = 90,
) -> animation.FuncAnimation:
    """
    Train both models, then return a three-panel FuncAnimation.

    Parameters
    ----------
    n_epochs : total training epochs
    save_every : snapshot frequency (determines number of animation frames)
    interval : milliseconds between frames
    """
    print("Training both models… (this may take a minute)")
    (
        t_eval, x_exact,
        t_data_np, x_data_np,
        nn_snaps, pinn_snaps,
        nn_losses, pinn_losses,
        epoch_list,
    ) = _train(n_epochs=n_epochs, save_every=save_every)
    print(f"Done. {len(epoch_list)} animation frames.")

    # Times synced to animation frames for the spring panel
    t_spring = np.linspace(0.0, T_END, len(epoch_list))
    x_spring = _exact(t_spring)

    # ── Figure ────────────────────────────────────────────────────────────────
    fig, (ax_s, ax_n, ax_p) = plt.subplots(
        1, 3, figsize=(16, 5.2),
        gridspec_kw={"width_ratios": [1, 1.4, 1.4]},
    )
    fig.subplots_adjust(wspace=0.38, left=0.05, right=0.97, top=0.88, bottom=0.12)

    Y_LIM  = (-1.55, 1.55)
    Y_WALL = 1.25
    Y_EQ   = 0.0
    MASS_W = 0.34
    MASS_H = 0.17

    # ── Spring axis (static decorations) ─────────────────────────────────────
    ax_s.set_xlim(-0.75, 0.75)
    ax_s.set_ylim(-1.45, 1.55)
    ax_s.set_aspect("equal")
    ax_s.axis("off")
    ax_s.set_title("Exact damped spring", fontsize=11, fontweight="bold", pad=8)

    # Ceiling hatch
    ax_s.plot([-0.6, 0.6], [Y_WALL, Y_WALL], "k-", lw=3)
    for xi in np.linspace(-0.55, 0.55, 10):
        ax_s.plot([xi, xi - 0.07], [Y_WALL, Y_WALL + 0.10], "k-", lw=1.2)

    # Equilibrium guide
    ax_s.axhline(Y_EQ, color="gray", lw=0.7, ls="--", alpha=0.5)
    ax_s.text(0.42, Y_EQ + 0.04, "eq.", fontsize=8, color="gray", alpha=0.7)

    # Dynamic elements
    spring_line, = ax_s.plot([], [], "k-", lw=2)
    mass_patch   = mpatches.Rectangle(
        (-MASS_W / 2, -MASS_H / 2), MASS_W, MASS_H,
        facecolor="steelblue", edgecolor="k", lw=1.5,
    )
    ax_s.add_patch(mass_patch)
    spring_info = ax_s.text(
        0, -1.25, "", ha="center", fontsize=9.5, color="#222222"
    )

    # ── NN / PINN axes (static decorations) ──────────────────────────────────
    for ax, title, col in (
        (ax_n, "Standard NN  (data only)", "tomato"),
        (ax_p, "PINN  (physics-informed)", "royalblue"),
    ):
        ax.fill_betweenx(Y_LIM, 0.0, T_DATA,
                         color="green", alpha=0.07, label="training data window")
        ax.fill_betweenx(Y_LIM, T_DATA, T_END,
                         color="red",   alpha=0.05, label="extrapolation zone")
        ax.axvline(T_DATA, color="gray", lw=1.0, ls=":", alpha=0.8)
        ax.plot(t_eval, x_exact, "k-", lw=1.6, label="Exact", zorder=3)
        ax.set_xlim(0.0, T_END)
        ax.set_ylim(*Y_LIM)
        ax.set_xlabel("t  (s)", fontsize=11)
        ax.set_ylabel("x(t)", fontsize=11)
        ax.set_title(title, fontsize=11, color=col, fontweight="bold", pad=8)

    ax_n.scatter(t_data_np, x_data_np, s=20, c="green",
                 zorder=5, label="Training data")
    ax_n.legend(fontsize=8, loc="upper right")
    ax_p.legend(fontsize=8, loc="upper right")

    # Dynamic prediction lines
    line_nn,   = ax_n.plot([], [], color="tomato",     lw=2.0, ls="--", label="NN")
    line_pinn, = ax_p.plot([], [], color="royalblue",  lw=2.0, ls="-",  label="PINN")

    # Loss / epoch text
    txt_nn   = ax_n.text(0.03, 0.07, "", transform=ax_n.transAxes,
                         fontsize=9, color="darkred",   family="monospace")
    txt_pinn = ax_p.text(0.03, 0.07, "", transform=ax_p.transAxes,
                         fontsize=9, color="darkblue",  family="monospace")

    # Shared epoch title
    epoch_title = fig.text(
        0.5, 0.945, "", ha="center", fontsize=12, fontweight="bold"
    )

    # ── Update function ───────────────────────────────────────────────────────
    def _update(i: int):
        ep  = epoch_list[i]
        t_s = t_spring[i]
        x_s = x_spring[i]
        y_m = Y_EQ + x_s * X_SCALE          # mass y-position (scaled)

        # Spring
        sx, sy = _spring_xy(Y_WALL, y_m)
        spring_line.set_data(sx, sy)
        mass_patch.set_xy((-MASS_W / 2, y_m - MASS_H / 2))
        spring_info.set_text(f"t = {t_s:.2f} s\nx = {x_s:+.3f}")

        # NN prediction
        line_nn.set_data(t_eval, nn_snaps[i])
        txt_nn.set_text(f"epoch {ep:5d}   loss {nn_losses[i]:.2e}")

        # PINN prediction
        line_pinn.set_data(t_eval, pinn_snaps[i])
        txt_pinn.set_text(f"epoch {ep:5d}   loss {pinn_losses[i]:.2e}")

        epoch_title.set_text(f"Training epoch: {ep} / {n_epochs}")

        return spring_line, mass_patch, spring_info, line_nn, line_pinn, txt_nn, txt_pinn, epoch_title

    ani = animation.FuncAnimation(
        fig,
        _update,
        frames=len(epoch_list),
        interval=interval,
        blit=False,
    )
    plt.close(fig)
    return ani


# ── Script entry point ────────────────────────────────────────────────────────
if __name__ == "__main__":
    out_path = sys.argv[1] if len(sys.argv) > 1 else "spring_comparison.mp4"
    ani = make_animation()
    print(f"Saving to {out_path} …")
    ani.save(out_path, writer="ffmpeg", fps=12, dpi=110,
             extra_args=["-vcodec", "libx264", "-pix_fmt", "yuv420p"])
    print("Done.")
