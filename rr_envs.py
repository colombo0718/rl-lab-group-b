"""
rr_envs.py — RR Platform Custom Gymnasium Environments
=======================================================
Five environments that exactly replicate the reward functions of
Rein Room (reinroom.leaflune.org). Visualization may differ
(this is the control group's Colab version), but game logic and
reward values are identical to the RR JavaScript source code.

Environments:
    MABEnv       — Multi-Armed Bandit (MAB.html)
    Maze1DEnv    — 1D Maze            (Maze1D.html)
    Maze2DEnv    — 2D Maze            (Maze2D.html)
    HeliEnv      — Helicopter Flappy  (heli.html)
    FighterEnv   — Fighter Plane      (fighter.html)

Usage:
    from rr_envs import MABEnv, Maze1DEnv, Maze2DEnv, HeliEnv, FighterEnv
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib import font_manager
from IPython.display import clear_output
# ─────────────────────────────────────────────────────────────
# Lightweight RGB render helpers for Colab / pytest
# ─────────────────────────────────────────────────────────────

_REWARD_EMOJI = {0: "❌", 1: "🍬", 3: "🪙", 10: "💎"}


def _figure_to_rgb_array(fig):
    """Convert a Matplotlib figure into an RGB uint8 image array."""
    fig.canvas.draw()
    width, height = fig.canvas.get_width_height()

    if hasattr(fig.canvas, "buffer_rgba"):
        rgba = np.asarray(fig.canvas.buffer_rgba(), dtype=np.uint8)
        rgb = rgba[:, :, :3].copy()
    else:
        buffer = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        rgb = buffer.reshape((height, width, 3))

    plt.close(fig)
    return rgb


def _blank_axes(figsize=(6, 4), xlim=None, ylim=None):
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")
    if xlim is not None:
        ax.set_xlim(*xlim)
    if ylim is not None:
        ax.set_ylim(*ylim)
    return fig, ax

def _available_font_names():
    return {font.name for font in font_manager.fontManager.ttflist}


def _pick_emoji_font():
    available = _available_font_names()
    for name in ("Segoe UI Emoji", "Apple Color Emoji", "Noto Color Emoji", "Segoe UI Symbol"):
        if name in available:
            return name
    return None


def _pick_ui_font():
    available = _available_font_names()
    for name in ("Microsoft JhengHei", "Microsoft YaHei", "Noto Sans CJK TC", "Arial Unicode MS", "DejaVu Sans"):
        if name in available:
            return name
    return None


_EMOJI_FONT = _pick_emoji_font()
_UI_FONT = _pick_ui_font()


def _emoji_text(ax, *args, **kwargs):
    if _EMOJI_FONT is not None:
        kwargs.setdefault("fontfamily", _EMOJI_FONT)
    return ax.text(*args, **kwargs)

def _emoji_title(ax, *args, **kwargs):
    if _EMOJI_FONT is not None:
        kwargs.setdefault("fontfamily", _EMOJI_FONT)
    return ax.set_title(*args, **kwargs)

def _ui_text(ax, *args, **kwargs):
    if _UI_FONT is not None:
        kwargs.setdefault("fontfamily", _UI_FONT)
    return ax.text(*args, **kwargs)

def _draw_slot_icon(ax, x, y, size=0.28):
    """Draw a small colorful slot-machine icon instead of relying on monochrome emoji fonts."""
    w = size
    h = size * 1.12
    ax.add_patch(patches.FancyBboxPatch(
        (x - w / 2, y - h / 2), w, h,
        boxstyle="round,pad=0.015,rounding_size=0.02",
        linewidth=1.2,
        edgecolor="#3d3d3d",
        facecolor="#4f8cff",
        zorder=3,
    ))
    ax.add_patch(patches.Rectangle((x - w * 0.30, y - h * 0.20), w * 0.60, h * 0.24,
                                   facecolor="#fff7d1", edgecolor="#333333", linewidth=0.8, zorder=4))
    for dx, color in ((-0.16, "#e84855"), (0.0, "#ffd447"), (0.16, "#43aa8b")):
        ax.add_patch(patches.Circle((x + dx * w, y - h * 0.08), w * 0.055, color=color, zorder=5))
    ax.plot([x + w * 0.42, x + w * 0.58], [y + h * 0.05, y + h * 0.20], color="#333333", linewidth=1.5, zorder=4)
    ax.add_patch(patches.Circle((x + w * 0.61, y + h * 0.23), w * 0.055, color="#e84855", zorder=5))
    ax.add_patch(patches.Rectangle((x - w * 0.24, y + h * 0.24), w * 0.48, h * 0.08,
                                   facecolor="#2f2f2f", edgecolor="none", zorder=4))


def _draw_reward_icon(ax, value, x, y, size=0.28):
    """Draw colorful RR reward symbols for ❌/🍬/🪙/💎 without font-color limitations."""
    value = int(value) if value != "?" else "?"
    if value == "?":
        ax.add_patch(patches.Circle((x, y), size * 0.42, facecolor="#eeeeee", edgecolor="#999999", linewidth=1.0, zorder=3))
        ax.text(x, y - size * 0.01, "?", ha="center", va="center", fontsize=12, color="#777777", zorder=4)
        return

    if value == 0:
        ax.plot([x - size * 0.32, x + size * 0.32], [y - size * 0.32, y + size * 0.32],
                color="#e03131", linewidth=3.2, solid_capstyle="round", zorder=4)
        ax.plot([x - size * 0.32, x + size * 0.32], [y + size * 0.32, y - size * 0.32],
                color="#e03131", linewidth=3.2, solid_capstyle="round", zorder=4)
        return

    if value == 1:
        ax.add_patch(patches.Polygon(
            [(x - size * 0.52, y), (x - size * 0.33, y + size * 0.22), (x - size * 0.33, y - size * 0.22)],
            closed=True, facecolor="#ff9ecb", edgecolor="#9b3f68", linewidth=0.8, zorder=3))
        ax.add_patch(patches.Polygon(
            [(x + size * 0.52, y), (x + size * 0.33, y + size * 0.22), (x + size * 0.33, y - size * 0.22)],
            closed=True, facecolor="#ff9ecb", edgecolor="#9b3f68", linewidth=0.8, zorder=3))
        ax.add_patch(patches.FancyBboxPatch(
            (x - size * 0.32, y - size * 0.18), size * 0.64, size * 0.36,
            boxstyle="round,pad=0.01,rounding_size=0.04",
            facecolor="#ff6fb1", edgecolor="#9b3f68", linewidth=0.9, zorder=4))
        ax.plot([x - size * 0.18, x + size * 0.18], [y - size * 0.14, y + size * 0.14],
                color="#ffffff", linewidth=1.2, zorder=5)
        return

    if value == 3:
        ax.add_patch(patches.Circle((x, y), size * 0.42, facecolor="#f2c94c", edgecolor="#a66a00", linewidth=1.2, zorder=3))
        ax.add_patch(patches.Circle((x, y), size * 0.28, facecolor="#ffd966", edgecolor="#c88a00", linewidth=0.8, zorder=4))
        ax.text(x, y, "3", ha="center", va="center", fontsize=8.5, color="#7a4b00", fontweight="bold", zorder=5)
        return

    if value == 10:
        ax.add_patch(patches.Polygon(
            [(x, y + size * 0.48), (x + size * 0.42, y + size * 0.05), (x + size * 0.24, y - size * 0.42),
             (x - size * 0.24, y - size * 0.42), (x - size * 0.42, y + size * 0.05)],
            closed=True, facecolor="#66d9ef", edgecolor="#168aad", linewidth=1.1, zorder=3))
        ax.plot([x - size * 0.24, x, x + size * 0.24], [y - size * 0.42, y + size * 0.48, y - size * 0.42],
                color="#dff9ff", linewidth=0.9, zorder=4)
        return

    ax.text(x, y, str(value), ha="center", va="center", fontsize=10, zorder=4)


def _draw_fire_icon(ax, x, y, size=0.30):
    ax.add_patch(patches.Polygon(
        [(x, y + size * 0.50), (x + size * 0.26, y + size * 0.05), (x + size * 0.10, y - size * 0.42),
         (x - size * 0.04, y - size * 0.12), (x - size * 0.24, y - size * 0.40), (x - size * 0.20, y + size * 0.10)],
        closed=True, facecolor="#ff7a00", edgecolor="#b74700", linewidth=0.9, zorder=4))
    ax.add_patch(patches.Polygon(
        [(x, y + size * 0.28), (x + size * 0.12, y + size * 0.02), (x, y - size * 0.22), (x - size * 0.12, y + size * 0.02)],
        closed=True, facecolor="#ffd447", edgecolor="none", zorder=5))


def _draw_person_icon(ax, x, y, size=0.36):
    ax.add_patch(patches.Circle((x, y - size * 0.20), size * 0.18, facecolor="#ffd6a5", edgecolor="#8a5a2b", linewidth=0.8, zorder=5))
    ax.add_patch(patches.Arc((x, y - size * 0.24), size * 0.32, size * 0.26, theta1=0, theta2=180,
                             color="#3d2b1f", linewidth=2.0, zorder=6))
    ax.add_patch(patches.FancyBboxPatch(
        (x - size * 0.22, y - size * 0.02), size * 0.44, size * 0.36,
        boxstyle="round,pad=0.01,rounding_size=0.03",
        facecolor="#4dabf7", edgecolor="#1864ab", linewidth=0.8, zorder=4))
    ax.plot([x - size * 0.18, x - size * 0.32], [y + size * 0.10, y + size * 0.30], color="#1864ab", linewidth=1.8, zorder=4)
    ax.plot([x + size * 0.18, x + size * 0.32], [y + size * 0.10, y + size * 0.30], color="#1864ab", linewidth=1.8, zorder=4)


def _draw_bomb_icon(ax, x, y, size=0.34):
    ax.add_patch(patches.Circle((x, y), size * 0.34, facecolor="#2b2d42", edgecolor="#111111", linewidth=1.0, zorder=4))
    ax.plot([x + size * 0.20, x + size * 0.36], [y - size * 0.24, y - size * 0.42], color="#5c4033", linewidth=2.0, zorder=5)
    ax.add_patch(patches.Circle((x + size * 0.41, y - size * 0.47), size * 0.07, facecolor="#ffd447", edgecolor="#f08c00", linewidth=0.8, zorder=6))
    ax.add_patch(patches.Circle((x - size * 0.12, y - size * 0.10), size * 0.08, facecolor="#6c757d", edgecolor="none", zorder=5))


def _draw_trophy_icon(ax, x, y, size=0.36):
    ax.add_patch(patches.FancyBboxPatch(
        (x - size * 0.20, y - size * 0.28), size * 0.40, size * 0.34,
        boxstyle="round,pad=0.01,rounding_size=0.025",
        facecolor="#ffd43b", edgecolor="#b08900", linewidth=1.0, zorder=5))
    ax.add_patch(patches.Arc((x - size * 0.25, y - size * 0.14), size * 0.26, size * 0.28, theta1=90, theta2=270,
                             color="#b08900", linewidth=1.5, zorder=4))
    ax.add_patch(patches.Arc((x + size * 0.25, y - size * 0.14), size * 0.26, size * 0.28, theta1=-90, theta2=90,
                             color="#b08900", linewidth=1.5, zorder=4))
    ax.add_patch(patches.Rectangle((x - size * 0.06, y + size * 0.06), size * 0.12, size * 0.24,
                                   facecolor="#d4a017", edgecolor="#8a6d00", linewidth=0.8, zorder=4))
    ax.add_patch(patches.Rectangle((x - size * 0.28, y + size * 0.28), size * 0.56, size * 0.10,
                                   facecolor="#8d6e00", edgecolor="#5f4b00", linewidth=0.8, zorder=4))


def _draw_pizza_icon(ax, x, y, size=0.38):
    ax.add_patch(patches.Polygon(
        [(x - size * 0.34, y - size * 0.26), (x + size * 0.34, y - size * 0.26), (x, y + size * 0.40)],
        closed=True, facecolor="#ffd166", edgecolor="#b5651d", linewidth=1.0, zorder=4))
    ax.plot([x - size * 0.28, x + size * 0.28], [y - size * 0.20, y - size * 0.20], color="#b5651d", linewidth=2.0, zorder=5)
    for dx, dy in ((-0.12, -0.06), (0.10, 0.02), (0.00, 0.20)):
        ax.add_patch(patches.Circle((x + size * dx, y + size * dy), size * 0.045, facecolor="#e03131", edgecolor="none", zorder=5))


def _draw_helicopter_icon(ax, x, y, size=34):
    ax.add_patch(patches.Ellipse((x, y), size * 0.92, size * 0.42, facecolor="#4dabf7", edgecolor="#1864ab", linewidth=1.4, zorder=5))
    ax.add_patch(patches.Circle((x - size * 0.18, y - size * 0.02), size * 0.12, facecolor="#d0ebff", edgecolor="#1864ab", linewidth=0.8, zorder=6))
    ax.plot([x - size * 0.50, x - size * 0.86], [y, y - size * 0.08], color="#1864ab", linewidth=2.0, zorder=5)
    ax.add_patch(patches.Circle((x - size * 0.92, y - size * 0.09), size * 0.08, facecolor="#ff6b6b", edgecolor="#a61e4d", linewidth=0.8, zorder=6))
    ax.plot([x - size * 0.50, x + size * 0.50], [y - size * 0.36, y - size * 0.36], color="#333333", linewidth=2.0, zorder=6)
    ax.plot([x, x], [y - size * 0.22, y - size * 0.36], color="#333333", linewidth=1.4, zorder=6)
    ax.plot([x - size * 0.28, x + size * 0.28], [y + size * 0.30, y + size * 0.30], color="#333333", linewidth=1.8, zorder=6)


def _draw_plane_icon(ax, x, y, size=34):
    ax.add_patch(patches.Polygon(
        [(x, y - size * 0.62), (x + size * 0.18, y + size * 0.30), (x, y + size * 0.18), (x - size * 0.18, y + size * 0.30)],
        closed=True, facecolor="#4dabf7", edgecolor="#1864ab", linewidth=1.2, zorder=6))
    ax.add_patch(patches.Polygon(
        [(x - size * 0.12, y - size * 0.05), (x - size * 0.58, y + size * 0.16), (x - size * 0.12, y + size * 0.18)],
        closed=True, facecolor="#74c0fc", edgecolor="#1864ab", linewidth=1.0, zorder=5))
    ax.add_patch(patches.Polygon(
        [(x + size * 0.12, y - size * 0.05), (x + size * 0.58, y + size * 0.16), (x + size * 0.12, y + size * 0.18)],
        closed=True, facecolor="#74c0fc", edgecolor="#1864ab", linewidth=1.0, zorder=5))
    ax.add_patch(patches.Circle((x, y - size * 0.20), size * 0.07, facecolor="#d0ebff", edgecolor="#1864ab", linewidth=0.7, zorder=7))


def _draw_rock_icon(ax, x, y, size=24):
    pts = [(x - size * 0.55, y + size * 0.05), (x - size * 0.35, y - size * 0.42),
           (x + size * 0.16, y - size * 0.55), (x + size * 0.52, y - size * 0.18),
           (x + size * 0.42, y + size * 0.36), (x - size * 0.08, y + size * 0.52)]
    ax.add_patch(patches.Polygon(pts, closed=True, facecolor="#868e96", edgecolor="#495057", linewidth=1.2, zorder=5))
    ax.plot([x - size * 0.28, x + size * 0.05], [y - size * 0.12, y - size * 0.32], color="#adb5bd", linewidth=1.2, zorder=6)
    ax.plot([x + size * 0.02, x + size * 0.30], [y + size * 0.18, y - size * 0.02], color="#6c757d", linewidth=1.1, zorder=6)


# ─────────────────────────────────────────────────────────────
# 1. Multi-Armed Bandit
# ─────────────────────────────────────────────────────────────

class MABEnv(gym.Env):
    """
    Multi-Armed Bandit — 5 machines, episode = 10 pulls.

    Reward pools match RR MAB.html exactly:
        ❌ = 0 pts  |  🍬 = 1 pt  |  🪙 = 3 pts  |  💎 = 10 pts

    Modes (same as RR):
        'same'    — all machines share the same pool
        'fixed'   — two 0s, two 1s, one 3
        'slight'  — slight expected-value differences between machines
        'jackpot' — one machine hides a 💎 pool
    """

    _POOLS = {
        "same": [
            [0,0,0,1,1,1,1,1,3,3],
            [0,0,0,1,1,1,1,1,3,3],
            [0,0,0,1,1,1,1,1,3,3],
            [0,0,0,1,1,1,1,1,3,3],
            [0,0,0,1,1,1,1,1,3,3],
        ],
        "fixed": [
            [0]*10, [0]*10,
            [1]*10, [1]*10,
            [3]*10,
        ],
        "slight": [
            [0,0,0,1,1,1,1,1,3,3],
            [0,0,0,1,1,1,1,1,3,3],
            [1,1,1,1,1,1,1,1,3,3],
            [1,1,1,1,1,1,1,1,3,3],
            [1,1,1,1,1,1,3,3,3,3],
        ],
        "jackpot": [
            [0,0,0,1,1,1,1,1,3,3],
            [0,0,0,1,1,1,1,1,3,3],
            [0,0,0,1,1,1,1,1,3,3],
            [0,0,0,1,1,1,1,1,3,3],
            [0,0,0,0,0,0,0,0,10,10],
        ],
    }

    def __init__(self, mode: str = "slight"):
        super().__init__()
        assert mode in self._POOLS, f"mode must be one of {list(self._POOLS)}"
        self.mode = mode
        self.n_machines = 5
        self.episode_step_limit = 10          # 10 pulls per episode (matches RR)

        self.observation_space = spaces.Box(
            low=np.array([0], dtype=np.float32),
            high=np.array([4], dtype=np.float32),
        )
        self.action_space = spaces.Discrete(self.n_machines)

    def _setup_pools(self):
        pools = self._POOLS[self.mode]
        idx = self.np_random.permutation(len(pools))
        self.machine_pools = [pools[i] for i in idx]

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if not hasattr(self, "_episode_count"):
            self._episode_count = 0
        if not hasattr(self, "machine_pools") or seed is not None or self._episode_count % 100 == 0:
            self._setup_pools()
        self._episode_count += 1
        self.selected = 0
        self.step_count = 0
        self.displayed_rewards = ["?"] * self.n_machines
        return np.array([self.selected], dtype=np.float32), {}

    def step(self, action: int):
        self.selected = int(action)
        pool = self.machine_pools[self.selected]
        reward = float(self.np_random.choice(pool))
        self.displayed_rewards = ["?"] * self.n_machines
        self.displayed_rewards[self.selected] = int(reward)
        self.step_count += 1
        done = self.step_count >= self.episode_step_limit
        return np.array([self.selected], dtype=np.float32), reward, done, False, {}
    def render(self):
        """Render MAB like RR MAB.html: 編號 / 獎勵池 / 選擇 / 獎勵."""
        fig, ax = _blank_axes(figsize=(9.4, 4.6), xlim=(0, 10), ylim=(0, 6.4))
        fig.patch.set_facecolor("white")
        ax.set_facecolor("white")

        left = 0.35
        top = 5.55
        row_h = 0.82
        col_w = [1.35, 5.35, 1.25, 1.45]
        headers = ["編號", "獎勵池", "選擇", "獎勵"]
        total_w = sum(col_w)

        _ui_text(ax, left, 6.05, f"Multi-Armed Bandit  mode={self.mode}  step={self.step_count}/{self.episode_step_limit}",
                ha="left", va="center", fontsize=13, fontweight="bold", color="#222222")

        x = left
        for label, width in zip(headers, col_w):
            ax.add_patch(patches.Rectangle((x, top), width, row_h, facecolor="#f0f0f0", edgecolor="#d4d4d4", linewidth=1.0))
            _ui_text(ax, x + width / 2, top + row_h / 2, label, ha="center", va="center", fontsize=12, fontweight="bold")
            x += width

        displayed = getattr(self, "displayed_rewards", ["?"] * self.n_machines)
        for i in range(self.n_machines):
            y = top - (i + 1) * row_h
            row_face = "#fff7d6" if i == getattr(self, "selected", 0) else "#ffffff"
            x = left
            for width in col_w:
                ax.add_patch(patches.Rectangle((x, y), width, row_h, facecolor=row_face, edgecolor="#dcdcdc", linewidth=1.0))
                x += width

            # 編號: RR uses 🎰 A0. Draw a colorful slot icon to avoid monochrome emoji rendering.
            _draw_slot_icon(ax, left + 0.36, y + row_h / 2, size=0.36)
            _ui_text(ax, left + 0.82, y + row_h / 2, f"A{i}", ha="left", va="center", fontsize=13, fontweight="bold", color="#222222")

            # 獎勵池: draw all ten RR reward symbols in color.
            pool_x0 = left + col_w[0] + 0.30
            for j, reward_value in enumerate(getattr(self, "machine_pools", [[0] * 10])[i]):
                _draw_reward_icon(ax, reward_value, pool_x0 + j * 0.47, y + row_h / 2, size=0.34)

            # 選擇: RR shows 👈 or 🔘. Use a yellow pointer or grey radio dot.
            select_cx = left + col_w[0] + col_w[1] + col_w[2] / 2
            select_cy = y + row_h / 2
            if i == getattr(self, "selected", 0):
                ax.add_patch(patches.Polygon(
                    [(select_cx - 0.24, select_cy), (select_cx + 0.18, select_cy + 0.22),
                     (select_cx + 0.08, select_cy + 0.06), (select_cx + 0.34, select_cy + 0.06),
                     (select_cx + 0.34, select_cy - 0.06), (select_cx + 0.08, select_cy - 0.06),
                     (select_cx + 0.18, select_cy - 0.22)],
                    closed=True, facecolor="#ffd447", edgecolor="#996f00", linewidth=1.0))
            else:
                ax.add_patch(patches.Circle((select_cx, select_cy), 0.14, facecolor="#e9ecef", edgecolor="#868e96", linewidth=1.0))

            # 獎勵: unknown until selected this step, matching RR's ❔ behavior.
            reward_cx = left + col_w[0] + col_w[1] + col_w[2] + col_w[3] / 2
            _draw_reward_icon(ax, displayed[i], reward_cx, y + row_h / 2, size=0.40)

        progress_y = top - (self.n_machines + 1) * row_h + 0.22
        ax.add_patch(patches.Rectangle((left, progress_y), total_w, 0.13, facecolor="#e5e5e5", edgecolor="none"))
        ax.add_patch(patches.Rectangle((left, progress_y), total_w * self.step_count / self.episode_step_limit, 0.13,
                                       facecolor="#3ca65c", edgecolor="none"))
        _ui_text(ax, left, progress_y - 0.26, "每 10 次拉霸為一回合", ha="left", va="center", fontsize=10, color="#555555")
        return _figure_to_rgb_array(fig)


# ─────────────────────────────────────────────────────────────
# 2. Maze 1D
# ─────────────────────────────────────────────────────────────

class Maze1DEnv(gym.Env):
    """
    1D Maze — 10 cells (0–9). Bomb at cell 0, Goal at cell 9.

    Reward (matches RR Maze1D.html):
        Reach Goal  🏆 : +10, done
        Reach Bomb  💣 : -10, done
        Pie mode    🍕 : goal gives +2 instead of +10
        Positive feedback: +1 moving right (not at goal), -1 moving left
        Negative feedback: -1 moving right, +1 moving left

    Actions:
        0 = stay  |  1 = right  |  2 = left
    """

    def __init__(self, start_pos: int = 4, feedback_mode: str = "none"):
        super().__init__()
        assert 1 <= start_pos <= 8, "start_pos must be between 1 and 8"
        assert feedback_mode in ("none", "positive", "negative", "pie")
        self.grid_size = 10
        self.start_pos = start_pos
        self.feedback_mode = feedback_mode

        self.observation_space = spaces.Discrete(self.grid_size)
        self.action_space = spaces.Discrete(3)   # 0=stay, 1=right, 2=left

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.pos = self.start_pos
        return int(self.pos), {}

    def step(self, action: int):
        prev = self.pos

        if action == 1 and self.pos < self.grid_size - 1:
            self.pos += 1
        elif action == 2 and self.pos > 0:
            self.pos -= 1

        reward = 0.0

        # Step feedback
        if self.feedback_mode == "positive":
            if self.pos > prev and self.pos < 9:
                reward += 1.0
            elif self.pos < prev and self.pos > 0:
                reward -= 1.0
        elif self.feedback_mode in ("negative", "pie"):
            if self.pos > prev and self.pos < 9:
                reward -= 1.0
            elif self.pos < prev and self.pos > 0:
                reward += 1.0

        done = False
        if self.pos == 0:                  # bomb
            reward -= 10.0
            done = True
        elif self.pos == self.grid_size - 1:   # goal
            reward += 2.0 if self.feedback_mode == "pie" else 10.0
            done = True

        return int(self.pos), reward, done, False, {}
    def render(self):
        """Render the 1D maze with colorful RR-style symbols."""
        fig, ax = _blank_axes(figsize=(8, 1.8), xlim=(0, self.grid_size), ylim=(0, 1))
        _ui_text(ax, 0, 1.12, f"Maze1D feedback={self.feedback_mode}", ha="left", va="center", fontsize=13, fontweight="bold")

        for x in range(self.grid_size):
            face = "#ffffff"
            if x == 0:
                face = "#ffe3e3"
            elif x == self.grid_size - 1:
                face = "#def7e5"
            elif self.feedback_mode == "positive" and x != self.pos:
                face = "#fff9db" if x > self.pos and x < 9 else "#fff0f0"
            elif self.feedback_mode in ("negative", "pie") and x != self.pos:
                face = "#fff9db" if 0 < x < self.pos else "#fff0f0"
            ax.add_patch(patches.Rectangle((x, 0), 1, 1, linewidth=1.2, edgecolor="#333333", facecolor=face))

        if self.feedback_mode == "positive":
            for x in range(1, 9):
                if x < self.pos:
                    _draw_fire_icon(ax, x + 0.5, 0.5, size=0.34)
                elif x > self.pos:
                    _draw_reward_icon(ax, 1, x + 0.5, 0.5, size=0.36)
        elif self.feedback_mode in ("negative", "pie"):
            for x in range(1, 9):
                if x < self.pos:
                    _draw_reward_icon(ax, 1, x + 0.5, 0.5, size=0.36)
                elif x > self.pos:
                    _draw_fire_icon(ax, x + 0.5, 0.5, size=0.34)

        _draw_bomb_icon(ax, 0.5, 0.5, size=0.54)
        if self.feedback_mode == "pie":
            _draw_pizza_icon(ax, self.grid_size - 0.5, 0.5, size=0.58)
        else:
            _draw_trophy_icon(ax, self.grid_size - 0.5, 0.5, size=0.58)
        _draw_person_icon(ax, self.pos + 0.5, 0.5, size=0.62)
        return _figure_to_rgb_array(fig)


# ─────────────────────────────────────────────────────────────
# 3. Maze 2D
# ─────────────────────────────────────────────────────────────

class Maze2DEnv(gym.Env):
    """
    2D Maze. Start: (0, 0).

    Reward (matches RR Maze2D.html):
        Reach goal : +10, done
        All other steps : 0

    State: [x, y] grid coordinates.

    Actions:
        0 = stay  |  1 = up  |  2 = down  |  3 = left  |  4 = right

    Levels (match RR Maze2D level selector):
        0 = Open Field  — 10×10 grid, goal (9, 9)
        1 = Walled In   —  5×5 grid, goal (4, 4), faster convergence
    """

    _LEVELS = {
        0: {"grid_size": 10, "goal": (9, 9), "name": "Open Field"},
        1: {"grid_size": 5,  "goal": (4, 4), "name": "Walled In"},
    }

    def __init__(self, level: int = 0):
        super().__init__()
        assert level in self._LEVELS, f"level must be one of {list(self._LEVELS)}"
        cfg = self._LEVELS[level]
        self.level      = level
        self.grid_size  = cfg["grid_size"]
        self.goal       = cfg["goal"]
        self.level_name = cfg["name"]

        self.observation_space = spaces.Box(
            low=np.array([0, 0], dtype=np.float32),
            high=np.array([self.grid_size - 1, self.grid_size - 1], dtype=np.float32),
        )
        self.action_space = spaces.Discrete(5)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.x, self.y = 0, 0
        return np.array([self.x, self.y], dtype=np.float32), {}

    def step(self, action: int):
        g = self.grid_size - 1
        if action == 1 and self.y < g:
            self.y += 1
        elif action == 2 and self.y > 0:
            self.y -= 1
        elif action == 3 and self.x > 0:
            self.x -= 1
        elif action == 4 and self.x < g:
            self.x += 1

        done = (self.x, self.y) == self.goal
        reward = 10.0 if done else 0.0
        return np.array([self.x, self.y], dtype=np.float32), reward, done, False, {}

    def render(self):
        """Render the 2D maze with colorful RR-style symbols."""
        g = self.grid_size
        fig, ax = _blank_axes(figsize=(6, 6), xlim=(0, g), ylim=(0, g))
        _ui_text(ax, 0, g + 0.35, f"Maze2D  level={self.level} ({self.level_name})",
                 ha="left", va="center", fontsize=14, fontweight="bold")

        gx, gy = self.goal
        for x in range(g):
            for y in range(g):
                face = "#def7e5" if (x, y) == self.goal else "#ffffff"
                ax.add_patch(patches.Rectangle((x, y), 1, 1, linewidth=0.8, edgecolor="#555555", facecolor=face))

        _draw_trophy_icon(ax, gx + 0.5, gy + 0.5, size=0.58)
        _draw_person_icon(ax, self.x + 0.5, self.y + 0.5, size=0.58)
        return _figure_to_rgb_array(fig)


# ─────────────────────────────────────────────────────────────
# 4. Heli (Flappy Helicopter)
# ─────────────────────────────────────────────────────────────

class HeliEnv(gym.Env):
    """
    Helicopter Flappy Bird — matches RR heli.html exactly.

    Reward:
        Survive each step  : +0.01
        Pass a wall        : +1.0
        Crash (wall/edge)  : -10.0, done

    State (3D, normalized to [-1, 1]):
        heliY        — helicopter vertical position
        wallDistance — horizontal distance to next wall
        gapCenterY   — vertical center of next gap

    Actions:
        0 = fall (gravity)  |  1 = flap (move up)

    Modes (match RR):
        'fixed' — gap always at center height
        'small' — gap center ± 100 px random
        'large' — gap center ± 200 px random
    """

    # Constants from heli.html
    CANVAS_W      = 500
    CANVAS_H      = 500
    HELI_X        = 120
    HELI_SIZE     = 34
    WALL_WIDTH    = 44
    WALL_SPACING  = 220
    GAP_HEIGHT    = 170
    FLY_SPEED     = 2.0
    WALL_SPEED    = 1.8
    FIXED_GAP_CTR = 250
    SURVIVAL_R    = 0.01
    PASS_R        = 1.0
    CRASH_P       = -10.0

    def __init__(self, mode: str = "fixed"):
        super().__init__()
        assert mode in ("fixed", "small", "large")
        self.mode = mode

        self.observation_space = spaces.Box(
            low=np.full(3, -1.0, dtype=np.float32),
            high=np.full(3,  1.0, dtype=np.float32),
        )
        self.action_space = spaces.Discrete(2)

    # ----------------------------------------------------------
    def _make_wall(self, x: float) -> dict:
        if self.mode == "fixed":
            center = self.FIXED_GAP_CTR
        elif self.mode == "small":
            center = self.np_random.uniform(
                self.FIXED_GAP_CTR - 100, self.FIXED_GAP_CTR + 100
            )
        else:
            center = self.np_random.uniform(
                self.FIXED_GAP_CTR - 200, self.FIXED_GAP_CTR + 200
            )
        lo = self.GAP_HEIGHT / 2 + 20
        hi = self.CANVAS_H - self.GAP_HEIGHT / 2 - 20
        center = float(np.clip(center, lo, hi))
        gap_top = center - self.GAP_HEIGHT / 2
        return {"x": x, "gap_top": gap_top, "gap_bottom": gap_top + self.GAP_HEIGHT, "passed": False}

    def _obs(self) -> np.ndarray:
        nw = next((w for w in self.walls if w["x"] + self.WALL_WIDTH >= self.HELI_X), self.walls[0])
        gap_center = (nw["gap_top"] + nw["gap_bottom"]) / 2
        wall_dist   = nw["x"] + self.WALL_WIDTH - self.HELI_X
        # normalize each to [-1, 1] and clip (wall may start outside canvas)
        hy  = self.heli_y / self.CANVAS_H * 2 - 1
        wd  = wall_dist  / self.CANVAS_W * 2 - 1
        gc  = gap_center / self.CANVAS_H * 2 - 1
        return np.clip(np.array([hy, wd, gc], dtype=np.float32), -1.0, 1.0)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.heli_y   = float(self.CANVAS_H / 2)
        self.heli_dir = 1        # 1 = falling, -1 = rising
        self.walls = [
            self._make_wall(self.CANVAS_W + 120),
            self._make_wall(self.CANVAS_W + 120 + self.WALL_SPACING),
            self._make_wall(self.CANVAS_W + 120 + self.WALL_SPACING * 2),
        ]
        self.score = 0
        return self._obs(), {}

    def step(self, action: int):
        self.heli_dir = -1 if action == 1 else 1
        self.heli_y  += self.heli_dir * self.FLY_SPEED
        reward        = self.SURVIVAL_R

        # Move walls, check pass
        for w in self.walls:
            w["x"] -= self.WALL_SPEED
            if not w["passed"] and w["x"] + self.WALL_WIDTH < self.HELI_X:
                w["passed"] = True
                self.score += 1
                reward += self.PASS_R

        # Recycle wall
        if self.walls[0]["x"] + self.WALL_WIDTH < -10:
            self.walls.pop(0)
            self.walls.append(self._make_wall(self.walls[-1]["x"] + self.WALL_SPACING))

        # Collision — boundary
        r = self.HELI_SIZE * 0.28   # ≈ 9.52
        if self.heli_y - r <= 0 or self.heli_y + r >= self.CANVAS_H:
            return self._obs(), reward + self.CRASH_P, True, False, {}

        # Collision — walls
        for w in self.walls:
            hit_x = self.HELI_X + r > w["x"] and self.HELI_X - r < w["x"] + self.WALL_WIDTH
            hit_y = self.heli_y - r < w["gap_top"] or self.heli_y + r > w["gap_bottom"]
            if hit_x and hit_y:
                return self._obs(), reward + self.CRASH_P, True, False, {}

        return self._obs(), reward, False, False, {}
    def render(self):
        """Render the helicopter game with a colorful RR-style helicopter."""
        fig, ax = _blank_axes(figsize=(6, 6), xlim=(0, self.CANVAS_W), ylim=(self.CANVAS_H, 0))
        _ui_text(ax, 8, -18, f"Heli score={self.score}", ha="left", va="center", fontsize=14, fontweight="bold")

        ax.add_patch(patches.Rectangle((0, 0), self.CANVAS_W, self.CANVAS_H, facecolor="#edf7ff", edgecolor="#333333"))
        for wall in self.walls:
            ax.add_patch(patches.Rectangle((wall["x"], 0), self.WALL_WIDTH, wall["gap_top"], facecolor="#46a05a", edgecolor="#26723a"))
            ax.add_patch(patches.Rectangle((wall["x"], wall["gap_bottom"]), self.WALL_WIDTH, self.CANVAS_H - wall["gap_bottom"], facecolor="#46a05a", edgecolor="#26723a"))
        _draw_helicopter_icon(ax, self.HELI_X, self.heli_y, size=self.HELI_SIZE)
        return _figure_to_rgb_array(fig)


# ─────────────────────────────────────────────────────────────
# 5. Fighter Plane
# ─────────────────────────────────────────────────────────────

class FighterEnv(gym.Env):
    """
    Fighter Plane — matches RR fighter.html exactly.

    Reward:
        Hit rock (bullet hits rock)  : +10
        Miss (bullet exits top)      : -1
        Rock crashes into player     : -100, done
        Clear (10 hits)              : done (success)

    State (5D, normalized to [-1, 1]):
        playerX, rockX, rockY, rockVX, rockVY

    Actions:
        0 = none  |  1 = left  |  2 = right  |  3 = shoot

    Modes (match RR 5 levels):
        'fixed'    — rock stationary at center top
        'randomX'  — rock random X each episode
        'randomXY' — rock random X and Y
        'falling'  — rock falls from top (moving target)
        'drifting' — falling + horizontal drift
    """

    CANVAS_W    = 500
    CANVAS_H    = 600
    LANE_MIN    = 36
    LANE_MAX    = 464          # 500 - 36
    PLAYER_Y    = 528          # 600 - 72
    MAX_SPEED   = 12.0
    ACCEL       = 2.5
    FRICTION    = 0.82
    BULLET_SPD  = 20.0
    HIT_R       = 10.0
    MISS_P      = -1.0
    CRASH_P     = -100.0
    CLEAR_HITS  = 10
    COOLDOWN    = 20           # frames

    def __init__(self, mode: str = "fixed"):
        super().__init__()
        assert mode in ("fixed", "randomX", "randomXY", "falling", "drifting")
        self.mode = mode

        self.observation_space = spaces.Box(
            low=np.full(5, -1.0, dtype=np.float32),
            high=np.full(5,  1.0, dtype=np.float32),
        )
        self.action_space = spaces.Discrete(4)

    # ----------------------------------------------------------
    def _spawn_rock(self) -> dict:
        cx = self.CANVAS_W / 2
        if self.mode == "fixed":
            return {"x": cx, "y": 60.0, "vx": 0.0, "vy": 0.0}
        if self.mode == "randomX":
            return {"x": float(self.np_random.uniform(self.LANE_MIN, self.LANE_MAX)),
                    "y": 60.0, "vx": 0.0, "vy": 0.0}
        if self.mode == "randomXY":
            return {"x": float(self.np_random.uniform(self.LANE_MIN, self.LANE_MAX)),
                    "y": float(self.np_random.uniform(30, 200)),
                    "vx": 0.0, "vy": 0.0}
        if self.mode == "falling":
            return {"x": float(self.np_random.uniform(self.LANE_MIN, self.LANE_MAX)),
                    "y": -20.0, "vx": 0.0,
                    "vy": float(self.np_random.uniform(1.5, 3.0))}
        # drifting
        return {"x": float(self.np_random.uniform(self.LANE_MIN, self.LANE_MAX)),
                "y": -20.0,
                "vx": float(self.np_random.uniform(-2.0, 2.0)),
                "vy": float(self.np_random.uniform(1.5, 3.0))}

    def _normalize(self) -> np.ndarray:
        cx = self.CANVAS_W / 2
        cy = self.CANVAS_H / 2
        px  = (self.player_x          - cx) / cx
        rx  = (self.rock["x"]         - cx) / cx
        ry  = (cy - self.rock["y"])         / cy   # positive = above center
        rvx = self.rock["vx"]               / self.MAX_SPEED
        rvy = self.rock["vy"]               / self.MAX_SPEED
        return np.clip(np.array([px, rx, ry, rvx, rvy], dtype=np.float32), -1.0, 1.0)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.player_x   = float(self.CANVAS_W / 2)
        self.player_vx  = 0.0
        self.bullet     = None
        self.rock       = self._spawn_rock()
        self.hits       = 0
        self.cooldown   = 0
        return self._normalize(), {}

    def step(self, action: int):
        reward = 0.0

        if self.cooldown > 0:
            self.cooldown -= 1

        # Action
        if action == 1:
            self.player_vx -= self.ACCEL
        elif action == 2:
            self.player_vx += self.ACCEL
        elif action == 3 and self.bullet is None and self.cooldown == 0:
            self.bullet   = {"x": self.player_x, "y": float(self.PLAYER_Y)}
            self.cooldown = self.COOLDOWN

        # Physics — player
        self.player_vx  *= self.FRICTION
        self.player_vx   = float(np.clip(self.player_vx, -self.MAX_SPEED, self.MAX_SPEED))
        self.player_x   += self.player_vx
        self.player_x    = float(np.clip(self.player_x, self.LANE_MIN, self.LANE_MAX))

        # Physics — rock (cylindrical X wrap, same as RR)
        self.rock["x"] += self.rock["vx"]
        self.rock["y"] += self.rock["vy"]
        if self.rock["x"] < self.LANE_MIN:
            self.rock["x"] = float(self.LANE_MAX)
        elif self.rock["x"] > self.LANE_MAX:
            self.rock["x"] = float(self.LANE_MIN)

        # Rock exits bottom (falling/drifting modes) → respawn
        if self.rock["y"] > self.CANVAS_H + 20:
            self.rock = self._spawn_rock()

        # Bullet movement & hit detection
        if self.bullet is not None:
            self.bullet["y"] -= self.BULLET_SPD
            rock_r = 20
            if (abs(self.bullet["x"] - self.rock["x"]) < rock_r and
                    abs(self.bullet["y"] - self.rock["y"]) < rock_r):
                reward     += self.HIT_R
                self.hits  += 1
                self.bullet = None
                self.rock   = self._spawn_rock()
                if self.hits >= self.CLEAR_HITS:
                    return self._normalize(), reward, True, False, {"result": "clear"}
            elif self.bullet["y"] < 0:
                reward     += self.MISS_P
                self.bullet = None

        # Rock crashes into player
        player_r = 20
        if (abs(self.rock["x"] - self.player_x) < player_r and
                abs(self.rock["y"] - self.PLAYER_Y) < player_r):
            reward += self.CRASH_P
            return self._normalize(), reward, True, False, {"result": "crash"}

        return self._normalize(), reward, False, False, {}
    def render(self):
        """Render the fighter game with colorful RR-style plane, rock, and bullet."""
        fig, ax = _blank_axes(figsize=(5.4, 6.5), xlim=(0, self.CANVAS_W), ylim=(self.CANVAS_H, 0))
        _ui_text(ax, 8, -20, f"Fighter hits={self.hits}/{self.CLEAR_HITS}", ha="left", va="center", fontsize=14, fontweight="bold")

        ax.add_patch(patches.Rectangle((0, 0), self.CANVAS_W, self.CANVAS_H, facecolor="#f7fbff", edgecolor="#333333"))
        ax.axvline(self.LANE_MIN, color="#999999", linewidth=1)
        ax.axvline(self.LANE_MAX, color="#999999", linewidth=1)
        _draw_rock_icon(ax, self.rock["x"], self.rock["y"], size=28)
        _draw_plane_icon(ax, self.player_x, self.PLAYER_Y, size=34)
        if self.bullet is not None:
            ax.add_patch(patches.FancyBboxPatch(
                (self.bullet["x"] - 3, self.bullet["y"] - 12), 6, 20,
                boxstyle="round,pad=0.0,rounding_size=2",
                facecolor="#ffd43b", edgecolor="#f08c00", linewidth=0.8, zorder=6))
        return _figure_to_rgb_array(fig)


# ─────────────────────────────────────────────────────────────
# Colab HTML renderers with native color emoji
# ─────────────────────────────────────────────────────────────

def _rr_reward_emoji(value) -> str:
    if value == "?":
        return "❔"
    return _REWARD_EMOJI.get(int(value), "❔")


def _rr_html_shell(title: str, body: str, width: str = "fit-content") -> str:
    return f"""
<div class="rr-env" style="width:{width}; font-family: system-ui, -apple-system, BlinkMacSystemFont,
            'Segoe UI Emoji', 'Apple Color Emoji', 'Noto Color Emoji', 'Microsoft JhengHei', sans-serif;
            color:#222; line-height:1.25;">
  <div style="font-weight:800; font-size:18px; margin:0 0 10px 0;">{title}</div>
  {body}
</div>
"""


def _mab_render_html(self) -> str:
    displayed = getattr(self, "displayed_rewards", ["?"] * self.n_machines)
    rows = []
    for i in range(self.n_machines):
        selected = i == getattr(self, "selected", 0)
        bg = "#fff7d6" if selected else "#fff"
        pool = "".join(_rr_reward_emoji(v) for v in getattr(self, "machine_pools", [[0] * 10])[i])
        selector = "👈" if selected else "🔘"
        reward = _rr_reward_emoji(displayed[i])
        rows.append(f"""
        <tr style="background:{bg};">
          <td style="border:1px solid #ddd; padding:10px 12px; text-align:center; white-space:nowrap;">🎰 A{i}</td>
          <td style="border:1px solid #ddd; padding:10px 12px; text-align:center; letter-spacing:2px;">{pool}</td>
          <td style="border:1px solid #ddd; padding:10px 12px; text-align:center;">{selector}</td>
          <td style="border:1px solid #ddd; padding:10px 12px; text-align:center;">{reward}</td>
        </tr>
        """)

    progress = int(100 * self.step_count / self.episode_step_limit)
    body = f"""
    <table style="border-collapse:collapse; font-size:28px; background:#fff; box-shadow:0 1px 2px rgba(0,0,0,.08);">
      <thead>
        <tr style="background:#f0f0f0;">
          <th style="border:1px solid #ddd; padding:10px 12px; text-align:center;">編號</th>
          <th style="border:1px solid #ddd; padding:10px 12px; text-align:center;">獎勵池</th>
          <th style="border:1px solid #ddd; padding:10px 12px; text-align:center;">選擇</th>
          <th style="border:1px solid #ddd; padding:10px 12px; text-align:center;">獎勵</th>
        </tr>
      </thead>
      <tbody>{''.join(rows)}</tbody>
    </table>
    <div style="height:8px; background:#e5e5e5; margin-top:10px; width:100%;">
      <div style="height:8px; background:#3ca65c; width:{progress}%;"></div>
    </div>
    <div style="font-size:13px; color:#555; margin-top:6px;">mode={self.mode} · step={self.step_count}/{self.episode_step_limit} · 每 10 次拉霸為一回合</div>
    """
    return _rr_html_shell("🎰 Multi-Armed Bandit", body)


def _maze1d_render_html(self) -> str:
    cells = []
    for i in range(self.grid_size):
        icon = ""
        bg = "#fff"
        if i == 0:
            icon = "💣"
            bg = "#ffe3e3"
        elif i == self.grid_size - 1:
            icon = "🍕" if self.feedback_mode == "pie" else "🏆"
            bg = "#def7e5"
        elif self.feedback_mode == "positive":
            if i < self.pos:
                icon = "🔥"
                bg = "#fff0f0"
            elif i > self.pos:
                icon = "🍬"
                bg = "#fff9db"
        elif self.feedback_mode in ("negative", "pie"):
            if i < self.pos:
                icon = "🍬"
                bg = "#fff9db"
            elif i > self.pos:
                icon = "🔥"
                bg = "#fff0f0"

        if i == self.pos:
            icon = "🧑"
            bg = "#d3f9d8"

        cells.append(f"""
        <div style="width:50px; height:50px; display:flex; align-items:center; justify-content:center;
                    border:1px solid #111; background:{bg}; font-size:38px; box-sizing:border-box;">{icon}</div>
        """)

    body = f"""
    <div style="display:flex; width:500px; height:50px;">{''.join(cells)}</div>
    <div style="font-size:13px; color:#555; margin-top:8px;">playerPos={self.pos} · feedback={self.feedback_mode}</div>
    """
    return _rr_html_shell("1D Maze", body, width="520px")


def _maze2d_render_html(self) -> str:
    g = self.grid_size
    cell_px = 42
    gx, gy = self.goal
    cells = []
    for y in reversed(range(g)):
        for x in range(g):
            icon = ""
            bg = "#fff"
            if (x, y) == self.goal:
                icon = "🏆"
                bg = "#def7e5"
            if x == self.x and y == self.y:
                icon = "🧑"
                bg = "#d3f9d8"
            cells.append(f"""
            <div style="width:{cell_px}px; height:{cell_px}px; display:flex; align-items:center; justify-content:center;
                        border:1px solid #555; background:{bg}; font-size:30px; box-sizing:border-box;">{icon}</div>
            """)

    grid_w = g * cell_px
    body = f"""
    <div style="display:grid; grid-template-columns:repeat({g}, {cell_px}px); grid-template-rows:repeat({g}, {cell_px}px);
                width:{grid_w}px; height:{grid_w}px;">{''.join(cells)}</div>
    <div style="font-size:13px; color:#555; margin-top:8px;">state=({self.x}, {self.y}) · goal={self.goal} · level={self.level} ({self.level_name})</div>
    """
    return _rr_html_shell("2D Maze", body, width=f"{grid_w + 10}px")


def _heli_render_html(self) -> str:
    walls = []
    for wall in self.walls:
        x = wall["x"]
        walls.append(f"""
        <div style="position:absolute; left:{x}px; top:0; width:{self.WALL_WIDTH}px; height:{wall['gap_top']}px;
                    background:#46a05a; border:1px solid #26723a; box-sizing:border-box;"></div>
        <div style="position:absolute; left:{x}px; top:{wall['gap_bottom']}px; width:{self.WALL_WIDTH}px;
                    height:{self.CANVAS_H - wall['gap_bottom']}px; background:#46a05a;
                    border:1px solid #26723a; box-sizing:border-box;"></div>
        """)

    body = f"""
    <div style="position:relative; width:{self.CANVAS_W}px; height:{self.CANVAS_H}px; background:#edf7ff;
                border:2px solid #333; overflow:hidden; box-sizing:content-box;">
      {''.join(walls)}
      <div style="position:absolute; left:{self.HELI_X - 20}px; top:{self.heli_y - 20}px; font-size:34px; line-height:1;">🚁</div>
    </div>
    <div style="font-size:13px; color:#555; margin-top:8px;">score={self.score} · mode={self.mode}</div>
    """
    return _rr_html_shell("Heli", body, width=f"{self.CANVAS_W + 8}px")


def _fighter_render_html(self) -> str:
    bullet = ""
    if self.bullet is not None:
        bullet = f"""<div style="position:absolute; left:{self.bullet['x'] - 5}px; top:{self.bullet['y'] - 12}px;
                         font-size:18px; line-height:1; color:#f2b705;">🟡</div>"""

    body = f"""
    <div style="position:relative; width:{self.CANVAS_W}px; height:{self.CANVAS_H}px; background:#f7fbff;
                border:2px solid #333; overflow:hidden; box-sizing:content-box;">
      <div style="position:absolute; left:{self.LANE_MIN}px; top:0; width:1px; height:{self.CANVAS_H}px; background:#999;"></div>
      <div style="position:absolute; left:{self.LANE_MAX}px; top:0; width:1px; height:{self.CANVAS_H}px; background:#999;"></div>
      <div style="position:absolute; left:{self.rock['x'] - 18}px; top:{self.rock['y'] - 18}px; font-size:34px; line-height:1;">🪨</div>
      <div style="position:absolute; left:{self.player_x - 22}px; top:{self.PLAYER_Y - 22}px; font-size:40px; line-height:1; transform:rotate(-45deg);">✈️</div>
      {bullet}
    </div>
    <div style="font-size:13px; color:#555; margin-top:8px;">hits={self.hits}/{self.CLEAR_HITS} · mode={self.mode}</div>
    """
    return _rr_html_shell("Fighter", body, width=f"{self.CANVAS_W + 8}px")


def show_env(env):
    """Display an RR-style color emoji game view in Colab/Jupyter."""
    from IPython.display import HTML, display
    display(HTML(env.render_html()))


MABEnv.render_html = _mab_render_html
Maze1DEnv.render_html = _maze1d_render_html
Maze2DEnv.render_html = _maze2d_render_html
HeliEnv.render_html = _heli_render_html
FighterEnv.render_html = _fighter_render_html


# ─────────────────────────────────────────────────────────────
# Shared Q-Learning Runner (for Colab use)
# ─────────────────────────────────────────────────────────────

def run_q_learning(env, alpha=0.5, gamma=0.95, epsilon=0.2,
                   n_episodes=500, bins=None, verbose=True, max_steps=2000):
    """
    Run tabular Q-learning on any rr_envs environment.

    For continuous state environments (Maze2D, Heli, Fighter),
    pass bins=N to discretize each dimension into N buckets.
    Discrete-state envs (MAB, Maze1D) ignore bins.

    Returns:
        episode_rewards  — list of total reward per episode
        episode_lengths  — list of steps per episode
        Q                — final Q-table (dict)
    """
    is_discrete = isinstance(env.observation_space, spaces.Discrete)
    obs_dim     = 1 if is_discrete else env.observation_space.shape[0]
    n_actions   = env.action_space.n

    if is_discrete:
        n_states = env.observation_space.n
        def to_key(obs):
            return int(obs)
    else:
        if bins is None:
            bins = 6
        low   = env.observation_space.low
        high  = env.observation_space.high
        edges = [np.linspace(low[i], high[i], bins + 1) for i in range(obs_dim)]
        def to_key(obs):
            return tuple(
                int(np.clip(np.digitize(obs[i], edges[i]) - 1, 0, bins - 1))
                for i in range(obs_dim)
            )

    Q = {}
    get_q  = lambda s, a: Q.get((s, a), 0.0)
    best_a = lambda s: max(range(n_actions), key=lambda a: get_q(s, a))

    episode_rewards, episode_lengths = [], []

    for ep in range(n_episodes):
        obs, _ = env.reset()
        state   = to_key(obs)
        total_r = 0.0
        steps   = 0
        done    = False

        while not done and steps < max_steps:
            if np.random.random() < epsilon:
                action = env.action_space.sample()
            else:
                action = best_a(state)

            obs, reward, terminated, truncated, _ = env.step(action)
            done       = terminated or truncated
            next_state = to_key(obs)

            old_q   = get_q(state, action)
            next_q  = max(get_q(next_state, a) for a in range(n_actions)) if not done else 0.0
            Q[(state, action)] = old_q + alpha * (reward + gamma * next_q - old_q)

            state   = next_state
            total_r += reward
            steps   += 1

        episode_rewards.append(total_r)
        episode_lengths.append(steps)

        if verbose and (ep + 1) % max(1, n_episodes // 10) == 0:
            avg = np.mean(episode_rewards[-50:])
            print(f"  Episode {ep+1:>5} / {n_episodes}  |  avg reward (last 50): {avg:.2f}")

    return episode_rewards, episode_lengths, Q


def plot_training(episode_rewards, episode_lengths,
                  label="", window=30):
    """Plot reward curve + episode length curve."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4))

    smooth_r = np.convolve(episode_rewards, np.ones(window) / window, mode="valid")
    ax1.plot(smooth_r, linewidth=2, label=label)
    ax1.set_title(f"Episode Reward  (smoothed, window={window})")
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Total Reward")
    ax1.grid(True, alpha=0.3)

    smooth_l = np.convolve(episode_lengths, np.ones(window) / window, mode="valid")
    ax2.plot(smooth_l, color="coral", linewidth=2)
    ax2.set_title("Episode Length (steps)")
    ax2.set_xlabel("Episode")
    ax2.set_ylabel("Steps")
    ax2.grid(True, alpha=0.3)

    plt.suptitle(label, fontsize=13)
    plt.tight_layout()
    plt.show()


def plot_maze2d_qtable(Q, bins=6):
    """Visualize Q-table for Maze2D as a heatmap with arrows."""
    import matplotlib.patches as mpatches

    action_arrows = {0: "", 1: "↑", 2: "↓", 3: "←", 4: "→"}
    grid = np.zeros((bins, bins))
    arrows = [["" for _ in range(bins)] for _ in range(bins)]

    for bx in range(bins):
        for by in range(bins):
            vals = [Q.get(((bx, by), a), 0.0) for a in range(5)]
            grid[bins - 1 - by, bx] = max(vals)
            arrows[bins - 1 - by][bx] = action_arrows[int(np.argmax(vals))]

    fig, ax = plt.subplots(figsize=(7, 7))
    im = ax.imshow(grid, cmap="YlOrRd", aspect="equal")
    plt.colorbar(im, ax=ax, label="Max Q-value (confidence)")

    for r in range(bins):
        for c in range(bins):
            ax.text(c, r, arrows[r][c], ha="center", va="center", fontsize=14)

    ax.set_xticks(range(bins))
    ax.set_yticks(range(bins))
    ax.set_xlabel("X bin (→ east)")
    ax.set_ylabel("Y bin (↑ north)")
    ax.set_title("Maze2D Q-Table — Best Action per State\n(color = confidence, arrow = preferred direction)")
    plt.tight_layout()
    plt.show()


def plot_maze1d_qtable(Q, grid_size=10):
    """
    Q-Value Slice for Maze1D: bar chart of best Q-value per state position.
    Equivalent to the Q-Value Slice line chart on the RR platform.
    """
    action_name = {0: "stay", 1: "right", 2: "left"}
    states = list(range(grid_size))
    best_vals = [max(Q.get((s, a), 0.0) for a in range(3)) for s in states]
    best_acts = [max(range(3), key=lambda a: Q.get((s, a), 0.0)) for s in states]

    colors = ["#e03131" if v < 0 else "#2f9e44" if v > 0 else "#ced4da" for v in best_vals]
    labels = ["💣" if s == 0 else "🏆" if s == grid_size - 1 else str(s) for s in states]

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(states, best_vals, color=colors, edgecolor="white", linewidth=0.8, zorder=3)
    ax.axhline(0, color="#333333", linewidth=1.2)

    for s, v, a in zip(states, best_vals, best_acts):
        if v != 0:
            ax.text(s, v + (0.3 if v >= 0 else -0.3), action_name[a],
                    ha="center", va="bottom" if v >= 0 else "top", fontsize=9, color="#333333")

    ax.set_xticks(states)
    ax.set_xticklabels(labels, fontsize=12)
    ax.set_xlabel("State (position)")
    ax.set_ylabel("Max Q-value")
    ax.set_title("Q-Value Slice — Best value per state position\n"
                 "Green = positive (goal direction)  |  Red = negative (bomb direction)  |  Grey = not yet learned")
    ax.grid(True, axis="y", alpha=0.3, zorder=0)
    plt.tight_layout()
    plt.show()









# ─────────────────────────────────────────────────────────────
# Colab animation helpers
# ─────────────────────────────────────────────────────────────

def animate_actions(env, actions, delay=0.35, reset=False, show_initial=True, stop_on_done=True):
    """
    Play an action sequence in Colab/Jupyter by repeatedly updating render_html().

    Example:
        env = Maze1DEnv(start_pos=4)
        env.reset()
        animate_actions(env, [1, 1, 1, 1, 1], delay=0.4)
    """
    import time
    from IPython.display import HTML, clear_output, display

    if reset:
        env.reset()

    if show_initial:
        clear_output(wait=True)
        display(HTML(env.render_html()))
        time.sleep(delay)

    last = None
    for step_idx, action in enumerate(actions, start=1):
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        last = (obs, reward, terminated, truncated, info)

        clear_output(wait=True)
        display(HTML(env.render_html()))
        print(f"step={step_idx}  action={action}  reward={reward}  done={done}")
        time.sleep(delay)

        if done and stop_on_done:
            break

    return last


def animate_random(env, steps=30, delay=0.25, reset=True, stop_on_done=True):
    """
    Play random actions with an RR-style color emoji view in Colab/Jupyter.

    This is useful as a quick visual smoke test before plugging in Q-learning.
    """
    if reset:
        env.reset()
    actions = [env.action_space.sample() for _ in range(steps)]
    return animate_actions(
        env,
        actions,
        delay=delay,
        reset=False,
        show_initial=True,
        stop_on_done=stop_on_done,
    )
