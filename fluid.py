"""
fluid.py — SPH Fluid Simulation (Phase 14 — Dynamic Memory & Performance Graph)
Changes applied:
  • Pre-flight Fix: MAX_PARTICLES renamed to INITIAL_CAPACITY; slider ceiling
    decoupled via SLIDER_MAX so the user can actually reach higher counts.
  • Pre-flight Fix: calibrate_rest_density now cached — only recomputes when
    particle_radius changes, not every frame.
  • Pre-flight Fix: grid_next grows in lockstep with particle arrays inside
    maybe_grow_arrays() to prevent silent out-of-bounds writes.
  • Pre-flight Fix: Removed redundant ays[num_active] = gravity_val from the
    pour loop (integrate() resets it on the very next tick anyway).
  • Phase 14: Dynamic array resizing — maybe_grow_arrays() doubles capacity
    whenever num_active is about to exceed it, preserving all live particles.
  • Phase 14: Real-time FPS graph drawn at the bottom of the UI panel using
    a collections.deque(maxlen=120) ring buffer.
"""

import pygame
import random
import math
import numpy as np
from numba import njit, prange
from collections import deque

# ---------------------------------------------------------------------------
# Display & Layout
# ---------------------------------------------------------------------------
SIM_WIDTH  = 800
UI_WIDTH   = 250
WIDTH      = SIM_WIDTH + UI_WIDTH
HEIGHT     = 600
FPS        = 60
BACKGROUND = (8, 12, 24)

# ---------------------------------------------------------------------------
# Array Capacity
# ---------------------------------------------------------------------------
INITIAL_CAPACITY = 10_000   # Starting allocation — NOT a hard limit anymore.
SLIDER_MAX       = 30_000   # Ceiling the UI slider is allowed to reach.

# Grid is bounded by the smallest possible smoothing radius, not particle count.
MIN_SMOOTHING_RADIUS = 5.0
MAX_GRID_COLS = math.ceil(SIM_WIDTH / MIN_SMOOTHING_RADIUS) + 2
MAX_GRID_ROWS = math.ceil(HEIGHT    / MIN_SMOOTHING_RADIUS) + 2

# ---------------------------------------------------------------------------
# Static Physics Defaults
# ---------------------------------------------------------------------------
WALL_FRICTION          = 0.05
INTERIOR_DENSITY_RATIO = 0.85

# Pour stream
POUR_X      = 180.0
POUR_Y      = 80.0
POUR_VX     = 10.0
POUR_VY     = 2.0
POUR_RATE   = 8
POUR_SPREAD = 3.0

# Mouse
MOUSE_RADIUS   = 80.0
MOUSE_STRENGTH = 3.0

# Metaball renderer
RENDER_SCALE     = 2
RENDER_THRESHOLD = 0.5

# ---------------------------------------------------------------------------
# Flask geometry
# ---------------------------------------------------------------------------
def build_flask():
    cx = SIM_WIDTH // 2
    cy = HEIGHT // 2 + 40
    hw, hh = 260, 200
    return (float(cx - hw), float(cx + hw), float(cy - hh), float(cy + hh))

FLASK_BOUNDS = build_flask()
F_LEFT, F_RIGHT, F_TOP, F_BOT = FLASK_BOUNDS

# ---------------------------------------------------------------------------
# Dynamic Array Management  (Phase 14 core)
# ---------------------------------------------------------------------------
def make_particle_arrays(capacity: int):
    """Allocate a fresh set of particle arrays at the given capacity."""
    return (
        np.zeros(capacity, dtype=np.float32),   # xs
        np.zeros(capacity, dtype=np.float32),   # ys
        np.zeros(capacity, dtype=np.float32),   # vxs
        np.zeros(capacity, dtype=np.float32),   # vys
        np.zeros(capacity, dtype=np.float32),   # axs
        np.zeros(capacity, dtype=np.float32),   # ays
        np.zeros(capacity, dtype=np.float32),   # densities
        np.full(capacity, -1, dtype=np.int32),  # grid_next
    )

def maybe_grow_arrays(num_active, capacity, xs, ys, vxs, vys, axs, ays, densities, grid_next):
    """
    If num_active has reached the current capacity, double every particle array
    and grid_next in one shot, copying live data into the new allocation.
    Returns (new_capacity, xs, ys, vxs, vys, axs, ays, densities, grid_next).
    Only allocates when necessary — no per-frame cost in steady state.
    """
    if num_active < capacity:
        return capacity, xs, ys, vxs, vys, axs, ays, densities, grid_next

    new_cap = capacity * 2
    print(f"[Phase 14] Growing arrays: {capacity} → {new_cap}")

    def grow(arr, fill=0):
        new = np.full(new_cap, fill, dtype=arr.dtype)
        new[:capacity] = arr[:capacity]
        return new

    return (
        new_cap,
        grow(xs),
        grow(ys),
        grow(vxs),
        grow(vys),
        grow(axs),
        grow(ays),
        grow(densities),
        grow(grid_next, fill=-1),
    )

# ---------------------------------------------------------------------------
# Colour
# ---------------------------------------------------------------------------
def speed_to_color(speed: float) -> tuple:
    MAX_SPEED = 14.0
    t = min(speed / MAX_SPEED, 1.0)
    if t < 0.33:
        s = t / 0.33
        return (0, int(150 * s + 105), 255)
    elif t < 0.66:
        s = (t - 0.33) / 0.33
        return (int(255 * s), 255, int(255 * (1 - s)))
    else:
        s = (t - 0.66) / 0.34
        return (255, int(255 * (1 - s)), 0)

# ---------------------------------------------------------------------------
# UI Classes
# ---------------------------------------------------------------------------
class Slider:
    def __init__(self, x, y, w, h, min_val, max_val, initial_val,
                 text, is_int=False, decimals=1):
        self.rect     = pygame.Rect(x, y, w, h)
        self.min_val  = min_val
        self.max_val  = max_val
        self.val      = initial_val
        self.text     = text
        self.is_int   = is_int
        self.decimals = decimals
        self.dragging = False

    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self.rect.collidepoint(event.pos):
                self.dragging = True
                return True
        elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
            self.dragging = False
        elif event.type == pygame.MOUSEMOTION and self.dragging:
            rel_x    = max(0, min(event.pos[0] - self.rect.x, self.rect.width))
            fraction = rel_x / self.rect.width
            self.val = self.min_val + fraction * (self.max_val - self.min_val)
            if self.is_int:
                self.val = int(self.val)
            return True
        return False

    def draw(self, screen, font):
        pygame.draw.rect(screen, (40, 50, 70), self.rect, border_radius=4)
        handle_x = self.rect.x + (
            (self.val - self.min_val) / (self.max_val - self.min_val)
        ) * self.rect.width
        pygame.draw.circle(
            screen, (200, 220, 255),
            (int(handle_x), self.rect.centery),
            self.rect.height // 2 + 4,
        )
        val_str = (
            f"{int(self.val)}"
            if self.is_int
            else f"{self.val:.{self.decimals}f}"
        )
        txt = font.render(f"{self.text}: {val_str}", True, (220, 230, 255))
        screen.blit(txt, (self.rect.x, self.rect.y - 22))


class Button:
    def __init__(self, x, y, w, h, text):
        self.rect   = pygame.Rect(x, y, w, h)
        self.text   = text
        self.active = False

    def draw(self, screen, font):
        color = (100, 150, 255) if self.active else (60, 70, 90)
        pygame.draw.rect(screen, color, self.rect, border_radius=4)
        pygame.draw.rect(screen, (200, 220, 255), self.rect, 1, border_radius=4)
        txt = font.render(self.text, True, (255, 255, 255))
        screen.blit(
            txt,
            (
                self.rect.centerx - txt.get_width()  // 2,
                self.rect.centery - txt.get_height() // 2,
            ),
        )

    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self.rect.collidepoint(event.pos):
                return True
        return False

# ---------------------------------------------------------------------------
# FPS Graph  (Phase 14)
# ---------------------------------------------------------------------------
FPS_HISTORY_LEN = 120   # 2 seconds at 60 fps

def draw_fps_graph(screen, font, fps_history, panel_x, panel_y, panel_w, panel_h):
    """
    Draws a filled line graph of recent FPS values at the bottom of the UI panel.
    panel_x/y = top-left corner of the graph area, panel_w/h = its size.
    """
    bg_rect = pygame.Rect(panel_x, panel_y, panel_w, panel_h)
    pygame.draw.rect(screen, (15, 20, 32), bg_rect, border_radius=4)
    pygame.draw.rect(screen, (40, 55, 80), bg_rect, 1, border_radius=4)

    label = font.render("FPS", True, (80, 110, 160))
    screen.blit(label, (panel_x + 4, panel_y + 4))

    history = list(fps_history)
    if len(history) < 2:
        return

    max_fps   = max(max(history), 60.0)
    graph_top = panel_y + 20
    graph_h   = panel_h - 24
    graph_w   = panel_w - 8

    # Build polyline points
    points = []
    for i, f in enumerate(history):
        x = panel_x + 4 + int(i / (FPS_HISTORY_LEN - 1) * graph_w)
        y = panel_y + panel_h - 4 - int((f / max_fps) * graph_h)
        points.append((x, y))

    # Filled area under the curve
    if len(points) >= 2:
        baseline = panel_y + panel_h - 4
        poly = [points[0]] + points + [(points[-1][0], baseline), (points[0][0], baseline)]
        fill_surf = pygame.Surface((panel_w, panel_h), pygame.SRCALPHA)
        local_poly = [(px - panel_x, py - panel_y) for px, py in poly]
        pygame.draw.polygon(fill_surf, (40, 100, 200, 60), local_poly)
        screen.blit(fill_surf, (panel_x, panel_y))
        pygame.draw.lines(screen, (80, 160, 255), False, points, 1)

    # Current FPS label (right-aligned)
    cur = font.render(f"{history[-1]:.0f}", True, (140, 200, 255))
    screen.blit(cur, (panel_x + panel_w - cur.get_width() - 4, panel_y + 4))

    # 60 FPS target line
    target_y = panel_y + panel_h - 4 - int((60.0 / max_fps) * graph_h)
    if graph_top <= target_y <= panel_y + panel_h:
        pygame.draw.line(
            screen, (60, 120, 60),
            (panel_x + 4, target_y),
            (panel_x + panel_w - 4, target_y),
            1,
        )

# ---------------------------------------------------------------------------
# Numba Math Kernels
# ---------------------------------------------------------------------------
@njit
def kernel(h: float, d: float) -> float:
    if d >= h: return 0.0
    q = 1.0 - (d / h) ** 2
    return q * q * q

@njit
def kernel_grad(h: float, d: float) -> float:
    if d >= h or d < 1e-6: return 0.0
    q = 1.0 - (d / h) ** 2
    return -6.0 * d / (h * h) * q * q

@njit
def kernel_laplacian(h: float, d: float) -> float:
    if d >= h: return 0.0
    h2 = h * h
    d2 = d * d
    q  = 1.0 - d2 / h2
    return (24.0 / h2) * q * (4.0 * d2 / h2 - q)

@njit
def calibrate_rest_density(h: float, r_part: float, mass: float) -> float:
    spacing = r_part * 2.2
    total   = 0.0
    r       = int(h / spacing) + 1
    for ix in range(-r, r + 1):
        for iy in range(-r, r + 1):
            d = math.hypot(ix * spacing, iy * spacing)
            total += mass * kernel(h, d)
    return max(total, 0.001)

# ---------------------------------------------------------------------------
# Numba SPH Passes
# ---------------------------------------------------------------------------
@njit
def build_grid(xs, ys, num_active, h, grid_cols, grid_rows, grid_head, grid_next):
    grid_head.fill(-1)
    grid_next[:num_active] = -1
    for i in range(num_active):
        cx = int(xs[i] // h)
        cy = int(ys[i] // h)
        if 0 <= cx < grid_cols and 0 <= cy < grid_rows:
            grid_next[i]       = grid_head[cx, cy]
            grid_head[cx, cy]  = i

@njit(parallel=True)
def compute_densities(xs, ys, densities, num_active, h, mass,
                      grid_cols, grid_rows, grid_head, grid_next):
    h2 = h * h
    for i in prange(num_active):
        rho = 0.0
        cx  = int(xs[i] // h)
        cy  = int(ys[i] // h)
        for dx in range(-1, 2):
            for dy in range(-1, 2):
                nx = cx + dx
                ny = cy + dy
                if 0 <= nx < grid_cols and 0 <= ny < grid_rows:
                    j = grid_head[nx, ny]
                    while j != -1:
                        dist2 = (xs[i] - xs[j])**2 + (ys[i] - ys[j])**2
                        if dist2 < h2:
                            rho += mass * kernel(h, math.sqrt(dist2))
                        j = grid_next[j]
        densities[i] = max(rho, 1e-4)

@njit(parallel=True)
def apply_pressure(xs, ys, axs, ays, densities, num_active, h, mass,
                   stiffness, rest_density,
                   grid_cols, grid_rows, grid_head, grid_next):
    h2 = h * h
    for i in prange(num_active):
        pres_i = max(stiffness * (densities[i] - rest_density), 0.0)
        cx = int(xs[i] // h)
        cy = int(ys[i] // h)
        for dx_ in range(-1, 2):
            for dy_ in range(-1, 2):
                nx = cx + dx_
                ny = cy + dy_
                if 0 <= nx < grid_cols and 0 <= ny < grid_rows:
                    j = grid_head[nx, ny]
                    while j != -1:
                        if i != j:
                            dx = xs[i] - xs[j]
                            dy = ys[i] - ys[j]
                            d2 = dx * dx + dy * dy
                            if 1e-10 < d2 < h2:
                                d      = math.sqrt(d2)
                                pres_j = max(stiffness * (densities[j] - rest_density), 0.0)
                                grad   = kernel_grad(h, d)
                                shared = (
                                    0.5 * mass * (pres_i + pres_j)
                                    / (2.0 * densities[j] + 1e-6) * grad
                                )
                                axs[i] -= shared * (dx / d)
                                ays[i] -= shared * (dy / d)
                        j = grid_next[j]

@njit(parallel=True)
def apply_viscosity(xs, ys, vxs, vys, densities, num_active, h, mass, viscosity,
                    grid_cols, grid_rows, grid_head, grid_next):
    h2 = h * h
    for i in prange(num_active):
        dvx, dvy = 0.0, 0.0
        cx = int(xs[i] // h)
        cy = int(ys[i] // h)
        for dx_ in range(-1, 2):
            for dy_ in range(-1, 2):
                nx = cx + dx_
                ny = cy + dy_
                if 0 <= nx < grid_cols and 0 <= ny < grid_rows:
                    j = grid_head[nx, ny]
                    while j != -1:
                        if i != j:
                            dx = xs[j] - xs[i]
                            dy = ys[j] - ys[i]
                            d2 = dx * dx + dy * dy
                            if d2 < h2:
                                d    = math.sqrt(d2)
                                coef = mass * kernel(h, d) / (densities[j] + 1e-6)
                                dvx += (vxs[j] - vxs[i]) * coef
                                dvy += (vys[j] - vys[i]) * coef
                        j = grid_next[j]
        vxs[i] += viscosity * dvx
        vys[i] += viscosity * dvy

@njit(parallel=True)
def apply_surface_tension(xs, ys, axs, ays, densities, num_active, h, mass,
                          surface_tension, cutoff,
                          grid_cols, grid_rows, grid_head, grid_next):
    h2        = h * h
    threshold = 0.3
    for i in prange(num_active):
        if densities[i] > cutoff:
            continue
        gcx, gcy, lap = 0.0, 0.0, 0.0
        cx = int(xs[i] // h)
        cy = int(ys[i] // h)
        for dx_ in range(-1, 2):
            for dy_ in range(-1, 2):
                nx = cx + dx_
                ny = cy + dy_
                if 0 <= nx < grid_cols and 0 <= ny < grid_rows:
                    j = grid_head[nx, ny]
                    while j != -1:
                        if i != j:
                            dx = xs[j] - xs[i]
                            dy = ys[j] - ys[i]
                            d2 = dx * dx + dy * dy
                            if 1e-6 < d2 < h2:
                                d    = math.sqrt(d2)
                                coef = mass / (densities[j] + 1e-6)
                                g    = kernel_grad(h, d)
                                gcx += coef * g * (dx / d)
                                gcy += coef * g * (dy / d)
                                lap += coef * kernel_laplacian(h, d)
                        j = grid_next[j]
        gm = math.hypot(gcx, gcy)
        if gm > threshold:
            axs[i] -= surface_tension * lap * gcx / gm
            ays[i] -= surface_tension * lap * gcy / gm

@njit(parallel=True)
def integrate(xs, ys, vxs, vys, axs, ays, num_active, gravity):
    for i in prange(num_active):
        vxs[i] += axs[i]
        vys[i] += ays[i]
        spd = math.hypot(vxs[i], vys[i])
        if spd > 25.0:
            f = 25.0 / spd
            vxs[i] *= f
            vys[i] *= f
        xs[i] += vxs[i]
        ys[i] += vys[i]
        axs[i] = 0.0
        ays[i] = gravity

@njit(parallel=True)
def resolve_overlaps(xs, ys, num_active, h, diam,
                     grid_cols, grid_rows, grid_head, grid_next):
    diam2 = diam * diam
    for i in prange(num_active):
        cx = int(xs[i] // h)
        cy = int(ys[i] // h)
        for dx_ in range(-1, 2):
            for dy_ in range(-1, 2):
                nx = cx + dx_
                ny = cy + dy_
                if 0 <= nx < grid_cols and 0 <= ny < grid_rows:
                    j = grid_head[nx, ny]
                    while j != -1:
                        if i != j:
                            dx = xs[i] - xs[j]
                            dy = ys[i] - ys[j]
                            d2 = dx * dx + dy * dy
                            if 1e-10 < d2 < diam2:
                                d       = math.sqrt(d2)
                                overlap = (diam - d) * 0.5
                                xs[i]  += (dx / d) * overlap
                                ys[i]  += (dy / d) * overlap
                        j = grid_next[j]

@njit(parallel=True)
def apply_mouse_push(xs, ys, vxs, vys, num_active,
                     mx, my, mouse_radius, mouse_strength):
    mr2 = mouse_radius * mouse_radius
    for i in prange(num_active):
        dx = xs[i] - mx
        dy = ys[i] - my
        d2 = dx * dx + dy * dy
        if 0 < d2 < mr2:
            d      = math.sqrt(d2)
            f      = (1.0 - d / mouse_radius) * mouse_strength
            vxs[i] += (dx / d) * f
            vys[i] += (dy / d) * f

@njit(parallel=True)
def resolve_flask(xs, ys, vxs, vys, num_active, r, wf, res,
                  f_left, f_right, f_top, f_bot):
    for i in prange(num_active):
        if xs[i] < f_left + r:
            xs[i] = f_left + r
            if vxs[i] < 0: vxs[i] = -vxs[i] * res
            vys[i] *= (1.0 - wf)
        if xs[i] > f_right - r:
            xs[i] = f_right - r
            if vxs[i] > 0: vxs[i] = -vxs[i] * res
            vys[i] *= (1.0 - wf)
        if ys[i] > f_bot - r:
            ys[i] = f_bot - r
            if vys[i] > 0: vys[i] = -vys[i] * res
            vxs[i] *= (1.0 - wf)
        if ys[i] < r:
            ys[i] = r
            if vys[i] < 0: vys[i] = -vys[i] * res

@njit(parallel=True)
def resolve_obstacles(xs, ys, vxs, vys, num_active, r,
                      obs_xs, obs_ys, obs_ws, obs_hs, obs_active, res, wf):
    for i in prange(num_active):
        px = xs[i]
        py = ys[i]
        for o in range(len(obs_active)):
            if obs_active[o] == 1:
                ox = obs_xs[o]
                oy = obs_ys[o]
                ow = obs_ws[o]
                oh = obs_hs[o]
                if px > ox - r and px < ox + ow + r and py > oy - r and py < oy + oh + r:
                    dist_left   = px - (ox - r)
                    dist_right  = (ox + ow + r) - px
                    dist_top    = py - (oy - r)
                    dist_bottom = (oy + oh + r) - py
                    min_dist    = min(dist_left, dist_right, dist_top, dist_bottom)
                    if min_dist == dist_left:
                        xs[i] = ox - r
                        if vxs[i] > 0: vxs[i] = -vxs[i] * res
                        vys[i] *= (1.0 - wf)
                    elif min_dist == dist_right:
                        xs[i] = ox + ow + r
                        if vxs[i] < 0: vxs[i] = -vxs[i] * res
                        vys[i] *= (1.0 - wf)
                    elif min_dist == dist_top:
                        ys[i] = oy - r
                        if vys[i] > 0: vys[i] = -vys[i] * res
                        vxs[i] *= (1.0 - wf)
                    elif min_dist == dist_bottom:
                        ys[i] = oy + oh + r
                        if vys[i] < 0: vys[i] = -vys[i] * res
                        vxs[i] *= (1.0 - wf)

# ---------------------------------------------------------------------------
# Numba Metaball Renderer
# ---------------------------------------------------------------------------
@njit(parallel=True)
def compute_metaballs(xs, ys, vxs, vys, num_active, lw, lh, scale,
                      r_render, threshold, f_left, f_right, f_bot,
                      obs_xs, obs_ys, obs_ws, obs_hs, obs_active):
    grid    = np.zeros((lw, lh), dtype=np.float32)
    vx_grid = np.zeros((lw, lh), dtype=np.float32)
    vy_grid = np.zeros((lw, lh), dtype=np.float32)
    r_sq    = r_render * r_render

    for i in range(num_active):
        px = xs[i] / scale
        py = ys[i] / scale
        min_x = max(0,      int(px - r_render))
        max_x = min(lw - 1, int(px + r_render))
        min_y = max(0,      int(py - r_render))
        max_y = min(lh - 1, int(py + r_render))
        for x in range(min_x, max_x + 1):
            for y in range(min_y, max_y + 1):
                dx = px - x
                dy = py - y
                d2 = dx * dx + dy * dy
                if d2 < r_sq:
                    w          = (1.0 - d2 / r_sq) ** 2
                    grid[x, y]    += w
                    vx_grid[x, y] += vxs[i] * w
                    vy_grid[x, y] += vys[i] * w

    pixels = np.zeros((lw, lh, 3), dtype=np.uint8)
    for x in prange(lw):
        for y in range(lh):
            sx = x * scale
            sy = y * scale
            inside_flask = sx >= f_left and sx <= f_right and sy <= f_bot
            inside_box   = False
            for o in range(len(obs_active)):
                if obs_active[o] == 1:
                    if (sx > obs_xs[o] and sx < obs_xs[o] + obs_ws[o]
                            and sy > obs_ys[o] and sy < obs_ys[o] + obs_hs[o]):
                        inside_box = True
                        break
            if not inside_flask or inside_box:
                pixels[x, y, 0] = 8
                pixels[x, y, 1] = 12
                pixels[x, y, 2] = 24
                continue
            val = grid[x, y]
            if val > threshold:
                vx  = vx_grid[x, y] / val
                vy  = vy_grid[x, y] / val
                spd = math.hypot(vx, vy)
                t   = min(spd / 12.0, 1.0)
                if t < 0.5:
                    s = t * 2.0
                    r = int(10 + 20  * s)
                    g = int(40 + 80  * s)
                    b = int(120 + 100 * s)
                else:
                    s = (t - 0.5) * 2.0
                    r = int(30  + 170 * s)
                    g = int(120 + 110 * s)
                    b = int(220 + 35  * s)
                if val < threshold + 0.2:
                    r = min(255, r + 40)
                    g = min(255, g + 50)
                    b = min(255, b + 60)
                pixels[x, y, 0] = r
                pixels[x, y, 1] = g
                pixels[x, y, 2] = b
            else:
                pixels[x, y, 0] = 8
                pixels[x, y, 1] = 12
                pixels[x, y, 2] = 24
    return pixels

# ---------------------------------------------------------------------------
# Graphics Helpers
# ---------------------------------------------------------------------------
def draw_ui_overlay(screen, font, title_font):
    overlay = pygame.Surface((SIM_WIDTH, HEIGHT), pygame.SRCALPHA)
    overlay.fill((0, 0, 0, 180))
    title = title_font.render("SPH FLUID PLAYGROUND", True, (255, 255, 255))
    overlay.blit(title, (SIM_WIDTH // 2 - title.get_width() // 2, 80))
    instructions = [
        "--- Controls ---",
        "[LMB Click + Hold]  Drag a Block OR Push the water",
        "         (Use the UI panel on the right!)",
        "[ C ]  Spawn a new Block at your mouse cursor",
        "[ X ]  Delete the Block under your mouse cursor",
        "[ M ]  Toggle Metaball rendering (Solid vs Particles)",
        "[SPACE]  Pause / Resume pouring water",
        "[ R ]  Reset Simulation",
        "[ H ]  Hide or Show this Help menu",
        "",
        "Press [H] to close this menu and play!",
    ]
    for i, line in enumerate(instructions):
        color = (255, 215, 0) if "Press [H]" in line else (200, 220, 255)
        text  = font.render(line, True, color)
        overlay.blit(text, (SIM_WIDTH // 2 - text.get_width() // 2, 160 + i * 25))
    screen.blit(overlay, (0, 0))

def draw_flask(surf):
    glow = (30,  70, 150)
    wall = (100, 150, 220)
    tg, tw = 6, 2
    pygame.draw.line(surf, glow, (F_LEFT,  F_BOT), (F_RIGHT, F_BOT), tg)
    pygame.draw.line(surf, wall, (F_LEFT,  F_BOT), (F_RIGHT, F_BOT), tw)
    pygame.draw.line(surf, glow, (F_LEFT,  F_TOP), (F_LEFT,  F_BOT), tg)
    pygame.draw.line(surf, wall, (F_LEFT,  F_TOP), (F_LEFT,  F_BOT), tw)
    pygame.draw.line(surf, glow, (F_RIGHT, F_TOP), (F_RIGHT, F_BOT), tg)
    pygame.draw.line(surf, wall, (F_RIGHT, F_TOP), (F_RIGHT, F_BOT), tw)

def draw_obstacles(surf, obs_xs, obs_ys, obs_ws, obs_hs, obs_active):
    for i in range(len(obs_active)):
        if obs_active[i] == 1:
            rect = pygame.Rect(int(obs_xs[i]), int(obs_ys[i]),
                               int(obs_ws[i]), int(obs_hs[i]))
            pygame.draw.rect(surf, (150, 150, 150), rect)
            pygame.draw.rect(surf, (200, 200, 200), rect, 2)

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("SPH Fluid — Phase 14: Dynamic Memory & FPS Graph")
    clock      = pygame.time.Clock()
    font       = pygame.font.SysFont("monospace", 14, bold=True)
    title_font = pygame.font.SysFont("monospace", 28, bold=True)
    small_font = pygame.font.SysFont("monospace", 12)

    # ------------------------------------------------------------------
    # Particle arrays — start at INITIAL_CAPACITY, grow as needed
    # ------------------------------------------------------------------
    current_capacity = INITIAL_CAPACITY
    (xs, ys, vxs, vys, axs, ays,
     densities, grid_next) = make_particle_arrays(current_capacity)

    grid_head = np.full((MAX_GRID_COLS, MAX_GRID_ROWS), -1, dtype=np.int32)

    num_active    = 0
    pouring       = True
    use_metaballs = False
    show_help     = True

    # ------------------------------------------------------------------
    # Rest-density cache  (pre-flight fix: no longer computed every frame)
    # ------------------------------------------------------------------
    last_radius  = -1.0
    rest_density = 1.0

    # ------------------------------------------------------------------
    # FPS history ring buffer  (Phase 14)
    # ------------------------------------------------------------------
    fps_history = deque(maxlen=FPS_HISTORY_LEN)

    # ------------------------------------------------------------------
    # UI Widgets
    # ------------------------------------------------------------------
    btn_basic = Button(SIM_WIDTH + 20,  20, 100, 30, "Basic")
    btn_adv   = Button(SIM_WIDTH + 130, 20, 100, 30, "Advanced")
    btn_basic.active = True

    # Basic tab — slider_count ceiling is now SLIDER_MAX, not the array size
    slider_size  = Slider(SIM_WIDTH + 20, 100, 210, 15,
                          1.0, 6.0, 2.0, "Particle Size")
    slider_count = Slider(SIM_WIDTH + 20, 170, 210, 15,
                          100, SLIDER_MAX, 6000, "Max Particles", is_int=True)
    basic_sliders = [slider_size, slider_count]

    # Advanced tab
    slider_gravity = Slider(SIM_WIDTH + 20, 100, 210, 15, -1.0, 1.0,  0.35, "Gravity",      decimals=2)
    slider_stiff   = Slider(SIM_WIDTH + 20, 170, 210, 15,  0.1, 5.0,  1.0,  "Stiffness",    decimals=1)
    slider_visc    = Slider(SIM_WIDTH + 20, 240, 210, 15,  0.0, 0.2,  0.05, "Viscosity",    decimals=3)
    slider_surf    = Slider(SIM_WIDTH + 20, 310, 210, 15,  0.0, 0.1,  0.02, "Surf Tension", decimals=3)
    slider_rest    = Slider(SIM_WIDTH + 20, 380, 210, 15,  0.0, 1.0,  0.3,  "Restitution",  decimals=2)
    adv_sliders    = [slider_gravity, slider_stiff, slider_visc, slider_surf, slider_rest]

    # ------------------------------------------------------------------
    # Obstacles
    # ------------------------------------------------------------------
    MAX_OBS    = 10
    obs_xs     = np.zeros(MAX_OBS, dtype=np.float32)
    obs_ys     = np.zeros(MAX_OBS, dtype=np.float32)
    obs_ws     = np.zeros(MAX_OBS, dtype=np.float32)
    obs_hs     = np.zeros(MAX_OBS, dtype=np.float32)
    obs_active = np.zeros(MAX_OBS, dtype=np.int32)

    obs_ws[0], obs_hs[0] = 80.0, 80.0
    obs_xs[0]  = SIM_WIDTH // 2 - 40.0
    obs_ys[0]  = HEIGHT - 130.0
    obs_active[0] = 1

    dragging_idx            = -1
    pushing_water           = False
    drag_offset_x           = 0.0
    drag_offset_y           = 0.0

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------
    running = True
    while running:
        ui_interacted  = False
        active_sliders = basic_sliders if btn_basic.active else adv_sliders

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            for s in active_sliders:
                if s.handle_event(event):
                    ui_interacted = True

            if btn_basic.handle_event(event):
                btn_basic.active = True;  btn_adv.active = False;  ui_interacted = True
            if btn_adv.handle_event(event):
                btn_adv.active   = True;  btn_basic.active = False; ui_interacted = True

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_h:
                    show_help = not show_help
                elif event.key == pygame.K_r:
                    num_active = 0
                    pouring    = True
                elif event.key == pygame.K_SPACE:
                    pouring = not pouring
                elif event.key == pygame.K_m:
                    use_metaballs = not use_metaballs
                elif event.key == pygame.K_c:
                    mx, my = pygame.mouse.get_pos()
                    if mx < SIM_WIDTH:
                        for i in range(MAX_OBS):
                            if obs_active[i] == 0:
                                obs_ws[i], obs_hs[i] = 60.0, 60.0
                                obs_xs[i] = float(mx - 30.0)
                                obs_ys[i] = float(my - 30.0)
                                obs_active[i] = 1
                                break
                elif event.key == pygame.K_x:
                    mx, my = pygame.mouse.get_pos()
                    for i in range(MAX_OBS):
                        if obs_active[i] == 1:
                            if (obs_xs[i] <= mx <= obs_xs[i] + obs_ws[i]
                                    and obs_ys[i] <= my <= obs_ys[i] + obs_hs[i]):
                                obs_active[i] = 0
                                break

            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1 and not ui_interacted:
                mx, my = pygame.mouse.get_pos()
                if mx < SIM_WIDTH:
                    clicked_block = False
                    for i in range(MAX_OBS):
                        if obs_active[i] == 1:
                            if (obs_xs[i] <= mx <= obs_xs[i] + obs_ws[i]
                                    and obs_ys[i] <= my <= obs_ys[i] + obs_hs[i]):
                                dragging_idx  = i
                                drag_offset_x = mx - obs_xs[i]
                                drag_offset_y = my - obs_ys[i]
                                clicked_block = True
                                break
                    if not clicked_block:
                        pushing_water = True

            if event.type == pygame.MOUSEBUTTONUP and event.button == 1:
                dragging_idx  = -1
                pushing_water = False

        # ------------------------------------------------------------------
        # Read slider values
        # ------------------------------------------------------------------
        particle_radius      = slider_size.val
        current_max_particles = int(slider_count.val)

        gravity_val      = slider_gravity.val
        stiffness_val    = slider_stiff.val
        viscosity_val    = slider_visc.val
        surf_tension_val = slider_surf.val
        restitution_val  = slider_rest.val

        # Slider dragged down: silently discard particles above the new limit.
        # NOTE: this is intentionally lossy — dragging back up does NOT restore
        # them. The data beyond num_active is stale and must not be trusted.
        if num_active > current_max_particles:
            num_active = current_max_particles

        smoothing_radius = particle_radius * 5.0
        mass             = particle_radius ** 2
        render_radius    = particle_radius * 2.0

        # Recompute rest density only when the size slider actually changes.
        if particle_radius != last_radius:
            rest_density = calibrate_rest_density(smoothing_radius, particle_radius, mass)
            last_radius  = particle_radius

        grid_cols = math.ceil(SIM_WIDTH / smoothing_radius) + 2
        grid_rows = math.ceil(HEIGHT    / smoothing_radius) + 2

        # ------------------------------------------------------------------
        # Dynamic resize  (Phase 14) — grow before the pour loop so we never
        # index beyond the allocated length.
        # ------------------------------------------------------------------
        (current_capacity,
         xs, ys, vxs, vys, axs, ays,
         densities, grid_next) = maybe_grow_arrays(
            num_active, current_capacity,
            xs, ys, vxs, vys, axs, ays, densities, grid_next,
        )

        # ------------------------------------------------------------------
        mx, my = pygame.mouse.get_pos()

        if dragging_idx != -1:
            obs_xs[dragging_idx] = float(mx - drag_offset_x)
            obs_ys[dragging_idx] = float(my - drag_offset_y)

        if pouring and num_active < current_max_particles and not show_help:
            for _ in range(POUR_RATE):
                if num_active >= current_max_particles:
                    break
                xs[num_active]        = POUR_X + random.uniform(-POUR_SPREAD, POUR_SPREAD)
                ys[num_active]        = POUR_Y + random.uniform(-POUR_SPREAD, POUR_SPREAD)
                vxs[num_active]       = POUR_VX + random.uniform(-0.3, 0.3)
                vys[num_active]       = POUR_VY + random.uniform(-0.2, 0.2)
                axs[num_active]       = 0.0
                # ays intentionally NOT set here — integrate() resets it anyway.
                densities[num_active] = rest_density
                num_active           += 1

        if pushing_water and num_active > 0 and not show_help:
            apply_mouse_push(xs, ys, vxs, vys, num_active,
                             mx, my, MOUSE_RADIUS, MOUSE_STRENGTH)

        # ------------------------------------------------------------------
        # Physics passes
        # ------------------------------------------------------------------
        if num_active > 0 and not show_help:
            build_grid(xs, ys, num_active, smoothing_radius,
                       grid_cols, grid_rows, grid_head, grid_next)
            compute_densities(xs, ys, densities, num_active, smoothing_radius, mass,
                              grid_cols, grid_rows, grid_head, grid_next)
            apply_pressure(xs, ys, axs, ays, densities, num_active, smoothing_radius, mass,
                           stiffness_val, rest_density,
                           grid_cols, grid_rows, grid_head, grid_next)
            apply_viscosity(xs, ys, vxs, vys, densities, num_active, smoothing_radius, mass,
                            viscosity_val, grid_cols, grid_rows, grid_head, grid_next)
            apply_surface_tension(xs, ys, axs, ays, densities, num_active, smoothing_radius, mass,
                                  surf_tension_val, rest_density * INTERIOR_DENSITY_RATIO,
                                  grid_cols, grid_rows, grid_head, grid_next)
            integrate(xs, ys, vxs, vys, axs, ays, num_active, gravity_val)

            resolve_obstacles(xs, ys, vxs, vys, num_active, particle_radius,
                              obs_xs, obs_ys, obs_ws, obs_hs, obs_active,
                              restitution_val, WALL_FRICTION)
            resolve_overlaps(xs, ys, num_active, smoothing_radius, particle_radius * 2.0,
                             grid_cols, grid_rows, grid_head, grid_next)
            resolve_obstacles(xs, ys, vxs, vys, num_active, particle_radius,
                              obs_xs, obs_ys, obs_ws, obs_hs, obs_active,
                              restitution_val, WALL_FRICTION)
            resolve_flask(xs, ys, vxs, vys, num_active, particle_radius,
                          WALL_FRICTION, restitution_val,
                          F_LEFT, F_RIGHT, F_TOP, F_BOT)

        # ------------------------------------------------------------------
        # Render — simulation area
        # ------------------------------------------------------------------
        screen.fill(BACKGROUND)

        if use_metaballs and num_active > 0:
            lw     = SIM_WIDTH // RENDER_SCALE
            lh     = HEIGHT    // RENDER_SCALE
            pixels = compute_metaballs(
                xs, ys, vxs, vys, num_active, lw, lh, RENDER_SCALE,
                render_radius, RENDER_THRESHOLD,
                F_LEFT, F_RIGHT, F_BOT,
                obs_xs, obs_ys, obs_ws, obs_hs, obs_active,
            )
            surf = pygame.surfarray.make_surface(pixels)
            surf = pygame.transform.smoothscale(surf, (SIM_WIDTH, HEIGHT))
            screen.blit(surf, (0, 0))
        else:
            for i in range(num_active):
                spd   = math.hypot(vxs[i], vys[i])
                color = speed_to_color(spd)
                pygame.draw.circle(screen, color,
                                   (int(xs[i]), int(ys[i])), int(particle_radius))

        draw_flask(screen)
        draw_obstacles(screen, obs_xs, obs_ys, obs_ws, obs_hs, obs_active)

        if pushing_water and not show_help:
            pygame.draw.circle(screen, (255, 255, 255, 100),
                               (mx, my), int(MOUSE_RADIUS), 1)

        hud = small_font.render(
            f"FPS: {clock.get_fps():.0f}  |  "
            f"{num_active}/{current_max_particles} particles  "
            f"(cap {current_capacity})  |  [H] Help",
            True, (150, 180, 220),
        )
        screen.blit(hud, (10, 10))

        if show_help:
            draw_ui_overlay(screen, font, title_font)

        # ------------------------------------------------------------------
        # Render — UI panel (right side)
        # ------------------------------------------------------------------
        pygame.draw.rect(screen, (22, 28, 40), (SIM_WIDTH, 0, UI_WIDTH, HEIGHT))
        pygame.draw.line(screen, (50, 60, 80), (SIM_WIDTH, 0), (SIM_WIDTH, HEIGHT), 2)

        btn_basic.draw(screen, font)
        btn_adv.draw(screen, font)

        for s in active_sliders:
            s.draw(screen, font)

        hint_text = "Auto-Scaling ensures stability." if btn_basic.active else "Raw physics variables."
        info = small_font.render(hint_text, True, (100, 150, 200))
        screen.blit(info, (SIM_WIDTH + 15, HEIGHT - 110))

        # FPS graph — sits at the very bottom of the panel
        fps_history.append(clock.get_fps())
        draw_fps_graph(
            screen, small_font, fps_history,
            panel_x = SIM_WIDTH + 10,
            panel_y = HEIGHT - 95,
            panel_w = UI_WIDTH - 20,
            panel_h = 88,
        )

        pygame.display.flip()
        clock.tick(FPS)

    pygame.quit()


if __name__ == "__main__":
    main()