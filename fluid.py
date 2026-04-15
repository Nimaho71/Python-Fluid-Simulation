"""SPH Fluid Simulation — interactive 2D fluid sandbox.

This module implements a real-time Smoothed Particle Hydrodynamics (SPH)
fluid simulation using Pygame for rendering and Numba for JIT-compiled
physics kernels.

Quick start::

    pip install pygame numpy numba
    python fluid.py

Controls:
    H          Toggle this help overlay
    Space      Pause / resume the particle stream
    R          Reset the simulation
    M          Toggle metaball (liquid blob) rendering
    C          Spawn a new obstacle block at the cursor
    X          Delete the obstacle block under the cursor
    LMB drag   Drag an obstacle block, or push the fluid

Architecture:
    Constants and config  — top of module, grouped by concern
    SimState dataclass    — all mutable runtime state in one place
    Numba kernels         — @njit physics passes (build_grid → densities →
                            pressure → viscosity → surface_tension →
                            integrate → resolve)
    UI helpers            — Slider, Button, draw_fps_graph
    Render helpers        — draw_flask, draw_obstacles, draw_help_overlay
    main()                — thin orchestrator: init → loop(events, physics,
                            render) → quit

SPH notation used throughout the kernels:
    h   smoothing radius (neighbourhood size)
    d   distance between two particles
    W   kernel weight  (how much j influences i at distance d)
    rho density
    p   pressure
    m   particle mass
"""

from __future__ import annotations

import math
import random
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Tuple

import numpy as np
import pygame
from numba import njit, prange


# ---------------------------------------------------------------------------
# Display constants
# ---------------------------------------------------------------------------

SIM_WIDTH: int = 800    # Width of the simulation viewport in pixels.
UI_WIDTH: int  = 250    # Width of the right-hand control panel in pixels.
WIDTH: int     = SIM_WIDTH + UI_WIDTH
HEIGHT: int    = 600
TARGET_FPS: int = 60
BACKGROUND_COLOR: Tuple[int, int, int] = (8, 12, 24)
FPS_HISTORY_LEN = 120


# ---------------------------------------------------------------------------
# Particle array capacities
# ---------------------------------------------------------------------------

# Starting array size. Arrays double automatically when full (see
# maybe_grow_arrays), so this is NOT a hard cap — just the initial allocation.
INITIAL_CAPACITY: int = 10_000

# Maximum value the "Max Particles" slider can reach.
SLIDER_MAX: int = 30_000

# Grid cell count is bounded by the smallest possible smoothing radius, not
# by particle count. Pre-allocating at the worst case avoids runtime resizing
# of the grid arrays, which are small (floats * grid_cols * grid_rows).
_MIN_SMOOTHING_RADIUS: float = 5.0
MAX_GRID_COLS: int = math.ceil(SIM_WIDTH / _MIN_SMOOTHING_RADIUS) + 2
MAX_GRID_ROWS: int = math.ceil(HEIGHT    / _MIN_SMOOTHING_RADIUS) + 2


# ---------------------------------------------------------------------------
# Physics defaults  (all overridable via the Advanced tab sliders at runtime)
# ---------------------------------------------------------------------------

DEFAULT_GRAVITY: float      = 0.20  # Pixels-per-frame^2 downward acceleration.
DEFAULT_STIFFNESS: float    = 0.80  # Pressure equation stiffness constant k.
DEFAULT_VISCOSITY: float    = 0.20  # Velocity-smoothing coefficient mu.
DEFAULT_SURF_TENSION: float = 0.10  # Surface cohesion force coefficient sigma.
DEFAULT_RESTITUTION: float  = 0.30  # Wall/obstacle bounce factor (0=none, 1=full).

# Wall friction damps the velocity component parallel to the wall on impact.
# Not exposed in the UI; tweak here if you want stickier or slipperier walls.
WALL_FRICTION: float = 0.05

# Surface tension only acts on particles whose density is below this fraction
# of the rest density. Higher values extend cohesion deeper into the fluid.
SURFACE_CUTOFF_RATIO: float = 0.85

# Velocity is clamped to this limit (pixels/frame) to prevent tunnelling
# through thin walls at very high particle counts or stiffness values.
MAX_PARTICLE_SPEED: float = 25.0


# ---------------------------------------------------------------------------
# Pour stream  — where new particles enter the simulation
# ---------------------------------------------------------------------------

POUR_X: float      = 180.0  # Horizontal origin of the stream (pixels).
POUR_Y: float      = 80.0   # Vertical origin of the stream (pixels).
POUR_VX: float     = 10.0   # Initial horizontal velocity (pixels/frame).
POUR_VY: float     = 2.0    # Initial vertical velocity (pixels/frame).
POUR_RATE: int     = 8      # Particles added per frame while pouring.
POUR_SPREAD: float = 3.0    # +/- pixels of random jitter applied to spawn pos.


# ---------------------------------------------------------------------------
# Mouse interaction
# ---------------------------------------------------------------------------

MOUSE_PUSH_RADIUS: float   = 80.0  # Influence radius of the push cursor (px).
MOUSE_PUSH_STRENGTH: float = 3.0   # Maximum impulse applied at the cursor.


# ---------------------------------------------------------------------------
# Metaball renderer
# ---------------------------------------------------------------------------

# Render at 1/METABALL_SCALE resolution then upscale, trading sharpness for
# speed. Value of 2 means physics is at full res, rendering at half res.
METABALL_SCALE: int       = 2
METABALL_THRESHOLD: float = 0.5  # Field value at which a pixel is "inside".

# Speed value (pixels/frame) that maps to fully saturated colour in the
# metaball renderer. Particles faster than this render at maximum saturation.
METABALL_COLOR_SPEED: float = 12.0


# ---------------------------------------------------------------------------
# Colour (particle heatmap mode)
# ---------------------------------------------------------------------------

# Speed at which the heatmap colour reaches its maximum (orange-red).
HEATMAP_MAX_SPEED: float = 14.0


# ---------------------------------------------------------------------------
# Flask (container) geometry
# ---------------------------------------------------------------------------

def _build_flask() -> Tuple[float, float, float, float]:
    """Return the (left, right, top, bottom) bounds of the container in pixels.

    The flask is centred horizontally in the simulation viewport and
    positioned slightly below the vertical midpoint to leave room for the
    pour stream above.

    Returns:
        A 4-tuple (left_x, right_x, top_y, bottom_y).
    """
    cx = SIM_WIDTH // 2
    cy = HEIGHT // 2 + 40
    half_width  = 260
    half_height = 200
    return (
        float(cx - half_width),
        float(cx + half_width),
        float(cy - half_height),
        float(cy + half_height),
    )


_FLASK_BOUNDS = _build_flask()
F_LEFT, F_RIGHT, F_TOP, F_BOT = _FLASK_BOUNDS


# ---------------------------------------------------------------------------
# Simulation state dataclass
# ---------------------------------------------------------------------------

@dataclass
class SimState:
    """All mutable runtime state for one simulation session.

    Grouping state here keeps main() readable and makes it easy to
    implement a full reset (just replace the instance).

    Attributes:
        capacity:            Current length of the particle arrays.
        num_active:          Number of particles currently alive.
        xs, ys:              Particle positions (pixels).
        vxs, vys:            Particle velocities (pixels per frame).
        axs, ays:            Particle accelerations accumulated this frame.
        densities:           SPH density estimate for each particle.
        grid_next:           Linked-list next-index for the spatial grid.
        grid_head:           Head-of-list for each grid cell.
        pouring:             True while the particle stream is active.
        use_metaballs:       True to use the blob renderer instead of circles.
        show_help:           True to show the help overlay.
        last_particle_radius: Cached radius used to skip rest-density
                             recalculation when the size slider has not moved.
        rest_density:        Cached result of calibrate_rest_density().
        fps_history:         Ring buffer of recent FPS samples for the graph.
        obs_xs, obs_ys:      Obstacle top-left corners (pixels).
        obs_ws, obs_hs:      Obstacle dimensions (pixels).
        obs_active:          1 if the obstacle slot is in use, else 0.
        dragging_idx:        Index of the obstacle being dragged, or -1.
        pushing_water:       True while the user is pushing fluid with LMB.
        drag_offset_x/y:     Cursor-to-obstacle-corner offset during a drag.
    """

    # Particle arrays
    capacity:   int
    num_active: int
    xs:         np.ndarray
    ys:         np.ndarray
    vxs:        np.ndarray
    vys:        np.ndarray
    axs:        np.ndarray
    ays:        np.ndarray
    densities:  np.ndarray
    grid_next:  np.ndarray
    grid_head:  np.ndarray

    # Simulation mode flags
    pouring: bool = True
    sim_paused: bool = False
    step_one_frame: bool = False
    use_metaballs: bool = False
    show_help: bool = True

    # Rest-density cache — recomputed only when particle_radius changes.
    last_particle_radius: float = -1.0
    rest_density:         float = 1.0

    # Performance history for the FPS graph (2 s at 60 fps).
    fps_history: Deque[float] = field(
        default_factory=lambda: deque(maxlen=FPS_HISTORY_LEN)
    )

    # Obstacles — fixed-size arrays; unused slots have obs_active[i] == 0.
    obs_xs:     np.ndarray = field(default_factory=lambda: np.zeros(10, dtype=np.float32))
    obs_ys:     np.ndarray = field(default_factory=lambda: np.zeros(10, dtype=np.float32))
    obs_ws:     np.ndarray = field(default_factory=lambda: np.zeros(10, dtype=np.float32))
    obs_hs:     np.ndarray = field(default_factory=lambda: np.zeros(10, dtype=np.float32))
    obs_active: np.ndarray = field(default_factory=lambda: np.zeros(10, dtype=np.int32))

    # Mouse interaction state
    dragging_idx: int = -1
    pushing_water: bool = False
    pulling_water: bool = False
    drag_offset_x: float = 0.0
    drag_offset_y: float = 0.0

    # Values written by _step_physics and read by _render_frame.
    _particle_radius: float = 2.0
    _render_radius:   float = 4.0
    _current_max:     int   = 6000


# ---------------------------------------------------------------------------
# Particle array helpers
# ---------------------------------------------------------------------------

def make_particle_arrays(capacity: int) -> tuple:
    """Allocate a fresh set of per-particle numpy arrays.

    All arrays use float32 to maximise Numba vectorisation and cache
    efficiency. grid_next is initialised to -1 (end-of-list sentinel).

    Args:
        capacity: Number of particle slots to allocate.

    Returns:
        An 8-tuple: (xs, ys, vxs, vys, axs, ays, densities, grid_next).
    """
    zeros = lambda: np.zeros(capacity, dtype=np.float32)
    return (
        zeros(),                                    # xs
        zeros(),                                    # ys
        zeros(),                                    # vxs
        zeros(),                                    # vys
        zeros(),                                    # axs
        zeros(),                                    # ays
        zeros(),                                    # densities
        np.full(capacity, -1, dtype=np.int32),      # grid_next
    )


def maybe_grow_arrays(state: SimState) -> None:
    """Double every particle array in-place if capacity is nearly exhausted.

    Growth is triggered when num_active is within one pour-burst of the
    array boundary. Without this buffer the pour loop would write past the
    end of the arrays before the next frame could resize them.

    Doubling strategy amortises allocation cost: at most O(log n) doublings
    occur over the lifetime of the simulation regardless of final count.

    Mutates state directly — no return value.
    """
    if state.num_active + POUR_RATE + 1 < state.capacity:
        return  # Enough room for at least one more full burst.

    new_capacity = state.capacity * 2
    print(f"[SPH] Growing particle arrays: {state.capacity:,} -> {new_capacity:,}")

    def _grow(arr: np.ndarray, fill: int = 0) -> np.ndarray:
        """Return a new array of new_capacity with live data copied in."""
        grown = np.full(new_capacity, fill, dtype=arr.dtype)
        grown[:state.capacity] = arr[:state.capacity]
        return grown

    state.xs        = _grow(state.xs)
    state.ys        = _grow(state.ys)
    state.vxs       = _grow(state.vxs)
    state.vys       = _grow(state.vys)
    state.axs       = _grow(state.axs)
    state.ays       = _grow(state.ays)
    state.densities = _grow(state.densities)
    state.grid_next = _grow(state.grid_next, fill=-1)
    state.capacity  = new_capacity


# ---------------------------------------------------------------------------
# Colour helpers
# ---------------------------------------------------------------------------

def speed_to_heatmap_color(speed: float) -> Tuple[int, int, int]:
    """Map a particle speed to an RGB colour on a blue to orange ramp.

    Slow particles are cool blue; fast particles are hot orange-red.

    Args:
        speed: Particle speed in pixels per frame.

    Returns:
        An (R, G, B) tuple with each component in [0, 255].
    """
    t = min(speed / HEATMAP_MAX_SPEED, 1.0)
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
# UI: Slider
# ---------------------------------------------------------------------------

class Slider:
    """A horizontal drag slider for the control panel.

    Supports both continuous float values and integer (snapped) values.
    The label and current value are rendered directly above the track.
    Clicking the slider focuses it, allowing keyboard entry.

    Attributes:
        rect:       Bounding rectangle of the slider track.
        min_val:    Minimum selectable value.
        max_val:    Maximum selectable value.
        val:        Current value.
        text:       Label shown above the slider.
        is_int:     If True, the value is rounded to the nearest integer.
        decimals:   Decimal places shown when is_int is False.
        focused:    True if the user is typing a custom value.
        input_text: Temporary string holding the typed value.
    """

    def __init__(
        self,
        x: int, y: int, w: int, h: int,
        min_val: float, max_val: float, initial_val: float,
        text: str,
        is_int: bool = False,
        decimals: int = 1,
    ) -> None:
        self.rect       = pygame.Rect(x, y, w, h)
        self.min_val    = min_val
        self.max_val    = max_val
        self.val        = initial_val
        self.text       = text
        self.is_int     = is_int
        self.decimals   = decimals
        self.dragging   = False
        self.focused    = False
        self.input_text = ""

    def handle_event(self, event: pygame.event.Event) -> bool:
        """Update the slider value from a mouse event or keyboard input."""
        
        # Create a hitbox that covers both the track and the label text above it
        hitbox = pygame.Rect(self.rect.x, self.rect.y - 25, self.rect.width, self.rect.height + 25)

        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if hitbox.collidepoint(event.pos):
                # If they clicked the physical track/handle, it's a DRAG
                if self.rect.collidepoint(event.pos):
                    self.dragging = True
                    self.focused = False # Prevent text input while dragging!
                # If they clicked the text label above the track, it's a TYPE
                else:
                    if not self.focused:
                        self.focused = True
                        self.input_text = ""
                return True
            else:
                # Clicked outside the slider entirely: apply value and unfocus
                if self.focused:
                    self._apply_input()
                    self.focused = False
                return False

        elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
            if self.dragging:
                self.dragging = False
                return True

        elif event.type == pygame.MOUSEMOTION and self.dragging:
            relative_x = max(0, min(event.pos[0] - self.rect.x, self.rect.width))
            fraction   = relative_x / self.rect.width
            self.val   = self.min_val + fraction * (self.max_val - self.min_val)
            if self.is_int:
                self.val = int(self.val)
            return True

        elif event.type == pygame.KEYDOWN and self.focused:
            if event.key in (pygame.K_RETURN, pygame.K_KP_ENTER):
                self._apply_input()
                self.focused = False
            elif event.key == pygame.K_ESCAPE:
                self.focused = False # Cancels input, reverts to previous slider val
            elif event.key == pygame.K_BACKSPACE:
                self.input_text = self.input_text[:-1]
            else:
                # Accept digits, period, and minus sign
                char = event.unicode
                if char in "0123456789.-":
                    self.input_text += char
            return True

        return False

    def _apply_input(self) -> None:
        """Parse the typed input text, clamp it, and update the slider value."""
        if not self.input_text:
            return
        try:
            new_val = float(self.input_text)
            if self.is_int:
                new_val = int(new_val)
            # Clamp the value between min_val and max_val safely!
            self.val = max(self.min_val, min(new_val, self.max_val))
        except ValueError:
            pass  # Invalid text (e.g., "-", "."), just ignore it.

    def draw(self, screen: pygame.Surface, font: pygame.font.Font) -> None:
        """Render the slider track, handle, and label to screen."""
        pygame.draw.rect(screen, (40, 50, 70), self.rect, border_radius=4)

        t        = (self.val - self.min_val) / (self.max_val - self.min_val)
        handle_x = self.rect.x + int(t * self.rect.width)

        handle_color = (255, 215, 0) if self.focused else (200, 220, 255)
        pygame.draw.circle(
            screen, handle_color,
            (handle_x, self.rect.centery),
            self.rect.height // 2 + 4,
        )

        if self.focused:
            # Create a blinking cursor natively in Pygame using get_ticks()
            cursor = "|" if (pygame.time.get_ticks() // 500) % 2 == 0 else ""
            val_str = f"{self.input_text}{cursor}"
            text_color = (255, 215, 0)
        else:
            val_str = f"{int(self.val)}" if self.is_int else f"{self.val:.{self.decimals}f}"
            text_color = (220, 230, 255)

        label = font.render(f"{self.text}: {val_str}", True, text_color)
        screen.blit(label, (self.rect.x, self.rect.y - 22))

# ---------------------------------------------------------------------------
# UI: Button
# ---------------------------------------------------------------------------

class Button:
    """A toggle button for the tab bar at the top of the control panel.

    The active flag controls the highlighted (selected) visual state.

    Attributes:
        rect:   Bounding rectangle.
        text:   Label rendered centred in the button.
        active: Whether this button is currently selected.
    """

    def __init__(self, x: int, y: int, w: int, h: int, text: str) -> None:
        self.rect   = pygame.Rect(x, y, w, h)
        self.text   = text
        self.active = False

    def draw(self, screen: pygame.Surface, font: pygame.font.Font) -> None:
        """Render the button, highlighted if active is True."""
        fill_color = (100, 150, 255) if self.active else (60, 70, 90)
        pygame.draw.rect(screen, fill_color,      self.rect, border_radius=4)
        pygame.draw.rect(screen, (200, 220, 255), self.rect, 1, border_radius=4)
        label = font.render(self.text, True, (255, 255, 255))
        screen.blit(label, (
            self.rect.centerx - label.get_width()  // 2,
            self.rect.centery - label.get_height() // 2,
        ))

    def handle_event(self, event: pygame.event.Event) -> bool:
        """Return True if this button was clicked.

        Args:
            event: A Pygame event.
        """
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self.rect.collidepoint(event.pos):
                return True
        return False


# ---------------------------------------------------------------------------
# UI: FPS graph
# ---------------------------------------------------------------------------

# The green horizontal reference line is drawn at this target FPS value.
_FPS_TARGET: float = 60.0


def draw_fps_graph(
    screen:      pygame.Surface,
    font:        pygame.font.Font,
    fps_history: Deque[float],
    panel_x: int, panel_y: int,
    panel_w: int, panel_h: int,
) -> None:
    """Draw a filled line graph of recent FPS values inside a panel rectangle.

    The graph scales vertically so the display is stable when running above
    target. A green dashed line marks 60 fps.

    Args:
        screen:      Destination surface.
        font:        Font for the FPS label and current-value readout.
        fps_history: Ring buffer of recent clock.get_fps() samples.
        panel_x/y:   Top-left corner of the graph area (pixels).
        panel_w/h:   Width and height of the graph area (pixels).
    """
    history_len = fps_history.maxlen or 120

    # Background and border.
    bg_rect = pygame.Rect(panel_x, panel_y, panel_w, panel_h)
    pygame.draw.rect(screen, (15, 20, 32), bg_rect, border_radius=4)
    pygame.draw.rect(screen, (40, 55, 80), bg_rect, 1,  border_radius=4)
    screen.blit(font.render("FPS", True, (80, 110, 160)), (panel_x + 4, panel_y + 4))

    history = list(fps_history)
    if len(history) < 2:
        return

    max_fps  = max(max(history), _FPS_TARGET)
    graph_h  = panel_h - 24
    graph_w  = panel_w - 8
    baseline = panel_y + panel_h - 4

    # Build a polyline from the ring buffer, oldest sample on the left.
    points = [
        (
            panel_x + 4 + int(i / (history_len - 1) * graph_w),
            panel_y + panel_h - 4 - int((f / max_fps) * graph_h),
        )
        for i, f in enumerate(history)
    ]

    # Filled area under the curve (semi-transparent blue).
    fill_poly  = [points[0]] + points + [(points[-1][0], baseline), (points[0][0], baseline)]
    fill_surf  = pygame.Surface((panel_w, panel_h), pygame.SRCALPHA)
    local_poly = [(px - panel_x, py - panel_y) for px, py in fill_poly]
    pygame.draw.polygon(fill_surf, (40, 100, 200, 60), local_poly)
    screen.blit(fill_surf, (panel_x, panel_y))
    pygame.draw.lines(screen, (80, 160, 255), False, points, 1)

    # Current FPS readout, right-aligned inside the panel.
    cur_label = font.render(f"{history[-1]:.0f}", True, (140, 200, 255))
    screen.blit(cur_label, (panel_x + panel_w - cur_label.get_width() - 4, panel_y + 4))

    # 60 fps target reference line.
    target_y = panel_y + panel_h - 4 - int((_FPS_TARGET / max_fps) * graph_h)
    if (panel_y + 20) <= target_y <= (panel_y + panel_h):
        pygame.draw.line(
            screen, (60, 120, 60),
            (panel_x + 4,           target_y),
            (panel_x + panel_w - 4, target_y),
            1,
        )


# ---------------------------------------------------------------------------
# Numba SPH kernels  (notation: h = smoothing radius, d = inter-particle dist)
# ---------------------------------------------------------------------------

@njit
def kernel(h: float, d: float) -> float:
    """Poly6 smoothing kernel W(h, d).

    Returns the weight of a neighbour at distance d within smoothing radius h.
    The cubic fall-off gives a smooth zero derivative at the boundary,
    avoiding discontinuous forces near the edge of the neighbourhood.
    """
    if d >= h:
        return 0.0
    q = 1.0 - (d / h) ** 2
    return q * q * q


@njit
def kernel_grad(h: float, d: float) -> float:
    """Radial gradient of the Poly6 kernel: dW/dd.

    Used in the pressure force accumulation. The negative sign means the
    gradient points inward, so subtracting it in the pressure loop correctly
    produces a repulsive force.
    """
    if d >= h or d < 1e-6:
        return 0.0
    q = 1.0 - (d / h) ** 2
    return -6.0 * d / (h * h) * q * q


@njit
def kernel_laplacian(h: float, d: float) -> float:
    """Laplacian of the Poly6 kernel: nabla^2 W(h, d).

    Used by the surface-tension pass to measure the curvature of the
    colour field at each surface particle.
    """
    if d >= h:
        return 0.0
    h2 = h * h
    d2 = d * d
    q  = 1.0 - d2 / h2
    return (24.0 / h2) * q * (4.0 * d2 / h2 - q)


@njit
def calibrate_rest_density(h: float, particle_radius: float, mass: float) -> float:
    """Compute the equilibrium density for a fully-surrounded particle.

    Samples a regular grid of phantom neighbours at the natural packing
    spacing (2.2 * radius) and sums their kernel contributions. This gives
    the density that a particle should have when surrounded by peers, so
    the pressure equation produces zero force at equilibrium.

    Args:
        h:               Smoothing radius.
        particle_radius: Radius of a single particle.
        mass:            Particle mass (= radius^2).

    Returns:
        The rest (equilibrium) density rho_0.
    """
    spacing = particle_radius * 2.2
    total   = 0.0
    grid_r  = int(h / spacing) + 1
    for ix in range(-grid_r, grid_r + 1):
        for iy in range(-grid_r, grid_r + 1):
            d = math.hypot(ix * spacing, iy * spacing)
            total += mass * kernel(h, d)
    return max(total, 0.001)


# ---------------------------------------------------------------------------
# Numba SPH simulation passes
# ---------------------------------------------------------------------------

@njit
def build_grid(
    xs: np.ndarray, ys: np.ndarray, num_active: int,
    h: float, grid_cols: int, grid_rows: int,
    grid_head: np.ndarray, grid_next: np.ndarray,
) -> None:
    """Insert all active particles into the spatial hash grid.

    Uses a linked-list per cell: grid_head[cx, cy] is the index of the first
    particle in that cell; grid_next[i] chains to the next particle in the
    same cell (-1 = end of list).

    Only clearing grid_next[:num_active] (not the whole array) avoids an
    O(capacity) fill on every frame.
    """
    grid_head.fill(-1)
    grid_next[:num_active] = -1
    for i in range(num_active):
        cx = int(xs[i] // h)
        cy = int(ys[i] // h)
        if 0 <= cx < grid_cols and 0 <= cy < grid_rows:
            grid_next[i]      = grid_head[cx, cy]  # Push onto front of list.
            grid_head[cx, cy] = i


@njit(parallel=True)
def compute_densities(
    xs: np.ndarray, ys: np.ndarray,
    densities: np.ndarray, num_active: int,
    h: float, mass: float,
    grid_cols: int, grid_rows: int,
    grid_head: np.ndarray, grid_next: np.ndarray,
) -> None:
    """Estimate SPH density at each particle: rho_i = sum_j m*W(h, |r_i - r_j|).

    Each particle sums kernel contributions of its neighbours. Only the 3x3
    cells surrounding particle i's own cell need to be checked because
    W(h, d) = 0 for d >= h and each cell has side-length h.
    """
    h2 = h * h
    for i in prange(num_active):
        rho = 0.0
        cx  = int(xs[i] // h)
        cy  = int(ys[i] // h)
        for dcx in range(-1, 2):
            for dcy in range(-1, 2):
                nx = cx + dcx
                ny = cy + dcy
                if 0 <= nx < grid_cols and 0 <= ny < grid_rows:
                    j = grid_head[nx, ny]
                    while j != -1:
                        dist2 = (xs[i] - xs[j]) ** 2 + (ys[i] - ys[j]) ** 2
                        if dist2 < h2:
                            rho += mass * kernel(h, math.sqrt(dist2))
                        j = grid_next[j]
        densities[i] = max(rho, 1e-4)  # Floor prevents division by zero downstream.


@njit(parallel=True)
def apply_pressure(
    xs: np.ndarray, ys: np.ndarray,
    axs: np.ndarray, ays: np.ndarray,
    densities: np.ndarray, num_active: int,
    h: float, mass: float,
    stiffness: float, rest_density: float,
    grid_cols: int, grid_rows: int,
    grid_head: np.ndarray, grid_next: np.ndarray,
) -> None:
    """Accumulate pressure forces onto each particle.

    Pressure equation (linear): p_i = max(k * (rho_i - rho_0), 0).
    The max(0) clamp means only compression creates repulsion — no tension.

    Force: a_i -= sum_j  m*(p_i+p_j)/(2*rho_j) * grad_W * r_hat_ij.
    Averaging pressures symmetrises the interaction so Newton's 3rd law is
    approximately preserved.
    """
    h2 = h * h
    for i in prange(num_active):
        pressure_i = max(stiffness * (densities[i] - rest_density), 0.0)
        cx = int(xs[i] // h)
        cy = int(ys[i] // h)
        for dcx in range(-1, 2):
            for dcy in range(-1, 2):
                nx = cx + dcx
                ny = cy + dcy
                if 0 <= nx < grid_cols and 0 <= ny < grid_rows:
                    j = grid_head[nx, ny]
                    while j != -1:
                        if i != j:
                            dx = xs[i] - xs[j]
                            dy = ys[i] - ys[j]
                            d2 = dx * dx + dy * dy
                            if 1e-10 < d2 < h2:
                                d          = math.sqrt(d2)
                                pressure_j = max(stiffness * (densities[j] - rest_density), 0.0)
                                grad       = kernel_grad(h, d)
                                f_shared   = (
                                    0.5 * mass * (pressure_i + pressure_j)
                                    / (2.0 * densities[j] + 1e-6)
                                    * grad
                                )
                                axs[i] -= f_shared * (dx / d)
                                ays[i] -= f_shared * (dy / d)
                        j = grid_next[j]


@njit(parallel=True)
def apply_viscosity(
    xs: np.ndarray, ys: np.ndarray,
    vxs: np.ndarray, vys: np.ndarray,
    densities: np.ndarray, num_active: int,
    h: float, mass: float, viscosity: float,
    grid_cols: int, grid_rows: int,
    grid_head: np.ndarray, grid_next: np.ndarray,
) -> None:
    """Smooth velocity differences between neighbouring particles.

    delta_v_i = sum_j (v_j - v_i) * m * W(h,d) / rho_j;  v_i += mu * delta_v_i.

    This is a velocity-averaging (XSPH variant) rather than a true Laplacian
    viscosity, but it is cheaper and stable at the timestep used here.
    Higher mu produces a thicker, more honey-like fluid.
    """
    h2 = h * h
    for i in prange(num_active):
        dv_x, dv_y = 0.0, 0.0
        cx = int(xs[i] // h)
        cy = int(ys[i] // h)
        for dcx in range(-1, 2):
            for dcy in range(-1, 2):
                nx = cx + dcx
                ny = cy + dcy
                if 0 <= nx < grid_cols and 0 <= ny < grid_rows:
                    j = grid_head[nx, ny]
                    while j != -1:
                        if i != j:
                            rx = xs[j] - xs[i]
                            ry = ys[j] - ys[i]
                            d2 = rx * rx + ry * ry
                            if d2 < h2:
                                d    = math.sqrt(d2)
                                coef = mass * kernel(h, d) / (densities[j] + 1e-6)
                                dv_x += (vxs[j] - vxs[i]) * coef
                                dv_y += (vys[j] - vys[i]) * coef
                        j = grid_next[j]
        vxs[i] += viscosity * dv_x
        vys[i] += viscosity * dv_y


@njit(parallel=True)
def apply_surface_tension(
    xs: np.ndarray, ys: np.ndarray,
    axs: np.ndarray, ays: np.ndarray,
    densities: np.ndarray, num_active: int,
    h: float, mass: float,
    surface_tension: float, cutoff_density: float,
    grid_cols: int, grid_rows: int,
    grid_head: np.ndarray, grid_next: np.ndarray,
) -> None:
    """Apply a cohesion force to particles near the fluid surface.

    Uses the colour-field method (Muller et al. 2003): treat the fluid as a
    scalar field C = 1 inside, 0 outside. At the surface grad(C) points
    outward and nabla^2(C) measures curvature. The force
        a_i -= sigma * nabla^2(C) * (grad(C) / |grad(C)|)
    pulls surface particles inward, creating a surface-tension-like effect.

    Only particles with density below cutoff_density are considered surface
    particles; interior particles skip this pass entirely.

    A minimum gradient magnitude guard suppresses the force at isolated
    particles where the gradient direction is numerically unreliable.
    """
    _MIN_GRADIENT = 0.3  # Suppress force when gradient direction is unreliable.
    h2 = h * h
    for i in prange(num_active):
        if densities[i] > cutoff_density:
            continue  # Interior particle — skip.

        grad_cx, grad_cy, laplacian = 0.0, 0.0, 0.0
        cx = int(xs[i] // h)
        cy = int(ys[i] // h)
        for dcx in range(-1, 2):
            for dcy in range(-1, 2):
                nx = cx + dcx
                ny = cy + dcy
                if 0 <= nx < grid_cols and 0 <= ny < grid_rows:
                    j = grid_head[nx, ny]
                    while j != -1:
                        if i != j:
                            rx = xs[j] - xs[i]
                            ry = ys[j] - ys[i]
                            d2 = rx * rx + ry * ry
                            if 1e-6 < d2 < h2:
                                d          = math.sqrt(d2)
                                coef       = mass / (densities[j] + 1e-6)
                                g          = kernel_grad(h, d)
                                grad_cx   += coef * g * (rx / d)
                                grad_cy   += coef * g * (ry / d)
                                laplacian += coef * kernel_laplacian(h, d)
                        j = grid_next[j]

        grad_mag = math.hypot(grad_cx, grad_cy)
        if grad_mag > _MIN_GRADIENT:
            axs[i] -= surface_tension * laplacian * grad_cx / grad_mag
            ays[i] -= surface_tension * laplacian * grad_cy / grad_mag


@njit(parallel=True)
def integrate(
    xs: np.ndarray, ys: np.ndarray,
    vxs: np.ndarray, vys: np.ndarray,
    axs: np.ndarray, ays: np.ndarray,
    num_active: int,
    gravity: float,
) -> None:
    """Advance positions and velocities by one timestep (semi-implicit Euler).

    Integration order: velocity is updated first from accumulated acceleration,
    then position is updated from the new velocity. This is "symplectic" Euler
    which conserves energy better than explicit Euler for oscillatory systems.

    After integration, acceleration is reset to zero except for gravity,
    which is loaded into ays so it acts as a constant body force next frame.

    A speed clamp prevents particles from moving more than MAX_PARTICLE_SPEED
    pixels per frame, which would otherwise cause tunnelling through walls.
    """
    for i in prange(num_active):
        vxs[i] += axs[i]
        vys[i] += ays[i]

        # Clamp speed — scale both components uniformly to preserve direction.
        spd = math.hypot(vxs[i], vys[i])
        if spd > MAX_PARTICLE_SPEED:
            scale   = MAX_PARTICLE_SPEED / spd
            vxs[i] *= scale
            vys[i] *= scale

        xs[i] += vxs[i]
        ys[i] += vys[i]

        axs[i] = 0.0      # Reset for next frame.
        ays[i] = gravity  # Gravity is the only persistent body force.


@njit(parallel=True)
def resolve_overlaps(
    xs: np.ndarray, ys: np.ndarray,
    num_active: int, h: float, particle_diameter: float,
    grid_cols: int, grid_rows: int,
    grid_head: np.ndarray, grid_next: np.ndarray,
) -> None:
    """Push overlapping particle pairs apart by half the overlap each.

    This positional correction pass runs after integration to prevent
    persistent inter-penetration. Each particle is nudged outward by its
    share of the overlap, keeping the correction symmetric.
    """
    diam2 = particle_diameter * particle_diameter
    for i in prange(num_active):
        cx = int(xs[i] // h)
        cy = int(ys[i] // h)
        for dcx in range(-1, 2):
            for dcy in range(-1, 2):
                nx = cx + dcx
                ny = cy + dcy
                if 0 <= nx < grid_cols and 0 <= ny < grid_rows:
                    j = grid_head[nx, ny]
                    while j != -1:
                        if i != j:
                            dx = xs[i] - xs[j]
                            dy = ys[i] - ys[j]
                            d2 = dx * dx + dy * dy
                            if 1e-10 < d2 < diam2:
                                d      = math.sqrt(d2)
                                push   = (particle_diameter - d) * 0.5
                                xs[i] += (dx / d) * push
                                ys[i] += (dy / d) * push
                        j = grid_next[j]


@njit(parallel=True)
def apply_mouse_push(
    xs: np.ndarray, ys: np.ndarray,
    vxs: np.ndarray, vys: np.ndarray,
    num_active: int,
    cursor_x: float, cursor_y: float,
    push_radius: float, push_strength: float, is_pulling: bool,
) -> None:
    """Apply a radial impulse to particles near the cursor position.

    Force falls off linearly from push_strength at the cursor to zero at
    push_radius. Particles outside the radius are unaffected.
    
    If is_pulling is True, the impulse is inverted to act as a vacuum.
    """
    radius2 = push_radius * push_radius
    direction = -1.0 if is_pulling else 1.0
    for i in prange(num_active):
        dx = xs[i] - cursor_x
        dy = ys[i] - cursor_y
        d2 = dx * dx + dy * dy
        if 0 < d2 < radius2:
            d       = math.sqrt(d2)
            impulse = (1.0 - d / push_radius) * push_strength * direction
            vxs[i] += (dx / d) * impulse
            vys[i] += (dy / d) * impulse


@njit(parallel=True)
def resolve_flask(
    xs: np.ndarray, ys: np.ndarray,
    vxs: np.ndarray, vys: np.ndarray,
    num_active: int,
    particle_radius: float,
    wall_friction: float,
    restitution: float,
    f_left: float, f_right: float, f_top: float, f_bot: float,
) -> None:
    """Clamp particles inside the flask boundaries and reflect wall velocity.

    On impact the normal velocity component is reversed and scaled by
    restitution (0 = fully inelastic, 1 = perfectly elastic). The tangential
    component is damped by wall_friction to simulate surface drag.
    """
    for i in prange(num_active):
        # Left wall
        if xs[i] < f_left + particle_radius:
            xs[i] = f_left + particle_radius
            if vxs[i] < 0:
                vxs[i] = -vxs[i] * restitution
            vys[i] *= (1.0 - wall_friction)

        # Right wall
        if xs[i] > f_right - particle_radius:
            xs[i] = f_right - particle_radius
            if vxs[i] > 0:
                vxs[i] = -vxs[i] * restitution
            vys[i] *= (1.0 - wall_friction)

        # Floor
        if ys[i] > f_bot - particle_radius:
            ys[i] = f_bot - particle_radius
            if vys[i] > 0:
                vys[i] = -vys[i] * restitution
            vxs[i] *= (1.0 - wall_friction)

        # Ceiling — prevents escape above the flask opening.
        if ys[i] < particle_radius:
            ys[i] = particle_radius
            if vys[i] < 0:
                vys[i] = -vys[i] * restitution


@njit(parallel=True)
def resolve_obstacles(
    xs: np.ndarray, ys: np.ndarray,
    vxs: np.ndarray, vys: np.ndarray,
    num_active: int,
    particle_radius: float,
    obs_xs: np.ndarray, obs_ys: np.ndarray,
    obs_ws: np.ndarray, obs_hs: np.ndarray,
    obs_active: np.ndarray,
    restitution: float,
    wall_friction: float,
) -> None:
    """Resolve collisions between particles and axis-aligned box obstacles.

    For each particle inside an obstacle's expanded bounding box, the
    shortest escape direction is chosen (minimum penetration depth among
    the four faces) and the particle is pushed to that face. The velocity
    component into the face is reflected and scaled by restitution; the
    tangential component is damped by friction.
    """
    for i in prange(num_active):
        px = xs[i]
        py = ys[i]
        for o in range(len(obs_active)):
            if obs_active[o] != 1:
                continue

            ox, oy = obs_xs[o], obs_ys[o]
            ow, oh = obs_ws[o], obs_hs[o]

            # Test centre-point overlap with the obstacle expanded by particle_radius.
            inside = (
                ox - particle_radius < px < ox + ow + particle_radius
                and oy - particle_radius < py < oy + oh + particle_radius
            )
            if not inside:
                continue

            # Depth from each face — smallest depth is the escape direction.
            d_left   = px - (ox - particle_radius)
            d_right  = (ox + ow + particle_radius) - px
            d_top    = py - (oy - particle_radius)
            d_bottom = (oy + oh + particle_radius) - py
            min_d    = min(d_left, d_right, d_top, d_bottom)

            if min_d == d_left:
                xs[i] = ox - particle_radius
                if vxs[i] > 0:
                    vxs[i] = -vxs[i] * restitution
                vys[i] *= (1.0 - wall_friction)
            elif min_d == d_right:
                xs[i] = ox + ow + particle_radius
                if vxs[i] < 0:
                    vxs[i] = -vxs[i] * restitution
                vys[i] *= (1.0 - wall_friction)
            elif min_d == d_top:
                ys[i] = oy - particle_radius
                if vys[i] > 0:
                    vys[i] = -vys[i] * restitution
                vxs[i] *= (1.0 - wall_friction)
            else:  # d_bottom
                ys[i] = oy + oh + particle_radius
                if vys[i] < 0:
                    vys[i] = -vys[i] * restitution
                vxs[i] *= (1.0 - wall_friction)


# ---------------------------------------------------------------------------
# Numba metaball renderer
# ---------------------------------------------------------------------------

@njit(parallel=True)
def compute_metaballs(
    xs: np.ndarray, ys: np.ndarray,
    vxs: np.ndarray, vys: np.ndarray,
    num_active: int,
    render_w: int, render_h: int,
    render_scale: int,
    render_radius: float,
    threshold: float,
    f_left: float, f_right: float, f_bot: float,
    obs_xs: np.ndarray, obs_ys: np.ndarray,
    obs_ws: np.ndarray, obs_hs: np.ndarray,
    obs_active: np.ndarray,
) -> np.ndarray:
    """Render particles as a metaball field and return an RGB pixel array.

    Each particle splats a radial weight function onto a downscaled grid.
    Pixels whose accumulated weight exceeds threshold are coloured by the
    average velocity of contributing particles (speed maps to hue). Pixels
    outside the flask or inside an obstacle get the background colour.

    The splat weight function is w = (1 - d^2/r^2)^2, which is smooth,
    non-negative, and zero at d = render_radius.

    Args:
        render_w/h:    Grid dimensions (= screen size / render_scale).
        render_scale:  Downscale factor (higher = faster, blurrier).
        render_radius: Splat radius in grid pixels.
        threshold:     Minimum field weight for a pixel to count as fluid.

    Returns:
        A (render_w, render_h, 3) uint8 array of RGB values.
    """
    weight_grid = np.zeros((render_w, render_h), dtype=np.float32)
    vx_grid     = np.zeros((render_w, render_h), dtype=np.float32)
    vy_grid     = np.zeros((render_w, render_h), dtype=np.float32)
    r_sq        = render_radius * render_radius

    # Splat pass: accumulate weight and velocity for each particle.
    for i in range(num_active):
        px = xs[i] / render_scale
        py = ys[i] / render_scale

        # Iterate only over the pixel patch within the splat radius.
        min_x = max(0,            int(px - render_radius))
        max_x = min(render_w - 1, int(px + render_radius))
        min_y = max(0,            int(py - render_radius))
        max_y = min(render_h - 1, int(py + render_radius))

        for gx in range(min_x, max_x + 1):
            for gy in range(min_y, max_y + 1):
                dx = px - gx
                dy = py - gy
                d2 = dx * dx + dy * dy
                if d2 < r_sq:
                    w = (1.0 - d2 / r_sq) ** 2
                    weight_grid[gx, gy] += w
                    vx_grid[gx, gy]     += vxs[i] * w
                    vy_grid[gx, gy]     += vys[i] * w

    # Shading pass: map field weight and average velocity to colour.
    pixels = np.zeros((render_w, render_h, 3), dtype=np.uint8)
    for gx in prange(render_w):
        for gy in range(render_h):
            screen_x = gx * render_scale
            screen_y = gy * render_scale

            inside_flask = f_left <= screen_x <= f_right and screen_y <= f_bot

            inside_obstacle = False
            for o in range(len(obs_active)):
                if obs_active[o] == 1:
                    if (obs_xs[o] < screen_x < obs_xs[o] + obs_ws[o]
                            and obs_ys[o] < screen_y < obs_ys[o] + obs_hs[o]):
                        inside_obstacle = True
                        break

            if not inside_flask or inside_obstacle:
                pixels[gx, gy, 0] = 8
                pixels[gx, gy, 1] = 12
                pixels[gx, gy, 2] = 24
                continue

            w = weight_grid[gx, gy]
            if w > threshold:
                avg_vx = vx_grid[gx, gy] / w
                avg_vy = vy_grid[gx, gy] / w
                spd    = math.hypot(avg_vx, avg_vy)
                t      = min(spd / METABALL_COLOR_SPEED, 1.0)

                # Two-stage speed-to-colour ramp: slow = dark blue, fast = cyan/white.
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

                # Brighten pixels near the surface to create a specular rim.
                if w < threshold + 0.2:
                    r = min(255, r + 40)
                    g = min(255, g + 50)
                    b = min(255, b + 60)

                pixels[gx, gy, 0] = r
                pixels[gx, gy, 1] = g
                pixels[gx, gy, 2] = b
            else:
                pixels[gx, gy, 0] = 8
                pixels[gx, gy, 1] = 12
                pixels[gx, gy, 2] = 24

    return pixels


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------

def draw_flask(surface: pygame.Surface) -> None:
    """Draw the container walls as glowing blue lines on surface."""
    glow_color = (30,  70, 150)
    wall_color = (100, 150, 220)
    glow_width = 6
    wall_width = 2

    # Bottom, left wall, right wall — no top, so particles can fall in.
    walls = [
        ((F_LEFT,  F_BOT), (F_RIGHT, F_BOT)),
        ((F_LEFT,  F_TOP), (F_LEFT,  F_BOT)),
        ((F_RIGHT, F_TOP), (F_RIGHT, F_BOT)),
    ]
    for start, end in walls:
        pygame.draw.line(surface, glow_color, start, end, glow_width)
        pygame.draw.line(surface, wall_color, start, end, wall_width)


def draw_obstacles(
    surface:    pygame.Surface,
    obs_xs:     np.ndarray,
    obs_ys:     np.ndarray,
    obs_ws:     np.ndarray,
    obs_hs:     np.ndarray,
    obs_active: np.ndarray,
) -> None:
    """Draw all active obstacle blocks as filled grey rectangles with an outline."""
    for i in range(len(obs_active)):
        if obs_active[i] == 1:
            rect = pygame.Rect(
                int(obs_xs[i]), int(obs_ys[i]),
                int(obs_ws[i]), int(obs_hs[i]),
            )
            pygame.draw.rect(surface, (150, 150, 150), rect)
            pygame.draw.rect(surface, (200, 200, 200), rect, 2)


def draw_help_overlay(
    screen:     pygame.Surface,
    font:       pygame.font.Font,
    title_font: pygame.font.Font,
) -> None:
    """Render a semi-transparent controls help screen over the simulation."""
    overlay = pygame.Surface((SIM_WIDTH, HEIGHT), pygame.SRCALPHA)
    overlay.fill((0, 0, 0, 180))

    title = title_font.render("SPH FLUID PLAYGROUND", True, (255, 255, 255))
    overlay.blit(title, (SIM_WIDTH // 2 - title.get_width() // 2, 80))

    controls =[
        "--- Controls ---",
        "[LMB / RMB]  Push / Pull the fluid",
        "         (Use the UI panel on the right!)",
        "[ C / X ]    Spawn / Delete a block under cursor",
        "[ M ]        Toggle metaball rendering",
        "[ SPACE ]    Pause / resume physics",
        "[ RIGHT ]    Step one frame (when paused)",
        "[ P ]        Pause / resume pouring",
        "[ R ]        Reset simulation",
        "[ H ]        Toggle this help screen",
        "",
        "Press [H] to close and play!",
    ]

    for i, line in enumerate(controls):
        color = (255, 215, 0) if "Press [H]" in line else (200, 220, 255)
        text  = font.render(line, True, color)
        overlay.blit(text, (SIM_WIDTH // 2 - text.get_width() // 2, 160 + i * 25))

    screen.blit(overlay, (0, 0))


# ---------------------------------------------------------------------------
# Main — initialisation helpers
# ---------------------------------------------------------------------------

def _init_sim_state() -> SimState:
    """Allocate particle arrays and create the initial SimState.

    One default obstacle block is placed near the flask floor so the user
    has something to interact with immediately.

    Returns:
        A freshly initialised SimState ready for the main loop.
    """
    capacity = INITIAL_CAPACITY
    xs, ys, vxs, vys, axs, ays, densities, grid_next = make_particle_arrays(capacity)
    grid_head = np.full((MAX_GRID_COLS, MAX_GRID_ROWS), -1, dtype=np.int32)

    state = SimState(
        capacity   = capacity,
        num_active = 0,
        xs = xs, ys = ys,
        vxs = vxs, vys = vys,
        axs = axs, ays = ays,
        densities = densities,
        grid_next = grid_next,
        grid_head = grid_head,
    )

    # Default obstacle: a square block centred at the flask floor.
    state.obs_ws[0]    = 80.0
    state.obs_hs[0]    = 80.0
    state.obs_xs[0]    = SIM_WIDTH / 2.0 - 40.0
    state.obs_ys[0]    = HEIGHT - 130.0
    state.obs_active[0] = 1

    return state


def _build_ui() -> dict:
    """Create and return all UI widgets as a plain dict.

    Default slider values are defined here — this is the one place to look
    if you want to change what value a slider starts at.

    Returns:
        A dict with keys: btn_basic, btn_adv, basic_sliders, adv_sliders.
        basic_sliders and adv_sliders are lists of Slider objects.
    """
    panel_x = SIM_WIDTH + 20  # Left edge of the control panel.

    btn_basic        = Button(panel_x,        20, 100, 30, "Basic")
    btn_adv          = Button(panel_x + 110,  20, 100, 30, "Advanced")
    btn_basic.active = True

    # Basic tab — Particle Size auto-scales smoothing radius, mass, and rest
    # density, keeping the simulation numerically stable at any size.
    slider_size  = Slider(panel_x, 100, 210, 15,
                          min_val=1.0, max_val=6.0, initial_val=2.0,
                          text="Particle Size")
    slider_count = Slider(panel_x, 170, 210, 15,
                          min_val=0, max_val=SLIDER_MAX, initial_val=6000,
                          text="Max Particles", is_int=True)
    basic_sliders = [slider_size, slider_count]

    # Advanced tab — these pass directly into the Numba kernels with no
    # scaling. Extreme values can make the simulation unstable; use with care.
    slider_gravity = Slider(panel_x, 100, 210, 15,
                            min_val=-1.0, max_val=1.0,
                            initial_val=DEFAULT_GRAVITY,
                            text="Gravity", decimals=2)
    slider_stiff   = Slider(panel_x, 170, 210, 15,
                            min_val=0.1, max_val=5.0,
                            initial_val=DEFAULT_STIFFNESS,
                            text="Stiffness", decimals=1)
    slider_visc    = Slider(panel_x, 240, 210, 15,
                            min_val=0.0, max_val=0.5,
                            initial_val=DEFAULT_VISCOSITY,
                            text="Viscosity", decimals=3)
    slider_surf    = Slider(panel_x, 310, 210, 15,
                            min_val=0.0, max_val=1.0,
                            initial_val=DEFAULT_SURF_TENSION,
                            text="Surf Tension", decimals=3)
    slider_rest    = Slider(panel_x, 380, 210, 15,
                            min_val=0.0, max_val=1.0,
                            initial_val=DEFAULT_RESTITUTION,
                            text="Restitution", decimals=2)
    adv_sliders = [slider_gravity, slider_stiff, slider_visc, slider_surf, slider_rest]

    return {
        "btn_basic":     btn_basic,
        "btn_adv":       btn_adv,
        "basic_sliders": basic_sliders,
        "adv_sliders":   adv_sliders,
    }


# ---------------------------------------------------------------------------
# Main — per-frame helpers
# ---------------------------------------------------------------------------

def _handle_events(state: SimState, ui: dict) -> bool:
    """Process the Pygame event queue for one frame."""
    btn_basic      = ui["btn_basic"]
    btn_adv        = ui["btn_adv"]
    active_sliders = ui["basic_sliders"] if btn_basic.active else ui["adv_sliders"]

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            return False

        # 1. Feed events to sliders first. If a slider consumes the event 
        # (like typing a number), ui_consumed becomes True.
        ui_consumed = False
        for slider in active_sliders:
            if slider.handle_event(event):
                ui_consumed = True

        # 2. Check tab buttons. If we switch tabs, automatically unfocus sliders!
        if btn_basic.handle_event(event):
            btn_basic.active = True
            btn_adv.active   = False
            ui_consumed      = True
            for s in ui["adv_sliders"]:
                if s.focused:
                    s._apply_input()
                    s.focused = False

        if btn_adv.handle_event(event):
            btn_adv.active   = True
            btn_basic.active = False
            ui_consumed      = True
            for s in ui["basic_sliders"]:
                if s.focused:
                    s._apply_input()
                    s.focused = False

        # 3. Keyboard Controls
        if event.type == pygame.KEYDOWN and not ui_consumed:
            if event.key == pygame.K_h:
                state.show_help = not state.show_help
            elif event.key == pygame.K_r:
                state.num_active = 0
                state.pouring    = True
            elif event.key == pygame.K_p:
                state.pouring = not state.pouring
            elif event.key == pygame.K_SPACE:
                state.sim_paused = not state.sim_paused
            elif event.key == pygame.K_RIGHT:
                state.step_one_frame = True
            elif event.key == pygame.K_m:
                state.use_metaballs = not state.use_metaballs
            elif event.key == pygame.K_c:
                _spawn_obstacle(state)
            elif event.key == pygame.K_x:
                _delete_obstacle_under_cursor(state)

        # 4. Mouse Click (Push / Pull / Drag)
        if event.type == pygame.MOUSEBUTTONDOWN and not ui_consumed:
            mx, my = pygame.mouse.get_pos()
            if mx < SIM_WIDTH:
                if event.button == 1:    # Left Click
                    _on_sim_click(state)
                elif event.button == 3:  # Right Click
                    state.pulling_water = True

        # 5. Mouse Release
        if event.type == pygame.MOUSEBUTTONUP:
            if event.button == 1:
                state.dragging_idx  = -1
                state.pushing_water = False
            elif event.button == 3:
                state.pulling_water = False

    return True


def _spawn_obstacle(state: SimState) -> None:
    """Spawn a 60x60 obstacle block centred on the current cursor position.

    Does nothing if all obstacle slots are already occupied or the cursor
    is over the UI panel.
    """
    mx, my = pygame.mouse.get_pos()
    if mx >= SIM_WIDTH:
        return  # Cursor is over the UI panel.
    for i in range(len(state.obs_active)):
        if state.obs_active[i] == 0:
            state.obs_ws[i]    = 60.0
            state.obs_hs[i]    = 60.0
            state.obs_xs[i]    = float(mx - 30)
            state.obs_ys[i]    = float(my - 30)
            state.obs_active[i] = 1
            return  # Only spawn one block per keypress.


def _delete_obstacle_under_cursor(state: SimState) -> None:
    """Remove the first obstacle block whose bounding box contains the cursor."""
    mx, my = pygame.mouse.get_pos()
    for i in range(len(state.obs_active)):
        if state.obs_active[i] == 1:
            hit = (
                state.obs_xs[i] <= mx <= state.obs_xs[i] + state.obs_ws[i]
                and state.obs_ys[i] <= my <= state.obs_ys[i] + state.obs_hs[i]
            )
            if hit:
                state.obs_active[i] = 0
                return


def _on_sim_click(state: SimState) -> None:
    """Handle a left-click in the simulation viewport.

    If the click lands on an obstacle, starts dragging it. Otherwise,
    starts pushing the fluid.
    """
    mx, my = pygame.mouse.get_pos()
    if mx >= SIM_WIDTH:
        return  # Click was in the UI panel.

    for i in range(len(state.obs_active)):
        if state.obs_active[i] == 1:
            hit = (
                state.obs_xs[i] <= mx <= state.obs_xs[i] + state.obs_ws[i]
                and state.obs_ys[i] <= my <= state.obs_ys[i] + state.obs_hs[i]
            )
            if hit:
                state.dragging_idx  = i
                state.drag_offset_x = mx - state.obs_xs[i]
                state.drag_offset_y = my - state.obs_ys[i]
                return  # Obstacle drag takes priority over fluid push.

    state.pushing_water = True


def _step_physics(state: SimState, ui: dict) -> None:
    """Run one complete physics tick: pour -> forces -> integrate -> resolve.

    Physics pass order:
        1. build_grid            hash particles into spatial grid
        2. compute_densities     estimate rho_i for each particle
        3. apply_pressure        pressure repulsion forces
        4. apply_viscosity       velocity smoothing
        5. apply_surface_tension surface cohesion
        6. integrate             advance positions and velocities
        7. resolve_obstacles     push out of boxes (pass 1)
        8. resolve_overlaps      push apart overlapping particles
        9. resolve_obstacles     push out of boxes (pass 2 catches re-entries)
        10. resolve_flask        clamp inside container

    The double obstacle-resolution pass (7+9) prevents particles from
    tunnelling through obstacle sides at high speeds.

    Args:
        state: Simulation state (mutated in place).
        ui:    UI widget dict (slider values are read each frame).
    """
    # Read slider values — updates take effect the very next frame.
    basic_s = ui["basic_sliders"]
    adv_s   = ui["adv_sliders"]

    particle_radius = basic_s[0].val        # Slider: Particle Size
    current_max     = int(basic_s[1].val)   # Slider: Max Particles

    gravity      = adv_s[0].val  # Slider: Gravity
    stiffness    = adv_s[1].val  # Slider: Stiffness
    viscosity    = adv_s[2].val  # Slider: Viscosity
    surf_tension = adv_s[3].val  # Slider: Surf Tension
    restitution  = adv_s[4].val  # Slider: Restitution

    # Silently drop particles if the cap was dragged down.
    # Intentionally lossy: particles are NOT restored when dragged back up.
    if state.num_active > current_max:
        state.num_active = current_max

    # Derived physics values — all scale with particle size for stability.
    smoothing_h    = particle_radius * 5.0
    mass           = particle_radius ** 2
    particle_diam  = particle_radius * 2.0
    cutoff_density = state.rest_density * SURFACE_CUTOFF_RATIO

    # Re-calibrate rest density only when particle size has changed.
    if particle_radius != state.last_particle_radius:
        state.rest_density         = calibrate_rest_density(smoothing_h, particle_radius, mass)
        state.last_particle_radius = particle_radius

    grid_cols = math.ceil(SIM_WIDTH / smoothing_h) + 2
    grid_rows = math.ceil(HEIGHT    / smoothing_h) + 2

    # Grow arrays before pouring so the pour loop never overflows.
    maybe_grow_arrays(state)

    # Move the dragged obstacle to follow the cursor.
    mx, my = pygame.mouse.get_pos()
    if state.dragging_idx >= 0:
        state.obs_xs[state.dragging_idx] = float(mx - state.drag_offset_x)
        state.obs_ys[state.dragging_idx] = float(my - state.drag_offset_y)

    # --- PAUSE & FRAME STEP LOGIC ---
    if state.sim_paused and not state.step_one_frame:
        state._particle_radius = particle_radius
        state._render_radius   = particle_radius * 2.0
        state._current_max     = current_max
        return  # Bypass all physics!

    if state.step_one_frame:
        state.step_one_frame = False  # Consume the frame step

    # Pour new particles
    if state.pouring and state.num_active < current_max and not state.show_help:
        for _ in range(POUR_RATE):
            if state.num_active >= current_max:
                break
            n = state.num_active
            state.xs[n]  = POUR_X + random.uniform(-POUR_SPREAD, POUR_SPREAD)
            state.ys[n]  = POUR_Y + random.uniform(-POUR_SPREAD, POUR_SPREAD)
            state.vxs[n] = POUR_VX + random.uniform(-0.3, 0.3)
            state.vys[n] = POUR_VY + random.uniform(-0.2, 0.2)
            state.axs[n] = 0.0
            state.densities[n] = state.rest_density
            state.num_active  += 1

    # Apply cursor push OR pull
    if (state.pushing_water or state.pulling_water) and state.num_active > 0 and not state.show_help:
        apply_mouse_push(
            state.xs, state.ys, state.vxs, state.vys, state.num_active,
            float(mx), float(my),
            MOUSE_PUSH_RADIUS, MOUSE_PUSH_STRENGTH,
            state.pulling_water
        )

    # Run the full physics pipeline
    if state.num_active > 0 and not state.show_help:
        build_grid(
            state.xs, state.ys, state.num_active, smoothing_h,
            grid_cols, grid_rows, state.grid_head, state.grid_next,
        )
        compute_densities(
            state.xs, state.ys, state.densities, state.num_active,
            smoothing_h, mass, grid_cols, grid_rows,
            state.grid_head, state.grid_next,
        )
        apply_pressure(
            state.xs, state.ys, state.axs, state.ays,
            state.densities, state.num_active,
            smoothing_h, mass, stiffness, state.rest_density,
            grid_cols, grid_rows, state.grid_head, state.grid_next,
        )
        apply_viscosity(
            state.xs, state.ys, state.vxs, state.vys,
            state.densities, state.num_active,
            smoothing_h, mass, viscosity,
            grid_cols, grid_rows, state.grid_head, state.grid_next,
        )
        apply_surface_tension(
            state.xs, state.ys, state.axs, state.ays,
            state.densities, state.num_active,
            smoothing_h, mass, surf_tension, cutoff_density,
            grid_cols, grid_rows, state.grid_head, state.grid_next,
        )
        integrate(
            state.xs, state.ys, state.vxs, state.vys,
            state.axs, state.ays, state.num_active, gravity,
        )
        resolve_obstacles(
            state.xs, state.ys, state.vxs, state.vys, state.num_active,
            particle_radius,
            state.obs_xs, state.obs_ys, state.obs_ws, state.obs_hs, state.obs_active,
            restitution, WALL_FRICTION,
        )
        resolve_overlaps(
            state.xs, state.ys, state.num_active,
            smoothing_h, particle_diam,
            grid_cols, grid_rows, state.grid_head, state.grid_next,
        )
        resolve_obstacles(
            state.xs, state.ys, state.vxs, state.vys, state.num_active,
            particle_radius,
            state.obs_xs, state.obs_ys, state.obs_ws, state.obs_hs, state.obs_active,
            restitution, WALL_FRICTION,
        )
        resolve_flask(
            state.xs, state.ys, state.vxs, state.vys, state.num_active,
            particle_radius, WALL_FRICTION, restitution,
            F_LEFT, F_RIGHT, F_TOP, F_BOT,
        )

    # Cache values the renderer needs this frame.
    state._particle_radius = particle_radius
    state._render_radius   = particle_radius * 2.0
    state._current_max     = current_max


def _render_frame(
    screen: pygame.Surface,
    state:  SimState,
    ui:     dict,
    fonts:  dict,
    clock:  pygame.time.Clock,
) -> None:
    """Draw the complete frame: simulation viewport, HUD, and UI panel.

    Args:
        screen: Destination display surface.
        state:  Current simulation state.
        ui:     Dict of UI widgets.
        fonts:  Dict with keys 'normal', 'title', 'small'.
        clock:  Pygame clock (used for FPS readout and graph).
    """
    font       = fonts["normal"]
    title_font = fonts["title"]
    small_font = fonts["small"]

    # --- Simulation viewport ---
    screen.fill(BACKGROUND_COLOR)

    if state.use_metaballs and state.num_active > 0:
        grid_w     = SIM_WIDTH // METABALL_SCALE
        grid_h     = HEIGHT    // METABALL_SCALE
        pixels     = compute_metaballs(
            state.xs, state.ys, state.vxs, state.vys,
            state.num_active, grid_w, grid_h, METABALL_SCALE,
            state._render_radius, METABALL_THRESHOLD,
            F_LEFT, F_RIGHT, F_BOT,
            state.obs_xs, state.obs_ys, state.obs_ws, state.obs_hs, state.obs_active,
        )
        fluid_surf = pygame.surfarray.make_surface(pixels)
        fluid_surf = pygame.transform.smoothscale(fluid_surf, (SIM_WIDTH, HEIGHT))
        screen.blit(fluid_surf, (0, 0))
    else:
        for i in range(state.num_active):
            spd   = math.hypot(state.vxs[i], state.vys[i])
            color = speed_to_heatmap_color(spd)
            pygame.draw.circle(
                screen, color,
                (int(state.xs[i]), int(state.ys[i])),
                int(state._particle_radius),
            )

    draw_flask(screen)
    draw_obstacles(
        screen,
        state.obs_xs, state.obs_ys,
        state.obs_ws, state.obs_hs,
        state.obs_active,
    )

    # Show the cursor influence circle while pushing or pulling.
    if (state.pushing_water or state.pulling_water) and not state.show_help:
        mx, my = pygame.mouse.get_pos()
        color = (255, 100, 100, 100) if state.pulling_water else (255, 255, 255, 100)
        pygame.draw.circle(screen, color, (mx, my), int(MOUSE_PUSH_RADIUS), 1)

    # HUD: FPS and particle count
    pause_text = "  [PAUSED]" if state.sim_paused else ""
    hud = (
        f"FPS: {clock.get_fps():.0f}  |  "
        f"{state.num_active}/{state._current_max} particles{pause_text}  |[H] Help"
    )
    screen.blit(small_font.render(hud, True, (150, 180, 220)), (10, 10))

    if state.show_help:
        draw_help_overlay(screen, font, title_font)

    # --- Control panel ---
    pygame.draw.rect(screen, (22, 28, 40), (SIM_WIDTH, 0, UI_WIDTH, HEIGHT))
    pygame.draw.line(screen, (50, 60, 80), (SIM_WIDTH, 0), (SIM_WIDTH, HEIGHT), 2)

    btn_basic = ui["btn_basic"]
    btn_adv   = ui["btn_adv"]
    btn_basic.draw(screen, font)
    btn_adv.draw(screen, font)

    active_sliders = ui["basic_sliders"] if btn_basic.active else ui["adv_sliders"]
    for slider in active_sliders:
        slider.draw(screen, font)

    hint = "Auto-Scaling ensures stability." if btn_basic.active else "Raw physics variables."
    screen.blit(small_font.render(hint, True, (100, 150, 200)), (SIM_WIDTH + 15, HEIGHT - 110))

    # FPS performance graph at the bottom of the control panel.
    state.fps_history.append(clock.get_fps())
    draw_fps_graph(
        screen, small_font, state.fps_history,
        panel_x = SIM_WIDTH + 10,
        panel_y = HEIGHT - 95,
        panel_w = UI_WIDTH - 20,
        panel_h = 88,
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Initialise Pygame, build the simulation, and run the main loop.

    Loop structure each frame:
        _handle_events  -> _step_physics  -> _render_frame  -> flip -> tick

    Numba JIT compilation happens on the first physics step, causing a brief
    pause (2-4 seconds) before the first particles appear. This is expected.
    """
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("SPH Fluid Simulation")
    clock  = pygame.time.Clock()

    fonts = {
        "normal": pygame.font.SysFont("monospace", 14, bold=True),
        "title":  pygame.font.SysFont("monospace", 28, bold=True),
        "small":  pygame.font.SysFont("monospace", 12),
    }

    state   = _init_sim_state()
    ui      = _build_ui()
    running = True

    while running:
        running = _handle_events(state, ui)
        _step_physics(state, ui)
        _render_frame(screen, state, ui, fonts, clock)
        pygame.display.flip()
        clock.tick(TARGET_FPS)

    pygame.quit()


if __name__ == "__main__":
    main()