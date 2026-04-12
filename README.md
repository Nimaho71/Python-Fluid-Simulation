# 🌊 Python 2D SPH Fluid Simulation

A high-performance, real-time **Smoothed Particle Hydrodynamics (SPH)** fluid simulation sandbox built with Python. 

This project uses **Pygame** for rendering and heavily leverages **Numba (@njit, parallel=True)** for JIT-compiled, multi-threaded CPU physics calculations. It is capable of simulating tens of thousands of interacting particles at 60 FPS.

## ✨ Features
* **Hardware Accelerated:** All physics passes (density, pressure, viscosity, surface tension) run on parallel CPU threads via Numba.
* **Dual Rendering Engines:** Toggle between a velocity-based heatmap particle view and a smooth "Liquid Blob" metaball renderer.
* **Interactive Sandbox UI:** Tweak gravity, stiffness, viscosity, and surface tension on the fly without pausing the simulation.
* **Dynamic Memory Management:** Arrays automatically scale up in the background as you increase the particle count up to 30,000+.
* **Real-time Profiling:** Built-in dynamic FPS graphing.

## 🚀 Installation & Quick Start

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Nimaho71/Python-Fluid-Simulation.git
   cd Python-Fluid-Simulation
   python -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   python fluid.py

🎮 Controls
[LMB Click + Hold] Drag an obstacle block OR push the water away radially.

[ C ] Spawn a new obstacle block at your cursor.

[ X ] Delete the obstacle block under your cursor.

[ M ] Toggle Metaball rendering (Solid Liquid vs. Particles).

[ SPACE ] Pause / Resume the pouring water stream.

[ R ] Reset the simulation.

[ H ] Hide / Show the help menu overlay.
