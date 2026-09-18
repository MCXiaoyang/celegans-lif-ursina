# C. elegans LIF Neural Simulation + Ursina 3D Environment

A C. elegans-inspired spiking neural simulation with 302 LIF neurons, a synthetic connectome, chemotaxis, body traveling wave, and a free-fly 3D camera built with Ursina.

![demo](demo.gif)

> This is a simplified neural-body simulation, not a faithful reproduction of the real C. elegans connectome.

---

## Features

- **302 LIF neurons** with leaky integrate-and-fire dynamics
- **Synthetic connectome**: random excitatory / inhibitory synapses plus hand-crafted sensory-inter-motor pathways
- **Chemotaxis**: left / right concentration probes steer the worm toward food
- **Motor decoding**: forward / left / right neuron pools drive movement
- **Body model**: spring-damper chain with lateral dorsal-ventral traveling wave
- **3D environment**: food, score, circular boundary, free-fly camera
- **Live UI**: membrane potential, spike counts, concentration gradient, camera position

---

## Controls

| Key | Action |
|---|---|
| `WASD` | Move |
| `Mouse` | Look around |
| `Space` / `Q` | Up / Down |
| `Shift` | Boost |
| `ESC` | Lock / unlock mouse |
| `R` | Reset camera |

---

## Requirements

- Windows 11 (tested)
- Python 3.12
- `ursina`
- `numpy`

Install:

```bash
pip install -r requirements.txt
```

Or manually:

```bash
pip install ursina numpy
```

---

## Run

```bash
python main.py
```

---

## How It Works

### Neuron model

Each neuron follows a leaky integrate-and-fire (LIF) rule:

```
dV/dt = (-(V - V_rest) + R_m * I_total) / tau_m
```

- `V_rest  = -65 mV`
- `V_thresh = -50 mV`
- `V_reset  = -70 mV`
- `tau_m    = 20 ms`
- `R_m      = 10 MΩ`

When `V >= V_thresh`, the neuron fires, resets to `V_reset`, and sends a spike through the connection matrix `W`.

### Synthetic connectome

- 80% excitatory, 20% inhibitory
- 5% random connection probability
- Hand-crafted pathways:
  - sensory left  → inter left  → left motor
  - sensory right → inter right → right motor
  - left inter → right motor (inhibitory)
  - right inter → left motor (inhibitory)

### Chemotaxis

Two probes sample food concentration at ±0.5 rad from the head direction. The difference drives sensory neurons on each side, which biases turning.

### Body

The worm is a chain of 17 segments:

- Distance constraint: each segment stays `SPACING` behind the previous
- Angle constraint: gentle follow to keep a smooth body
- Lateral traveling wave: sinusoidal offset perpendicular to the body axis, phase coupled to speed
- Head swing: small oscillation added to the head angle

### Boundary

Soft push-back plus gentle steering toward the center when approaching the world edge.

---

## Project Structure

```
.
├── main.py             # everything: neurons, body, world, camera
├── requirements.txt
├── README.md
└── .gitignore
```

The whole simulation is intentionally in a single file for easy reading and running.

---

## Notes

- The connectome is **synthetic**, not the real 302-neuron wiring diagram.
- This project is for learning, visualization, and interactive demo purposes.
- No external fonts or assets required.

---

## License

MIT