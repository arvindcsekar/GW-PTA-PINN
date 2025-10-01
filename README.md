# SMBHB PINN Modeling — Final Project Submission

## Overview

This repository contains a physics-informed neural network (PINN) framework for modeling the orbital evolution of Supermassive Black Hole Binaries (SMBHBs). The project focuses on the time evolution of orbital frequency ω(t) and phase φ(t) under relativistic conditions, incorporating spin-aligned binary parameters and pulsar timing effects.

The work was carried out as part of a research module on gravitational wave modeling using machine learning techniques. The PINN architecture was designed to reproduce post-Newtonian (PN) dynamics and enable waveform recovery for pulsar timing arrays (PTAs).

---

## ✅ Achievements

### 1. **Multi-parameter PINN for ω(t)**
- Developed a robust PINN architecture that models the evolution of orbital frequency ω(t) over time.
- Inputs: normalized time τ ∈ [0, 1], spin parameters χ₁, χ₂, and normalized masses m₁, m₂.
- Output: ω(t) in physical units (rad/s), trained to match 2PN dynamics.
- Hard boundary condition enforced: ω(τ=0) = 5e−7 rad/s.
- Residual loss derived from 2PN evolution equations, including spin terms and chirp mass scaling.

### 2. **Validation Against Analytic Models**
- Compared PINN predictions against analytic 2PN and Newtonian models.
- Achieved high fidelity across multiple spin configurations:

| Configuration | 2PN ω(t)        | Newtonian ω(t) | ω-PINN Output     |
|---------------|------------------|----------------|-------------------|
| χ₁ = χ₂ = 0   | —                | —              | [5.644e−7]        |
| χ₁ = χ₂ = 0.5 | 5.648e−7         | 5.679e−7       | [5.644e−7]        |
| χ₁ = χ₂ = −0.5| 5.683e−7         | 5.679e−7       | [5.683e−7]        |

- ω-PINN consistently matched the expected relativistic evolution across spin-aligned cases.

### 3. **Architecture Extension for φ(t)**
- Implemented a second PINN to model orbital phase φ(t), trained via residual:
  

\[
  \frac{d\phi}{dt} = \omega(t)
  \]


- Enforced φ(0) = 0 boundary condition.
- Used autograd to compute dφ/dτ and scaled to physical time.
- Architecture structurally complete and integrated with ω-PINN.

### 4. **Pulsar Timing Residuals**
- Incorporated Earth and Pulsar terms using light travel delay Δt.
- Evaluated ω(t) and φ(t) at both Earth and Pulsar epochs.
- Computed h₊ and h× polarizations:
  

\[
  h_+ \propto \omega^{2/3} \cos(2\phi), \quad h_× \propto \omega^{2/3} \sin(2\phi)
  \]


- Derived timing residuals:
  

\[
  R_+(t) = \frac{1}{2} \frac{h_+(t_{\text{Earth}}) - h_+(t_{\text{Pulsar}})}{1 + \cos\theta}
  \]



---

## ⚠️ Limitations

### φ(t) Evolution
- Residual logic implemented correctly, but φ-PINN output remains misaligned.
- Phase evolution shows incorrect slope and scale, likely due to output scaling and training instability.
- Attempts to correct via bias terms and loss rebalancing were unsuccessful within time constraints.

### Pulsar Term Fidelity
- Earth/Pulsar ω(t) evaluations structurally correct, but frequency evolution is too flat over narrow slices.
- Timing residuals computed, but waveform fidelity is compromised by phase inaccuracies.

---

## File Structure

