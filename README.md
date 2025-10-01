# Using Physics-Informed Neural Networks (PINNs) to Model Parameter Evolution in a Spin-Aligned Post-Newtonian Non-Eccentric Supermassive Black Hole (SMBHB) System

## Overview

This repository contains a PINN framework for modeling the orbital evolution of Supermassive Black Hole Binaries (SMBHBs), focusing on the time evolution of orbital frequency ω(t) and phase φ(t) under relativistic conditions, incorporating spin-aligned binary parameters and pulsar timing effects. The PINN takes 5 input neurons - which are later hardcoded in the samlping - of BH spin parameters, masses, and time itself. The PINN architecture was designed to reproduce post-Newtonian (PN) dynamics and enable waveform reconstruction for pulsar timing arrays (PTAs) using the forward problem approach.

---

## Achievements

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
| χ₁ = χ₂ = 0   | 5.665e-7          | 5.679e-7      | 5.664e−7          |
| χ₁ = χ₂ = 0.5 | 5.648e−7         | 5.679e−7       | 5.644e−7          |
| χ₁ = χ₂ = −0.5| 5.683e−7         | 5.679e−7       | 5.683e−7          |

ω-PINN consistently matched the expected relativistic evolution across spin-aligned cases. The figures for the above table can be seen below:

<img width="630" height="470" alt="Evolution of SMBHB Orbital Frequency with Time at 0 X1, X2" src="https://github.com/user-attachments/assets/2842e845-43cf-4d94-8034-6d285b479353" />

<img width="682" height="470" alt="5 40e-07" src="https://github.com/user-attachments/assets/ccd30249-25e9-4ba3-a87d-c11c57417650" />

<img width="685" height="470" alt="Evolution of SMBHB Orbital Frequency with Time at -0 5 x1  X2, 1e9 ml, m2" src="https://github.com/user-attachments/assets/ae7cd544-50b2-4b38-aaab-8e37b9116df2" />

---

## 3. **Architecture Extension for φ(t)**

### 🔧 Motivation and Background
For circular SMBHB systems, the orbital phase φ(t) is a critical observable for gravitational wave detection and timing residual recovery. The phase evolution is governed by:



\[
\frac{d\phi}{dt} = \omega(t)
\]



This project extends the PINN framework to solve this ODE using a second neural network, trained via residual minimization. The goal was to recover φ(t) directly from the learned ω(t), enabling full waveform reconstruction.

### 🧠 Architectural Design
- A second PINN (`PINN2`) was constructed with a single input neuron (normalized time τ) and a single output neuron (φ).
- The residual loss was computed using autograd to obtain \( \frac{d\phi}{d\tau} \), scaled by the physical time range.
- ω(t) was dynamically evaluated from the trained ω-PINN during φ-PINN training.
- A boundary condition φ(0) = 0 was enforced to anchor the phase evolution.

### 📉 Results and Observations
- The φ-PINN architecture was successfully implemented and trained.
- However, the output φ(t) showed a negative slope and incorrect amplitude, despite correct residual logic.
- Attempts to correct via output scaling, bias terms, and loss rebalancing were unsuccessful within the project timeline.

### 📊 Plots

<img width="680" height="470" alt="Evolution of SMBHB Orbital Phase Earth" src="https://github.com/user-attachments/assets/hjR6DjrVb3CsioLYJoqDM.png" />

<img width="680" height="470" alt="Evolution of SMBHB Orbital Phase Pulsar" src="https://github.com/user-attachments/assets/a5Cqv8pgGeutPKDUxwcgv.png" />

These plots show the φ(t) evolution at Earth and Pulsar epochs. While the structure is present, the slope and scale are incorrect, indicating training instability.

---

## 4. **Pulsar Timing Residuals**

### 🌌 Physical Setup
- Earth and Pulsar terms were evaluated using a light travel delay Δt corresponding to 1 kpc.
- ω(t) and φ(t) were evaluated at both epochs using normalized time inputs.
- Polarizations h₊ and h× were computed using:



\[
h_+ \propto \omega^{2/3} \cos(2\phi), \quad h_× \propto \omega^{2/3} \sin(2\phi)
\]



### ⏱️ Residuals Computed
- Timing residuals were derived using:



\[
R_+(t) = \frac{1}{2} \frac{h_+(t_{\text{Earth}}) - h_+(t_{\text{Pulsar}})}{1 + \cos\theta}
\]



- Residuals were computed for h₊, h×, and RMS.

### 📊 Plots

<img width="680" height="470" alt="Timing Residual h+" src="https://github.com/user-attachments/assets/7TqcTCXR9XBDov1A23Wfg.png" />

<img width="680" height="470" alt="Timing Residual hx" src="https://github.com/user-attachments/assets/rUU5TKKVuaijBq4aHi84c.png" />

<img width="680" height="470" alt="Total Timing Residual" src="https://github.com/user-attachments/assets/kVvPsh5Eacevib2nRFpn7.png" />

While the residuals are structurally correct, waveform fidelity is compromised due to phase inaccuracies.

---

## 🧪 RK4 Ground Truth Comparison

To validate the PINN outputs, a Runge-Kutta 4th order integrator was implemented for both ω(t) and φ(t). The RK4 evolution served as a ground truth reference for the PINN predictions.

- ω(t) RK4 evolution matched analytic 2PN and Newtonian models.
- φ(t) RK4 evolution was used to compare against PINN phase recovery, highlighting the misalignment.

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

## 📁 File Structure

