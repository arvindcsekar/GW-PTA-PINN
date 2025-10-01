# Using Physics-Informed Neural Networks (PINNs) to Model Parameter Evolution in a Spin-Aligned Post-Newtonian Non-Eccentric Supermassive Black Hole (SMBHB) System

## Repository Structure

This repository contains all code, notebooks, and documentation developed for modelling the relativistic evolution of SMBHB systems using Physics-Informed Neural Networks (PINNs). Each file is modular and serves a distinct purpose in the overall pipeline.

### `PINN_GW_multiparam_pulsar.ipynb`
Main notebook implementing the full PINN architecture for ω(t) and φ(t), including the pulsar term. Contains training loops, residual definitions, waveform recovery, and visualisations. The pulsar term is structurally included but remains physically inaccurate due to phase misalignment.

### `README.md`
Project documentation detailing the objectives, methodology, results, limitations, and architectural commentary. Includes validation plots, analytic comparisons, and a full breakdown of the PINN framework.

### `RK4_Analytic_GW.ipynb`
Notebook implementing Runge-Kutta 4th order integration for ω(t) and φ(t), used as ground truth for validating PINN outputs. Includes comparisons with analytic 2PN and Newtonian models, and overlays with PINN predictions.

### `current_multiparam_PTA.py`
Script version of the full PINN pipeline, including ω-PINN, φ-PINN, and pulsar timing residuals. Designed for reproducible execution outside the notebook environment. Contains all architectural components and inference logic.

### `working_multiparam.ipynb`
identical, with minor parameter changes


## Overview

This repository contains a PINN framework for modeling the orbital evolution of Supermassive Black Hole Binaries (SMBHBs), focusing on the time evolution of orbital frequency ω(t) and phase φ(t) under relativistic conditions, incorporating spin-aligned binary parameters and pulsar timing effects. The PINN takes 5 input neurons - which are later hardcoded in the sampling - of BH spin parameters, masses, and time itself. The PINN architecture was designed to reproduce post-Newtonian (PN) dynamics and enable waveform reconstruction for pulsar timing arrays (PTAs) using the forward problem approach.

---

## Orbital Frequency PINN (ω-PINN)

### Implementation Summary

The ω-PINN was designed to solve the relativistic evolution of orbital frequency for circular spin-aligned SMBHB systems. The network takes normalised time, spin parameters, and normalised masses as input, and outputs ω(t) in physical units. It comprises a hard boundary condition at τ = 0 and a residual loss derived from the 2PN evolution equation, which includes spin-orbit coupling, chirp mass scaling, and post-Newtonian corrections.

The PINN was trained over a normalised time domain τ ∈ [0, 1], corresponding to a physical range from first 0 to + 10 years and then from −Δt to +10 years, where Δt is the light travel delay for a pulsar at a distance of 1 kpc. The residual loss was computed using autograd to obtain dω/dτ, scaled appropriately to physical time. The network was trained using Adam optimisation with gradient clipping and high residual weighting to enforce physical fidelity.

## RK4 Ground Truth Comparison

To validate the PINN outputs, a Runge-Kutta 4th order integrator was implemented for both ω(t) and φ(t). The RK4 evolution served as a ground truth reference for the PINN predictions - primarily for the φ-PINN which had none. The ω evolution matched analytic 2PN and Newtonian models, confirming the correctness of the PINN training. The φ evolution was used to compare against the φ-PINN output, highlighting the misalignment and guiding architectural debugging.

### Validation Against Analytic Models

The ω-PINN was validated against both Newtonian and 2PN analytic models across multiple spin configurations. The results demonstrate excellent agreement with analytic predictions, confirming the physical correctness of the learned evolution.

| Configuration | 2PN ω(t)        | Newtonian ω(t) | ω-PINN Output     |
|---------------|------------------|----------------|-------------------|
| χ₁ = χ₂ = 0   | 5.665e-7          | 5.679e-7      | 5.664e−7          |
| χ₁ = χ₂ = 0.5 | 5.648e−7         | 5.679e−7       | 5.644e−7          |
| χ₁ = χ₂ = −0.5| 5.683e−7         | 5.679e−7       | 5.683e−7          |

The figures for the above table can be seen below:

<img width="630" height="470" alt="Evolution of SMBHB Orbital Frequency with Time at 0 X1, X2" src="https://github.com/user-attachments/assets/2842e845-43cf-4d94-8034-6d285b479353" />

<img width="682" height="470" alt="5 40e-07" src="https://github.com/user-attachments/assets/ccd30249-25e9-4ba3-a87d-c11c57417650" />

<img width="685" height="470" alt="Evolution of SMBHB Orbital Frequency with Time at -0 5 x1  X2, 1e9 ml, m2" src="https://github.com/user-attachments/assets/ae7cd544-50b2-4b38-aaab-8e37b9116df2" />

---

## Orbital Phase PINN (φ-PINN)

### Motivation and Physical Context

The orbital phase φ(t) evolution is governed by the instantaneous orbital frequency ω(t), and its accurate modeling is essential for waveform reconstruction. The φ-PINN was designed to learn this evolution by solving the linked differential equation dφ/dt = ω(t), using the previously trained ω-PINN as input. The lack of a specific analytic solution for the φ-PINN led to the construction of an RK4 numerical model to be used as the ground truth. The plots can be seen below:

χ₁ = χ₂ = 0

<img width="571" height="455" alt="Evolution of SMBHB Orbital Phase" src="https://github.com/user-attachments/assets/d5b3a357-b58b-470c-80be-138c12c5cccb" />
<img width="771" height="455" alt="download-2" src="https://github.com/user-attachments/assets/7faf87a9-2113-4dfc-9729-d5e6f55a8bc6" />

χ₁ = χ₂ = 0.5

<img width="571" height="455" alt="Evolution of SMBHB Orbital Phase" src="https://github.com/user-attachments/assets/0177e890-29d3-41d9-b78e-d83630b9bece" />
<img width="787" height="455" alt="time (in years)" src="https://github.com/user-attachments/assets/fe090a13-5094-40d1-9b98-2f66edd7121f" />

χ₁ = χ₂ = -0.5

<img width="571" height="455" alt="Evolution of SMBHB Orbital Phase" src="https://github.com/user-attachments/assets/44e7353e-6a3a-40c0-a6b2-b2e88bc33ead" />
<img width="793" height="455" alt="time (in years)" src="https://github.com/user-attachments/assets/ed51c919-2f6d-4f44-a54d-3964cc26b3b6" />


### Results and Observations

The ω-PINN and φ-PINN architecture were successfully implemented and trained over a multiparameter system, and the output was then extended to be evaluated at both Earth and Pulsar epochs. The structure of the phase evolution is present, but the slope and scale are incorrect, indicating training instability and architectural limitations.

Attempts to correct the output using bias terms, output scaling, and loss rebalancing were unsuccessful within the project timeline. The residual logic remains valid, and the architecture is extensible for future refinement.

---

## Pulsar Timing Residuals

### Physical Setup

To model pulsar timing residuals, the Earth and Pulsar terms were evaluated using a light travel delay Δt corresponding to a pulsar at 1 kpc. ω(t) and φ(t) were evaluated at both epochs using normalised time inputs, and the gravitational wave polarisations h₊ and h× were computed using standard waveform expressions.

The timing residuals were derived from the difference in strain between Earth and Pulsar epochs, scaled by a geometric factor based on the angle between the pulsar and the GW source.

### Results

Residuals were computed for h₊, h×, and RMS, and plotted over the Earth time domain. While the residuals are structurally correct, waveform fidelity is compromised due to phase inaccuracies in φ(t).

<img width="630" height="470" alt="download-1" src="https://github.com/user-attachments/assets/ac784e72-8835-4e69-b0d0-8de7b400b736" />
<img width="630" height="470" alt="download" src="https://github.com/user-attachments/assets/2f95247c-5c2e-4c5a-b361-ed3918bdd6a3" />
<img width="630" height="470" alt="download-2" src="https://github.com/user-attachments/assets/d4074eac-13c8-4466-a77e-4f179136fff5" />

---

## Architectural Commentary

This project demonstrates the power and limitations of PINNs in modeling relativistic astrophysical systems. The ω-PINN successfully captured the dynamics of SMBHB orbital frequency evolution, including spin-aligned effects and post-Newtonian corrections. The φ-PINN, while structurally complete, revealed the sensitivity of residual-based training to output scaling and initialisation. The integration of pulsar timing effects and waveform recovery showcases the extensibility of the architecture, even if the final results remain imperfect.

---

## Limitations

- φ-PINN output remains misaligned despite correct residual logic.
- Pulsar timing residuals are structurally correct but lack waveform fidelity due to frequency and phase inaccuracies.
- Further refinement of architecture and training strategy is required for full waveform recovery.

