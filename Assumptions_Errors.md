Assumptions made: 

SMBHB comprises 2 SMBHs, of 10^9 ☉ each, and 0.1pc apart
omg_a is calculated using Kepler's relation (for omega at t_0) and initialised as 3.03* 1e-9 Hz (valid, in the nHz range)
nu = 1 (as instructed)
Used a PN expansion for tau(t) to convert omega(tau) into omega(t) to plot: tau = 1.0 - ((256.0 / 5.0) * omega_a * ((G * m_chirp) / (c**3))**(5.0 / 3.0) * t), as found online (by leading order ON expansion)


Errors:

Analytic solution produces the same values of tau and omega(t) for all values of t. Maybe the difference is too negligible to count? Code produces a straight line graph
PINN converges very early (<200 epochs) - notably with the residual losses. Possible error in learning rate/weight assignment?

Output and plots:
tau tensor:  tensor([[1.],
omg:  tensor([[3.0300e-09],

# Training Loss Progress

This table shows the progression of the total loss, boundary condition (BC) loss, and residual loss over different training epochs.

| Epoch | Total Loss | BC Loss | Residual Loss |
|-------|------------|---------|---------------|
| 0     | 0.00170    | 0.00297 | 0.00147       |
| 100   | 0.00002    | 0.00023 | 0.00000       |
| 200   | 0.00000    | 0.00000 | 0.00000       |
| 300   | 0.00000    | 0.00000 | 0.00000       |
| 400   | 0.00000    | 0.00000 | 0.00000       |
| 500   | 0.00000    | 0.00000 | 0.00000       |
| 600   | 0.00000    | 0.00000 | 0.00000       |
| 700   | 0.00000    | 0.00000 | 0.00000       |
| 800   | 0.00000    | 0.00000 | 0.00000       |
| 900   | 0.00000    | 0.00000 | 0.00000       |
| 1000  | 0.00000    | 0.00000 | 0.00000       |
| 1100  | 0.00000    | 0.00000 | 0.00000       |
| 1200  | 0.00000    | 0.00000 | 0.00000       |
| 1300  | 0.00000    | 0.00000 | 0.00000       |
| 1400  | 0.00000    | 0.00000 | 0.00000       |
| 1500  | 0.00000    | 0.00000 | 0.00000       |
| 1600  | 0.00000    | 0.00000 | 0.00000       |
| 1700  | 0.00000    | 0.00000 | 0.00000       |
| 1800  | 0.00000    | 0.00000 | 0.00000       |
| 1900  | 0.00000    | 0.00000 | 0.00000       |

<img width="565" height="455" alt="download" src="https://github.com/user-attachments/assets/6711446f-9ba7-46aa-8376-d849892b79c3" />

