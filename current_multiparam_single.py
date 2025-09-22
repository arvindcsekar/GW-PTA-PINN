import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import torch
import torch.nn as nn
import torch.optim as optim
from astropy import units as u
from astropy import constants as ac

torch.manual_seed(42)
np.random.seed(42)

G_SI = ac.G.value #m^3 kg^-1 s^-2
c_SI = ac.c.value #m s^-1
M_sun_SI = ac.M_sun.value #kg

G = torch.tensor(G_SI, dtype=torch.float64)
c = torch.tensor(c_SI, dtype=torch.float64)
t_end = (10 * u.yr).to(u.s).value
t_end_torch = torch.tensor(t_end, dtype=torch.float64)

#normalisation: scale masses by 1e9 Msun
M_scale = 1e9 * M_sun_SI

class PINN(nn.Module):
    def __init__(self):
        super(PINN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(5, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1, bias=True)
        )
        self.register_buffer('omega0', torch.tensor(5e-7, dtype=torch.float64))  #baseline near BC
        self.register_buffer('scale',  torch.tensor(1e-6, dtype=torch.float64))  # allows ~1e-6 variation
        self.double()

    def forward(self, x):
        z = self.net(x)
        omega = self.omega0 + torch.nn.functional.softplus(z) * self.scale
        return omega

def compute_residual_loss(model, num_points):
    # tau in [0,1]
    t_res = torch.rand(num_points, 1, dtype=torch.float64)
    # FIX 1: spins and masses fixed to target system
    chi_1_res = torch.zeros(num_points, 1, dtype=torch.float64)
    chi_2_res = torch.zeros(num_points, 1, dtype=torch.float64)
    m1_phys = (1e9 * M_sun_SI) * torch.ones(num_points, 1, dtype=torch.float64)
    m2_phys = (1e9 * M_sun_SI) * torch.ones(num_points, 1, dtype=torch.float64)
    m1_norm = m1_phys / M_scale
    m2_norm = m2_phys / M_scale

    batch_inputs = torch.cat([t_res, chi_1_res, chi_2_res, m1_norm, m2_norm], dim=1).requires_grad_()

    chi_A = 0.5 * (chi_1_res - chi_2_res)
    chi_S = 0.5 * (chi_1_res + chi_2_res)

    m_total = m1_phys + m2_phys
    eta = (m1_phys * m2_phys) / (m_total**2)
    m_chirp = (eta ** (3.0/5.0)) * m_total

    omega = model(batch_inputs)
    delta = torch.sqrt(torch.clamp(1.0 - 4.0 * eta, min=0.0))
    Q_15 = (-113.0/12.0) * chi_A * delta + (19.0/3.0) * chi_S * eta - (113.0/12.0) * chi_S + 4.0 * torch.pi

    tN_omg = (G * m_chirp * omega) / (c**3)  #dimensionless
    tm_omg = (G * m_total * omega) / (c**3)  #dimensionless

    RHS_t = (96.0/5.0) * (
        1.0
        + ( (59.0/18.0) * (eta**2) + (13661.0/2016.0) * eta + (34103.0/18144.0) ) * (tm_omg ** (4.0/3.0))
        + tm_omg * (Q_15)
        + ( -11.0/4.0 * eta - 743.0/336.0 ) * (tm_omg ** (2.0/3.0))
    ) * (omega**2) * (tN_omg ** (5.0/3.0))

    grads = torch.autograd.grad(
        outputs=omega,
        inputs=batch_inputs,
        grad_outputs=torch.ones_like(omega),
        create_graph=True,
    )[0]
    domega_dtau = grads[:, 0:1]  #derivative wrt t (first column)

    residual = domega_dtau - (t_end_torch * RHS_t)
    residual_loss = torch.mean(residual**2)
    return residual_loss

num_points = 30000
epoch_num = 3000

model = PINN()
optimiser = optim.Adam(model.parameters(), lr=1.55e-4)
loss_MSE = nn.MSELoss()

w1 = 1e15 #4e12
w2 = 3e11

loss_arr = []

for epoch in range(epoch_num):
    optimiser.zero_grad()

    t_bc = torch.tensor([[0.0]], dtype=torch.float64)
    chi0 = torch.zeros_like(t_bc)
    m0_norm = torch.ones_like(t_bc)

    bc_inputs = torch.cat([t_bc, chi0, chi0, m0_norm, m0_norm], dim=1)
    omg_pred_bc = model(bc_inputs)
    omg_target_bc = torch.tensor([[5e-7]], dtype=torch.float64)
    bc_loss = loss_MSE(omg_pred_bc, omg_target_bc)

    residual_loss = compute_residual_loss(model, num_points)

    total_loss = w1 * bc_loss + w2 * residual_loss
    total_loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.2) #adjust norm
    optimiser.step()

    loss_arr.append(total_loss.item())

    if epoch % 100 == 0:
        print(f"Epoch {epoch:4d} | Total: {total_loss.item():.6e} | BC: {bc_loss.item():.6e} | Residual: {residual_loss.item():.6e}")

t_values = np.linspace(0.0, 1.0, 1000)
t_tensor = torch.tensor(t_values, dtype=torch.float64).view(-1, 1)

sample_chi1 = torch.zeros_like(t_tensor)
sample_chi2 = torch.zeros_like(t_tensor)
sample_m1_norm = torch.ones_like(t_tensor)
sample_m2_norm = torch.ones_like(t_tensor)

inference_inputs = torch.cat([t_tensor, sample_chi1, sample_chi2, sample_m1_norm, sample_m2_norm], dim=1)
omega_tensor = model(inference_inputs) #physical rad/s

t_years = (t_tensor.detach().numpy() * t_end) / u.yr.to(u.s)
omega_np = omega_tensor.detach().numpy()

plt.figure()
plt.plot(t_years, omega_np, label="Predicted Evolution of SMBHB Orbital Frequency with Time", color="orange")
plt.title("Evolution of SMBHB Orbital Frequency with Time")
plt.xlabel('Time (years)')
plt.ylabel('Orbital Frequency ω (rad/s)')
plt.gca().yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2e'))
plt.legend()
plt.tight_layout()
plt.show()

plt.figure()
plt.plot(range(epoch_num), loss_arr, label="Evolution of Loss with Epoch", color="red")
plt.title("Evolution of Loss with Epoch")
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.yscale('log')
plt.legend()
plt.tight_layout()
plt.show()

print(omega_np[0])
print(omega_np[999])

