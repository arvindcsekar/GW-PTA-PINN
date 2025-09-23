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
        self.register_buffer('scale',  torch.tensor(1e-6, dtype=torch.float64))
        self.double()

    def forward(self, x):
        z = self.net(x)  # unconstrained
        f = torch.nn.functional.softplus(z) * self.scale  # positive variation
        t = x[:, 0:1]  #normalised time in [0,1]
        omega = self.omega0 + t * f  #hard-enforce BC: omega(tau=0) = omega0
        return omega

def compute_residual_loss(model, num_points):
    t_res = torch.rand(num_points, 1, dtype=torch.float64)
    #need to hard code chi and masses here so network learns exact solution
    chi_1_res = torch.full((num_points, 1), -0.5, dtype=torch.float64)
    chi_2_res = torch.full((num_points, 1), -0.5, dtype=torch.float64)


    m1_phys = torch.full((num_points, 1), 1e9 * M_sun_SI, dtype=torch.float64)
    m2_phys = torch.full((num_points, 1), 1e9 * M_sun_SI, dtype=torch.float64)


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
    domega_dtau = grads[:, 0:1]  #derivative wrt tau (first column)

    residual = domega_dtau - (t_end_torch * RHS_t)
    residual_loss = torch.mean(residual**2)
    return residual_loss

num_points = 30000
epoch_num = 1000

model = PINN()
optimiser = optim.Adam(model.parameters(), lr=1.7e-4)
loss_MSE = nn.MSELoss()

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
    w1 = 1
    w2 = 1e8

    total_loss = w1 * bc_loss + w2 * residual_loss
    total_loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.2) #adjust norm
    optimiser.step()

    loss_arr.append(total_loss.item())

    if epoch % 100 == 0:
        print(f"Epoch {epoch:4d} | Total: {total_loss.item():.6e} | BC: {bc_loss.item():.6e} | Residual: {residual_loss.item():.6e}")

t_values = np.linspace(0.0, 1.0, 1000)
t_tensor = torch.tensor(t_values, dtype=torch.float64).view(-1, 1)

sample_chi1 = torch.full_like(t_tensor, -0.5, dtype=torch.float64)
sample_chi2 = torch.full_like(t_tensor, -0.5, dtype=torch.float64)
sample_m1_norm = torch.ones_like(t_tensor)
sample_m2_norm = torch.ones_like(t_tensor)

inference_inputs = torch.cat([t_tensor, sample_chi1, sample_chi2, sample_m1_norm, sample_m2_norm], dim=1)
omega_tensor = model(inference_inputs) #physical rad/s

t_years = (t_tensor.detach().numpy() * t_end) / u.yr.to(u.s)
omega_np = omega_tensor.detach().numpy()
print(omega_np[0])
print(omega_np[-1])

plt.figure()
plt.plot(t_years, omega_np, label="Predicted Evolution of SMBHB Orbital Frequency with Time", color="orange")
plt.title("Evolution of SMBHB Orbital Frequency with Time")
plt.xlabel('Time (years)')
plt.ylabel('Orbital Frequency ω (rad/s)')
plt.gca().yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2e'))
plt.legend()
plt.tight_layout()

GMsun = ac.GM_sun.value
c = ac.c.value
dsun = GMsun/(c**2)
tsun = GMsun/(c**3)
pc = ac.pc.value
yr = (365.25)*(24)*(60)*(60)

def get_omg(t,t_a,omg_a,m_c,eta,nu,chi_A,chi_S):
    m=m_c/eta**(3/5)
    tN_ = m_c*omg_a*tsun
    tm_ = m*omg_a*tsun
    tmta_ = t-t_a
    delta=np.sqrt(1-4*eta)
    tau = 1-256/5*omg_a*tN_**(5/3)*tmta_
    Q15=(-113/12*chi_A*delta+19/3*chi_S*eta-113/12*chi_S+4*np.pi)
    ww=(omg_a*(((371/128*eta**2+56975/16128*eta+1855099/903168)/tau**(7/8)+(-3058673/
    1354752-617/192*eta**2-5429/1344*eta)/tau**(11/8)+(-605/192*eta**2-40865/8064*eta-\
    2760245/1354752)/tau**(13/8)+(1331/384*eta**2+89903/16128*eta+6072539/2709504)/
    tau**(19/8))*tm_**(4/3)*nu**2+(3/5/tau**(11/8)*Q15-3/5/tau**(3/4)*Q15)*tm_*nu**(3/2)+
    ((11/8*eta+743/672)/tau**(5/8)+(-11/8*eta-743/672)/tau**(11/8))*tm_**(2/3)*nu+1/
    tau**(3/8)))
    return ww

t=np.linspace(0,10,100)*yr
t_a=0
omg_a= 5e-7
m_c=(1e9**2)**(3/5) / (2*1e9)**(1/5)
eta=1/4
nu=1
chi1 = -0.5
chi2 = -0.5
chi_A = 0.5*(chi1-chi2)
chi_S = 0.5*(chi1+chi2)
#chi_A ranges from (-0.5,0.5)
#chi_S ranges from (0,1)

omg_arr=get_omg(t,t_a,omg_a,m_c,eta,nu,chi_A,chi_S)
omg_arr_Q=get_omg(t,t_a,omg_a,m_c,eta,0,chi_A,chi_S)

plt.plot(t/yr,omg_arr)
plt.plot(t/yr,omg_arr_Q)
plt.legend(["ω-PINN", "2PN Spin", "Newtonian"])
plt.title("Evolution of SMBHB Orbital Frequency with Time at -0.5 χ1, χ2, 1e9 m1, m2")
plt.xlabel("time (in years)")
plt.ylabel(r"$\omega$")
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

class PINN2(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )
        self.double()

    def forward(self, t):
        return torch.exp(self.net(t))


def compute_residual_loss_phi(model2, t_batch, model):
    t_pred = t_batch.requires_grad_()
    phi_pred = model2(t_pred)
    dphi_dtau = torch.autograd.grad(phi_pred, t_pred, grad_outputs=torch.ones_like(phi_pred), create_graph=True)[0]

    chi1 = torch.zeros_like(t_pred)
    chi2 = torch.zeros_like(t_pred)
    m1_norm = torch.ones_like(t_pred)
    m2_norm = torch.ones_like(t_pred)

    omega_here = model(torch.cat([t_pred, chi1, chi2, m1_norm, m2_norm], dim=1))

    residual = dphi_dtau - (t_end_torch * omega_here)
    return torch.mean(residual**2)


phi_enforced = torch.tensor([[0.0]], dtype=torch.float64)
model2 = PINN2()
opt_phi = optim.Adam(model2.parameters(), lr=1e-3)
loss_arr_phi = []
w1_phi = 1e11
w2_phi = 1e7

tau_pool = torch.linspace(0.0, 1.0, steps=num_points, dtype=torch.float64).view(-1, 1)

for epoch in range(epoch_num):
    opt_phi.zero_grad()

    bc_inputs2 = torch.tensor([[0.0]], dtype=torch.float64)
    phi_pred_bc = model2(bc_inputs2)
    bc_loss_phi = loss_MSE(phi_pred_bc, phi_enforced)

    idx = torch.randperm(num_points)[:1024]
    tau_batch = tau_pool[idx]
    residual_loss_phi = compute_residual_loss_phi(model2, tau_batch, model)

    total_loss_phi = w1_phi * bc_loss_phi + w2_phi * residual_loss_phi
    total_loss_phi.backward()
    torch.nn.utils.clip_grad_norm_(model2.parameters(), 1.0)
    opt_phi.step()

    loss_arr_phi.append(total_loss_phi.item())
    if epoch % 100 == 0:
        print(f"[φ] Epoch {epoch} | Total {total_loss_phi.item():.3e} | BC {bc_loss_phi.item():.3e} | Resid {residual_loss_phi.item():.3e}")

t_vals = torch.linspace(0, 1, 1000, dtype=torch.float64).view(-1,1)
t_phys = t_vals * t_end_torch
phi_pred = model2(t_vals).detach().numpy()
t_years_phi = (t_phys / u.yr.to(u.s)).detach().numpy()

plt.figure()
plt.plot(t_years_phi, phi_pred, label="φ(t)", color="blue")
plt.xlabel("Time (years)")
plt.ylabel("Orbital Phase φ (rad)")
plt.title("Evolution of SMBHB Orbital Phase")
plt.grid()
plt.legend()
plt.show()

plt.figure()
plt.plot(range(epoch_num), loss_arr_phi, label = "Evolution of φ Loss with Epoch", color="Red")
plt.title("Evolution of Loss with Epoch (φ)")
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid()
plt.show()

