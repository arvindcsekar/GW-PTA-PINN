import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

torch.manual_seed(42)
np.random.seed(42)
G = torch.tensor(6.6743e-11, dtype=torch.float64) #big G constant
c = torch.tensor(3e8, dtype=torch.float64) #light
m = 1e9*1.989e30
m_chirp = torch.tensor((m**2)**(3/5) / ((2 * m )**(1/5)), dtype=torch.float64)  # chirp mass
m_total = torch.tensor(2*m, dtype=torch.float64) #total mass
nu = torch.tensor(1.0, dtype=torch.float64) #stated
eta = torch.tensor(0.25, dtype=torch.float64) #assuming both SMBHs of identical mass (1e9 solar masses)
chi = torch.tensor(0.5, dtype=torch.float64) #high spin SMBH assumed (equal, aligned spin)
Q_15 = 19/3*chi*eta -113/12*chi +4*np.pi #aligned spin case, 3rd term (delta) vanishes
omg_a = 3.03 * 1e-9 #from Kepler relation

class PINN(nn.Module):
  def __init__(self):
    super(PINN, self).__init__()
    self.net = nn.Sequential(
      nn.Linear(1, 64), #input 1 neuron (time), then 32 neurons in each hidden layer
      nn.Tanh(), #activation function
      nn.Linear(64, 64),
      nn.Tanh(),
      nn.Linear(64, 64),
      nn.Tanh(),
      nn.Linear(64, 64),
      nn.Tanh(),
      nn.Linear(64, 1) #output layer, 1 neuron (solution, omega)
    )
    self.double() #Ensure the model parameters are of type float64


  def forward(self, x):
    return torch.exp(self.net(x)) 

num_points = 10000
t_end = 10 * 365.25 * 24 * 3600  #10 years in seconds
t_0 = 0
t = torch.rand(num_points, 1, dtype=torch.float64) #b/w t=0 and coalescence time, NORMALISED
input = t.requires_grad_() #tells PyTorch to track gradients wrt inputs
omg_enforced = torch.tensor(3.03* 1e-9, dtype=torch.float64) #kepler's relation for typical omg value at t=0, with M = 1e9 and R = 0.1pc

t_bc2 = torch.tensor([0.0], dtype=torch.float64).view(-1, 1)
loss_MSE = nn.MSELoss() #mean squared


#weights
w1 = 1e6
w2 = 7

def compute_residual_loss(model, num_points):
  t_residual = torch.rand(num_points, 1, dtype=torch.float64) #normalised
  t_pred = t_residual.requires_grad_()
  omg_pred = model(t_pred)

  omg_positive = omg_pred
  t_physical = t_pred * t_end
  tN_omg = (G * m_chirp * omg_positive / (c**3)) ** (5/3)
  tm_omg_2by3 = (G * m_total * omg_positive / (c**3)) ** (2/3)
  tm_omg_1 = (G * m_total * omg_positive / (c**3))  # For Q_15 term
  tm_omg_4by3 = (G * m_total * omg_positive / (c**3)) ** (4/3)

  domg_dt = torch.autograd.grad(omg_positive, t_pred, grad_outputs=torch.ones_like(omg_positive),create_graph=True)[0]/t_end
  rhs = rhs = ((96/5) * omg_positive**2 * tN_omg * (
    1
    + (-743/336 - (11/4)*eta) * tm_omg_2by3 * nu
    + Q_15 * tm_omg_1 * nu**(3/2)
    + (34103/18144 + (13661/2016)*eta + (59/18)*eta**2) * tm_omg_4by3 * nu**2
))
  residual = domg_dt - rhs #LHS - RHS of provided equation
  residual_loss = torch.mean(residual**2) #mean squared
  return residual_loss

model = PINN()
optimiser = optim.Adam(model.parameters(), lr=1e-2) #Adam optimiser w learning rate

epoch_num = 3000
for epoch in range(epoch_num):
  optimiser.zero_grad() #clears old grads

  omg_pred_bc2 = model(t_bc2) #runs BC sampling point through PINN
  omg_target_bc2 = omg_enforced.view(-1,1) #omg = enforced value at t_0

  bc_loss = loss_MSE(omg_pred_bc2, omg_target_bc2)

  residual_loss = compute_residual_loss(model, num_points)

  total_loss = w1 * bc_loss + w2 * residual_loss
  total_loss.backward() #back propagation, calculates how much weights should change
  torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

  optimiser.step() #updates weights using gradients computed in back propagation

  if epoch % 300 == 0:
        print(f"Epoch {epoch} | Total: {total_loss.item():.5f} | BC: {bc_loss.item():.5f} | Residual: {residual_loss.item():.5f}")

t_values = np.linspace(0, 1, 1000) #normalised
t_tensor = torch.tensor(t_values, dtype=torch.float64).view(-1, 1)
t_tensor_upscale = t_tensor * t_end #upscale
omg_tensor = model(t_tensor)
tau = np.zeros(1000)
omega_arr = np.zeros(1000)
print("Sample predicted omega (Hz):", model(torch.tensor([[0.0], [0.5], [1.0]], dtype=torch.float64)).detach().numpy())

tm = (G * m_total * omg_a) / (c**3)
with torch.no_grad():
  tau_tensor = 1.0 - ((256.0 / 5.0) * omg_a * ((G * m_chirp) / (c**3))**(5.0/3.0) * t_tensor_upscale)
  tau_tensor = torch.clamp(tau_tensor, min=1e-12)  # avoid division by zero or negative roots
  omega = omg_a * (((371/128)*eta**2 + (56975/16128)*eta + (1855099/903168)) * tm**(4/3) * nu**2 / tau_tensor**(7/8) + ((-3058673/1354752 - (617/192)*eta**2 - (5429/1344)*eta)) * tm**(4/3) * nu**2 / tau_tensor**(11/8) + ((-605/192)*eta**2 - (40865/8064)*eta - (2760245/1354752)) * tm**(4/3) * nu**2 / tau_tensor**(13/8) + ((1331/384)*eta**2 + (89903/16128)*eta + (6072539/2709504)) * tm**(4/3) * nu**2 / tau_tensor**(19/8) + (3/5) * tm * nu**(3/2) * (1 / tau_tensor**(11/8) - 1 / tau_tensor**(3/4)) * Q_15 + ( (11/8)*eta + (743/672) ) * tm**(2/3) * nu / tau_tensor**(5/8) - ( (11/8)*eta + (743/672) ) * tm**(2/3) * nu / tau_tensor**(11/8) + 1 / tau_tensor**(3/8))
  tau = tau_tensor.detach().numpy().flatten()
  omega_arr = omega.detach().numpy().flatten()

t_np = (t_tensor_upscale / (365.25 * 24 * 3600)).detach().numpy() #converting back
omg_np = omg_tensor.detach().numpy()

plt.plot(t_np, omg_np, label="Predicted Evolution of SMBHB Orbital Frequency with Time", color="orange")
plt.title("Evolution of SMBHB Orbital Frequency with Time")
plt.xlabel('Time (years)')
plt.ylabel('Orbital Frequency ω (Hz)')
plt.legend()

plt.grid(True)
plt.show()
