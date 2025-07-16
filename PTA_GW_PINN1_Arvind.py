import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

G = torch.tensor(6.6743e-11, dtype=torch.float32) #big G constant
c = torch.tensor(3e8, dtype=torch.float32) #light
m_chirp = torch.tensor((1e9**2)**(3/5) / (2*1e9)**(1/5), dtype=torch.float32) #chirp mass
m_total = torch.tensor(2*1e9, dtype=torch.float32) #total mass
nu = torch.tensor(1.0, dtype=torch.float32) #stated
eta = torch.tensor(0.25, dtype=torch.float32) #assuming both SMBHs of identical mass (1e9 solar masses)
chi = torch.tensor(0.9, dtype=torch.float32) #high spin SMBH assumed
Q_15 = (((19/3) * eta - (113/12)) * chi + 4 * np.pi).float() #aligned spin case, 3rd term vanishes

class PINN(nn.Module):
  def __init__(self):
    super(PINN, self).__init__()
    self.net = nn.Sequential(
      nn.Linear(1, 32), #input 1 neuron (time), then 32 neurons in each hidden layer
      nn.Tanh(), #activation function
      nn.Linear(32, 32),
      nn.Tanh(),
      nn.Linear(32, 32),
      nn.Tanh(),
      nn.Linear(32, 1) #output layer, 1 neuron (solution, omega)
    )

  def forward(self, x):
    return self.net(x) #sets up network

num_points = 10000
t_coal = 1.05*10**15
t_0 = 0
t = torch.rand(num_points, 1)*t_coal #b/w t=0 and coalescence time
input = t.requires_grad_() #tells PyTorch to track gradients wrt inputs
omg_enforced = torch.tensor(3.03* 1e-9, dtype=torch.float32) #kepler's relation for typical omg value at t=0, with M = 1e9 and R = 0.1pc

t_bc1 = torch.tensor([t_coal], dtype=torch.float32).view(-1, 1)
t_bc2 = torch.tensor([t_0], dtype=torch.float32).view(-1, 1)
loss_MSE = nn.MSELoss() #mean squared



#weights
w1 = 0.5
w2 = 0.7

def compute_residual_loss(model, num_points):
  t_residual = torch.rand(num_points, 1)*t_coal
  t_pred = t_residual.requires_grad_()
  omg_pred = model(t_pred)

  omg_clamped = torch.clamp(omg_pred, min=1e-15, max = 1e5)  #prevents zero/negatives, CHATGPT SUGGESTED FIX
  tN_omg = G * m_chirp * omg_clamped / (c**3) #post-Newtonian relation
  tm_omg = G * m_total * omg_clamped / (c**3) #post-Newtonian relation

  domg_dt = torch.autograd.grad(omg_clamped, t_pred, grad_outputs=torch.ones_like(omg_clamped), create_graph=True, retain_graph=True)[0] #domg/dt using autograd
  residual = domg_dt - (96/5) * omg_clamped**2 * tN_omg**(5/3) * (1 + (-743/336 - (11/4)*eta) * tm_omg**(2/3) * nu + Q_15 * tm_omg * nu**(3/2) + (34103/18144 + (13661/2016)*eta + (59/18)*eta**2) * tm_omg**(4/3) * nu**2) #LHS - RHS of provided equation
  residual_loss = torch.mean(residual**2) #mean squared
  return residual_loss

model = PINN()
optimiser = optim.Adam(model.parameters(), lr=0.001) #Adam optimiser, learning rate

epoch_num = 100
for epoch in range(epoch_num):
  optimiser.zero_grad() #clears old grads

  omg_pred_bc1 = model(t_bc1) #runs BC sampling point through PINN
  omg_target_bc1 = torch.zeros_like(omg_pred_bc1) #omg = 0 at t_coal
  omg_pred_bc2 = model(t_bc2) #runs BC sampling point through PINN
  omg_target_bc2 = omg_target_bc2 = omg_enforced.view(-1,1) #omg = enforced value at t_0

  bc_loss1 = loss_MSE(omg_pred_bc1, omg_target_bc1)
  bc_loss2 = loss_MSE(omg_pred_bc2, omg_target_bc2)
  bc_loss = bc_loss1 + bc_loss2

  residual_loss = compute_residual_loss(model, num_points)

  total_loss = w1 * bc_loss + w2 * residual_loss
  total_loss.backward() #back propagation, calculates how much weights should change

  optimiser.step() #updates weights using gradients computed in back propagation

  if epoch % 1 == 0:
        print(f"Epoch {epoch} | Total: {total_loss.item():.5f} | BC: {bc_loss.item():.5f} | Residual: {residual_loss.item():.5f}")

t_values = np.linspace(0, t_coal, 1000)
t_tensor = torch.tensor(t_values, dtype=torch.float32).view(-1, 1)
omg_tensor = model(t_tensor)

t_np = t_tensor.detach().numpy()
omg_np = omg_tensor.detach().numpy()

plt.plot(t_np, omg_np, label='Predicted Evolution of SMBHB Orbital Frequency with Time')
plt.xlabel('Time (s)')
plt.ylabel('Orbital Frequency ω (Hz)')
plt.legend()
plt.show()
