import numpy as np
import matplotlib.pyplot as plt

G = 6.6743e-11 #big G constant
c = 3e8 #light
m = 1e9*1.989e30
m_chirp = (m**2)**(3/5) / ((2 * m)**(1/5))  # chirp mass
m_total = 2*m #total mass
nu = 1 #stated
eta = 0.25 #assuming both SMBHs of identical mass (1e9 solar masses)
chi = 0.5 #high spin SMBH assumed (equal, aligned spin)
Q_15 = 19/3*chi*eta -113/12*chi +4*np.pi #aligned spin case, 3rd term (delta) vanishes
omg_a = 3.03 * 1e-9 #from Kepler relation
omg_positive = 3.03 * 1e-9

def f(t, omg):
  tN_omg = G * m_chirp * omg / c**3 #post-Newtonian relation
  tm_omg = G * m_total * omg / c**3 #post-Newtonian relation
  return ((96/5) * omg**2 * tN_omg**(5/3) * (1 + (-743/336 - (11/4)*eta) * tm_omg**(2/3) * nu + Q_15 * tm_omg * nu**(3/2) + (34103/18144 + (13661/2016)*eta + (59/18)*eta**2) * tm_omg**(4/3) * nu**2))

def rk4(f, omg_0, t_f, t_0, h):
  n_steps = int((t_f - t_0)/h + 1) #define number of steps
  t = np.linspace(t_0, t_f, n_steps) #uses a in the linspace
  omg = np.zeros(n_steps) #initialises empty omg array
  omg[0] = omg_0 #omg_0 at time t initialised

  for i in range(1, n_steps):
    k1 = h * f(t[i-1], omg[i-1])  #initial slope
    k2 = h * f(t[i-1] + h/2, omg[i-1] + k1/2)  #slope at midpoint using k1
    k3 = h * f(t[i-1] + h/2, omg[i-1] + k2/2)  #slope at midpoint using k2
    k4 = h * f(t[i-1] + h, omg[i-1] + k3)      #end slope using k3

    omg[i] = omg[i-1] + (k1 + 2*k2 + 2*k3 + k4)/6  #combine weighted slopes

  return t, omg

omg_0 = 3.03 * 1e-9
t_0 = 0
t_f = 10 * 365.25 * 24 * 3600
h = 10000

t, omg = rk4(f, omg_0, t_f, t_0, h)

t_years = t / (365.25 * 24 * 3600)

plt.plot(t_years, omg)
plt.xlabel('t')
plt.ylabel('ω(t)')
plt.title('RK4')
plt.legend()
plt.grid(True)
plt.show()
