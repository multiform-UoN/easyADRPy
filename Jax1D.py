import jax
import jax.numpy as jnp
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# Parameters
alpha = 0.1
L = 10.0  # Length of the domain
Nx = 100  # Number of spatial points
Tmax = 1.0  # Maximum time
dx = L / (Nx - 1)
x = jnp.linspace(0, L, Nx)

# Initial condition
u0 = jnp.exp(-(x - L/2)**2)

# Coefficients for advection, diffusion, and reaction (define as global functions)
def D(u):
    return 0.1 + 0.01 * u

def a(u):
    return 0.2 * u

def r(u):
    return 0.1 * u**2

# Define the PDE function
def pde_rhs(t, u_flat):
    u = u_flat.reshape((Nx,))
    
    # Compute the spatial derivatives using jax.numpy.gradient
    u_x = jnp.gradient(u, dx)
    
    # Compute diffusion term
    diffusion_term = jnp.gradient(D(u) * u_x, dx)
    
    # Compute advection term
    advection_term = jnp.gradient(a(u) * u, dx)
    
    # Compute reaction term
    reaction_term = r(u)
    
    # Compute the time derivative of u
    du_dt = diffusion_term - advection_term + reaction_term
    return du_dt.flatten()

# Define the Jacobian of the PDE function
jacobian_pde = jax.jacfwd(pde_rhs, argnums=1)

# Time integration
t_span = (0, Tmax)
u0_flat = u0.flatten()
t_eval = jnp.linspace(0, Tmax, 100)

sol = solve_ivp(
    pde_rhs, 
    t_span, 
    u0_flat, 
    method='Radau', 
    jac=jacobian_pde,
    t_eval=t_eval,
    vectorized=True
)

# Plot the solution
plt.figure(figsize=(8, 6))
for i, t in enumerate(sol.t[::10]):
    plt.plot(x, sol.y[:, i*10], label=f't={t:.2f}')
plt.xlabel('x')
plt.ylabel('u')
plt.title('Solution of the Nonlinear PDE with Implicit Time Stepping using solve_ivp')
plt.legend()
plt.show()
