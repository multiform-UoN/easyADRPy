import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

def L(t, u, p):
    """PDE Operator L(u) = - D(u)*d^2u/dx^2 + V(u)*du/dx + R(u)*u.

    Args:
        t: Current time.
        u: Array representing the temperature distribution at time t.
        p: Dictionary containing physical and discretization parameters.

    Returns:
        dudt: The time derivative of the temperature distribution.
    """
    x = p['x']
    
    # Impose Dirichlet boundary conditions
    if p['bc_left']['type'] == 'dirichlet':
        u[0] = p['bc_left']['f'](t)  # Time-dependent f
    if p['bc_right']['type'] == 'dirichlet':
        u[-1] = p['bc_right']['f'](t)  # Time-dependent f

    # Compute the first spatial derivative 
    dudx = np.gradient(u, x)

    # Impose Neumann or Robin boundary conditions 
    if p['bc_left']['type'] == 'neumann':
        dudx[0] = p['bc_left']['k'](u[0], t) + p['bc_left']['f'](t)  # Time-dependent k and f
    if p['bc_right']['type'] == 'neumann':
        dudx[-1] = p['bc_right']['k'](u[-1], t) + p['bc_right']['f'](t)  # Time-dependent k and f

    # Calculate the flux
    flux = -p['D'](u, t)*dudx + p['V'](u, t)*u  # Time-dependent D and V

    # Compute the right-hand side  
    dudt = -np.gradient(flux, x) + p['R'](u, t)*u  # Time-dependent R


    return dudt


if __name__ == "__main__":
    # Parameters
    p = {
        # Grid
        'x': np.linspace(0, 1, 50)**2  # Example non-equispaced grid
        ,
        # Boundary conditions
        'bc_left': { 
            'type': 'dirichlet',
            'k': lambda u, t: 0.1,   # Can be time-dependent
            'f': lambda t: np.sin(t)  # Example time-dependent f
        },
        'bc_right': { 
            'type': 'neumann',
            'k': lambda u, t: 0.1,   # Can be time-dependent
            'f': lambda t: np.sin(t)  # Example time-dependent f
        },
        # Initial condition
        'ic': lambda x: 0*x
        ,
        # Physical parameters
        'D': lambda u, t: 1 + 0.2*np.cos(t),   # Example time-dependent D
        'V': lambda u, t: 0.1 - 0.05*t,        # Example time-dependent V 
        'R': lambda u, t: -0.1*u               # Example time-dependent R 
    }

    # Grid spacing
    x = p['x']
    
    # Time span and plot interval
    t = np.linspace(0, 1.0, 10)  # Times at which the solution is evaluated and stored

    # Solve the heat equation
    try:
        sol = solve_ivp(L, (t[0], t[-1]), p['ic'](x), method='BDF', args=(p,), t_eval=t) 

        if not sol.success:  # Check the success flag
            raise RuntimeError("solve_ivp failed: " + sol.message)

    except RuntimeError as err:
        print(f"Solution Error: {err}") 
        # Handle the error, e.g., log it, retry with different settings, etc.

    # Plotting 
    plt.figure()
    for i,tt in enumerate(t):
        plt.plot(x, sol.y[:, i], label=f't={tt:.2f}')
    plt.xlabel('x')
    plt.ylabel('Temperature')
    plt.title('Heat Diffusion')
    plt.legend()
    plt.show()