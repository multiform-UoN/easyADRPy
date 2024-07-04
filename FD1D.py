import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

TOL = 1e-6

def divergence(f, x, dim):
    """Redefines np.gradient for polar coordinates

    Args:
        f: Array representing the function values.
        x: Array representing the grid points.
        dim: Dimension of the non-radial coordinates

    Returns:
        dfdx: Array representing the gradient of the function.
    """
    dfdx = np.gradient(f, x, edge_order=2)
    if x[0]<TOL:
        dfdx[1:] += dim*f[1:]/x[1:]
        dfdx[0] += dim*dfdx[0]
    else:
        dfdx += dim*f/x
    # print(dfdx)
    return dfdx

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
    
    # Compute the first spatial derivative 
    rhs = np.gradient(u, x, edge_order=2)

    # Calculate the flux
    flux = -p['D'](u, t, x)*rhs + p['V'](u, t, x)*u  # Time-dependent D and V

    # Compute the right-hand side  
    rhs = -divergence(flux, x, p['polar']) + p['R'](u, t, x)  # Time-dependent R

    # Impose boundary conditions
    dxL = x[1] - x[0]
    dxR = x[-1] - x[-2]
    rhs[0] = rhs[1] * (p['bc_left']['d'](t) - p['bc_left']['df'](u[0],t)*dxL )/(p['bc_left']['d'](t) - dxL*p['bc_left']['k'](t))
    rhs[-1] = rhs[-2] * (p['bc_right']['d'](t) + p['bc_right']['df'](u[-1],t)*dxR )/(p['bc_right']['d'](t) + dxR*p['bc_right']['k'](t))

    return rhs


if __name__ == "__main__":
    # Parameters
    p = {
        # Time
        't': np.linspace(0, 1.0, 10)  # Times at which the solution is evaluated and stored
        ,
        # Grid
        # 'x': np.linspace(1e-2, 1, 50)**2  # Example non-equispaced grid
        'x': np.linspace(0, 1, 20) # Equispaced grid
        ,
        # Polar coordinates # 0: Cartesian, 1: Cylinder, 2: Spherical
        'polar': 0
        ,
        # Boundary conditions
        # d u' + k u = f(u,t)
        # NOTE: the dependences below are implemented neglecting time derivatives terms so works only for small gradients in time
        # Basically it only works for constant coefficients (f goes away when taking the time derivative)
        'bc_left': { 
            'd': lambda t: 0,   # Neumann coefficient, Can be time-dependent
            'k': lambda t: 1,   # Dirichlet coefficient, Can be time-dependent
            'df': lambda u, t: 0, # time-derivative of the non-linear forcing coefficient, Can be time-dependent
        },
        'bc_right': { 
            'd': lambda t: 1,   # Neumann coefficient, Can be time-dependent
            'k': lambda t: 0,   # Dirichlet coefficient, Can be time-dependent
            'df': lambda u, t: 0, # time derivative of the non-linear forcing coefficient, Can be time-dependent
        },
        # Initial condition
        'ic': lambda x: x>0
        ,
        # Physical parameters
        'D': lambda u, t, x: 1,   # Example time-dependent D
        'V': lambda u, t, x: 0,        # Example time-dependent V 
        'R': lambda u, t, x: x**2               # Example time-dependent R 
    }

    # Grid spacing
    x = p['x']
    t = p['t']
    
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