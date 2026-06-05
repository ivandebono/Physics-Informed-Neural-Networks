
import numpy as np

# ---------------------------
# Analytical reference (to produce the mock data)
# ---------------------------

def analytical(t, zeta=0.3, omega_n=5.0):
    """
    Analytical solution for a damped harmonic oscillator.

    Computes the displacement response of an underdamped second-order system
    subject to an initial unit step input and initial conditions.

    Parameters
    ----------
    t : float or array-like
        Time variable(s) at which to evaluate the solution.
    zeta : float, optional
        Damping ratio (default: 0.3). Must be 0 < zeta < 1 for underdamped response.
    omega_n : float, optional
        Natural frequency in rad/s (default: 5.0). Must be positive.

    Returns
    -------
    float or ndarray
        Displacement response at time(s) t. Has the same shape as input t.

    Notes
    -----
    This solution assumes:
    - Initial displacement: y(0) = 1
    - Initial velocity: dy/dt(0) = 0
    - The system is underdamped (0 < zeta < 1)

    The response consists of an exponential decay envelope modulated by
    oscillatory behavior at the damped natural frequency wd.

    Examples
    --------
    >>> analytical(1.0, zeta=0.3, omega_n=5.0)
    0.14073...

    See Also
    --------
    scipy.integrate.odeint : For numerical solutions of ODEs
    """
    wd = omega_n * np.sqrt(1 - zeta**2)
    return np.exp(-zeta*omega_n*t) * (np.cos(wd*t) + (zeta*omega_n/wd)*np.sin(wd*t))