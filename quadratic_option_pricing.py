import math
from scipy.optimize import fsolve
import matplotlib.pyplot as plt

from pricing_common import (
    payoff_by_sabr_mc_without_discout,
    sigma_normal,
    v_qc,
)


def main():
    t_0 = 0
    beta = 0.9
    r_0 = 0.05
    strikes = [0.044, 0.046, 0.048, 0.050, 0.052, 0.054, 0.056, 0.058]
    # strikes = [0.058]
    tau_0 = 0.5
    tau_1 = 1
    rho = -0.5
    nu = 0.5
    alpha = 0.1
    q = 1
    num_grid = 512
    num_mc_path = 10**6

    tau = 2*q*tau_0+tau_1

    def payoff_function(forward_rate, strike):
        return (max(forward_rate-strike, 0))**2

    payoff_mc = payoff_by_sabr_mc_without_discout(
        t_0, beta, r_0, tau_0, tau_1, strikes, payoff_function, rho, nu, alpha, q, num_mc_path, num_grid)

    ##################################################################
    # for backward-looking parameters
    gamma = tau*(2*tau**3+tau_1**3+(4*q**2-2*q)*tau_0**3+6*q*tau_0**2*tau_1)/(4*q+3)/(2*q+1)\
        + 3*q*rho**2*(tau_1-tau_0)**2*(3*tau**2-tau_1**2+5*q *
                                       tau_0**2+4*tau_0*tau_1)/(4*q+3)/(3*q+2)**2
    nu_hat = math.sqrt(nu**2*gamma*(2*q+1)/(tau**3*tau_1))
    rho_hat = rho*(3*tau**2+2*q*tau_0**2+tau_1**2)/math.sqrt(gamma)/(6*q+4)
    H = nu**2*(tau**2+2*q*tau_0**2+tau_1**2)/(2*tau_1*tau*(q+1))-nu_hat**2
    alpha_hat = math.sqrt((alpha**2/(2*q+1))*tau/tau_1*math.exp(1/2*H*tau_1))
    ##################################################################

    payoff_approximation = list()
    for strike in strikes:
        payoff_approximation.append(v_qc(
            t_0, tau_1, strike, r_0, alpha_hat, beta, rho_hat, nu_hat))

    sigma_mc_solution = list()
    sigma_approximation_solution = list()
    for i in range(0, len(strikes)):
        def func(sigma): return sigma_normal(
            t_0, tau_1, strikes[i], r_0, sigma)-payoff_mc[i]
        sigma_initial_guess = 0.1
        sigma_mc_solution.append(fsolve(func, sigma_initial_guess))

        def func(sigma): return sigma_normal(
            t_0, tau_1, strikes[i], r_0, sigma)-payoff_approximation[i]
        sigma_initial_guess = 0.1
        sigma_approximation_solution.append(fsolve(func, sigma_initial_guess))

    import winsound
    frequency = 500
    duration = 1000
    winsound.Beep(frequency, duration)
    import numpy as np
    plt.plot(strikes, np.abs(payoff_approximation-payoff_mc), label='diff')
    plt.plot(strikes, payoff_approximation, label='MC')
    plt.plot(strikes, payoff_mc, label='all effective')
    plt.xlabel("K")
    plt.grid()
    plt.legend()
    plt.show()

    print("end")


if __name__ == "__main__":
    main()
