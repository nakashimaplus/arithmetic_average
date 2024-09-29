import math
import numpy as np
from scipy.stats import norm

from arithmetic_average_common import (
    g_function,
    g_prime_function,
    g_double_prime_function,
    k_n_hat,
)


# derivative value at time t_0  without discout by Backward-looking SABR model with montecarlo simulation
def payoff_by_sabr_mc_without_discout(t_0, beta, r_0, tau_0, tau_1, strikes, payoff_function, rho, nu, alpha, q,
                                      num_mc_path, num_grid):

    payoff = np.zeros((len(strikes), num_mc_path))

    for mc_path_counter in range(0, num_mc_path):
        forward_rates = np.zeros((num_grid+1))
        forward_rates[0] = r_0
        sigma = np.zeros(num_grid+1)
        sigma[0] = alpha
        time_step = 0
        for grid_counter in range(1, num_grid+1):
            rnd_1 = np.random.normal(0, math.sqrt(1/num_grid))
            rnd_2 = np.random.normal(0, math.sqrt(1/num_grid))

            forward_rates[grid_counter] = forward_rates[grid_counter-1] + \
                (min(1, ((tau_1-time_step)/(tau_1-tau_0))) ** q) \
                * sigma[grid_counter-1]*forward_rates[grid_counter-1]**beta*rnd_1

            sigma[grid_counter] = sigma[grid_counter-1] + nu*sigma[grid_counter-1] * \
                (rho*rnd_1+math.sqrt(1-rho**2)*rnd_2)
            time_step = time_step+(tau_1-t_0)/num_grid

        for k in range(0, len(strikes)):
            payoff[k][mc_path_counter] = payoff_function(
                forward_rates[num_grid], strikes[k])

    return np.mean(payoff, axis=1)


def black_formula(t_0, T, K, forward_rate, sigma):
    x = math.log(forward_rate/K)

    def d_plus(x, T, sigma):
        return (x+sigma**2/2*(T-t_0))/(sigma*math.sqrt(T-t_0))

    def d_minus(x, T, sigma):
        return (x-sigma**2/2*(T-t_0))/(sigma*math.sqrt(T-t_0))
    return forward_rate*norm.cdf(d_plus(x, T-t_0, sigma))-K*norm.cdf(d_minus(x, T-t_0, sigma))


def normal_formula(t_0, T, K, R, sigma):
    def d_normal(t_0, T, K, R, sigma):
        return (R-K)/(sigma*math.sqrt(T-t_0))
    return (R-K)*norm.cdf(d_normal(t_0, T, K, R, sigma))+sigma*math.sqrt(T-t_0)*norm.pdf(d_normal(t_0, T, K, R, sigma))


def sigma_sabr_black(t_0, T, K, forward_rate, alpha, beta, rho, nu):
    x = math.log(forward_rate/K)
    zeta = nu/alpha*(forward_rate*K)**((1-beta)/2)*x

    def chi(zeta):
        return math.log(((math.sqrt(1-2*rho*zeta+zeta**2))+zeta-rho)/(1-rho))
    sigma = alpha/((forward_rate*K)**((1-beta)/2)) * (1+(((1-beta)**2)/24*alpha**2/((forward_rate*K)**(1-beta))
                                                         + 1/4*rho*beta*nu*alpha/((forward_rate*K)**((1-beta)/2))+(2-3*rho**2)/24*nu**2)*(T-t_0))
    if forward_rate == K:
        return sigma
    else:
        return sigma*zeta/chi(zeta)


def sigma_sabr_normal(t_0, T, strike, forward_rate, alpha, beta, rho, nu):

    if beta > 0 and beta < 1:
        if forward_rate != strike:
            f_mid = (forward_rate+strike)/2
            zeta = nu/alpha*(forward_rate**(1-beta)-strike**(1-beta))/(1-beta)
            xi = math.log((math.sqrt(1-2*rho*zeta+zeta**2)+zeta-rho)/(1-rho))
            return nu*(forward_rate-strike)/xi*(1+(beta*(beta-2)*alpha**2/(24*f_mid**(2-2*beta))+rho*beta*nu*alpha/(4*f_mid**(1-beta))+(2-3*rho**2)/24*rho**2)*(T-t_0))
        else:
            return alpha*forward_rate**beta*(1+(beta*(beta-2)*alpha**2/(24*forward_rate**(2-2*beta))+rho*beta*nu*alpha/(4*forward_rate**(1-beta))+(2-3*rho**2)/24*rho**2)*(T-t_0))

    elif beta == 0:
        if forward_rate != strike:
            eta = nu/alpha*(forward_rate-strike)
            xi = xi = math.log((math.sqrt(1-2*rho*eta+eta**2)+eta-rho)/(1-rho))
            return nu*(forward_rate-strike)/xi*(1+(2-3*rho**2)/24*nu**2*(T-t_0))
        else:
            return alpha*(1+(2-3*rho**2)/24*nu**2*(T-t_0))
    elif beta == 1:
        if forward_rate != strike:
            eta = nu/alpha*math.log(forward_rate/strike)
            xi = xi = math.log((math.sqrt(1-2*rho*eta+eta**2)+eta-rho)/(1-rho))
            return nu*(forward_rate-strike)/xi*(1+(-alpha**2/24+rho*nu*alpha/4+(2-3*rho**2)/24*nu**2)*(T-t_0))
        else:
            return alpha*forward_rate*(1+(-alpha**2/24+rho*nu*alpha/4+(2-3*rho**2)/24*nu**2)*(T-t_0))


def sigma_sabr_normal_for_negative_rate(t_0, T, K, R, alpha, beta, rho, nu):
    theta = (2-3*rho**2)/24*nu**2+alpha**2/24*(2*np.abs(R)**beta *
                                               beta*(beta-1)*np.abs(R)**(beta-2)-(beta*np.abs(R)**(beta-1))**2)
    if R != K:  # and beta!=0:
        I = 0
        if K >= 0 and R >= 0:
            I = (R**(1-beta)-K**(1-beta))/(1-beta)
        elif K <= 0 and R >= 0:
            I = (R**(1-beta)+(-K)**(1-beta))/(1-beta)
        elif K >= 0 and R <= 0:
            I = (-(-R)**(1-beta)-K**(1-beta))/(1-beta)
        elif K <= 0 and R <= 0:
            I = (-(-R)**(1-beta)+(-K)**(1-beta))/(1-beta)

        zeta = -nu/alpha * I
        Y = math.log((rho+zeta+np.sqrt(1+2*rho*zeta+zeta**2))/(1+rho))
        return alpha*(R-K)/I*zeta/Y*(1+theta*(T-t_0))
    elif R == K:
        return alpha*np.abs(R)**beta*(1+theta*(T-t_0))


def s_q(t_0, T, K, forward_rate, alpha, beta, rho, nu):
    _s_q = (1+(1/24*(beta*(11*beta-4)*alpha**2)/((forward_rate)**(2-2*beta))+3/4*(
        rho*nu*alpha*beta)/(forward_rate**(1-beta))+1/24*(4+3*rho**2)*nu**2)*(T-t_0))
    return sigma_sabr_normal(t_0, T, K, forward_rate, alpha, beta, rho, nu) * _s_q


def sigma_q(t_0, T, K, forward_rate, alpha, beta, rho, nu):
    _sigma_q = 1+(1/6*(beta*(2*beta-1)*alpha**2)/(forward_rate**(2-2*beta)) +
                  1/2*rho*nu*alpha*beta/(forward_rate**(1-beta))+1/6*nu**2)*(T-t_0)
    return sigma_sabr_normal(t_0, T, K, forward_rate, alpha, beta, rho, nu) * _sigma_q


def v_qc(t_0, T,  strike, forward_rate, alpha, beta, rho, nu):
    sigma_q_k = sigma_q(t_0, T, strike, strike, alpha, beta, rho, nu)
    sigma_q_f = sigma_q(t_0, T, strike, forward_rate, alpha, beta, rho, nu)
    _s_q = s_q(t_0, T, strike, forward_rate, alpha, beta, rho, nu)

    if strike != forward_rate:
        f_adj = forward_rate+1/2*(sigma_q_k**2-sigma_q_f**2) / \
            (strike-forward_rate)*(T-t_0)
    else:
        f_adj = forward_rate
    return ((forward_rate-strike)**2+_s_q**2*(T-t_0))*norm.cdf((f_adj-strike)/(sigma_q_k*math.sqrt(T-t_0))
                                                               )+(f_adj-strike)*sigma_q_k*math.sqrt(T-t_0)*norm.pdf((f_adj-strike)/(sigma_q_k*math.sqrt(T-t_0)))


def v_aa_caplet(t_0, t_s, t_e, d_n,   strike, forward_rate, alpha, beta, rho, nu):
    strike_hat = k_n_hat(t_s, t_e, d_n, strike)
    g_r = g_function(t_s, t_e, d_n, forward_rate)
    g_k = g_function(t_s, t_e, d_n, strike_hat)
    g_r_prime = g_prime_function(t_s, t_e, d_n, forward_rate)
    g_r_double_prime = g_double_prime_function(t_s, t_e, d_n, forward_rate)

    # first_term = 0
    # if forward_rate > strike_hat:
    #     first_term = (g_r-g_k - g_r_prime*(forward_rate-strike_hat) +
    #                   1/2 * g_r_double_prime*(forward_rate-strike_hat)**2)

    second_term = (g_r_prime-g_r_double_prime*(forward_rate-strike_hat))*normal_formula(t_0, t_e, strike_hat,
                                                                                        forward_rate, sigma_sabr_normal(t_0, t_e, strike_hat, forward_rate, alpha, beta, rho, nu))
    third_term = 1/2*g_r_double_prime * \
        v_qc(t_0, t_e,  strike_hat, forward_rate, alpha, beta, rho, nu)

    return second_term + third_term
