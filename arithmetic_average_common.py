import math


def g_function(t_s, t_e, d_n, x):
    return 1/(t_e-t_s)*math.log(1+(t_e-d_n)*x)


def delta_x(t_e, d_n, x):
    return 1/(1+(t_e-d_n)*x)


def g_prime_function(t_s, t_e, d_n, x):
    return (t_e-d_n)/(t_e-t_s)*delta_x(t_e, d_n, x)


def g_double_prime_function(t_s, t_e, d_n, x):
    return -(t_e-d_n)**2/(t_e-t_s)*delta_x(t_e, d_n, x)**2


def k_n_hat(t_s, t_e, d_n, k_n):
    return (math.exp((t_e-t_s)*k_n)-1)/(t_e-d_n)
