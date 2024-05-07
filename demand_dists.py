import numpy as np



def true_lamb_poisson(p, t, upper_t, alpha=50.0, a0=1.0, a1=5.0):
     return alpha * np.exp(- (p + a0 * t) / a1)

def true_lamb_poisson_v2(p, t, upper_t, alpha=50.0, a0=0.5, a1=5.0):
     return alpha * np.exp(- p / (a0 + a1 * t / upper_t))

def true_fail_prob_PA(p, t, upper_t, a0=1.0, a1=10.0):
     return np.exp(- (p + a0 * t) / a1)

def true_mean_demand_PA(p, t, upper_t, n_p=10):
    p_f = true_fail_prob_PA(p, t, upper_t)
    return n_p * p_f / (1.0 - p_f) 

def true_fail_prob_PB(p, t, upper_t, a0=0.5, a1=5.0):
     return np.exp(- p / (a0 + a1 * t / upper_t))

def true_mean_demand_PB(p, t, upper_t, n_p=10):
    p_f = true_fail_prob_PB(p, t, upper_t)
    return n_p * p_f / (1.0 - p_f) 

def observe_demand_poisson(p, t, upper_t, random_state= None):
    return random_state.poisson(true_lamb_poisson(p, t, upper_t))

def observe_demand_poisson_v2(p, t, upper_t, random_state= None):
    return random_state.poisson(true_lamb_poisson_v2(p, t, upper_t))

def observe_demand_negbinom_PA(p, t, upper_t, n=10, random_state= None):
    return random_state.negative_binomial(n, 1 - true_fail_prob_PA(p, t, upper_t))

def observe_demand_negbinom_PB(p, t, upper_t, n=10, random_state= None):
    return random_state.negative_binomial(n, 1 - true_fail_prob_PB(p, t, upper_t))