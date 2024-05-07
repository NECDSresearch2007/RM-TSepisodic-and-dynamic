"""
This work © 2024 by NEC Corporation is licensed under Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International.
"""

import GPy
import cvxopt
import numpy as np
import gurobipy as grbp
from gurobipy import GRB 


class TS_episodic:

    @classmethod
    def _setup_upper_x(cls, range_p, range_t):
        """
        Return all grid points consisting of prices and time periods.
        """
        return np.vstack([np.asarray([[t, p] for p in range_p])
                        for t in range_t])

    @classmethod
    def _sample_prior_f_gp(cls, upper_x, kernel, random_state):
        """
        Initial samples of mean demand using the initial kernel.
        """
        return random_state.multivariate_normal(
            mean=np.zeros(upper_x.shape[0]), cov=kernel.K(upper_x))

    @classmethod
    def _transf(cls, f):
        return GPy.util.misc.safe_exp(f)

    @classmethod
    def _sample_demand_gp(cls, upper_x0, lamb_dist, sample_slice, kernel_init, kernel_param,
    random_state):
        """
        Sample mean demand from the Gaussian process. 
        """
        upper_x = upper_x0[sample_slice, :]
        # Initial samples 
        if lamb_dist is None:
            f_sample = cls._sample_prior_f_gp(
                upper_x, kernel_init(**kernel_param), random_state)
        # sampled latent functions using Gaussian process 
        else:
            f_sample = lamb_dist.posterior_samples_f(upper_x, size=1)[:, 0, 0]
        # return sampled mean demand
        return np.clip(cls._transf(f_sample), a_min=0.0, a_max=None), \
            lamb_dist
    
    @classmethod
    def _sample_demand_gamma(cls, lamb_dist, sample_slice, range_p,
                            range_t, init_alpha, init_beta, random_state):
        """
        Sample mean demand from the posterior Gamma distributions. 
        """
        # Initial sample
        if lamb_dist is None:
            lamb_dist = cls._setup_gamma_dist(range_p, range_t, init_alpha, init_beta)
        # set gamma scale parameters
        alpha_part = lamb_dist['alpha'][sample_slice]
        beta_part = lamb_dist['beta'][sample_slice]
        # sample mean demand using the gamma distribution
        lamb_sample = [random_state.gamma(shape=alpha_part[cnt_sample],
                                    scale=1.0 / beta_part[cnt_sample]
                                    )
                    for cnt_sample in range(len(alpha_part))]
        return lamb_sample, lamb_dist
    
    @classmethod
    def _sample_demand_beta(cls, lamb_dist, sample_slice, range_p,
                            range_t, initial_a, initial_b, n,  random_state):
        
        if lamb_dist is None:
            lamb_dist = cls._setup_beta_dist(range_p, range_t, 
                                             initial_a, initial_b, n)
        
        alpha_part = lamb_dist['a'][sample_slice]
        beta_part = lamb_dist['b'][sample_slice]

        #lamb_sample = random_state.beta(a=alpha_part, b= beta_part)
        success_pro = random_state.beta(a=alpha_part, b=beta_part)
        # average number of successes
        mean_sample = lamb_dist['n'] * (1.0 - success_pro) / success_pro
        return mean_sample, lamb_dist

    @classmethod
    def _sample_demand(cls, range_p, range_t, sample_slice, lamb_dist, 
                        dist_param, dist_type, random_state):
        """
        Return sampled mean demand from the posterior distribution.  
        """
        # get mean demands through the Gaussian process 
        if dist_type == 'gp':
            return cls._sample_demand_gp(
                    dist_param['upper_x0'], lamb_dist, sample_slice,
                    dist_param['kernel_init'], dist_param['kernel_param'], random_state)
        # get mean demands through the independent Gamma distribution
        elif dist_type == 'gamma':
            return cls._sample_demand_gamma(
                lamb_dist, sample_slice, range_p, range_t, dist_param['alpha'], 
                dist_param['beta'], random_state)
        # get mean demand through the independent beta distribution.
        elif dist_type == "beta":
            return cls._sample_demand_beta(lamb_dist, sample_slice, range_p, range_t, 
                                           dist_param['initial_a'], dist_param['initial_b'], 
                                           dist_param['n'], random_state)
        else:
            raise NotImplementedError

    @classmethod
    def _setup_c(cls, lamb_sample, upper_k, range_p, range_t):
        """
        Set up the objective of the LP.
        """
        return np.hstack([- range_p
                          * lamb_sample[cnt * upper_k : (cnt + 1) * upper_k]
                          for cnt in range(len( range_t )) ])
 
    @classmethod
    def _setup_upper_g(cls, lamb_sample, upper_k, range_t, use_gurobi):
        """
        Set up the left-hand side of the constraints of the LP.
        """
        # inventory constraint 
        block_lamb = np.hstack(
            [lamb_sample[cnt * upper_k:upper_k * (cnt + 1)]
            for cnt in range(len(range_t))])
        # constraint for the solution 
        block_e = np.vstack(
            [np.hstack([np.ones(upper_k) if cnt_c == cnt_r
                        else np.zeros(upper_k)
                        for cnt_c in range(len(range_t))])
                        for cnt_r in range(len(range_t))])
        if use_gurobi:
            return np.vstack([block_lamb, block_e])
        return np.vstack([block_lamb, block_e,
                          -np.eye(upper_k * len(range_t))
                          ])
    
    @classmethod
    def _setup_h(cls, upper_k, n_t, range_t, use_gurobi):
        """
        Set up the right-hand side of the constraints of the LP.
        """
        if use_gurobi:
                return np.hstack([[n_t], np.ones(len(range_t))]
                          )
        return np.hstack([[n_t], np.ones(len(range_t)),
                          np.zeros(upper_k * len(range_t))
                          ])

    @classmethod
    def _solve_lp(cls, c, upper_g, h):
        sol = cvxopt.solvers.lp(cvxopt.matrix(c), cvxopt.matrix(upper_g),
                                cvxopt.matrix(h), options={'show_progress': False})
        if sol['status'] != 'optimal':
            print(sol['status'])
        return np.asarray(sol['x']).ravel(), sol['status']
    
    @classmethod
    def _solve_gurobi(cls, c, upper_g, h):
        grbp.setParam('LogtoConsole', 0)
        grbp.setParam('LogFile', "")
        model = grbp.Model(name='calc_x')
        indexes = np.arange(len(c))
        vtype = GRB.CONTINUOUS
        vars = model.addVars(indexes, lb = 0.0, ub = 1.0, vtype = vtype, name='x')

        # Add objective
        model.setObjective(grbp.LinExpr(c, vars.values()), GRB.MINIMIZE)

        #Add constrain(ts:
        for i, g_i in enumerate(upper_g):
            model.addConstr(grbp.LinExpr(g_i, vars.values()) <= h[i], name='inv')
        
        # optimize
        model.optimize()

        #set return 
        status = model.status
        x = np.array([v.x for v in model.getVars()])
        if status != GRB.OPTIMAL:
            print(f'non-optimal solution: Gurobi status code = {status}')
        model.close()
        return x, status

    @classmethod
    def _calc_x(cls, lamb_sample, upper_k, n_t, range_p, range_t, use_gurobi):
        """
        Set up and solve the LP.
        """
        # set the objective function
        c = cls._setup_c(lamb_sample, upper_k, range_p, range_t) 
        # set the left-hand side of the constraints
        upper_g = cls._setup_upper_g(lamb_sample, upper_k, range_t, use_gurobi)
        # set the right-hand side of the constraints 
        h = cls._setup_h(upper_k, n_t, range_t, use_gurobi)

        if use_gurobi:
            return cls._solve_gurobi(c, upper_g, h)
        return cls._solve_lp(c, upper_g, h)
        
    @classmethod
    def _calc_x_ext(cls, x):
        """
        Add the probability for the shut-off price 
        """
        x_ext = np.append(x, 1.0 - np.sum(x))
        x_ext[x_ext < 0.0] = 0.0
        return x_ext / np.sum(x_ext)

    @classmethod
    def _setup_p_ext(cls, range_p, p_inf):
        """
        Add the shut-off price to the price candidates set.
        """
        return np.append(range_p, p_inf)

    @classmethod
    def _choice_p(cls, p_ext, x_ext, random_state):
        """ 
        Choose a price following the solution of the LP.
        """
        k = random_state.choice(len(p_ext), p=x_ext)
        return p_ext[k], k

    @classmethod
    def _sample_p(cls, range_p, p_ext, x, random_state):
        """
        Choose a price following the solution of the LP.
        """
        x_ext = cls._calc_x_ext(x)
        return cls._choice_p(p_ext, x_ext, random_state)

    @classmethod
    def _update_n_and_revenue(cls, upper_d, n_t, p, revenue):
        """ 
        Consume inventory and obtain revenue observed demand based on observed demand.
        """
        # actual sales
        upper_d_prime = upper_d if n_t - upper_d >= 0.0 else n_t
        # inventory consumption
        n_t -= upper_d_prime
        # obtain revenue
        revenue += p * upper_d_prime
        return n_t, revenue
    

    @classmethod
    def _setup_beta_dist(cls, range_p, range_t, initial_a, initial_b, n):
        """
        Initial set up for beta prior.
        """
        a = initial_a * np.ones(len(range_p) * len(range_t) )
        b = initial_b * np.ones(len(range_p) * len(range_t))
        return {'a': a, 'b': b, 'n': n}

    
    @classmethod
    def _setup_gamma_dist(cls, range_p, range_t, init_alpha, init_beta):
        """
        Initial set up for gamma prior.  
        """
        alpha = init_alpha * np.ones(len(range_p) * len(range_t))
        beta = init_beta * np.ones(len(range_p) * len(range_t))
        return {'alpha': alpha, 'beta': beta}

    @classmethod
    def _update_beta(cls, upper_d, k, cnt_t, upper_k, p_ext, range_t,
                      lamb_dist):
        """
        Update the beta posterior distribution
        """
        if not isinstance(upper_d, (list, np.ndarray)):
            upper_d = [upper_d]
            k = [k]
            cnt_t = [cnt_t]
        
        if lamb_dist is None:
            lamb_dist = cls._setup_beta_dist(p_ext[:upper_k], range_t)
        # Update beta distribution parameters
        for cnt_d in range(len(upper_d)):
            if k[cnt_d] >= upper_k:
                continue
            idx = cnt_t[cnt_d] * upper_k + k[cnt_d]
            lamb_dist['a'][idx] += lamb_dist['n']
            lamb_dist['b'][idx] += upper_d[cnt_d]
        return lamb_dist
    
    @classmethod
    def _update_gamma(cls, upper_d, k, cnt_t, upper_k, p_ext, range_t,
                      lamb_dist):
        """
        Update the gamma posterior distribution.
        """
        if not isinstance(upper_d, (list, np.ndarray)):
            upper_d = [upper_d]
            k = [k]
            cnt_t = [cnt_t]

        if lamb_dist is None:
            lamb_dist = cls._setup_gamma_dist(p_ext[:upper_k], range_t)

        for cnt_d in range(len(upper_d)):
            if k[cnt_d] >= upper_k:
                continue
            idx = cnt_t[cnt_d] * upper_k + k[cnt_d]
            lamb_dist['alpha'][idx] += upper_d[cnt_d]
            lamb_dist['beta'][idx] += 1.0
        
        return lamb_dist
    
    @classmethod
    def _update_gp(cls, upper_d, p, t, gp, kernel_init, kernel_param):
        """
        Update the Gaussian posterior distribution. 
        """
        if isinstance(upper_d, np.ndarray):
            if upper_d.shape[1] != 1 or p.shape[1] != 1 or t.shape[1] != 1:
                raise RuntimeError
            x = np.hstack([t, p])
            y = upper_d
        else:
            x = np.asarray([[t, p]])
            y = np.asarray([[upper_d]])

        if gp is None:
            upper_x = x
            upper_y = y
        else:
            upper_x = np.vstack([gp.X, x])
            upper_y = np.vstack([gp.Y, y])
        # Set an inference method.
        inference_method = GPy.inference.latent_function_inference.Laplace()
        # Set a GPy instance
        gp = GPy.core.GP(
            X=upper_x, Y=upper_y, kernel=kernel_init(**kernel_param),
            likelihood=GPy.likelihoods.Poisson(),
            inference_method=inference_method)
        return gp

    @classmethod
    def _update_gp_once(cls, history, gp, kernel_init, kernel_param):
        """
        UPdate the Gaussian process at the beginning of each episode. 
        """
        # get the observed demand 
        upper_d = np.asarray([history[cnt]['upper_d'] for cnt in
                              range(len(history))]).reshape(-1, 1)
        # get the offered price
        p = np.asarray([history[cnt]['p'] for cnt in
                        range(len(history))]).reshape(-1, 1)
        # get the corresponding time period
        t = np.asarray([history[cnt]['t'] for cnt in
                        range(len(history))]).reshape(-1, 1)
        return cls._update_gp(upper_d, p, t, gp, kernel_init, kernel_param)

    @classmethod
    def _update_gamma_once(cls, history, lamb_dist, upper_k, p_ext, range_t):
        """
        Update the gamma distribution at the beginning of each episode.
        """
        upper_d = np.asarray([history[cnt]['upper_d'] for cnt in
                              range(len(history))])
        k = np.asarray([history[cnt]['k'] for cnt in range(len(history))])
        cnt_t = np.asarray([history[cnt]['cnt_t'] for cnt in
                            range(len(history))])
        return cls._update_gamma(upper_d, k, cnt_t, upper_k, p_ext, range_t,
                                 lamb_dist)
    
    @classmethod
    def _update_beta_once(cls, history, lamb_dist, upper_k, p_ext, range_t):
        """
        Update the beta distribution at the beginning of each episode.
        """
        upper_d = np.asarray([history[cnt]['upper_d'] for cnt in
                              range(len(history))])
        k = np.asarray([history[cnt]['k'] for cnt in range(len(history))])
        cnt_t = np.asarray([history[cnt]['cnt_t'] for cnt in
                            range(len(history))])
        return cls._update_beta(upper_d, k, cnt_t, upper_k, p_ext, range_t,
                                 lamb_dist)
    

    @classmethod
    def _update_dist_once(cls, history, upper_k, p_ext, range_t, lamb_dist,
                          dist_param, dist_type):
        """
        Update the posterior distribution at the beginning of each episode.
        """
        if dist_type == 'gp':
            return cls._update_gp_once(history, lamb_dist,
                                       dist_param['kernel_init'],
                                       dist_param['kernel_param']
                                        )
        elif dist_type == 'gamma':
            return cls._update_gamma_once(history, lamb_dist, upper_k, p_ext,
                                          range_t)
        elif dist_type == "beta":
            return cls._update_beta_once(history, lamb_dist, upper_k, p_ext,
                                          range_t)
        else:
            raise NotImplementedError

    @classmethod
    def episode(cls, n_t0, upper_k, range_p, p_ext,
                range_t, lamb_dist, observe_demand,
                 dist_param, dist_type, use_gurobi, random_state):
        """
        Run an episode of an episodic revenue management problem.
        """
        n_t = n_t0
        revenue_episode = 0.0
        history = []
        for cnt_t, t in enumerate(range_t):
            if cnt_t == 0:
                # Sample demand
                lamb_sample, lamb_dist = cls._sample_demand(
                    range_p, range_t, slice(None), lamb_dist,
                    dist_param, dist_type, random_state)
                
                # Optimize prices given sampled demand
                x, x_status = cls._calc_x(
                    lamb_sample, upper_k, n_t, range_p, range_t, 
                    use_gurobi)
            # Restrict the solution for a suitable form at this period.  
            x_t = x[cnt_t * upper_k:(cnt_t + 1) * upper_k]

            # Offer price
            p, k = cls._sample_p(range_p, p_ext, x_t, random_state)

            # Update estimate of parameter
            upper_d = observe_demand(p, t, range_t[-1])
            inventory_begin = n_t
            n_t, revenue_episode = cls._update_n_and_revenue(
                upper_d, n_t, p, revenue_episode)
            # Add the obtained data to the history. 
            history.append(
                {'t': t, 'cnt_t': cnt_t, 'lamb_sample': lamb_sample, 'x': x,
                 'x_status': x_status, 'p': p, 'k': k, 'upper_d': upper_d,
                 'inventory_begin': inventory_begin, 'inventory_end': n_t,
                 'revenue': revenue_episode})
            if n_t == 0.0:
                break
        #Update the posterior distribution.
        lamb_dist = cls._update_dist_once(
                history, upper_k, p_ext, range_t, lamb_dist, dist_param,
                dist_type)
        return lamb_dist, revenue_episode, history

    @classmethod
    def run(cls, n_t0, range_p, p_inf, range_t, range_n_episodes,
            observe_demand, dist_param, dist_type, 
            use_gurobi, random_state=None):
        """
        Run an episodic revenue management problem.
        """
        
        if not random_state:
            random_state = np.random.RandomState[None]
        
        upper_k = len(range_p)
        p_ext = cls._setup_p_ext(range_p, p_inf)

        lamb_dist = None
        revenue_n_episodes = 0.0
        history = []
        print('---Starts---')
        for cnt_episode in range_n_episodes:
            lamb_dist, revenue_episode, history_episode = cls.episode(
                n_t0, upper_k, range_p, p_ext, range_t,
                lamb_dist, observe_demand,
                dist_param, dist_type, use_gurobi, random_state)
            revenue_n_episodes += revenue_episode
            history.append(history_episode)
            if (cnt_episode + 1 )% 100 == 0:
                print(f'progress: {cnt_episode + 1}')
        print('---End---')
        return lamb_dist, revenue_n_episodes, history

class TS_dynamic(TS_episodic):

    @classmethod
    def _update_dist(cls, upper_d, k, cnt_t, upper_k, p_ext, range_t,
                     lamb_dist, dist_param, dist_type):
        if dist_type == 'gp':
            return cls._update_gp(upper_d, p_ext[k], range_t[cnt_t], lamb_dist,
                                  dist_param['kernel_init'],
                                  dist_param['kernel_param'])
        elif dist_type == 'gamma':
            return cls._update_gamma(upper_d, k, cnt_t, upper_k, p_ext,
                                     range_t, lamb_dist)
        elif dist_type == "beta":
            return cls._update_beta(upper_d, k, cnt_t, upper_k, p_ext,
                                     range_t, lamb_dist)
        else:
            raise NotImplementedError
    

    @classmethod
    def episode(cls, n_t0, upper_k, range_p, p_ext,
                range_t, lamb_dist, observe_demand,
                dist_param, dist_type, use_gurobi, random_state):
        n_t = n_t0
        revenue_episode = 0.0
        history = []
        for cnt_t, t in enumerate(range_t):
            # Sample demand
            lamb_sample, lamb_dist = cls._sample_demand(
                range_p, range_t, slice(cnt_t * upper_k, None),
                lamb_dist, dist_param, dist_type, random_state)

            # Optimize prices given sampled demand
            x, x_status = cls._calc_x(
                lamb_sample, upper_k, n_t, range_p,
                range_t[cnt_t:], use_gurobi)

            x_t = x[:upper_k]
            
            # Offer price
            p, k = cls._sample_p(range_p, p_ext, x_t, random_state)

            # Update estimate of parameter
            upper_d = observe_demand(p, t, range_t[-1])
            inventory_begin = n_t
            n_t, revenue_episode = cls._update_n_and_revenue(
                upper_d, n_t, p, revenue_episode)
            lamb_dist = cls._update_dist(
                    upper_d, k, cnt_t, upper_k, p_ext, range_t, lamb_dist,
                    dist_param, dist_type)

            history.append(
                {'t': t, 'cnt_t': cnt_t, 'lamb_sample': lamb_sample, 'x': x,
                 'x_status': x_status, 'p': p, 'k': k, 'upper_d': upper_d,
                 'inventory_begin': inventory_begin, 'inventory_end': n_t,
                 'revenue': revenue_episode})
            if n_t == 0.0:
                break
    
        return lamb_dist, revenue_episode, history

    @classmethod
    def run(cls, n_t0, range_p, p_inf, range_t, range_n_episodes,
            observe_demand, dist_param,
            dist_type, use_gurobi, random_state=None):
        
        if not random_state:
            random_state = np.random.RandomState[None]
        
        upper_k = len(range_p)
        p_ext = cls._setup_p_ext(range_p, p_inf)

        lamb_dist = None
        revenue_n_episodes = 0.0
        history = []
        print('--Start--')
        for cnt_episode in range_n_episodes:
            lamb_dist, revenue_episode, history_episode = cls.episode(
                n_t0, upper_k, range_p, p_ext, range_t,
                lamb_dist, observe_demand,
                dist_param, dist_type, use_gurobi, random_state)
            revenue_n_episodes += revenue_episode
            history.append(history_episode)
            if (cnt_episode + 1 ) % 100 == 0:
                print(f'progress: {cnt_episode + 1}')
        print('--End--')
        return lamb_dist, revenue_n_episodes, history


class TS_Fixed(TS_episodic):
    @classmethod
    def _setup_c(cls, lamb_sample, range_p):
        return -range_p * lamb_sample

    @classmethod
    def _setup_upper_g(cls, lamb_sample, upper_k):
        return np.vstack([lamb_sample, np.ones(upper_k), -np.eye(upper_k)])

    @classmethod
    def _setup_h(cls, upper_k, n_t0, range_t):
        return np.hstack([np.asarray([n_t0 / float(len(range_t)), 1.0]),
                          np.zeros(upper_k)])

    @classmethod
    def _calc_x(cls, lamb_sample, upper_k, n_t0, range_p, range_t, use_gurobi):
        c = cls._setup_c(lamb_sample, range_p) 
        upper_g = cls._setup_upper_g(lamb_sample, upper_k)
        h = cls._setup_h(upper_k, n_t0, range_t)
        if use_gurobi:
            return cls._solve_gurobi(c, upper_g, h)
        return cls._solve_lp(c, upper_g, h)
    
    @classmethod
    def episode(cls, n_t0, upper_k, range_p, p_ext, range_t,
                lamb_dist, observe_demand,
                dist_param, dist_type, use_gurobi, random_state):
        n_t = n_t0
        revenue_episode = 0.0
        history = []
        for cnt_t, t in enumerate(range_t):

            if cnt_t == 0:
                # Sample demand
                lamb_sample0, lamb_dist = cls._sample_demand(
                    range_p, range_t, slice(None), lamb_dist,
                    dist_param, dist_type, random_state)

            lamb_sample \
                = lamb_sample0[cnt_t * upper_k:(cnt_t + 1) * upper_k]

            # Optimize prices given sampled demand
            x, x_status = cls._calc_x(
                lamb_sample, upper_k, n_t0, range_p, range_t, use_gurobi
                )

            # Offer price
            p, k = cls._sample_p(range_p, p_ext, x, random_state)

            # Update estimate of parameter
            upper_d = observe_demand(p, t, range_t[-1])
            inventory_begin = n_t
            n_t, revenue_episode = cls._update_n_and_revenue(
                upper_d, n_t, p, revenue_episode)

            history.append(
                {'t': t, 'cnt_t': cnt_t, 'lamb_sample': lamb_sample, 'x': x,
                 'x_status': x_status, 'p': p, 'k': k, 'upper_d': upper_d,
                 'inventory_begin': inventory_begin, 'inventory_end': n_t,
                 'revenue': revenue_episode})
            if n_t == 0.0:
                break

        lamb_dist = cls._update_dist_once(
            history, upper_k, p_ext, range_t, lamb_dist, dist_param,
            dist_type)
        return lamb_dist, revenue_episode, history

    @classmethod
    def run(cls, n_t0, range_p, p_inf, range_t, range_n_episode,
            observe_demand, dist_param,
            dist_type, use_gurobi, random_state=None):
        upper_k = len(range_p)
        p_ext = cls._setup_p_ext(range_p, p_inf)

        lamb_dist = None
        revenue_n_episodes = 0.0
        history = []
        print('episode')
        for cnt_episode in range_n_episode:
            lamb_dist, revenue_episode, history_episode = cls.episode(
                n_t0, upper_k, range_p, p_ext, range_t, lamb_dist,
                observe_demand, dist_param,
                dist_type, use_gurobi, random_state)
            revenue_n_episodes += revenue_episode
            history.append(history_episode)
        return lamb_dist, revenue_n_episodes, history


class TS_Update(TS_Fixed):

    @classmethod
    def episode(cls, n_t0, upper_k, range_p, p_ext, range_t,
                lamb_dist, observe_demand,
                dist_param, dist_type, use_gurobi, random_state=None):
        n_t = n_t0
        revenue_episode = 0.0
        history = []
        for cnt_t, t in enumerate(range_t):

            if cnt_t == 0:
                # Sample demand
                lamb_sample0, lamb_dist = cls._sample_demand(
                    range_p, range_t, slice(None), lamb_dist,
                    dist_param, dist_type, random_state)

            lamb_sample \
                = lamb_sample0[cnt_t * upper_k:(cnt_t + 1) * upper_k]

            # Optimize prices given sampled demand
            x, x_status = cls._calc_x(
                lamb_sample, upper_k, n_t, range_p, range_t[cnt_t:], use_gurobi
                )

            # Offer price
            p, k = cls._sample_p(range_p, p_ext, x, random_state)

            # Update estimate of parameter
            upper_d = observe_demand(p, t, range_t[-1])
            inventory_begin = n_t
            n_t, revenue_episode = cls._update_n_and_revenue(
                upper_d, n_t, p, revenue_episode)

            history.append(
                {'t': t, 'cnt_t': cnt_t, 'lamb_sample': lamb_sample, 'x': x,
                 'x_status': x_status, 'p': p, 'k': k, 'upper_d': upper_d,
                 'inventory_begin': inventory_begin, 'inventory_end': n_t,
                 'revenue': revenue_episode})
            if n_t == 0.0:
                break
        
        lamb_dist = cls._update_dist_once(
                history, upper_k, p_ext, range_t, lamb_dist, dist_param,
                dist_type)
        return lamb_dist, revenue_episode, history


class TS_episodic_oracle(TS_episodic):
    
    @classmethod
    def _sample_demand(cls, sample_slice, 
                        dist_param,  true_lamb):
        """
        Get true mean demand.
        """
        upper_x = dist_param['upper_x0'][sample_slice, :]
        upper_t = upper_x[-1][0]
        return np.asarray([true_lamb(x[1], x[0], upper_t) for x in upper_x ])
    

    @classmethod
    def run(cls, n_t0, range_p, p_inf, range_t, range_n_episodes,
            observe_demand, true_lamb, dist_param, use_gurobi, random_state=None):
        upper_k = len(range_p)
        p_ext = cls._setup_p_ext(range_p, p_inf)

        lamb_dist = None
        revenue_n_episodes = 0.0
        history = []
        for cnt_episode in range_n_episodes:
            lamb_dist, revenue_episode, history_episode = cls.episode(
                n_t0, upper_k, range_p, p_ext, range_t,
                lamb_dist, observe_demand, true_lamb, dist_param, 
                use_gurobi, random_state=random_state)
            revenue_n_episodes += revenue_episode
            history.append(history_episode)
        return lamb_dist, revenue_n_episodes, history
    
    @classmethod
    def episode(cls, n_t0, upper_k, range_p, p_ext, range_t,
                lamb_dist, observe_demand, true_lamb, dist_param,
                use_gurobi, random_state=None):
        n_t = n_t0
        revenue_episode = 0.0
        history = []
        for cnt_t, t in enumerate(range_t):
            if cnt_t == 0:
                # Sample demand
                lamb_sample = cls._sample_demand(slice(None), dist_param, true_lamb)
                x0, x_status = cls._calc_x(lamb_sample, upper_k, n_t, 
                                            range_p, range_t[cnt_t:], use_gurobi
                                            )

            x = x0[cnt_t * upper_k: (cnt_t+ 1) * upper_k]

            # Offer price
            p, k = cls._sample_p(range_p, p_ext, x, random_state)

            # Update estimate of parameter
            upper_d = observe_demand(p, t, range_t[-1])
            inventory_begin = n_t
            n_t, revenue_episode = cls._update_n_and_revenue(
                upper_d, n_t, p, revenue_episode)

            history.append(
                {'t': t, 'cnt_t': cnt_t, 'lamb_sample': lamb_sample, 'x': x,
                 'x_status': x_status, 'p': p, 'k': k, 'upper_d': upper_d,
                 'inventory_begin': inventory_begin, 'inventory_end': n_t,
                 'revenue': revenue_episode})
            if n_t == 0.0:
                break
        return lamb_dist, revenue_episode, history


class TS_dynamic_oracle(TS_dynamic):
    
    @classmethod
    def _sample_demand(cls, sample_slice, dist_param, true_lamb):
        """
        Get the true mean demand.  
        """
        upper_x = dist_param['upper_x0'][sample_slice, :]
        upper_t = upper_x[-1][0]
        return np.asarray([true_lamb(x[1],x[0], upper_t) for x in upper_x ])
    

    @classmethod
    def episode(cls, n_t0, upper_k, range_p, p_ext,
                range_t, lamb_dist, observe_demand, true_lamb, dist_param, 
                use_gurobi, random_state):
        n_t = n_t0
        revenue_episode = 0.0
        history = []
        for cnt_t, t in enumerate(range_t):
            # Sample demand
            lamb_sample = cls._sample_demand(slice(cnt_t * upper_k, None), 
                                                        dist_param, true_lamb)

            # Optimize prices given sampled demand
            x, x_status = cls._calc_x(
                lamb_sample, upper_k, n_t, range_p,
                range_t[cnt_t:], use_gurobi)

            x_t = x[:upper_k]

            # Offer price
            p, k = cls._sample_p(range_p, p_ext, x_t, random_state)

            # Update estimate of parameter
            upper_d = observe_demand(p, t, range_t[-1])
            inventory_begin = n_t
            n_t, revenue_episode = cls._update_n_and_revenue(
                upper_d, n_t, p, revenue_episode)

            history.append(
                {'t': t, 'cnt_t': cnt_t, 'lamb_sample': lamb_sample, 'x': x,
                 'x_status': x_status, 'p': p, 'k': k, 'upper_d': upper_d,
                 'inventory_begin': inventory_begin, 'inventory_end': n_t,
                 'revenue': revenue_episode})
            if n_t == 0.0:
                break
    
        return lamb_dist, revenue_episode, history


    @classmethod
    def run(cls, n_t0, range_p, p_inf, range_t, range_n_episodes,
            observe_demand, true_lamb, dist_param, use_gurobi, random_state=None):
        upper_k = len(range_p)
        p_ext = cls._setup_p_ext(range_p, p_inf)

        lamb_dist = None
        revenue_n_episodes = 0.0
        history = []
        for cnt_episode in range_n_episodes:
            lamb_dist, revenue_episode, history_episode = cls.episode(
                n_t0, upper_k, range_p, p_ext, range_t,
                lamb_dist, observe_demand, true_lamb, dist_param, 
                use_gurobi, random_state=random_state)
            revenue_n_episodes += revenue_episode
            history.append(history_episode)
        return lamb_dist, revenue_n_episodes, history