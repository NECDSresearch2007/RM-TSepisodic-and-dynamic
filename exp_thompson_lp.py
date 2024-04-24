"""
This work © 2024 by NEC Corporation is licensed under Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International.
"""

import functools
import time
from multiprocessing import Pool

# Third party
import numpy as np
import GPy


from thompson_lp import TS_episodic, TS_dynamic, TS_Fixed, TS_Update
from demand_dists import observe_demand_poisson, observe_demand_poisson_v2, observe_demand_negbinom
from utils import save_data


class exp():

    @classmethod
    def create_agent(self, param):
        """
        return: class constructor
        """
        agent_label = param['agent']
        agent = self.agent_dic[agent_label]
        return agent

    @classmethod
    def get_history(self, random_state, param):
        """
        Run an experiment to obtain the resulting history.
        """
        # Set the true demand distribution
        observe_demand = self.true_demand_dist[param['true_demand_dist']]
        # Form the true demand distribution as a function of p
        observe_demand_p = functools.partial(observe_demand, random_state=random_state)
        # Set an algorithm
        agent = self.create_agent(param)
        print('run')
        # Get history
        lamb_dist, revenue_episode, history = agent.run(param['n_t0'], 
        param['range_p'], param['p_inf'], param['range_t'], param['range_n_episodes'],
        observe_demand_p, param['dist_param'], param['dist_param']['dist_type'], param['use_gurobi'],
        random_state=random_state
        )
        return history
    
    @classmethod
    def repeat(self, param, processes):
        """
        For parallel computation of experiments 
        """
        # generate different seeds for independent trials.
        random_states = [np.random.RandomState(seed) 
                        for seed in param['seeds']] 
        map_func = functools.partial(self.get_history, param=param)
        # Parallel computing
        with Pool(processes=processes) as pool:
            repeated_history = pool.map(map_func, random_states)
        return repeated_history


    @classmethod
    def element(self, param, processes):
        """
        Get data from all independent trials and save it.
        """
        et = - time.process_time()
        repeated_history = self.repeat(param, processes)
        et += time.process_time()
        data = {'repeated_history': repeated_history,
                'elapsed_time': et}
        data.update(param)
        save_data('data/' + param['id'] + '.pickle', data)

    @classmethod
    def main(self):
        """
        Run experiments
        """
        # Time horizon
        upper_t = 10.0
        # set of candidate prices.
        range_p = np.arange(1, 10, 1)
        # set time periods of each episode.
        range_t = np.arange(1, upper_t + 1, 1)
        # gird points 
        upper_x0 = TS_episodic._setup_upper_x(range_p, range_t)
        # Algorithm name list
        self.agent_dic = {'TSE' : TS_episodic, 'TSD': TS_dynamic,
        'TSF': TS_Fixed, 'TSU': TS_Update}
        # True demand distribution list
        self.true_demand_dist = {'poisson': observe_demand_poisson, 
                                 'poisson_v2': observe_demand_poisson_v2,
                                  'negbinom': observe_demand_negbinom
                                  }
        params = []
        # Initial inventory
        n_t0 = 1000
        n_t01 = 10
        # Number of independent trials.
        n_tri = 25
        # shut-off price
        p_inf = 100000.0
        # Episode length
        n_episodes_gamma = 5000
        n_episodes_gp = 20
        # If False, use cvxopt to solve the LP instead of Gurobi.
        use_gurobi = True
        # Number of CPUs used for parallel computing.  
        processes = 1

        
        #Hyper Parameters for a Gaussian process
        dist_param_gp = {'kernel_init': GPy.kern.RBF,
                      'kernel_param': {'input_dim': 2, 'ARD': True, 'lengthscale':[3, 2.5]},
                      'dist_type': 'gp',
                      'upper_x0':upper_x0
                      }
        #Hyper Parameters for the prior gamma
        dist_param_gamma = {'dist_type':'gamma',
                            'alpha':10.0, 
                            'beta':1.0, 
                            }
        #Hyper Parameters for the prior beta with the negative-binomial demand distribution.
        dist_param_beta = {'dist_type':'beta',
                            'initial_a':1.0, 
                            'initial_b':1.0, 
                            'n':10
                            }
        # Common settings
        param_common = {'p_inf':p_inf, 
                        'n_trials':n_tri, # the behavior at the end of learning process seems stable even when n_tri=25. 
                        'use_gurobi': use_gurobi,
                        'true_demand_dist': 'poisson', # Choose the true demand distribution. 
                        'range_p': range_p,
                        'range_t': range_t,
                        'upper_t': upper_t,
                        'experiment_id': np.random.randint(0, 2 ** 32 -1),
                        'dist_param': dist_param_gp, # If you use Gamma prior, use "dist_param_gamma" instead.
                        'range_n_episodes': range(n_episodes_gp), # If you use Gamma prior, use "n_episode_gamma" instead. 
                        'id_prefix': f'gp_Poisson'
                        }

        params.append(dict(**{'id': f'{param_common["experiment_id"]}_{param_common["id_prefix"]}_n50_LPE' + f'_T{upper_t}',
                                'seeds': [ _ for _ in range(param_common['n_trials'])],
                                'n_t0': n_t01,
                                'agent': 'TSE'
                                },
                            **param_common)
                            ) 
        params.append(dict(**{'id': f'{param_common["experiment_id"]}_{param_common["id_prefix"]}_n50_LPD'  + f'_T{upper_t}',
                                 'seeds': [ _ for _ in range(param_common['n_trials'])],
                                 'n_t0': n_t01, 
                                 'agent': 'TSD' 
                                 },
                             **param_common)
                        )
        params.append(dict(**{'id': f'{param_common["experiment_id"]}_{param_common["id_prefix"]}_n50_TSF'  + f'_T{upper_t}',
                                'seeds': [ _ for _ in range(param_common['n_trials'])],
                                'n_t0': n_t01,
                                'agent': 'TSF'
                                },
                            **param_common)
                            ) 
        params.append(dict(**{'id': f'{param_common["experiment_id"]}_{param_common["id_prefix"]}_n50_TSU'  + f'_T{upper_t}',
                                'seeds': [ _ for _ in range(param_common['n_trials'])],
                                'n_t0': n_t01,
                                'agent': 'TSU'}
                                ,
                            **param_common)
                            )
        
        # Run the experiments for the setting given in every item of the list "params".
        for param in params:
            self.element(param=param, processes = processes)


if __name__ == '__main__':
    exp.main()
