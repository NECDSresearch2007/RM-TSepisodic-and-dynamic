"""
This work © 2024 by NEC Corporation is licensed under Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International.
"""
import functools
import time
from multiprocessing import Pool

import numpy as np

from thompson_lp import TS_episodic, TS_episodic_oracle, TS_dynamic_oracle
from demand_dists import observe_demand_poisson, observe_demand_poisson_v2, observe_demand_negbinom
from demand_dists import true_lamb_poisson, true_lamb_poisson_v2, true_mean_demand
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
        
        observe_demand = self.true_demand_dist[param['true_demand_dist']]
        true_lamb = self.true_demand[param['true_demand_dist']]
        observe_demand_p = functools.partial(observe_demand, random_state=random_state)
        agent = self.create_agent(param)
        print('run')
        lamb_dist, revenue_episode, history = agent.run(
        param['n_t0'], param['range_p'], param['p_inf'], param['range_t'],
        param['range_n_episodes'], observe_demand_p, true_lamb,
        param['dist_param'], param['use_gurobi'], random_state=random_state)
        return history
    

    @classmethod
    def repeat(self, param, processes):
        random_states = [np.random.RandomState(seed) 
                        for seed in param['seeds']] 
        map_func = functools.partial(self.get_history, param=param)
        print('map function')
        with Pool(processes=processes) as pool:
            repeated_history = pool.map(map_func, random_states)
        return repeated_history

    @classmethod
    def element(self, param, processes):
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
        Run experiments.
        """
        upper_t = 10.0
        range_p = np.arange(1, 10, 1)
        range_t = np.arange(1, upper_t + 1, 1)
        upper_x0 = TS_episodic._setup_upper_x(range_p, range_t)

        # Set algorithms' name.
        self.agent_dic = {'TSE*': TS_episodic_oracle, 'TSD*': TS_dynamic_oracle}
        # Set true_demand parameter of the demand distribution.
        self.true_demand_dist = {'poisson': observe_demand_poisson, 
                                 'poisson_v2': observe_demand_poisson_v2,
                                  'negbinom': observe_demand_negbinom
                                  }
        # Set true_mean_demand the oracle algorithms use
        self.true_demand = {'poisson': true_lamb_poisson, 
                                 'poisson_v2': true_lamb_poisson_v2,
                                  'negbinom': true_mean_demand
                                  }

        params = []
        # Initial inventory
        n_t0 = 1000
        n_t1 = 50
        # the number of independent trial
        n_tri = 10
        # shut-off price
        p_inf = 100000.0
        # number of episodes
        n_episodes = 1
        # If False, use cvxopt to solve the LP instead of Gurobi.
        use_gurobi = True
        # Number of CPUs used for parallel computing. 
        processes = 1

        dist_param = {'upper_x0':upper_x0}
        
        param_common = {'p_inf':p_inf, 'n_trials':n_tri,
                        'range_n_episodes': range(n_episodes),
                        'true_demand_dist': 'poisson',
                        'range_p': range_p,
                        'range_t': range_t,
                        'use_gurobi': use_gurobi,
                        'dist_param': dist_param,
                        }


        params.append(dict(**{'id': f'{np.random.randint(0, 2 ** 32 -1)}_n1000_LPE*'  + f'_T{upper_t}',
                                'seeds': [ _
                                for _ in range(param_common['n_trials'])],
                                'n_t0': n_t0,
                                'agent': 'TSE*'
                                },
                            **param_common)
                            )
        params.append(dict(**{'id': f'{np.random.randint(0, 2 ** 32 -1)}_n1000_LPD*'  + f'_T{upper_t}',
                                'seeds': [ _
                                for _ in range(param_common['n_trials'])],
                                'n_t0': n_t0,
                                'agent': 'TSD*'
                                },
                            **param_common)
                            )
        
        params.append(dict(**{'id': f'{np.random.randint(0, 2 ** 32 -1)}_n50_LPE*'  + f'_T{upper_t}',
                                'seeds': [ _
                                for _ in range(param_common['n_trials'])],
                                'n_t0': n_t1,
                                'agent': 'TSE*'
                                },
                            **param_common)
                            )
        
        params.append(dict(**{'id': f'{np.random.randint(0, 2 ** 32 -1)}_n50_LPD*'  + f'_T{upper_t}',
                                'seeds': [ _
                                for _ in range(param_common['n_trials'])],
                                'n_t0': n_t1,
                                'agent': 'TSD*'
                                },
                            **param_common)
                            )
                            
        
        for param in params:
            self.element(param=param, processes = processes)

            
    
if __name__ == '__main__':
    exp.main()