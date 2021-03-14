import mlrose
import numpy as np

import pandas as pd

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.metrics import accuracy_score
import os

from alg_runner import sim_annealing_runner, rhc_runner, ga_runner, mimic_runner
from plotting import plot_montecarlo_sensitivity

import pickle
from datetime import datetime


import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

np.random.seed(1)

def run_knapsack():

    if not os.path.exists('./output/Knapsack/'):
        os.mkdir('./output/Knapsack/')

    logger = logging.getLogger(__name__)
    problem_size = 25
    prob_size_int = int(problem_size)
    weights = [int(np.random.randint(1, prob_size_int/2)) for _ in range(prob_size_int)]
    values = [int(np.random.randint(1, prob_size_int/2)) for _ in range(prob_size_int)]
    flip_fit = mlrose.Knapsack(weights, values)
    flop_state_gen = lambda: np.random.randint(0, 1, size=prob_size_int)
    init_state = flop_state_gen()
    problem = mlrose.DiscreteOpt(length=prob_size_int, fitness_fn=flip_fit, maximize=True, max_val=2)

    all_results = {}
    """
    print("Running simulated annealing montecarlos")
    sa_results, sa_timing = sim_annealing_runner(problem)
    sa_best_params=plot_montecarlo_sensitivity('Knapsack', 'sim_anneal', sa_results)
    plot_montecarlo_sensitivity('Knapsack', 'sim_anneal_timing', sa_timing)
    all_results['SA'] = [sa_results, sa_timing]


    print("Running random hill montecarlos")
    rhc_results, rhc_timing = rhc_runner(problem)
    rhc_best_params = plot_montecarlo_sensitivity('Knapsack', 'rhc', rhc_results)
    plot_montecarlo_sensitivity('Knapsack', 'rhc_timing', rhc_timing)
    all_results['RHC'] = [rhc_results, rhc_timing]

    print("Running genetic algorithm montecarlos")
    ga_results, ga_timing = ga_runner(problem, init_state)
    ga_best_params = plot_montecarlo_sensitivity('Knapsack', 'ga', ga_results)
    plot_montecarlo_sensitivity('Knapsack', 'ga_timing', ga_timing)
    all_results['GA'] = [ga_results, ga_timing]

    print("Running MIMIC montecarlos")
    mimic_results, mimic_timing = mimic_runner(problem, init_state)
    MIMIC_best_params = plot_montecarlo_sensitivity('Knapsack', 'mimic', mimic_results)
    plot_montecarlo_sensitivity('Knapsack', 'mimic_timing', mimic_timing)
    all_results['MIMIC'] = [mimic_results, mimic_timing]

    """
    with open('./output/Knapsack/flipflip_data.pickle', 'wb') as handle:
        pickle.dump(all_results, handle, protocol=pickle.HIGHEST_PROTOCOL)

    problem_size_space = np.linspace(5, 80, 10, dtype=int)

    best_fit_dict = {}
    best_fit_dict['Problem Size'] = problem_size_space
    best_fit_dict['Random Hill Climbing'] = []
    best_fit_dict['Simulated Annealing'] = []
    best_fit_dict['Genetic Algorithm'] = []
    best_fit_dict['MIMIC'] = []


    times = {}
    times['Problem Size'] = problem_size_space
    times['Random Hill Climbing'] = []
    times['Simulated Annealing'] = []
    times['Genetic Algorithm'] = []
    times['MIMIC'] = []


    fits_per_iteration = {}
    fits_per_iteration['Random Hill Climbing'] = []
    fits_per_iteration['Simulated Annealing'] = []
    fits_per_iteration['Genetic Algorithm'] = []
    fits_per_iteration['MIMIC'] = []


    for prob_size in problem_size_space:
        logger.info("---- Problem size: " + str(prob_size) + " ----")
        prob_size_int = int(prob_size)
        weights = [int(np.random.randint(1, prob_size/2)) for _ in range(prob_size_int)]
        values = [int(np.random.randint(1, prob_size/2)) for _ in range(prob_size_int)]
        flip_fit = mlrose.Knapsack(weights, values, max_weight_pct=0.5)
        flop_state_gen = lambda: np.random.randint(0, 1, size=prob_size_int)
        init_state = flop_state_gen()
        problem = mlrose.DiscreteOpt(length=prob_size_int, fitness_fn=flip_fit, maximize=True, max_val=2)

        start = datetime.now()
        best_state_sa, best_fitness_sa,fitness_curve_sa = mlrose.simulated_annealing(problem,
                                                                   schedule=mlrose.ExpDecay(exp_const=.701,
                                                                                            init_temp=4.6,
                                                                                            min_temp=.401),
                                                                   max_attempts=260,
                                                                   max_iters=1100,curve=True)
        best_fit_dict['Simulated Annealing'].append(best_fitness_sa)
        end = datetime.now()
        times['Simulated Annealing'].append((end-start).total_seconds())


        start = datetime.now()
        best_state_rhc, best_fitness_rhc,fitness_curve_rhc = mlrose.random_hill_climb(problem, max_attempts=260,
                                                                    max_iters=1100,
                                                                  restarts=40,curve=True)
        best_fit_dict['Random Hill Climbing'].append(best_fitness_rhc)
        end = datetime.now()
        times['Random Hill Climbing'].append((end-start).total_seconds())


        start = datetime.now()
        best_state_ga, best_fitness_ga,fitness_curve_ga =  mlrose.genetic_alg(problem, pop_size=99,
                                                            mutation_prob=.01,
                                                             max_attempts=410,max_iters=1100,curve=True)
        best_fit_dict['Genetic Algorithm'].append(best_fitness_ga)
        end = datetime.now()
        times['Genetic Algorithm'].append((end-start).total_seconds())


        start = datetime.now()
        best_state_mimic, best_fitness_mimic,fitness_curve_mimic= mlrose.mimic(problem, pop_size=90,
                                                        keep_pct=.21,
                                                        max_attempts=10,max_iters=1100,curve=True)
        best_fit_dict['MIMIC'].append(best_fitness_mimic)
        end = datetime.now()
        times['MIMIC'].append((end-start).total_seconds())


    # For the last fit that occurs, save off the fit arrays that are generated. We will plot fitness/iteration.
    fits_per_iteration['Random Hill Climbing'] =fitness_curve_rhc
    fits_per_iteration['Simulated Annealing'] = fitness_curve_sa
    fits_per_iteration['Genetic Algorithm'] =  fitness_curve_ga
    fits_per_iteration['MIMIC'] =  fitness_curve_mimic

    fit_frame = pd.DataFrame.from_dict(best_fit_dict, orient='index').transpose()
    # fit_frame.pop('Unnamed: 0') # idk why this shows up.
    time_frame = pd.DataFrame.from_dict(times, orient='index').transpose()
    # time_frame.pop('Unnamed: 0') # idk why this shows up.
    fit_iteration_frame = pd.DataFrame.from_dict(fits_per_iteration, orient='index').transpose()

    fit_frame.to_csv('./output/Knapsack/problem_size_fit.csv')
    time_frame.to_csv('./output/Knapsack/problem_size_time.csv')
    fit_iteration_frame.to_csv('./output/Knapsack/fit_per_iteration.csv')

if __name__ == "__main__":
    run_knapsack()