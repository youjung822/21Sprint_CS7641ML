import os
from cpeaks import run_cpeaks
from flipflop import run_flipflop
from knapsack import run_knapsack

from plot_toy_results import plot_toy_problems


if __name__ == '__main__':
    # Run the neural net stuff
    if not os.path.exists('./output/'):
        os.mkdir('./output/')
    
    # Run toy problems
    run_cpeaks()
    run_flipflop()
    run_knapsack()

    # Plot toy problems
    problems = ['FlipFlop', 'CPeaks', 'Knapsack']
    plot_toy_problems(problems)
