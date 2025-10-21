import os

# CONFIGURATION
number_of_experiments = 1
number_of_repetitions = 50
do_simplex = False
do_steady_state = True

# SIMPLEX EXPERIMENTS
if do_simplex:
    if os.path.exists("logs/simplex.txt"):
        os.system("rm logs/simplex.txt")
    for i in range(number_of_experiments):
        os.system(f"python3 main_simplex.py --runs {number_of_repetitions} --workers {number_of_repetitions}")
    os.system("python3 create_governancecontrolplot.py")

# STEADY STATE EXPERIMENTS
if do_steady_state:
    if os.path.exists("logs/steady_state.txt"):
        os.system("rm logs/steady_state.txt")
    for i in range(number_of_experiments):
        os.system(f"python3 main_one_instance.py --runs {number_of_repetitions} --workers {number_of_repetitions}")
    os.system("python3 showsteadystate.py")
