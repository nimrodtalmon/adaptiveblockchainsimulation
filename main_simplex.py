import os

number_of_experiments = 3

if os.path.exists("logs/simplex.txt"):
    os.system("del logs/simplex.txt")
for i in range(number_of_experiments):
    os.system(f"python3 main_put_simplex_lines_in_file.py")
os.system("python3 main_plot_simplex_from_file.py")