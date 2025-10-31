import os
from pathlib import Path

number_of_experiments = 3

path = Path("logs") / "simplex.txt"
if path.exists():
    path.unlink()
    
for i in range(number_of_experiments):
    os.system(f"python3 main_put_simplex_lines_in_file.py")
os.system("python3 main_plot_simplex_from_file.py")