import os
import numpy as np

file_name = "results/forecast_raw.txt"
c1 = np.array([1,2,3])
c2 = np.array([2,3,4])
value = np.column_stack([c1,c2])
c3 = np.array([1,2,3])

path_to_file = file_name
os.makedirs(os.path.dirname(path_to_file), exist_ok=True)
with open(path_to_file, "a") as f:
    f.write(str(value)+"\n")
    f.close()

