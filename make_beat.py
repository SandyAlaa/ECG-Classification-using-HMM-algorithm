import numpy as np
from ecg_data import generate_beat

samples = [
    ("N", 101, "record_01.csv"),
    ("S", 102, "record_02.csv"),
    ("V", 103, "record_03.csv"),
    ("F", 104, "record_04.csv"),
    ("Q", 105, "record_05.csv"),
]

for cls, sd, filename in samples:
    rng = np.random.RandomState(sd)
    beat = generate_beat(cls, rng)

    np.savetxt(filename, beat, delimiter=",")
    print(f"saved {filename}")