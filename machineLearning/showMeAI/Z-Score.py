import numpy as np
import pandasHelloWorld as pd


def detect_outliers(data, threshold=4):
    mean_d = np.mean(data)
    print(mean_d)
    std_d = np.std(data)
    print(std_d)
    outliers = []
    for y in data:
        z_score = (y - mean_d) / std_d
        if np.abs(z_score) > threshold:
            outliers.append(y)
    return outliers


randomArray = np.random.normal(5.0, 1.0, 100000)

print(detect_outliers(randomArray))
