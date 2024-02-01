import pandasHelloWorld as pd
import numpy as np

date1 = pd.Series([-10,1, 2, 3, 4, 7, 9, 10, 13, 24, 30, 40, 50])
#df = pd.DataFrame(np.array([[1, 1], [2, 10], [3, 100], [4, 100]]),columns=['a', 'b'])

def detect_outliers(sr):
    print(len(sr))
    # 1+(n-1)*0.25
    q1 = sr.quantile(0.25)
    print(q1)
    q3 = sr.quantile(0.75)
    print(q3)
    iqr = q3 - q1  # Interquartile range
    print(iqr)
    fence_low = q1 - 1.5 * iqr
    print(fence_low)
    fence_high = q3 + 1.5 * iqr
    print(fence_high)
    print(sr.loc[sr<q1])
    outliers = sr.loc[(sr < fence_low) | (sr > fence_high)]
    return outliers


print(detect_outliers(date1))
