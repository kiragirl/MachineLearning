import numpy as np
import pandas as pd

company = ["A", "B", "C"]
data = pd.DataFrame({
    "company": [company[x] for x in np.random.randint(0, len(company), 10)],
    "salary": np.random.randint(5, 50, 10),
    "age": np.random.randint(15, 50, 10)})
group = data.groupby("company")
print(list(group))
print(data.groupby("company").agg('mean'))
print(data.groupby("company")['age'].agg('mean'))
print(data.groupby('company').agg({'salary': 'median', 'age': 'mean'}))

print("-------------transform变换------------")
# avg_salary_dict = data.groupby('company')['salary'].mean().to_dict()
# data['avg_salary'] = data['company'].map(avg_salary_dict)
data['avg_salary'] = data.groupby('company')['salary'].transform('mean')
print(data)

print("-------------apply方法------------")


def get_oldest_staff(x):
    df = x.sort_values(by='age', ascending=True)
    return df.iloc[-1, :]


oldest_staff = data.groupby('company', as_index=False).apply(get_oldest_staff)
print(oldest_staff)
