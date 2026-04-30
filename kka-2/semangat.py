import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
data = pd.read_csv('job_salary_prediction_dataset.csv')
print(data.head())
print(data.info())
print(data.isnull().sum())