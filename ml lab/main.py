import pandas as pd

file_path = "iris.data"

names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class']
iris_data = pd.read_csv(file_path, names=names)

summary = iris_data.describe()
class_correlation = iris_data.groupby('class')[['sepal_length', 'petal_length']].mean()

print("Statistical Summary:")
print(summary[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']])
print("\nClass Correlation (Mean):")
print(class_correlation)

min_values = iris_data.groupby('class').min()[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
max_values = iris_data.groupby('class').max()[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
mean_values = iris_data.groupby('class').mean()[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
std_values = iris_data.groupby('class').std()[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]

print("\nAdditional Information:")
print("\nMin Values:")
print(min_values)
print("\nMax Values:")
print(max_values)
print("\nMean Values:")
print(mean_values)
print("\nStandard Deviation Values:")
print(std_values)
