import os

import pandas

# Deactivation of the maximum number of columns of the dataframe to be displayed
pandas.set_option('display.max_columns', None)

path = os.path.join(os.path.dirname(__file__), '../csv/Pokemon_dataset.csv')

observer = pandas.read_csv(path)

# print(observer.columns.values)
# Display of the first line
# print(observer.head(10))
# Combien de caractéristiques descriptives
print(observer.describe())
