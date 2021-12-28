import os

import numpy
import pandas
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

pandas.set_option('display.max_columns', None)

path = os.path.join(os.path.dirname(__file__), '../csv/Pokemon_dataset.csv')

observer = pandas.read_csv(path)
# print(observer.columns.values)
# Display of the first line
# print(observer.info())
#  Combien de caractéristiques descriptives ? De quels types ?
# print(pokedex.describe())  # types()

observer.boxplot()

"""Separation of data into training and test databases"""

X = observer.iloc[:, 4:11].values
Y = observer.iloc[:, 16].values

# Création des jeux d'apprentissage et de tests
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
Y_test = numpy.nan_to_num(Y_test)
Y_train = numpy.nan_to_num(Y_train)
# Choix de l'algorithme
algo = LinearRegression()
# # Apprentissage à l'aide de la fonction fit
algo.fit(X_train, Y_train)
print("Train score : " + str(algo.score(X_train, Y_train)))
print("Test score : " + str(algo.score(X_test, Y_test)))
