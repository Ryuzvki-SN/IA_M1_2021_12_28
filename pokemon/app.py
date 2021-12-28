import os
from sklearn.linear_model import LinearRegression

import numpy
import matplotlib.pyplot as plt
import pandas
from sklearn.model_selection import train_test_split

# Deactivation of the maximum number of columns of the dataframe to be displayed
pandas.set_option('display.max_columns', None)

path = os.path.join(os.path.dirname(__file__), '../csv/Pokemon_dataset.csv')

print("----------------------------------------------------------\n\n----------------------------------------------------------")
observer = pandas.read_csv(path)
print(observer.columns.values)
print("----------------------------------------------------------\n\n----------------------------------------------------------")
# Display of the first line
#HEAD
# Display of the first line
print(observer.head(1))
print("----------------------------------------------------------\n\n----------------------------------------------------------")

# Combien de caractéristiques descriptives ? De quels types ?
print(observer.describe())  # types(count, mean, std, min, max)
print("----------------------------------------------------------\n\n----------------------------------------------------------")


#Tracer les boxplots de toutes les variables
observer.boxplot()

#Calculer et tracer la matrice de corrélation des différentes features
features = observer[["POINTS_DE_VIE", "NIVEAU_ATTAQUE", "NIVEAU_DEFENSE", "NIVEAU_ATTAQUE_SPECIALE", "NIVEAU_DEFENSE_SPECIALE", "VITESSE","GENERATION"]]
var = features.corr()
print("FEATURES")
print(var)
print("----------------------------------------------------------\n\n----------------------------------------------------------")

plt.matshow(var)
plt.show()

"""Separation of data into training and test databases"""

# Split dataset into train and test
X = observer.iloc[:, 5:11].values
Y = observer.iloc[:, 17].values

# Création des jeux d'apprentissage et de tests
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
Y_test = numpy.nan_to_num(Y_test)
Y_train = numpy.nan_to_num(Y_train)
# Choix de l'algorithme
algo = LinearRegression()
# Apprentissage à l'aide de la fonction fit
algo.fit(X_train, Y_train)
print("Train score : " + str(algo.score(X_train, Y_train)))
print("Test score : " + str(algo.score(X_test, Y_test)))

#test prediction
predictions = algo.predict(X_test)
plt.scatter(Y_test, predictions)

#train prediction
prediction = algo.predict(X_train)
plt.scatter(Y_train, prediction)
#affichage des nuages de points
plt.show()

#coefficients de corrélation
print("coefficient de corrélationn test : " +str(numpy.corrcoef(Y_test,predictions, rowvar=False)))
print("coefficient de corrélationn train : " +str(numpy.corrcoef(Y_train,prediction, rowvar=False)))

