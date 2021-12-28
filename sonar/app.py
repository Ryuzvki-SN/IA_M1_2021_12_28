import pandas
import os
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix


# Deactivation of the maximum number of columns of the dataframe to be displayed

pandas.set_option('display.max_columns', None)

path = os.path.join(os.path.dirname(__file__), '../csv/sonar.all-data.csv')

observer = pandas.read_csv(path, names=["F1", "F2", "F3", "F4", "F5", "F6", "F7", "F8", "F9", "F10",
                                        "F11 ", "F12", "F13", "F14", "F15", "F16", "F17", "F18", "F19",
                                        "F20", "F21", "F22", "F23", "F24", "F25", "F26", "F27",
                                        "F28", "F29", "F30", "F31", "F32", "F33", "F34", "F35",
                                        "F36", "F37", "F38", "F39", "F40", "F41", "F42", "F43",
                                        "F44", "F45", "F46", "F47", " F48", "F49", "F50",
                                        "F51", "F52", "F53", "F54", "F55", "F56", "F57", "F58",
                                        "F59", "F60", "OBJET"])

"""Classification"""
print(observer.columns.values)
print("----------------------------------------------------------\n\n----------------------------------------------------------")

# Display of the first line
print(observer.head(1))
print("----------------------------------------------------------\n\n----------------------------------------------------------")

# display dimensions of the dataframe
print(observer.shape)
print("----------------------------------------------------------\n\n----------------------------------------------------------")

# display infos of the dataframe
print(observer.info())
print("----------------------------------------------------------\n\n----------------------------------------------------------")

# classes == 2 (R and M)
# Combien de caractéristiques descriptives ? De quels types ?
print(observer.describe())  # types(count, mean, std, min, max)
print("----------------------------------------------------------\n\n----------------------------------------------------------")

# Calculer les statistiques de base des variables 2 à 7
stat_base = observer[["F2", "F3", "F4", "F5", "F6", "F7"]]
print(stat_base.describe())
print("----------------------------------------------------------\n\n----------------------------------------------------------")

# Ex: 208 elements
# Combien d’exemples de chaque classe
values_expl = observer['OBJET'].value_counts()
print("Exemples de chaque classe  : " + str(values_expl))
print("----------------------------------------------------------\n\n----------------------------------------------------------")

# Comment sont organisés les exemples


"""Separation of data into training and test databases"""

# Split dataset into train and test
array = observer.values

# Convertion des données en type decimal
X = array[:, 0:-1].astype(float)

# On choisit la dernière colonne comme feature de prédiction
Y = array[:, -1]

# Création des jeux d'apprentissage et de tests
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state=42)

# #instanciation et définition du k
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train, Y_train)
print("Train score : " + str(knn.score(X_train, Y_train)))
print("Test score : " + str(knn.score(X_test, Y_test)))
predictions = knn.predict(X_test)
print(confusion_matrix(predictions,Y_test))
print("----------------------------------------------------------\n\n-----------------------------"-----------------------------)
print("Matrix de confusion : " + str(confusion_matrix(predictions, Y_test)))
print("----------------------------------------------------------\n\n----------------------------------------------------------")

def accuracy(k, x_train, y_train, x_test, y_test):
    """
    compute accuracy of the classification based on k values
    """
    # instantiate learning model and fit data
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(x_train, y_train)

    # predict the response
    pred = model.predict(x_test)

    # evaluate and return  accuracy
    return accuracy_score(y_test, pred)


"""Tableau de Score"""
rows_nbr = observer.shape[0]
tab_score = np.array([accuracy(k, X_train, Y_train, X_test, Y_test)
                      for k in range(1, int(rows_nbr / 2))])
print(tab_score)
print("----------------------------------------------------------\n\n-----------------------------"-----------------------------)

"""Tableau de K"""
tab_k = []
for item in range(1, int(rows_nbr / 2)):
    tab_k.append(item)

print(tab_k)
print("----------------------------------------------------------\n\n-----------------------------"-----------------------------)

plt.plot(tab_score, linewidth=2)
plt.title("la courbe de k en fonction des scores", fontsize=16)
plt.xlabel("Nombre d'iterations", fontsize=14)
plt.ylabel("Score", fontsize=14)
plt.axis([0, 104, 0, 1])
plt.grid()
plt.show()

