import pandas
import os

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

path = os.path.join(os.path.dirname(__file__), '../csv/sonar.all-data.csv')

observer = pandas.read_csv(path, names=["F1", "F2", "F3", " F4 ", "F5 ", "F6", "F7 ", "F8 ", "F9 ", "F10",
                                        "F11 ", "F12", "F13", "F14", "F15", "F16", "F17", "F18", "F19",
                                        "F20", "F21", "F22", "F23", "F24", "F25", "F26", "F27",
                                        "F28", "F29", "F30", "F31", "F32", "F33", "F34", "F35",
                                        "F36", "F37", "F38", "F39", "F40", "F41", "F42", "F43",
                                        "F44", "F45", "F46", "F47", " F48", "F49", "F50",
                                        "F51", "F52", "F53", "F54", "F55", "F56", "F57", "F58",
                                        "F59", "F60", "OBJET"])

"""Les k plus proches voisins Classification"""

print(observer.columns.values)

# Deactivation of the maximum number of columns of the dataframe to be displayed
pandas.set_option('display.max_columns', None)

# Display of the first line
# print(observer.head(1))

# display dimensions of the dataframe
# print(observer.shape)

# display infos of the dataframe
# classes == 208 (0 to 207)
# print(observer.info())


"""Séparation des données en bases d’apprentissage et de test"""

# Split dataset into train and test
array = observer.values

# Convertion des données en type decimal
X = array[:, 0:-1].astype(float)

# On choisit la dernière colonne comme feature de prédiction
Y = array[:, -1]

# Création des jeux d'apprentissage et de tests
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# K PLUS PROCHES VOISINS
knn = KNeighborsClassifier()
knn.fit(X_train, Y_train)
predictions = knn.predict(X_test)
print("K plus proches voisins : " + str(accuracy_score(predictions, Y_test)))
