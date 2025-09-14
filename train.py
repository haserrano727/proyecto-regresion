# Importar librerías
from sklearn.datasets import load_diabetes
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeRegressor

# Datos de entrada
X = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])

y = np.dot(X, np.array([1, 2])) + 3

# Creación y entrenamiento del modelo
reg = LinearRegression().fit(X, y)
