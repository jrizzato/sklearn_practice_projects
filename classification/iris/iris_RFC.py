# Goal: Build a FastAPI service that:
# 1. Loads a pretrained ML model
# 2. Accepts JSON input for inference

from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
import joblib

iris = load_iris()
X = iris.data
y = iris.target

model = RandomForestClassifier()
model.fit(X, y)

joblib.dump(model, './classification/iris/model/iris_model.pkl')