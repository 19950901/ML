import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import numpy as np
"""
data=pd.read_csv(os.path.join("data","titanic_openml.csv"),na_values="?")
y=data["survived"]
X=data.drop(columns="survived")
X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=42)
clf=LogisticRegression()
clf.fit(X_train,y_train)
scores=clf.score()
"""
