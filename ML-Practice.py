from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import balanced_accuracy_score

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_validate

#加载数据集
X_breast,Y_breast=load_breast_cancer(return_X_y=True)

#对数据集划分为训练集和测试集
X_breast_train,X_breast_test,y_breast_train,y_breast_test=train_test_split(X_breast,Y_breast,stratify=Y_breast,
                                            random_state=0,test_size=0.3) #TestDataset account for 30% of DataSet

#使用训练数据训练监督模型"梯度提升分类器"
clf=GradientBoostingClassifier(n_estimators=100,random_state=0)
clf.fit(X_breast_train,y_breast_train)

#使用拟合分类器预测测试集的分类标签,为模型预测到的标签
y_pred=clf.predict(X_breast_test)

#计算测试集的balanced_accuracy_score
#accuracy=balanced_accuracy_score(y_breast_test,y_pred)
#print("Accuracy score of the {} is {:.2f}".format(clf.__class__.__name__,accuracy))

pipe=make_pipeline(StandardScaler(),SGDClassifier(max_iter=1000))

scores=cross_validate(pipe,X_breast,Y_breast,scoring="balanced_accuracy",cv=3,return_train_score=True)
df_scores=pd.DataFrame(scores)
df_scores[["train_score","test_score"]].boxplot()
plt.show()
"""
pipe.fit(X_breast_train,y_breast_train)
y_pred=pipe.predict(X_breast_test)
accuracy=balanced_accuracy_score(y_breast_test,y_pred)
print("Accuracy score of the {} is {:.2f}".format(pipe.__class__.__name__,accuracy))
"""