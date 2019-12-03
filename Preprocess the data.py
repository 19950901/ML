"""
pipeline管道机制：流水线的输入为一连串的数据挖掘步骤，
其中最后一步必须是估计器，可理解成分类器，前几步是转换器，
输入的数据集经过转换器的处理后，输出的结果作为下一步的输入，
最后，用位于流水线最后一步的估计器对数据进行分类。

Pipeline可以将许多算法模型串联起来，比如将特征提取、归一化、分类等组织在一起形成一个典型的机器学习问题工作流。
"""
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_validate
import pandas as pd

#通过归一化数据，模型的收敛速度要比未归一化的数据快得多(迭代次数变少了)。
X,y=load_digits(return_X_y=True)
X_train,X_test,y_train,y_test=train_test_split(X,y,stratify=y,random_state=42)
"""
#MinMaxScaler变换器用于规范化数据
scaler=MinMaxScaler()
X_train_scaled=scaler.fit_transform(X_train)
X_test_scaled=scaler.transform(X_test)

clf=LogisticRegression(solver="lbfgs",multi_class="auto",max_iter=1000,random_state=42)
clf.fit(X_train_scaled,y_train)
accuracy=clf.score(X_test_scaled,y_test)
print("Accuracy score of the {} is {:.2f}".format(clf.__class__.__name__,accuracy))
#得到训练模型所需要的迭代次数
print("{} required {} iterations to be fitted".format(clf.__class__.__name__,clf.n_iter_[0]))
"""
#pipe=Pipeline(steps=[("scaler",MinMaxScaler()),
#                    ("clf",LogisticRegression(solver="lbfgs",multi_class="auto",random_state=42))])
#  OR
pipe=make_pipeline(MinMaxScaler(),
                   LogisticRegression(solver="lbfgs",multi_class="auto",random_state=42,max_iter=1000))

pipe.fit(X_train,y_train)
accuracy=pipe.score(X_test,y_test)
print("Accuracy score of the {} is {:.2f}".format(pipe.__class__.__name__,accuracy))

scores=cross_validate(pipe,X,y,cv=3,return_train_score=True)

df_scores=pd.DataFrame(scores)
print(df_scores)
