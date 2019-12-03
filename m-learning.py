import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

X,y=load_digits(return_X_y=True)
#.imshow()函数负责对图像进行处理，进行绘图，但是不能显示图像
#plt.imshow(X[0].reshape(8,8),cmap="gray")
#.show()用来显示处理之后的图像
#plt.show()
#将数据集划分为训练集和测试集
X_train,X_test,y_train,y_test=train_test_split(X,y,stratify=y,random_state=42)
"""
#构造回归模型分类器
clf=LogisticRegression(solver="lbfgs",multi_class="ovr",max_iter=5000,random_state=42)
clf.fit(X_train,X_test)

#对训练所得模型进行测试,得到其测试精度
accuracy=clf.score(y_train,y_test)
print("Accuracy score of the {} is {:.2f}".format(clf.__class__.__name__,accuracy))
"""
#构造随机森林分类器
clf=RandomForestClassifier(n_estimators=100,n_jobs=-1,random_state=42)
clf.fit(X_train,y_train)
accuracy=clf.score(X_test,y_test)
print("Accuracy score of the {} is {:.2f}".format(clf.__class__.__name__,accuracy))
















