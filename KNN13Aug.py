import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split


cell_df=pd.read_excel("Seed.xlsx")


print(cell_df['Class'].value_counts())
print(cell_df.shape)

cell_df=cell_df[cell_df['Class']<3]
print(cell_df.shape)

class1_df=cell_df[cell_df['Class']==1][:100]
class2_df=cell_df[cell_df['Class']==2][:100]


axes=class1_df.plot(kind='scatter',x='Area',y='Length of kernel groove',color='blue',label='class1')
class2_df.plot(kind='scatter',x='Area',y='Length of kernel groove',color='red',label='class2',ax=axes)
plt.show()

print(cell_df.dtypes)


feature_df=cell_df[['Area', 'Perimeter', 'Compactness', 'Length of kernel','Width of kernel', 'Asymmetry coefficient', 'Length of kernel groove']]
X=np.asarray(feature_df)#independent variables
y=np.asarray(cell_df['Class'])#dependent variables
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=4)

knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train,y_train)
#KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',metric_params=None, n_jobs=None, n_neighbors=1, p=2,weights='uniform')
pred = knn.predict(X_test)

print(cell_df)
#choosing k value 1-40
accuracy_rate = []

for i in range(1, 40):
    knn = KNeighborsClassifier(n_neighbors=i)
    score = cross_val_score(knn, feature_df, cell_df['Class'], cv=10)
    accuracy_rate.append(score.mean())
#---------------
error_rate = []

for i in range(1, 40):
    knn = KNeighborsClassifier(n_neighbors=i)
    score = cross_val_score(knn, feature_df, cell_df['Class'], cv=10)
    error_rate.append(1 - score.mean())



#plotting error rates
plt.figure(figsize=(10,6))
plt.plot(range(1,40),accuracy_rate,color='blue', linestyle='dashed', marker='o',
         markerfacecolor='red', markersize=10)
plt.title('Error rate vs. K count')
plt.xlabel('K')
plt.ylabel('Error Rate')
plt.show()

#for K=20
knn = KNeighborsClassifier(n_neighbors=20)

knn.fit(X_train,y_train)
pred = knn.predict(X_test)

print('for K=20')
print('\n')
print(confusion_matrix(y_test,pred))
print('\n')
print(classification_report(y_test,pred))