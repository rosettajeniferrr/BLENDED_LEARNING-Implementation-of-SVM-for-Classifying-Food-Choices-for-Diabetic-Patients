# BLENDED LEARNING
# Implementation of Support Vector Machine for Classifying Food Choices for Diabetic Patients

## AIM:
To implement a Support Vector Machine (SVM) model to classify food items and optimize hyperparameters for better accuracy.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Load the dataset and select features and target variable.
2. Split the data into training and testing sets and apply feature scaling.
3. Train the SVM model using GridSearchCV to find the best parameters.
4. Predict the test data and evaluate the model using accuracy and classification report.
5. Generate and display the confusion matrix to visualize model performance.

## Program:
```
/*
Program to implement SVM for food classification for diabetic patients.
Developed by: Rosetta Jenifer.C
RegisterNumber:    212225230230
*/
import pandas as pd
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
data=pd.read_csv('food_items_binary.csv')
print(data.head())
print(data.columns)
features = ['Calories','Total Fat','Saturated Fat','Sugars','Dietary Fiber','Protein']
target = 'class'
X=data[features]
y=data[target]
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
scaler = StandardScaler()
X_train=scaler.fit_transform(X_train)
X_test=scaler.transform(X_test)
svm=SVC()
param_grid={
    'C':[0.1,1,100],
    'kernel':['linear','rbf'],
    'gamma':['scale','auto']
}
grid_search=GridSearchCV(svm,param_grid,cv=5,scoring='accuracy')
grid_search.fit(X_train,y_train)
best_model = grid_search.best_estimator_
print("Name:Rosetta Jenifer.C")
print("Register Number:212225230230")
print("Best Parameters:",grid_search.best_params_)
y_pred=best_model.predict(X_test)
accuracy=accuracy_score(y_test,y_pred)
print("Name:Rosetta Jenifer.C")
print("Register Number:212225230230")
print("Accuracy:",accuracy)
print("Classification Report:\n",classification_report(y_test,y_pred))
conf_matrix=confusion_matrix(y_test,y_pred)
sns.heatmap(conf_matrix,annot=True,fmt="d",cmap="Blues")
plt.xlabel('Predicted')
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()
```

## Output:
<img width="922" height="747" alt="image" src="https://github.com/user-attachments/assets/ccb1ec2b-0d2a-48a7-a137-8c4182cd5bf4" />
<img width="456" height="147" alt="image" src="https://github.com/user-attachments/assets/1a76a3f7-35c1-4b24-b737-35104f4e32ca" />
<img width="817" height="102" alt="image" src="https://github.com/user-attachments/assets/2ed03f80-84ea-4152-a8f7-cdabe03f8a64" />
<img width="722" height="322" alt="image" src="https://github.com/user-attachments/assets/5ad45ee8-0071-4263-a46c-841ce75ddc77" />
<img width="880" height="652" alt="image" src="https://github.com/user-attachments/assets/f9be5d00-fe08-465f-8016-11642c08b599" />



## Result:
Thus, the SVM model was successfully implemented to classify food items for diabetic patients, with hyperparameter tuning optimizing the model's performance.
