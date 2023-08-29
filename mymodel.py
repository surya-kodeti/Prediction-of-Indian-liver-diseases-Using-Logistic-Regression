

# %% [code]
import numpy as np
import pandas as pd
import pickle 



# %% [code]

patients=pd.read_csv('indian_liver_patient.csv')


patients['Gender']=patients['Gender'].apply(lambda x:1 if x=='Male' else 0)

patients['Albumin_and_Globulin_Ratio'].mean()

# %% [code]
patients=patients.fillna(0.94)


# %% [code]
patients.isnull().sum()


# %% [code]
from sklearn.model_selection import train_test_split

# %% [code]
patients.columns



# %% [code]
X=patients[['Age', 'Gender', 'Total_Bilirubin', 'Direct_Bilirubin',
       'Alkaline_Phosphotase', 'Alamine_Aminotransferase',
       'Aspartate_Aminotransferase', 'Total_Protiens', 'Albumin',
       'Albumin_and_Globulin_Ratio']]
y=patients['Dataset']

# %% [code]
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=123)

# %% [markdown]
# **We split the training and testing  in a certain ratio as 70 for training and 30 for testing.**

# %% [markdown]
# **Now inorder to build our model we use Logistic Regression**

# %% [code]
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
logmodel = LogisticRegression()
logmodel.fit(X_train,y_train)
y_pred = logmodel.predict(X_test)
print("accuracy LR= ",accuracy_score(y_test,y_pred))



'''
from sklearn.ensemble import RandomForestClassifier
rforest=RandomForestClassifier()
rforest.fit(X_train,y_train)
y_pred_rforest=rforest.predict(X_test)
print("accuracy Random Forest= ",accuracy_score(y_test,y_pred_rforest))
'''






# Saving model to disk
pickle.dump(logmodel, open('model.pkl','wb'))


# Loading model to compare the results
# model = pickle.load(open('model.pkl','rb'))
# print(model.predict([[20,1,1.1,0.5,128,20,30,3.9,1.9,0.95]]))
