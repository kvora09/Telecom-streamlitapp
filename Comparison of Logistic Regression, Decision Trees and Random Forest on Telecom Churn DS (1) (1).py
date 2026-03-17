#!/usr/bin/env python
# coding: utf-8

# # Telecom Churn Prediction

# You will use 21 variables related to customer behaviour (such as monthly bill, internet usage, etc.) to predict whether a particular customer will switch to another telecom provider or not, i.e., whether they will churn or not.

# ## Step 1: Importing and Merging Data

# In[1]:


#Supressing warnings
import warnings
warnings.filterwarnings("ignore")


# In[2]:


#Importing Pandas and Numpy
import pandas as pd,numpy as np


# In[3]:


# Importing all datasets
churn_data=pd.read_csv("churn_data.csv")


# In[4]:


churn_data.head()


# In[5]:


customer_data=pd.read_csv("customer_data.csv")


# In[6]:


customer_data.head()


# In[7]:


internet_data=pd.read_csv("internet_data.csv")
internet_data.head()


# ### Combining all the data files into one consolidated dataframe

# In[8]:


# merging on customer_id
df_1=pd.merge(churn_data,customer_data,how="inner",on="customerID")


# In[ ]:


#Final dataframe
telecom=pd.merge(df_1,internet_data,how="inner",on="customerID")


# ## Inspecting the dataframe

# In[ ]:


telecom.head()


# In[ ]:


telecom.shape


# In[ ]:


telecom.info()


# In[ ]:


telecom.describe()


# ## Data preparation

# ### Converting some binary variables to 0/1

# In[ ]:


# List of variables to map
varlist=["PhoneService","PaperlessBilling","Churn","Partner","Dependents"]

def binary_map(x):
    return x.map({"Yes":1,"No":0})

telecom[varlist]=telecom[varlist].apply(binary_map)


# In[ ]:


telecom.head()


# In[ ]:


##For cataegorical variables with multiple levels, creating dummy variables, one hot encoded.


dummy1= pd.get_dummies(telecom[["Contract", "gender", "InternetService"]], drop_first=True)
# Adding the results to the master dataframe
telecom = pd.concat([telecom, dummy1], axis=1)


# In[ ]:


telecom.head()


# In[ ]:


# Creating dummy variables for the variable 'MultipleLines'
ml = pd.get_dummies(telecom['MultipleLines'], prefix='MultipleLines')
# Dropping MultipleLines_No phone service column
ml1 = ml.drop(['MultipleLines_No phone service'],axis=1)
#Adding the results to the master dataframe
telecom = pd.concat([telecom,ml1], axis=1)

# Creating dummy variables for the variable 'OnlineSecurity'.
os = pd.get_dummies(telecom['OnlineSecurity'], prefix='OnlineSecurity')
os1 = os.drop(['OnlineSecurity_No internet service'], axis=1)
# Adding the results to the master dataframe
telecom = pd.concat([telecom,os1], axis=1)

# Creating dummy variables for the variable 'OnlineBackup'.
ob = pd.get_dummies(telecom['OnlineBackup'], prefix='OnlineBackup')
ob1 = ob.drop(['OnlineBackup_No internet service'], axis=1)
# Adding the results to the master dataframe
telecom = pd.concat([telecom,ob1], axis=1)

# Creating dummy variables for the variable 'DeviceProtection'. 
dp = pd.get_dummies(telecom['DeviceProtection'], prefix='DeviceProtection')
dp1 = dp.drop(['DeviceProtection_No internet service'], axis=1)
# Adding the results to the master dataframe
telecom = pd.concat([telecom,dp1], axis=1)

# Creating dummy variables for the variable 'TechSupport'. 
ts = pd.get_dummies(telecom['TechSupport'], prefix='TechSupport')
ts1 = ts.drop(['TechSupport_No internet service'], axis=1)
# Adding the results to the master dataframe
telecom = pd.concat([telecom,ts1], axis=1)

# Creating dummy variables for the variable 'StreamingTV'.
st =pd.get_dummies(telecom['StreamingTV'], prefix='StreamingTV')
st1 = st.drop(['StreamingTV_No internet service'], axis=1)
# Adding the results to the master dataframe
telecom = pd.concat([telecom,st1], axis=1)

# Creating dummy variables for the variable 'StreamingMovies'. 
sm = pd.get_dummies(telecom['StreamingMovies'], prefix='StreamingMovies')
sm1 = sm.drop(['StreamingMovies_No internet service'], axis=1)
# Adding the results to the master dataframe
telecom = pd.concat([telecom,sm1], axis=1)


# In[ ]:


pm= pd.get_dummies(telecom["PaymentMethod"], drop_first=True)
# Adding the results to the master dataframe
telecom = pd.concat([telecom, pm], axis=1)


# In[ ]:


telecom.head()


# ### Dropping repeated variables as we have created dummy variables

# In[ ]:


telecom = telecom.drop(['Contract','PaymentMethod','gender','MultipleLines','InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
       'TechSupport', 'StreamingTV', 'StreamingMovies'], axis=1)


# In[ ]:


telecom.isnull().sum()


# In[ ]:


(telecom['TotalCharges'] == " ").sum()


# In[ ]:


telecom=telecom[~(telecom.TotalCharges==" ")]


# In[ ]:


telecom.info()


# In[ ]:


## Convert total charges obj type into numeric
telecom.TotalCharges=telecom.TotalCharges.astype(float,errors="ignore")


# In[ ]:


telecom.info()


# ## Checking for outliers

# In[ ]:


telecom.describe()


# In[ ]:


telecom.head()


# In[ ]:


#tenure, MonthlyCharges, TotalCharges
# Checking outliers at 25%, 50%, 75%, 90%, 95% and 99%
num_telecom = telecom[['tenure','MonthlyCharges','TotalCharges']]
num_telecom.describe(percentiles=[.25, .5, .75, .90, .95, .99])


# ### From the distribution shown above, you can see that there no outliers in your data. The numbers are gradually increasing.

# ## Checking for Missing Values and Inputing Them

# In[ ]:


telecom.isnull().sum()


# #### No missing values in dataset

# In[ ]:


(telecom.TotalCharges==" ").sum()


# ## Train-Test Split

# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


# Updating values of X and y
## Why we drop customerID? a)customerID is just an identifier, like a name or serial number. b) It has no relationship with whether a customer churns or not.

X=telecom.drop(["Churn","customerID"],axis=1)
X.head()


# In[ ]:


y=telecom["Churn"]
y.head()


# In[ ]:


# Splitting the data into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, test_size=0.3, random_state=100)


# ## Feature Scaling

# In[ ]:


from sklearn.preprocessing import StandardScaler


# In[ ]:


scaler = StandardScaler()

X_train[['tenure','MonthlyCharges','TotalCharges']] = scaler.fit_transform(X_train[['tenure','MonthlyCharges','TotalCharges']])

X_train.head()


# In[ ]:


# Checking the churn rate
churn= (sum(telecom["Churn"])/len(telecom["Churn"].index))*100


# In[ ]:


churn


# ## Checking the correlations

# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


# Let's see the correlation matrix 
plt.figure(figsize = (20,10))        # Size of the figure
sns.heatmap(telecom.corr(),annot = True)
plt.show()


# In[ ]:


telecom.head()


# In[ ]:


telecom.info()


# In[ ]:


corr_matrix = telecom.corr()


# In[ ]:


corr_matrix


# In[ ]:


X_test = X_test.drop(['MultipleLines_No','OnlineSecurity_No','OnlineBackup_No','DeviceProtection_No','TechSupport_No',
                       'StreamingTV_No','StreamingMovies_No'],axis=1)
X_train = X_train.drop(['MultipleLines_No','OnlineSecurity_No','OnlineBackup_No','DeviceProtection_No','TechSupport_No',
                         'StreamingTV_No','StreamingMovies_No'], axis=1)


# In[ ]:


plt.figure(figsize = (20,10))
sns.heatmap(X_train.corr(),annot = True)
plt.show()


# ## Model Building

# In[ ]:


## First Model

import statsmodels.api as sm


# In[ ]:


# Logistic regression model
logm1 = sm.GLM(y_train,(sm.add_constant(X_train)), family = sm.families.Binomial())
logm1.fit().summary()


# ## Feature selection using RFE

# In[ ]:


from sklearn.linear_model import LogisticRegression
logreg=LogisticRegression()


# In[ ]:


from sklearn.feature_selection import RFE
rfe = RFE(estimator=logreg, n_features_to_select=15)
rfe=rfe.fit(X_train,y_train)


# In[ ]:


rfe.support_


# In[ ]:


list(zip(X_train.columns,rfe.support_,rfe.ranking_))


# In[ ]:


Col=X_train.columns[rfe.support_]


# In[ ]:


X_train.columns[~rfe.support_]


# In[ ]:


# Assessing the model with statsmodels
X_train_sm=sm.add_constant(X_train[Col])
logm2=sm.GLM(y_train,X_train_sm,family=sm.families.Binomial())
res=logm2.fit()
res.summary()


# In[ ]:


# p values are under threshold
## Getting the predicted values on the train set
y_train_pred=res.predict(X_train_sm)
y_train_pred[:10]


# In[ ]:


y_train_pred.shape


# In[ ]:


y_train_pred = y_train_pred.values.reshape(-1)
y_train_pred[:10]


# ### Creating dataframe with the actual churn flag and the predicted probabilities

# In[ ]:


y_train_pred_final=pd.DataFrame({'Churn': y_train.values,'Churn_Prob':y_train_pred})
y_train_pred_final["CustID"]=y_train.index
y_train_pred_final.head()


# ### Creating new column 'predicted' with 1 if Churn_Prob > 0.5 else 0

# In[ ]:


y_train_pred_final['predicted']=y_train_pred_final.Churn_Prob.map(lambda x:1 if x>0.5 else 0)
y_train_pred_final.head()


# In[ ]:


from sklearn import metrics
# Confusion Matrix
confusion=metrics.confusion_matrix(y_train_pred_final.Churn,y_train_pred_final.predicted)


# In[ ]:


print(confusion)


# In[ ]:


# Accusracy of the model
print(metrics.accuracy_score(y_train_pred_final.Churn,y_train_pred_final.predicted))


# ## Checking VIFs

# In[ ]:


from statsmodels.stats.outliers_influence import variance_inflation_factor


# In[ ]:


#Create a dataframe that will contain the names of all the feature variables
vif=pd.DataFrame()
vif['Features']=X_train[Col].columns
vif['VIF']=[variance_inflation_factor(X_train[Col].values,i)for i in range(X_train[Col].shape[1])]
vif['VIF']=round(vif['VIF'],2)
vif=vif.sort_values(by="VIF",ascending=False)


# In[ ]:


vif


# #### There are a few variables with high VIF. It's best to drop these variables as they aren't helping much with prediction and unnecessarily making the model complex. The variable 'TotalCharges' is derived from tenure and monthly charges.. So let's start by dropping that.

# In[ ]:


Col=Col.drop("TotalCharges",1)
Col


# In[ ]:


## Let's rerun the model
X_train_sm=sm.add_constant(X_train[Col])
logm3=sm.GLM(y_train,X_train_sm,family=sm.families.Binomial())
res=logm3.fit()
res.summary()


# In[ ]:


y_train_pred = res.predict(X_train_sm).values.reshape(-1)


# In[ ]:


y_train_pred[:10]


# In[ ]:


y_train_pred_final['Churn_Prob'] = y_train_pred


# In[ ]:


# Creating new column 'predicted' with 1 if Churn_Prob > 0.5 else 0
y_train_pred_final['predicted'] = y_train_pred_final.Churn_Prob.map(lambda x: 1 if x > 0.5 else 0)
y_train_pred_final.head()


# In[ ]:


# Let's check the overall accuracy.
print(metrics.accuracy_score(y_train_pred_final.Churn, y_train_pred_final.predicted))


# In[ ]:


### Let's check the VIFs again
vif = pd.DataFrame()
vif['Features'] = X_train[Col].columns
vif['VIF'] = [variance_inflation_factor(X_train[Col].values, i) for i in range(X_train[Col].shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif


# In[ ]:


### All variables have a good value of VIF. So we need not drop any more variables and we can proceed with making predictions using this model only. Monthly charges is imp factor to understand churn. So no need to drop as p value is ok.


# In[ ]:


# Let's take a look at the confusion matrix again 
confusion = metrics.confusion_matrix(y_train_pred_final.Churn, y_train_pred_final.predicted )
confusion


# ## Receiver Operating Characteristic Curve

# In[ ]:


def draw_roc( actual, probs ):
    fpr, tpr, thresholds = metrics.roc_curve( actual, probs,
                                              drop_intermediate = False)
    auc_score = metrics.roc_auc_score( actual, probs )
    plt.figure(figsize=(5, 5))
    plt.plot( fpr, tpr, label='ROC curve (area = %0.2f)' % auc_score )
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate or [1 - True Negative Rate]')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()

    return None


# In[ ]:


fpr, tpr, thresholds = metrics.roc_curve( y_train_pred_final.Churn, y_train_pred_final.Churn_Prob, drop_intermediate = False )


# In[ ]:


draw_roc(y_train_pred_final.Churn, y_train_pred_final.Churn_Prob)


# ## Finding Optimal Cutoff Points

# Optimal cutoff probability is that prob where we get balanced sensitivity and specificity

# In[ ]:


#Let's create the columns with different probability cutoffs
numbers=[float(x)/10 for x in range(10)]
for i in numbers:
    y_train_pred_final[i]=y_train_pred_final.Churn_Prob.map(lambda x:1 if x>i else 0)
y_train_pred_final.head()


# In[ ]:


# Now let's calculate accuracy sensitivity and specificity for various probability cutoffs.
cutoff_df = pd.DataFrame( columns = ['prob','accuracy','sensi','speci'])


# In[ ]:


from sklearn.metrics import confusion_matrix

# TP = confusion[1,1] # true positive 
# TN = confusion[0,0] # true negatives
# FP = confusion[0,1] # false positives
# FN = confusion[1,0] # false negatives

num = [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
for i in num:
    cm1 = metrics.confusion_matrix(y_train_pred_final.Churn, y_train_pred_final[i] )
    total1=sum(sum(cm1))
    accuracy = (cm1[0,0]+cm1[1,1])/total1
    
    speci = cm1[0,0]/(cm1[0,0]+cm1[0,1])
    sensi = cm1[1,1]/(cm1[1,0]+cm1[1,1])
    cutoff_df.loc[i] =[ i ,accuracy,sensi,speci]
print(cutoff_df)


# In[ ]:


# Let's plot accuracy sensitivity and specificity for various probabilities.
cutoff_df.plot.line(x='prob', y=['accuracy','sensi','speci'])
plt.show()


# From the curve above, 0.3 is the optimum point to take it as a cutoff probability.

# In[ ]:


y_train_pred_final['final_predicted'] = y_train_pred_final.Churn_Prob.map( lambda x: 1 if x > 0.3 else 0)

y_train_pred_final.head()


# In[ ]:


# Let's check the overall accuracy.
metrics.accuracy_score(y_train_pred_final.Churn, y_train_pred_final.final_predicted)


# In[ ]:


confusion2 = metrics.confusion_matrix(y_train_pred_final.Churn, y_train_pred_final.final_predicted )
confusion2


# In[ ]:


TP = confusion2[1,1] # true positive 
TN = confusion2[0,0] # true negatives
FP = confusion2[0,1] # false positives
FN = confusion2[1,0] # false negatives


# In[ ]:


# Let's see the sensitivity of our logistic regression model
TP / float(TP+FN)


# In[ ]:


# Let us calculate specificity
TN / float(TN+FP)


# In[ ]:


# Calculate false postive rate - predicting churn when customer does not have churned
print(FP/ float(TN+FP))


# In[ ]:


# Positive predictive value 
print (TP / float(TP+FP))


# In[ ]:


# Negative predictive value
print (TN / float(TN+ FN))


# In[ ]:


# Precision( True predictive value)
confusion[1,1]/(confusion[0,1]+confusion[1,1])


# In[ ]:


#Recall( Sensitivity)
confusion[1,1]/(confusion[1,0]+confusion[1,1])


# ## Precision and Recall Tradeoff

# In[ ]:


from sklearn.metrics import precision_recall_curve


# In[ ]:


y_train_pred_final.Churn, y_train_pred_final.predicted


# In[ ]:


p, r, thresholds = precision_recall_curve(y_train_pred_final.Churn, y_train_pred_final.Churn_Prob)


# In[ ]:


plt.plot(thresholds, p[:-1], "g-")
plt.plot(thresholds, r[:-1], "r-")
plt.show()


# ## Making predictions on the test set

# In[ ]:


X_test[['tenure','MonthlyCharges','TotalCharges']] = scaler.transform(X_test[['tenure','MonthlyCharges','TotalCharges']])


# In[ ]:


X_test = X_test[Col]
X_test.head()


# In[ ]:


X_test_sm = sm.add_constant(X_test)


# Making predictions on the test set

# In[ ]:


y_test_pred = res.predict(X_test_sm)


# In[ ]:


y_test_pred[:10]


# In[ ]:


# Converting y_pred to a dataframe which is an array
y_pred_1 = pd.DataFrame(y_test_pred)


# In[ ]:


y_pred_1.head()


# In[ ]:


# Converting y_test to dataframe
y_test_df = pd.DataFrame(y_test)


# In[ ]:


# Putting CustID to index
y_test_df['CustID'] = y_test_df.index


# In[ ]:


# Removing index for both dataframes to append them side by side 
y_pred_1.reset_index(drop=True, inplace=True)
y_test_df.reset_index(drop=True, inplace=True)


# In[ ]:


# Appending y_test_df and y_pred_1
y_pred_final = pd.concat([y_test_df, y_pred_1],axis=1)


# In[ ]:


y_pred_final.head()


# In[ ]:


# Renaming the column 
y_pred_final= y_pred_final.rename(columns={ 0 : 'Churn_Prob'})


# In[ ]:


# Rearranging the columns
y_pred_final = y_pred_final[['CustID','Churn','Churn_Prob']]


# In[ ]:


# Let's see the head of y_pred_final
y_pred_final.head()


# In[ ]:


y_pred_final['final_predicted'] = y_pred_final.Churn_Prob.map(lambda x: 1 if x > 0.42 else 0)


# In[ ]:


y_pred_final.head()


# In[ ]:


# Let's check the overall accuracy.
metrics.accuracy_score(y_pred_final.Churn, y_pred_final.final_predicted)


# In[ ]:


confusion2 = metrics.confusion_matrix(y_pred_final.Churn, y_pred_final.final_predicted )
confusion2


# In[ ]:


TP = confusion2[1,1] # true positive 
TN = confusion2[0,0] # true negatives
FP = confusion2[0,1] # false positives
FN = confusion2[1,0] # false negatives


# In[ ]:


# Let's see the sensitivity of our logistic regression model
TP / float(TP+FN)


# In[ ]:


# Let us calculate specificity
TN / float(TN+FP)


# # Using Decision Trees

# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=100)


# In[ ]:


X_train.shape, X_test.shape


# In[ ]:


from sklearn.tree import DecisionTreeClassifier


# In[ ]:


dt_base = DecisionTreeClassifier(random_state=42, max_depth=4)


# In[ ]:


dt_base.fit(X_train, y_train)


# In[ ]:


y_train_pred = dt_base.predict(X_train)
y_test_pred = dt_base.predict(X_test)


# In[ ]:


from sklearn.metrics import classification_report


# In[ ]:


print(classification_report(y_test, y_test_pred))


# In[ ]:


from sklearn.metrics import RocCurveDisplay

RocCurveDisplay.from_estimator(dt_base, X_train, y_train)
plt.show()


# ### Hyper-parameter tuning for the Decision Tree

# In[ ]:


from sklearn.model_selection import GridSearchCV


# In[ ]:


dt = DecisionTreeClassifier(random_state=42)


# In[ ]:


params = {
    "max_depth": [2,3,5,10,20],
    "min_samples_leaf": [5,10,20,50,100,500]
}


# In[ ]:


grid_search = GridSearchCV(estimator=dt,
                           param_grid=params,
                           cv=4,
                           n_jobs=-1, verbose=1, scoring="accuracy")


# In[ ]:


get_ipython().run_cell_magic('time', '', 'grid_search.fit(X_train, y_train)\n')


# In[ ]:


grid_search.best_score_


# In[ ]:


dt_best = grid_search.best_estimator_
dt_best


# In[ ]:


from sklearn.metrics import RocCurveDisplay

RocCurveDisplay.from_estimator(dt_best, X_train, y_train)
plt.show()


# ## Using Random Forest

# In[ ]:


from sklearn.ensemble import RandomForestClassifier


# In[ ]:


rf = RandomForestClassifier(n_estimators=10, max_depth=4, max_features=5, random_state=100, oob_score=True)


# In[ ]:


rf.fit(X_train, y_train)


# In[ ]:


rf.oob_score_


# In[ ]:


from sklearn.metrics import RocCurveDisplay

RocCurveDisplay.from_estimator(rf, X_train, y_train)
plt.show()


# ### Hyper-parameter tuning for the Random Forest

# In[ ]:


rf = RandomForestClassifier(random_state=42, n_jobs=-1)


# In[ ]:


params = {
    'max_depth': [2,3,5,10,20],
    'min_samples_leaf': [5,10,20,50,100,200],
    'n_estimators': [10, 25, 50, 100]
}


# In[ ]:


grid_search = GridSearchCV(estimator=rf,
                           param_grid=params,
                           cv = 4,
                           n_jobs=-1, verbose=1, scoring="accuracy")


# In[ ]:


grid_search.fit(X_train, y_train)


# In[ ]:


grid_search.best_score_


# In[ ]:


rf_best = grid_search.best_estimator_
rf_best


# In[ ]:


rf_best.feature_importances_


# In[ ]:


feat_importances = pd.Series(rf_best.feature_importances_, index=X_train.columns)
feat_importances.sort_values(ascending=False)


# In[ ]:


index


# In[ ]:





# In[ ]:




