# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

df = pd.read_excel(r"E:\Pending Assignments\Ensemble_Techniques_Problem Statement\Datasets_ET\Coca_Rating_Ensemble.xlsx")
df


# %%
df.info()


# %%
categorical_features=[feature for feature in df.columns if df[feature].dtypes=='O']
categorical_features


# %%
for var in categorical_features:
    
    print(var, ' contains ', len(df[var].unique()), ' labels')


# %%
lb = LabelEncoder()

for i in categorical_features:
    df[i] = lb.fit_transform(df[i])


# %%
df.head()


# %%
df.info()


# %%
df.columns.to_list

# %% [markdown]
# # **Bagging**

# %%
df['Cocoa_Percent'] = df['Cocoa_Percent'] *100
df['Cocoa_Percent'] = df['Cocoa_Percent'].astype(int)


# %%
df['Cocoa_Percent']


# %%
# Input and Output Split
predictors =df.loc[:, df.columns!='Cocoa_Percent']
type(predictors)


# %%
target = df['Cocoa_Percent']
type(target)


# %%
# Train Test partition of the data
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(predictors, target, test_size = 0.2, random_state=0)


# %%
from sklearn import tree
clftree = tree.DecisionTreeClassifier()
from sklearn.ensemble import BaggingClassifier


# %%

bag_clf = BaggingClassifier(base_estimator = clftree, n_estimators = 500,
                            bootstrap = True, n_jobs = 1, random_state = 42)


# %%
bag_clf.fit(x_train, y_train)


# %%

from sklearn.metrics import accuracy_score, confusion_matrix


# %%
# Evaluation on Testing Data
confusion_matrix(y_test, bag_clf.predict(x_test))
acc_test = accuracy_score(y_test, bag_clf.predict(x_test))


# %%
# Evaluation on Training Data

acc_train = accuracy_score(y_train, bag_clf.predict(x_train))


# %%
confusion_matrix(y_train, bag_clf.predict(x_train))


# %%
results = pd.DataFrame([['BaggingClassifier', acc_train,acc_test]],columns = ['Model', 'Accuracy test','Accuracy train'])
results

# %% [markdown]
# # **Gradient Boosting**

# %%
from sklearn.ensemble import GradientBoostingClassifier

boost_clf = GradientBoostingClassifier()

boost_clf.fit(x_train, y_train)


# %%
confusion_matrix(y_test, boost_clf.predict(x_test))
accuracy_score(y_test, boost_clf.predict(x_test))


# %%
# Hyperparameters
boost_clf2 = GradientBoostingClassifier(learning_rate = 0.02, n_estimators = 1000, max_depth = 1)
boost_clf2.fit(x_train, y_train)


# %%

# Evaluation on Testing Data
confusion_matrix(y_test, boost_clf2.predict(x_test))
acc_test =accuracy_score(y_test, boost_clf2.predict(x_test))


# %%
# Evaluation on Training Data
acc_train = accuracy_score(y_train, boost_clf2.predict(x_train))


# %%
model_results = pd.DataFrame([['"GradientBoostingClassifier', acc_test,acc_train]],
            columns = ['Model', 'Accuracy test','Accuracy train'])
results = results.append(model_results, ignore_index = True)
results

# %% [markdown]
# ## **XGBoosting**

# %%
import xgboost as xgb


# %%
xgb_clf = xgb.XGBClassifier(max_depths = 5, n_estimators = 10000, learning_rate = 0.3, n_jobs = -1)


# %%
xgb_clf.fit(x_train, y_train)


# %%
# Evaluation on Testing Data
confusion_matrix(y_test, xgb_clf.predict(x_test))
accuracy_score(y_test, xgb_clf.predict(x_test))


# %%

xgb.plot_importance(xgb_clf)

# %% [markdown]
# ## **Adaboosting**

# %%
from sklearn.ensemble import AdaBoostClassifier


# %%

ada_clf = AdaBoostClassifier(learning_rate = 0.02, n_estimators = 5000)


# %%
ada_clf.fit(x_train, y_train)


# %%
# Evaluation on Testing Data
confusion_matrix(y_test, ada_clf.predict(x_test))
acc_test = accuracy_score(y_test, ada_clf.predict(x_test))


# %%
# Evaluation on Training Data
acc_train = accuracy_score(y_train, ada_clf.predict(x_train))


# %%
model_results1 = pd.DataFrame([['Adaboosting', acc_test,acc_train]],
            columns = ['Model', 'Accuracy test','Accuracy train'])
results = results.append(model_results1, ignore_index = True)
results


