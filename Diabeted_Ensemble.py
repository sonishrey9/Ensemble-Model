# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from IPython import get_ipython

# %%
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv("Diabeted_Ensemble.csv")
df


# %%
df.info()


# %%
categorical_features=[feature for feature in df.columns if df[feature].dtypes=='O']
categorical_features


# %%
lb = LabelEncoder()

for i in categorical_features:
    df[i] = lb.fit_transform(df[i])


# %%
df


# %%
df.columns.to_list


# %%
#split data into inputs and targets
X = df.drop(columns = [' Class variable'])
y = df[' Class variable']


# %%
from sklearn.model_selection import train_test_split
#split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y)

# %% [markdown]
# ## **Boosting**

# %%
from sklearn.ensemble import AdaBoostClassifier


# %%
ada_clf = AdaBoostClassifier(learning_rate = 0.02, n_estimators = 5000)


# %%
ada_clf.fit(X_train, y_train)


# %%
from sklearn.metrics import accuracy_score, confusion_matrix


# %%
# Evaluation on Testing Data
print(confusion_matrix(y_test, ada_clf.predict(X_test)))
print(accuracy_score(y_test, ada_clf.predict(X_test)))


# %%
from sklearn.metrics import plot_confusion_matrix
import matplotlib.pyplot as plt


# %%
plot_confusion_matrix(ada_clf, X_test, y_test)
plt.title("Confusion matrix of ADA Boosting")
plt.show()

# %% [markdown]
# ## **Bagging**

# %%
from sklearn import tree
clftree = tree.DecisionTreeClassifier()
from sklearn.ensemble import BaggingClassifier


# %%
bag_clf = BaggingClassifier(base_estimator = clftree, n_estimators = 500,
                            bootstrap = True, n_jobs = 1, random_state = 42)


# %%
bag_clf.fit(X_train, y_train)


# %%
# Evaluation on Testing Data
print(confusion_matrix(y_test, bag_clf.predict(X_test)))
print(accuracy_score(y_test, bag_clf.predict(X_test)))


# %%
# Evaluation on Training Data
print(confusion_matrix(y_train, bag_clf.predict(X_train)))
print(accuracy_score(y_train, bag_clf.predict(X_train)))


# %%
plot_confusion_matrix(bag_clf, X_test, y_test)
plt.title("Confusion matrix of Bagging Boosting")
plt.show()

# %% [markdown]
# # **Stacking**

# %%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble  import AdaBoostClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble  import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import  accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.linear_model import Perceptron
from mlxtend.classifier import StackingClassifier
from sklearn.ensemble import VotingClassifier
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn import model_selection
import os

# %% [markdown]
# ## **Feature Scaling**

# %%
min_train = X_train.min()
range_train = (X_train - min_train).max()
X_train_scaled = (X_train - min_train)/range_train


# %%
min_test = X_test.min()
range_test = (X_test - min_test).max()
X_test_scaled = (X_test - min_test)/range_test


# %%
from sklearn.linear_model import LogisticRegression
logi = LogisticRegression()
logi.fit(X_train_scaled, y_train)


# %%
y_predict = logi.predict(X_test_scaled)
from sklearn.metrics import  accuracy_score, f1_score, precision_score, recall_score
from sklearn.metrics import roc_auc_score
roc=roc_auc_score(y_test, y_predict)

acc = accuracy_score(y_test, y_predict)
prec = precision_score(y_test, y_predict)
rec = recall_score(y_test, y_predict)
f1 = f1_score(y_test, y_predict)

results = pd.DataFrame([['Logistic Regression', acc,prec,rec, f1,roc]],
            columns = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score','ROC'])
results


# %%
from xgboost import XGBClassifier
xgb_classifier = XGBClassifier()
xgb_classifier.fit(X_train_scaled, y_train)


# %%
y_predict = xgb_classifier.predict(X_test_scaled)
roc=roc_auc_score(y_test, y_predict)
acc = accuracy_score(y_test, y_predict)
prec = precision_score(y_test, y_predict)
rec = recall_score(y_test, y_predict)
f1 = f1_score(y_test, y_predict)

model_results = pd.DataFrame([['XGBOOST', acc,prec,rec, f1,roc]],
            columns = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score','ROC'])
results = results.append(model_results, ignore_index = True)
results


# %%
from sklearn.ensemble import RandomForestClassifier
random_forest = RandomForestClassifier()
random_forest.fit(X_train_scaled, y_train)


# %%
y_predict = random_forest.predict(X_test_scaled)
roc=roc_auc_score(y_test, y_predict)
acc = accuracy_score(y_test, y_predict)
prec = precision_score(y_test, y_predict)
rec = recall_score(y_test, y_predict)
f1 = f1_score(y_test, y_predict)

model_results = pd.DataFrame([['Random Forest', acc,prec,rec, f1,roc]],
            columns = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score','ROC'])
results = results.append(model_results, ignore_index = True)
results


# %%
sgd = SGDClassifier(max_iter=1000)

sgd.fit(X_train_scaled, y_train)
y_predict = sgd.predict(X_test_scaled)


# %%
roc=roc_auc_score(y_test, y_predict)
acc = accuracy_score(y_test, y_predict)
prec = precision_score(y_test, y_predict)
rec = recall_score(y_test, y_predict)
f1 = f1_score(y_test, y_predict)

model_results = pd.DataFrame([['SGD', acc,prec,rec, f1,roc]],
            columns = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score','ROC'])
results = results.append(model_results, ignore_index = True)
results


# %%
adaboost =AdaBoostClassifier()
adaboost.fit(X_train_scaled, y_train)
y_predict = adaboost.predict(X_test_scaled)


# %%
roc=roc_auc_score(y_test, y_predict)
acc = accuracy_score(y_test, y_predict)
prec = precision_score(y_test, y_predict)
rec = recall_score(y_test, y_predict)
f1 = f1_score(y_test, y_predict)

model_results = pd.DataFrame([['Adaboost', acc,prec,rec, f1,roc]],
            columns = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score','ROC'])
results = results.append(model_results, ignore_index = True)
results


# %%
gboost =GradientBoostingClassifier()
gboost.fit(X_train_scaled, y_train)
y_predict = gboost.predict(X_test_scaled)


# %%
roc=roc_auc_score(y_test, y_predict)
acc = accuracy_score(y_test, y_predict)
prec = precision_score(y_test, y_predict)
rec = recall_score(y_test, y_predict)
f1 = f1_score(y_test, y_predict)

model_results = pd.DataFrame([['Gboost', acc,prec,rec, f1,roc]],
            columns = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score','ROC'])
results = results.append(model_results, ignore_index = True)
results


# %%
knn = KNeighborsClassifier(n_neighbors = 7)
knn.fit(X_train_scaled, y_train)
y_predict = knn.predict(X_test_scaled)


# %%
roc=roc_auc_score(y_test, y_predict)
acc = accuracy_score(y_test, y_predict)
prec = precision_score(y_test, y_predict)
rec = recall_score(y_test, y_predict)
f1 = f1_score(y_test, y_predict)

model_results = pd.DataFrame([['KNN7', acc,prec,rec, f1,roc]],
            columns = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score','ROC'])
results = results.append(model_results, ignore_index = True)
results


# %%
from sklearn.svm import SVC 


svc_model = SVC(kernel='linear')
svc_model.fit(X_train, y_train)
y_predict = svc_model.predict(X_test_scaled)
roc=roc_auc_score(y_test, y_predict)
acc = accuracy_score(y_test, y_predict)
prec = precision_score(y_test, y_predict)
rec = recall_score(y_test, y_predict)
f1 = f1_score(y_test, y_predict)

model_results = pd.DataFrame([['SVC Linear', acc,prec,rec, f1,roc]],
            columns = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score','ROC'])
results = results.append(model_results, ignore_index = True)
results

# %% [markdown]
# # **Voting Classifier**

# %%
clf1=LogisticRegression()
clf2 = RandomForestClassifier()
clf3=AdaBoostClassifier()
clf4=XGBClassifier()
clf5=SGDClassifier(max_iter=1000,loss='log')
clf6=KNeighborsClassifier(n_neighbors = 7)
clf7=GradientBoostingClassifier()

eclf1 = VotingClassifier(estimators=[('lr', clf1), ('rf', clf2), ('ada', clf3),('xgb',clf4),('sgd',clf5),('knn',clf6),('gboost',clf7)], voting='soft', weights=[1,1,2,2,1,3,2])
eclf1.fit(X_train_scaled,y_train)


# %%
eclf_predictions = eclf1.predict(X_test_scaled)
acc = accuracy_score(y_test, eclf_predictions)
prec = precision_score(y_test, eclf_predictions)
rec = recall_score(y_test, eclf_predictions)
f1 = f1_score(y_test, eclf_predictions)
from sklearn.metrics import roc_auc_score
roc=roc_auc_score(y_test, eclf_predictions)
model_results = pd.DataFrame([['Voting Classifier ', acc,prec,rec, f1,roc]],
            columns = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score','ROC'])
results = results.append(model_results, ignore_index = True)
results

# %% [markdown]
# ## **Stacking**

# %%
clf1=LogisticRegression()
clf2 = RandomForestClassifier()
clf3=AdaBoostClassifier()
clf4=XGBClassifier()
clf5=SGDClassifier(max_iter=1000,loss='log')
clf6=GradientBoostingClassifier()
knn=KNeighborsClassifier(n_neighbors = 7)


sclf = StackingClassifier(classifiers=[clf1,clf2, clf3, clf4,clf5,clf6], 
                        meta_classifier=knn)

print('10-fold cross validation:\n')

for clf, label in zip([clf1,clf2, clf3, clf4,clf5,clf6, sclf], 
                    ['Logistic Regression'
                    'Random Forest', 
                    'Adaboost',
                        'XGB','SGD','Gradient',
                    'StackingClassifier']):

    scores = model_selection.cross_val_score(clf, X_test_scaled, y_test,
                                            cv=10, scoring='accuracy')
    print("Accuracy: %0.2f (+/- %0.2f) [%s]" 
        % (scores.mean(), scores.std(), label))


# %%
import time
parameters = {
        'min_child_weight': [1, 5,7, 10],
        'max_depth': [2,3, 5,7,10,12],
        'n_estimators':[10,50,100,200]
        }

from sklearn.model_selection import GridSearchCV
grid_search = GridSearchCV(estimator = xgb_classifier, # Make sure classifier points to the RF model
                        param_grid = parameters,
                        scoring = "accuracy",
                        cv = 5,
                        n_jobs = -1)

t0 = time.time()
grid_search.fit(X_train_scaled, y_train)
t1 = time.time()
print("Took %0.2f seconds" % (t1 - t0))


# %%
grid_search.best_params_


# %%
grid_predictions = grid_search.predict(X_test_scaled)


# %%
from sklearn.metrics import classification_report, confusion_matrix
cm = confusion_matrix(y_test, grid_predictions)


# %%
sns.heatmap(cm, annot=True)


# %%
acc = accuracy_score(y_test, grid_predictions)
prec = precision_score(y_test, grid_predictions)
rec = recall_score(y_test, grid_predictions)
f1 = f1_score(y_test, grid_predictions)
from sklearn.metrics import roc_auc_score
roc=roc_auc_score(y_test, grid_predictions)
model_results = pd.DataFrame([['XGBoost Optimized', acc,prec,rec, f1,roc]],
            columns = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score','ROC'])
results = results.append(model_results, ignore_index = True)
results


# %%
import time
parameters = {
        'C':[0.1, 1, 10, 100,1000],
        'gamma':[1, 0.1, 0.01, 0.001,0.0001],
    'kernel':['rbf','linear']
        }

from sklearn.model_selection import GridSearchCV
grid_search = GridSearchCV(estimator = svc_model, # Make sure classifier points to the RF model
                        param_grid = parameters,
                        scoring = "accuracy",
                        cv = 5,
                        n_jobs = -1)

t0 = time.time()
grid_search.fit(X_train_scaled, y_train)
t1 = time.time()
print("Took %0.2f seconds" % (t1 - t0))


# %%
grid_search.best_params_


# %%
grid_predictions = grid_search.predict(X_test_scaled)
acc = accuracy_score(y_test, grid_predictions)
prec = precision_score(y_test, grid_predictions)
rec = recall_score(y_test, grid_predictions)
f1 = f1_score(y_test, grid_predictions)
from sklearn.metrics import roc_auc_score
roc=roc_auc_score(y_test, grid_predictions)
model_results = pd.DataFrame([['SVC Optimized', acc,prec,rec, f1,roc]],
            columns = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score','ROC'])
results = results.append(model_results, ignore_index = True)
results


# %%
import lightgbm
train_data = lightgbm.Dataset(X_train_scaled, label=y_train)
test_data = lightgbm.Dataset(X_test_scaled, label=y_test)


#
# Train the model
#

parameters = {
    'application': 'binary',
    'objective': 'binary',
    'metric': 'binary_logloss',
    'max_bin': 200,
    'boosting': 'gbdt',
    'num_leaves': 10,
    'bagging_freq': 20,
    'learning_rate': 0.003,
    'verbose': 0
}

model = lightgbm.train(parameters,
                    train_data,
                    valid_sets=test_data,
                    num_boost_round=5000,
                    early_stopping_rounds=100)


# %%



