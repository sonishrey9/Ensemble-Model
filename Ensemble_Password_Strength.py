# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder


# %%
df = pd.read_excel(r"E:\Pending Assignments\Ensemble_Techniques_Problem Statement\Datasets_ET\Ensemble_Password_Strength.xlsx")
df


# %%
df.isna().sum() # check for missing values


# %%
df.shape


# %%
df.info()


# %%
#Shuffle data
from sklearn.utils import shuffle
df1=shuffle(df)


# %%
#reset index
df1=df1.reset_index(drop=True)
df1


# %%
df1.info()


# %%
import seaborn as sns
import matplotlib.pyplot as plt


# %%
x=df1['characters']
y=df1['characters_strength']


# %%
sns.countplot(y,data=df1)
plt.title("Distribution of password strength, 0 for not good, 1 for good")
plt.show()


# %%
df1.groupby(['characters_strength']).count()/len(df1)


# %%
#Let us make a list of characters of password
def word(password):
    character=[]
    
    for i in password:
        character.append(i)
        
    return character


# %%
import datetime


# %%
#convert password into vectors
from sklearn.feature_extraction.text import TfidfVectorizer
vector = TfidfVectorizer(tokenizer=word)


# %%
df1['characters']


# %%
df1['characters'] = df1['characters'].astype(str)
x = df1['characters'] 


# %%
x_vec = vector.fit_transform(x)


# %%
#dictionary
vector.vocabulary_


# %%
#getting  tf-idf vector for first password

feature_names=vector.get_feature_names()
first_password=x_vec[0]
vec=pd.DataFrame(first_password.T.todense(),index=feature_names,columns=['tfidf'])
vec.sort_values(by=['tfidf'],ascending=False)


# %%
#split the data into train and test
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x_vec,y,test_size=0.2,random_state=0)


# %%
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import cross_val_score


# %%
classifier=[]
classifier.append(LogisticRegression(multi_class='ovr'))
classifier.append(LogisticRegression(multi_class='multinomial',solver='newton-cg'))
classifier.append(xgb.XGBClassifier())
classifier.append(MultinomialNB())


# %%
#result
result=[]
for model in classifier:
    a=model.fit(x_train,y_train)
    result.append(a.score(x_test,y_test))


# %%
result1=pd.DataFrame({'score':result,
                    'algorithms':['logistic_regr_ovr',
                                    'logistic_regr_mutinomial',
                                    'xgboost','naive bayes']})


# %%
a=sns.barplot('score','algorithms',data=result1)
a.set_label('accuracy')
a.set_title('cross-val-score')


# %%
#prediction
x_pred=np.array(['123abc'])
x_pred=vector.transform(x_pred)
model=xgb.XGBClassifier()
model.fit(x_train,y_train)
y_pred=model.predict(x_pred)
y_pred


# %%



