
# coding: utf-8

# In[38]:


from sklearn.datasets import load_breast_cancer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[39]:


cancer = load_breast_cancer()
print(cancer.DESCR)


# In[40]:


print(cancer.feature_names)
print(cancer.target_names)


# In[41]:


type(cancer.data) 


# In[42]:


cancer.data


# In[43]:


cancer.data.shape


# In[44]:


import pandas as pd
raw_data = pd.read_csv('breast-cancer-wisconsin-data.csv')
raw_data.tail(10)


# In[45]:


#!pip install mglearn


# In[46]:


import mglearn
mglearn.plots.plot_knn_classification(n_neighbors=6)


# In[47]:


X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target,
                                                    stratify=cancer.target, random_state=42)
knn = KNeighborsClassifier()
knn.fit(X_train, y_train)



# In[48]:


print('Accuracy of KNN n-5, on training set: {:.2f}'.format(knn.score(X_train, y_train)))
print('Accuracy of KNN n-5, on test set: {:.2f}'.format(knn.score(X_test, y_test)))



# In[50]:


X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target,
                                                    stratify=cancer.target, random_state=65)
training_accuracy = []
test_accuracy = []

neighbors_settings = range(1,11)
for n_neighbors in neighbors_settings:
    clf = KNeighborsClassifier(n_neighbors=n_neighbors)
    clf.fit(X_train, y_train)
    training_accuracy.append(clf.score(X_train, y_train))
    test_accuracy.append(clf.score(X_test,y_test))
    
    plt.plot(neighbors_settings, training_accuracy, label='Accuracy Training Set')
    plt.plot(neighbors_settings, test_accuracy, label='Accuracy Test Set')
    plt.ylabel('Accuracy')
    plt.xlabel('Number of Neighbors')
    plt.legend()

