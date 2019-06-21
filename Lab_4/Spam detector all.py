 
# coding: utf-8

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

# In[157]:

# load all dataset
# d = pd.read_csv("dataset/Youtube01-Psy.csv")
d = pd.concat([pd.read_csv("dataset/Youtube01-Psy.csv"),
               pd.read_csv("dataset/Youtube02-KatyPerry.csv"),
               pd.read_csv("dataset/Youtube03-LMFAO.csv"),
               pd.read_csv("dataset/Youtube04-Eminem.csv"),
               pd.read_csv("dataset/Youtube05-Shakira.csv")])



# In[158]:


len(d)


# In[159]:


len(d.query('CLASS == 1'))


# In[160]:


len(d.query('CLASS == 0'))


# In[161]:


dshuf = d.sample(frac=1)
d_content = dshuf['CONTENT']
d_label = dshuf['CLASS']


# In[162]:


# set up a pipeline
from sklearn.pipeline import Pipeline, make_pipeline
pipeline = make_pipeline(CountVectorizer(), RandomForestClassifier())
pipeline.steps

# In[163]:


# or: pipeline = make_pipeline(CountVectorizer(), RandomForestClassifier())
#make_pipeline(CountVectorizer(), RandomForestClassifier())


# In[164]:


pipeline.fit(d_content[:1600],d_label[:1600])


# In[165]:


pipeline.score(d_content[1600:], d_label[1600:])


# In[166]:


print(pipeline.predict(["what a neat video!"]))


# In[167]:


print(pipeline.predict(["plz subscribe to my channel"]))


# In[168]:


scores = cross_val_score(pipeline, d_content, d_label, cv=5)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

parameters = {
    'countvectorizer__max_features': (None, 1000, 2000),
    'countvectorizer__ngram_range': ((1, 1), (1, 2)),  # unigrams or bigrams
    'countvectorizer__stop_words': ('english', None),
    'randomforestclassifier__n_estimators': (20, 50, 100)
}
from sklearn.model_selection import GridSearchCV
grid_search = GridSearchCV(pipeline, parameters, n_jobs=-1, verbose=1)


# In[185]:


grid_search.fit(d_content, d_label)


# In[186]:


print("Best score 1: %0.3f" % grid_search.best_score_)
print("Best parameters set:")
best_parameters = grid_search.best_estimator_.get_params()
for param_name in sorted(parameters.keys()):
    print("\t%s: %r" % (param_name, best_parameters[param_name]))


# In[181]:


# add tfidf
from sklearn.feature_extraction.text import TfidfTransformer
pipeline2 = make_pipeline(CountVectorizer(),
                          TfidfTransformer(norm=None),
                          RandomForestClassifier())


# In[182]:


scores = cross_val_score(pipeline2, d_content, d_label, cv=5)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))


# In[183]:


pipeline2.steps

pipeline2.fit(d_content[:1600],d_label[:1600])

pipeline2.score(d_content[1600:], d_label[1600:])

print(pipeline2.predict(["what a neat video!"]))


print(pipeline2.predict(["plz subscribe to my channel"]))

# In[184]:


# parameter search
parameters = {
    'countvectorizer__max_features': (None, 1000, 2000),
    'countvectorizer__ngram_range': ((1, 1), (1, 2)),  # unigrams or bigrams
    'countvectorizer__stop_words': ('english', None),
    'tfidftransformer__use_idf': (True, False), # effectively turn on/off tfidf
    'randomforestclassifier__n_estimators': (20, 50, 100)
}
from sklearn.model_selection import GridSearchCV
grid_search = GridSearchCV(pipeline2, parameters, n_jobs=-1, verbose=1)


# In[185]:


grid_search.fit(d_content, d_label)


# In[186]:


print("Best score 2: %0.3f" % grid_search.best_score_)
print("Best parameters set:")
best_parameters = grid_search.best_estimator_.get_params()
for param_name in sorted(parameters.keys()):
    print("\t%s: %r" % (param_name, best_parameters[param_name]))