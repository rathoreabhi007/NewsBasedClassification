import json
import pandas as pd

data=[]
for line in open(r'./data.json','r'):
    data.append(json.loads(line))
    
modData = {
    'short_description': [],
    'headline': [],
    'category': []
}
for dict in data:
    modData['short_description'].append(dict['short_description'])
    modData['headline'].append(dict['headline'])
    modData['category'].append(dict['category'])

df = pd.DataFrame(modData)
print(df.head())

from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()
y = encoder.fit_transform(df['category'])
print(y[0:])

CATEGORY = df['category']
SHORT_DESCRIPTION = df['short_description']
N = len(SHORT_DESCRIPTION)
print('Number of news',N)

labels = list(set(CATEGORY))
print('possible categories',labels)

for l in labels:
    print('number of ',l,' news',len(df.loc[df['category'] == l]))

from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
nCATEGORY = encoder.fit_transform(CATEGORY)

Ntrain = int(N * 0.7)
from sklearn.utils import shuffle
SHORT_DESCRIPTION, nCATEGORY = shuffle(SHORT_DESCRIPTION, nCATEGORY, random_state=0)

X_train = SHORT_DESCRIPTION[:Ntrain]
print('X_train.shape',X_train.shape)
y_train = nCATEGORY[:Ntrain]
print('y_train.shape',y_train.shape)
X_test = SHORT_DESCRIPTION[Ntrain:]
print('X_test.shape',X_test.shape)
y_test = nCATEGORY[Ntrain:]
print('y_test.shape',y_test.shape)

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

print('Training...')

text_clf = Pipeline([('vect', CountVectorizer()),
                     ('tfidf', TfidfTransformer()),
                     ('clf', MultinomialNB()),
                     ])

text_clf = text_clf.fit(X_train, y_train)
print('Predicting...')
predicted = text_clf.predict(X_test)

from sklearn import metrics

print('accuracy_score',metrics.accuracy_score(y_test,predicted))
print('Reporting...')
print(metrics.classification_report(y_test, predicted, target_names=labels))




