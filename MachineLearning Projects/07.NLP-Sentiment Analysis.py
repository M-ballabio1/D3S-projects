'''CLASSIFICATORE
il dataset contiene dei testi che sono recensioni di film.
Questi vengono analizzati i target ossia se sono commenti positivi o negativi

Il risultato ottenuto presenta dell'overfitting, ma l'idea principale di questo prima prova è valutare come si implementa
un metodo per il riconoscimento del "sentiment di un testo"

Train acc. 0.939, test acc. 0.720

Dataset kaggle: https://www.kaggle.com/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews

'''

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

df = pd.read_csv('5_movie_review.csv')               #dataset dei commenti positivi/negativi a film

print(df.head())                                     #stampa prime 5 righe circa

X = df['text']                                       #x conterrà la lista di tutti i testi
y = df['tag']

vect = CountVectorizer(ngram_range=(1,2))             #facendo ngram_range (1,2) non considera solo parole singole parole ma anche le coppie.
X = vect.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size =0.1)

#using model Bernoulli
model = BernoulliNB()
model.fit(X_train, y_train)

p_train = model.predict(X_train)
p_test = model.predict(X_test)

acc_train = accuracy_score(y_train, p_train)
acc_test = accuracy_score(y_test, p_test)

print(f'Train acc. {acc_train}, test acc. {acc_test}')
print(confusion_matrix(y_test,p_test))
print(classification_report(y_test,p_test))
