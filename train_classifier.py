import sys

# import libraries
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import pickle

#import text cleasing libraries
import string
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords

#import text preprocessing and modeling libraries
from sklearn.model_selection import train_test_split

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier

from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import MultinomialNB

from sklearn.metrics import classification_report

# load data from database
def load_data(database_filepath):
    engine = create_engine('sqlite:///'+ str(database_filepath))
    df = pd.read_sql("SELECT * FROM df", engine)
    X = df['message']
    Y = df.iloc[:, 4:]
    category_names = list(Y.columns)
    
    return X, Y, category_names

# Write a tokenization function to process text data
def tokenize(text):
    return [i for i in ''.join([c for c in text if c not in string.punctuation]).split()
            if i.lower() not in stopwords.words('english')]

# Build a machine learning pipeline
def build_model():
    pipeline = Pipeline([
    ('bow', CountVectorizer(analyzer=tokenize)),
    ('tfidf', TfidfTransformer()),
    ('classifier', MultiOutputClassifier(MultinomialNB()))
    ])

    parameters = { 'classifier__estimator__alpha': [0.2, 1.0]}
    cv = GridSearchCV(pipeline, param_grid=parameters, verbose=3)
    
    return cv

def evaluate_model(model, X_test, Y_test, category_names):
    predictions = model.predict(X_test)
    
    for i in range(0, len(category_names)):
        print(classification_report(Y_test.values[:,i], predictions[:,i]))


def save_model(model, model_filepath):
    pickle.dump(model,open(str(model_filepath),'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()