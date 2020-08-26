import sys
import pandas as pd
import numpy as np

import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
from sqlalchemy import create_engine
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, cross_validate, RandomizedSearchCV
from sklearn.metrics import classification_report, accuracy_score
import pickle

def load_data(database_filepath):
    '''
    loads data from database.

    Parameters:
    database_filepath (str): filepath of the database.

    Returns:
    X (list): features data.
    Y (list): Target data.
    '''
    # load data from database
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table('tDisasterResponse', engine)
    X = df['message']
    Y = df[df.columns[4:]]
    category_names = list(df.columns[4:])
    return X,Y,category_names


def tokenize(text):
    '''
    Cleans and tokenizes the input text.

    Parameters:
    text (str): Text to clean.

    Returns:
    clean_tokens (list): all cleaned tokens
    '''
    # Convert all characters to lower case
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9]", " ", text)
    # Tokenize text
    tokens = word_tokenize(text)
    # Lemmatize text
    lemmatizer = WordNetLemmatizer()
    clean_tokens = []
    for token in tokens:
        cleaned_token = lemmatizer.lemmatize(token).lower().strip()
        clean_tokens.append(cleaned_token)
    return clean_tokens


def build_model():
    '''
    Uses the pipeline and grid search to train a model.
    '''
    # Create ML pipeline
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(OneVsRestClassifier(GradientBoostingClassifier(random_state=0))))
    ])
    # Use Grid search to find best parameters
    parameters = {
        'clf__estimator__estimator__min_samples_leaf': [2, 5],
        'clf__estimator__estimator__min_samples_split': [8, 10],
        'clf__estimator__estimator__n_estimators': [100,200]
    }

    cv = GridSearchCV(pipeline, parameters)
    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    '''
    Creates a classification report for each category.

    Parameters:
    model: trained model.
    X_test: Trained features.
    Y_test: Target data.
    category_names (list): list of categories to create classification report for.
    '''
    # Create a classification report for each category
    y_pred = model.predict(X_test)
    for category in range(len(category_names)):
        print("Classification report for: ", category_names[category])
        print(classification_report(Y_test.iloc[:, category].values, y_pred[:, category]))



def save_model(model, model_filepath):
    '''
    Saves model as a pickle file.

    Parameters:
    model: trained model.
    model_filepath (str): Name of pickle file to save to
    '''
    pickle.dump(model, open(model_filepath, 'wb'))


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
