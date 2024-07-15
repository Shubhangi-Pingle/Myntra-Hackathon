import pandas as pd
import numpy as np
import re
import os
import emoji
import joblib
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import AdaBoostClassifier
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import classification_report, confusion_matrix
import warnings
warnings.filterwarnings("ignore", category=UserWarning)


def load_data(filepath):
    
    df = pd.read_excel(filepath)
    return df


def clean_data(df):
   
    df = df.drop_duplicates(subset='comments', keep='first')
    df = df.drop(columns=608, axis=1)
    return df


def tokenize(text):
    
    if text is None:
        return None
    
    # Convert emoji to text
    text = emoji.demojize(text, delimiters=(" ", " "))
    
    # Replace unrecognized emoji (left as ::) with space
    text = re.sub('::', ' ', text)
    
    # Tokenize
    tokens = word_tokenize(text)
    
    # Remove English stopwords
    tokens = [token for token in tokens if token not in stopwords.words('english')]
    
    return tokens


def split_data(df):
    
    df2 = df[df['category'].notnull()]
    X = df2.comments
    y = df2.category
    return train_test_split(X, y, test_size=0.2)


def build_model():

    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('moc', AdaBoostClassifier())
    ])

    params = {
    'tfidf__use_idf': (True, False),
    'moc__n_estimators': [50, 60, 70]
    }   

    return GridSearchCV(pipeline, param_grid=params)


def evaluate_model(model, X_test, y_test):
    
    y_pred = model.predict(X_test)

    print('Classification Report:')
    print(classification_report(y_test, y_pred, zero_division=0))

    accuracy = (y_test == y_pred).mean()
    print(f'accuracy: {accuracy}')


def save_model(model, filename):
    joblib.dump(model, filename)


def predict_unlabelled_data(model):
    df_comments = pd.read_csv('../data/posts_comments.csv')
    df_comments['comments'] = df_comments['comments'].fillna('unknown')
    df_comments['category'] = model.predict(df_comments['comments'])
    df_comments.to_csv('data/posts_comments.csv', sep=';')


def main():
    df = load_data('data/labelled_comments_train.xlsx')
    df = clean_data(df)
    X_train, X_test, y_train, y_test = split_data(df)
    model = build_model()
    model.fit(X_train, y_train)
    evaluate_model(model, X_test, y_test)
    save_model(model, 'models/sentiment_analysis_model.joblib')
    predict_unlabelled_data(model)


if __name__ == "__main__":
    main()
