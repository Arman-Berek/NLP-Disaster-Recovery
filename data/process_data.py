import sys
from sqlalchemy import create_engine
import pandas as pd
import numpy as np

def load_data(messages_filepath, categories_filepath):
    '''
    load data sets from csv to dataframe and merge.

    Parameters:
    messages_filepath (str): filepath for messages dataset.
    categories_filepath (str): filepath for categories dataset.

    Returns:
    df (pandas.DataFrame): merged dataframe
    '''
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = pd.merge(messages, categories, left_on = 'id', right_on = 'id', how = 'outer')
    return df

def clean_data(df):
    '''
    cleans the dataframe.

    Parameters:
    df (pandas.DataFrame): merged dataframe.

    Returns:
    df (pandas.DataFrame): cleaned dataframe
    '''
    categories = df.categories.str.split(';', expand=True)
    row = categories.loc[0]
    category_colnames = row.apply(lambda x:x[:-2])
    categories.columns = category_colnames
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].str[-1]
        # convert column from string to numeric
        categories[column] = pd.to_numeric(categories[column])
    # drop the original categories column from `df`
    df.drop(['categories'], axis = 1, inplace=True)
    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], axis = 1)
    df = df.drop_duplicates()
    df['related'] = df['related'].astype('str').str.replace('2', '1')
    df['related'] = df['related'].astype('int')
    return df

def save_data(df, database_filename):
    '''
    saves the dataframe to sql db.

    Parameters:
    df (pandas.DataFrame): dataframe to save.
    database_filename (str): name of database to save to.
    '''
    engine = create_engine('sqlite:///' + database_filename)
    df.to_sql('tDisasterResponse', engine, index=False)


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)

        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)

        print('Cleaned data saved to database!')

    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
