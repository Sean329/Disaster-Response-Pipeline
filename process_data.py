import sys
import numpy as np
import pandas as pd
from sqlalchemy import create_engine

# Read two CSV files into two dataframes, then merge into df
def load_data(messages_filepath, categories_filepath):
    messages = pd.read_csv(str(messages_filepath))
    messages.drop_duplicates(inplace=True)
    
    categories = pd.read_csv(str(categories_filepath))
    categories.drop_duplicates(inplace=True)
    
    df = messages.merge(categories, how='inner', on='id')
    
    return df, categories

#Split the categories and place them each as a column
def clean_data(df, categories):
    categories1 = categories['categories'].str.split(';', expand=True)
    
    # select the first row of the categories dataframe
    # use this row to extract a list of new column names for categories.
    row = categories1.iloc[0]
    category_colnames = row.str[:-2]
    
    # rename the columns of `categories`
    categories1.columns = category_colnames
    
    # Convert category values to just numbers 0 or 1.
    for column in categories1:
        categories1[column] = categories1[column].str[-1]
        categories1[column] = pd.to_numeric(categories1[column])
    
    categories = pd.concat([categories['id'], categories1], axis=1)
    
    # Replace categories column in df with new category columns.
    df.drop('categories', axis=1, inplace=True)
    df = df.merge(categories, on='id')
    
    # Remove duplicates.
    df.drop_duplicates(inplace=True)
    
    return df


def save_data(df, database_filename):
    engine = create_engine('sqlite:///'+ str(database_filename))
    df.to_sql('df', engine, index=False)


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df, categories = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df, categories)
        
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