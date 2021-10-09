# Import Libraries
import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    '''
    This function reads csv files and merges them
    
    Parameters
    ----------
    messages_filepath : Path to csv file containing messages
    categories_filepath : Path to csv file containing categories


    Returns
    -------
    df : Merged data frame

    '''
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = messages.merge(categories, on = 'id')
    return df


def clean_data(df):
    '''
    This function cleans the data by splitting categories into individual
    columns, and retaining only the score as integer.
    It also converts each column into binary.
    
    Parameters
    ----------
    df : Merged dataframe to be cleaned


    Returns
    -------
    df : Cleaned dataframe

    '''
    categories = df.categories.str.split(pat = ";", expand = True)
    row = categories.iloc[0]
    category_colnames = list(row.str.split(pat = "-", expand = True)[0])
    categories.columns = category_colnames

    for column in categories.columns:
        # set each value to be the last character of the string
        categories[column] = categories[column].astype(str).str[-1]

        # convert column from string to numeric
        categories[column] = categories[column].astype(int)
        
        # convert values greater than 0 to 1
        categories[column] = np.where(categories[column] > 0, 1, 0)
        
    df.drop(['categories'], axis = 1, inplace = True)
    df = pd.concat([df, categories], axis = 1)
    
    df = df.drop_duplicates()
    
    return df
    

def save_data(df, database_filename):
    '''
    This function saves the dataframe in sqllite database
    
    Parameters
    ----------
    df : Cleaned dataframe to be saved
    database_filename: Path of database where the dataframe is to be saved


    Returns
    -------
    None

    '''
    engine = create_engine('sqlite:///' + database_filename)
    df.to_sql('DRResponse_Ajay', engine, index=False, if_exists = 'replace') 


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