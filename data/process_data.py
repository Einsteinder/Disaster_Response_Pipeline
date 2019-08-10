# import libraries
import sys
import pandas
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """
    Loads the messages and categories datasets from the specified filepaths
    Args:
        messages_filepath: Filepath to the messages dataset
        categories_filepath: Filepath to the categories dataset
    Returns:
        (DataFrame) df: Merged Pandas dataframe
    """

    # load messages dataset
    messages = pandas.read_csv(messages_filepath)

    # load categories dataset
    categories = pandas.read_csv(categories_filepath)

    # merge datasets
    df = pandas.merge(messages,categories,on='id')
    
    return df


def clean_data(df):
    """
    Cleans the merged dataset
    Args:
        df: Merged pandas dataframe
    Returns:
        (DataFrame) df: Cleaned dataframe
    """
    # create a dataframe of the 36 individual category columns
    category_data = df.categories.str.split(';',expand=True)
    
    # select the first row of the categories dataframe
    # rename the columns of `categories`
    df[category_data.iloc[0].map(lambda x:x.split("-")[0])] = category_data
    df.drop(['categories'],axis=1,inplace=True)
    for column in df:
        # set each value to be the last character of the string
        if column in ['id' , 'message',	'original',	'genre']:
            continue
        df[column] = df[column].map(lambda x:x[-1])
        
        # convert column from string to numeric
        df[column] = df[column].map(lambda x:int(x))

    index,value=df.duplicated(subset=None, keep='first').index,df.duplicated(subset=None, keep='first').values
    keyValue = zip(index,value)

    # drop duplicates
    row_num = [key for key,value in keyValue if value]

    df.drop(row_num,inplace=True)

    return df


def save_data(df, database_filename):
    """
    Saves clean dataset into an sqlite database
    Args:
        df:  Cleaned dataframe
        database_filename: Name of the database file
    """
    engine = create_engine('sqlite:///'+database_filename)
    df.to_sql('disaster', engine, index=False,if_exists='replace')


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