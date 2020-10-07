""" 
	utils-gather-assess.py: 
		- Collection of python functions used in Data Science Projects
		- # functions: 6 
"""

# Load modules
import os 
import pandas as pd


def check_download(url, filename, path_data):
    """
    Checks for the datafile, and downloads if NOT present
    
    Args:
        url (TYPE): web url, where datafile is downloaded from
        filename (TYPE): name of the file which is being downloaded
        path_data (TYPE): local path, where the datafile is downloaded to
    """
    
    # Join file name to download
    url_data = url + '/' + filename
    
    # Check and create directory if doesn't exist
    if not os.path.exists(path_data):
        os.mkdir(path_data)
        print(" Folder created:",path_data)
    else:
        print(" Folder exists:",path_data)
    
    # Check and download file if doesn't exist
    if os.path.isfile(path_data+'/'+filename):
        print(" Datafile already present:",path_data+'/'+filename)
    else:
        print(" Downloading data... <START>")
        datafile = wget.download(url_data, path_data)
        print(" - {}".format(filename))
        print(" Downloading data... <FINISH>")



def show_files_in_datasets(path):
    """
    	Prints to terminal names of files, if present 
    
    Args:
        path (TYPE): local path, where the datafiles are located
    """
    print(" datasets/")
    for ifile in os.listdir(path):
        print("       -",ifile)



def load_csv_df(path_datafile):
    """
       Loads a csv data file into Pandas DataFrame  
    
    Args:
        path_dataset (TYPE): Path to the dataset
        filename_dataset (TYPE): Name of the dataset file
    
    Returns:
        TYPE: Pandas DataFrame 
    """
    return pd.read_csv(path_datafile, header=None)


def show_features_datatypes(df):
    """
       Prints a table of Features and their DataTypes
    
    Args:
        df (TYPE): Pandas DataFrame
    
    Returns:
        None
    """
    for inum,icol in enumerate(df.columns):
        print('Column id: {0:3d} \tName: {1:12s} \tDataType: {2}'.format(inum, icol, df[icol].dtypes))



def drop_duplicate_rows(df):
    """
       Checks for duplicate rows/instances and drops the rows from dataframe
    
    Args:
        df (TYPE): Pandas DataFrame
    
    Returns:
        TYPE: DataFrame with no duplicates (if found!)
    """

    ndup_rows = df.duplicated().sum()
    print('There are {} duplicated rows in the dataset.'.format(ndup_rows))
    if (ndup_rows > 0):
        return df.drop_duplicates().reset_index(inplace=True, drop=True)
        print('Dropped {} rows from the dataset.'.format(ndup_rows))
