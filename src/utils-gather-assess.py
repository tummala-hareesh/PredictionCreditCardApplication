""" 
	utils.py: 
		- Collection of python functions used in Data Science Projects
		- # functions: 6 
"""

# Load modules
import os 


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






