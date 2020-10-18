""" 
	utils-gather-assess.py: Collection of python functions used in Data Science Projects
"""
#!/usr/bin/env python3


# Load modules
import os 
import pandas as pd


def check_download(url, filename, path_data):
	"""
		Checks for the datafile, and downloads if NOT present
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



def show_files_datasets(path):
	"""
		Prints to terminal names of files, if present     
	"""
	print(" datasets/")
	for ifile in os.listdir(path):
		print("       -",ifile)



def load_csv_df(path_datafile):
	"""
	   Loads a csv data file into Pandas DataFrame  
	"""
	return pd.read_csv(path_datafile, header=None)


def drop_duplicate_rows(df):
	"""
	   Checks for duplicate rows/instances and drops the rows from dataframe
	"""

	# No. of duplicated rows
	ndup_rows = get_duplicate_rows(df)

	print('There are {} duplicated rows in the dataset.'.format(ndup_rows))
	if (ndup_rows > 0):
		return df.drop_duplicates().reset_index(inplace=True, drop=True)
		print('Dropped {} rows from the dataset.'.format(ndup_rows))


def get_duplicate_rows(df):
	"""
	   Returns duplicate rows/instances in the dataframe
	"""
	return df.duplicated().sum()


def get_unique_values(df, colname):
	"""
		Returns a list with all unique values in the column of datafram df
	"""
	return list(dict(df[colname].value_counts(ascending=False, dropna=False)).keys())


def get_unique_counts(df, colname):
	"""
		Returns a list with all counts of unique values in the column of datafram df
	"""
	return list(dict(df[colname].value_counts(ascending=False, dropna=False)).values())


def show_features_datatypes(df):
	"""
	   Prints a table of Features and their DataTypes
	"""
	for inum,icol in enumerate(df.columns):
		print('Column id: {0:3d} \tName: {1:12s} \tDataType: {2}'.format(inum, icol, df[icol].dtypes))


def show_feature_summary(df, colname):
	"""
		Prints all necessary information to fix missing data
	"""
	print(' Details of feature:',colname)
	print('         - datatype:',df[colname].dtypes)
	print('         - col.size:',df[colname].shape)
	print('         - NaN.vals:',df[colname].isnull().sum())
	print('         - uniqvals:',get_unique_values(df, colname))
	print('         - cnt.vals:',get_unique_counts(df, colname))


def change_feature_datatype(df, colname, dtype_new):
	"""
		Function to modify data type of a column 
	"""
	if dtype_new in [object, int, float]:
		print(' Details of column:',colname)
		print('        - dtype(o):',df[colname].dtypes)
		# Change of data type is done here!
		df[colname] = df[colname].astype(dtype_new)
		print('        - dtype(n):',df[colname].dtypes)
	else:
		print(' Details of column:',colname)
		print('        - >>>Error:',dtype_new) 


def replace_feature_missingvalues(df, colname, old_val, new_val=None):
	"""
		Function to replace old_val with new_val (according to datatype) in a column of DataFrame
	"""
	if (new_val is None):
		
		if (old_val in df[colname].unique()):
			print(' Details of column:',colname)
			print('      - uniqval(o):',get_unique_values(df, colname))
			print('      - cnt.val(o):',get_unique_counts(df, colname))
			
			# Replace old_val with new_val in df[colname]
			# ---- Object Datatype -----
			if (df[colname].dtype == object):
				new_val = df[colname].value_counts(ascending=False).index[0]
			else: # ---- Int or Float Datatype -----
				new_val = df[colname].mean()
			df[colname].replace(old_val, new_val, inplace=True)       
			print('      - uniqval(n):',get_unique_values(df, colname))
			print('      - cnt.val(n):',get_unique_counts(df, colname))
			
		else:
			print(' Details of column:',colname)
			print('        - >>>Error:',old_val)
	
	else:
		
		df[colname].replace(old_val, new_val, inplace=True)       

