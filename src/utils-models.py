""" 
	utils-models.py: Collection of python functions used in Data Science Projects
"""

# Load modules
import os 
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.utils import shuffle
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA


def one_hot_encode(df_cat):
	"""
		Performs and Returns one hot encoding on categorical variables
	"""
	return pd.get_dummies(df_cat)

def combine_two_dfs(df1, df2):
	"""
		Returns a combined dataframe using df_num and df_cat
	"""
	return pd.concat([df1, df2], axis=1)


def get_train_test_dfs(feature_df, target_df, nsize=0.33, lshuffle=True, irand=19):
	"""
		Returns training and testing (split) dataframes for models
	"""
	feature_train, feature_test, target_train, target_test \
		= train_test_split(feature_df, target_df, train_size=nsize \
												, shuffle=lshuffle \
												, random_state=irand)

	return feature_train,feature_test,target_train,target_test


def train_model(model, feature_df, target_df, nproc, mean_mse, cv_std):
	"""
		Returns mean mse and std on each model
	"""
	neg_mse 		= cross_val_score(model, feature_df, target_df
								, cv=2
								, n_jobs=nproc
								, scoring='neg_mean_squared_error')

	mean_mse[model] = -1.0*np.mean(neg_mse)
	cv_std[model]   = np.std(neg_mse)


def print_model_summary(model, mean_mse, cv_std):
	"""
		Prints model trainig summary
	""" 
	print('\n Model:\n ', model)
	print(' Average MSE:\n', mean_mse[model])
	print(' Standard deviation during CV:\n', cv_std[model])
	




