from sklearn.preprocessing import OneHotEncoder
from category_encoders import TargetEncoder
import pandas as pd
import numpy as np
import copy
import pickle

class ClusterPredictor():
  def __init__(self, transfer_data_path, transfer_df_path, k):
    # Load transfer data from the pickle file
    transfer_data = pd.read_pickle(transfer_data_path)
    
    # Load df_final from the pickle file
    df_final = pd.read_pickle(transfer_df_path)

    self.num_features = transfer_data['num_features']
    self.cat_features = transfer_data['cat_features']
    self.df_final = copy.deepcopy(df_final)
    self.cols_to_use = transfer_data['cols_to_use']
    self.cols_to_use_pca = transfer_data['cols_to_use_pca']

    self.k = k
    self.model = transfer_data['models'][k]
    self.scaler = transfer_data['scaler']
    self.target_encoder = transfer_data['target_encoder']
    self.pca = transfer_data['pca']
    default_center = [np.mean(self.df_final[col]) for col in self.df_final if col != 'region' and col not in self.cat_features]
    self.default_center = pd.DataFrame([default_center], columns=[col for col in self.df_final.columns if col != 'region' and col not in self.cat_features])
  
  def get_df_sample(self, website_data):

    # Helper function
    def underscore_formatter(words):
      import string
      words = str(words)
      for punct in string.punctuation:
        words=words.replace(punct, " ")
      spaces = [' '*i for i in range(10,1,-1)]
      words = words.lower().strip()
      for space in spaces:
        words=words.replace(space, ' ')
      return '_'.join(words.split(' '))

    df_sample = pd.DataFrame([website_data])

    # Use data from displayed value instead of slider
    display_cols = []
    for col in df_sample.columns:
      if '-display' in col:
        # Drop empty columns
        if df_sample[col][0] == '':
          df_sample.drop(col.replace('-display',''), axis=1)
          display_cols.append(col)
          continue

        df_sample[col.replace('-display','')] = df_sample[col]
        display_cols.append(col)
    
    if len(display_cols)>0: df_sample = df_sample.drop(columns = display_cols)
    df_sample.columns = [underscore_formatter(col) for col in df_sample.columns]
    df_sample.replace({'electricity': {'Has electricity': 1, 'No electricity': 0}}, inplace=True)

    # Fill out missing numerical columns
    missing_cols = [col for col in self.df_final[self.num_features].columns if col not in df_sample.columns]
    missing_cols_values = [self.default_center.at[0, col] for col in missing_cols]
    df_missing_cols = pd.DataFrame([missing_cols_values], columns=missing_cols)
    df_sample = df_sample.join(df_missing_cols)
    
    to_target_encode = ['region']
    # One-hot encoding on categorical features
    onehot_encoder = OneHotEncoder(sparse_output=False)
    df_onehot_encoded = pd.DataFrame(onehot_encoder.fit_transform(df_sample[self.cat_features]))
    df_onehot_encoded.columns = [underscore_formatter(col) for col in onehot_encoder.get_feature_names_out(self.cat_features)]

    # Target encoding on 'region' feature
    df_target_encoded = pd.DataFrame(self.target_encoder.transform(df_sample[to_target_encode], df_sample['total_household_income']))

    # Standardize the numerical features and target encoded feature
    df_standardized = pd.concat([df_sample[self.num_features], df_target_encoded], axis=1)
    df_standardized = self.scaler.transform(df_standardized)
    df_standardized = pd.DataFrame(df_standardized, columns=self.num_features + df_target_encoded.columns.tolist())
    df_standardized.columns = [underscore_formatter(col) + '_final' for col in self.num_features + df_target_encoded.columns.tolist()]

    # Concatenate original dataframe, standardized dataframe, and the encoded dataframe
    df_clean = pd.concat([df_sample, df_standardized, df_onehot_encoded], axis=1)

    # Fill out the rest of the missing columns 
    missing_cols = [col for col in self.df_final[self.cols_to_use].columns if col not in df_clean.columns]
    missing_cols_values = [self.default_center.at[0, col] for col in missing_cols]
    df_missing_cols = pd.DataFrame([missing_cols_values], columns=missing_cols)
    df_clean = df_clean.join(df_missing_cols)

    # Apply PCA
    df_clean_pca = self.pca.transform(df_clean[self.cols_to_use])
    df_clean_pca = pd.DataFrame(df_clean_pca, columns=self.cols_to_use_pca)

    return df_clean_pca

  def predict(self, website_data):
    df_sample = self.get_df_sample(website_data)
    return int(self.model.predict(df_sample)) + 1
  