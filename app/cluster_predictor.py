import pandas as pd
import numpy as np
import copy
import pickle

class ClusterPredictor():
  def __init__(self, transfer_data_path, transfer_df_income_path, best_k, pca_reduced=True):
    # Load needed data from the pickle file
    with open(transfer_data_path, 'rb') as file:
      transfer_data = pickle.load(file)
    
    # Load df_income from the pickle file
    with open(transfer_df_income_path, 'rb') as file:
      df_income = pickle.load(file)

    self.num_features = copy.copy(transfer_data['num_features'])
    self.cat_features = copy.copy(transfer_data['cat_features'])
    self.df_income = copy.deepcopy(df_income)
    self.std_cols = copy.copy(transfer_data['std_cols'])
    self.rstd_cols = copy.copy(transfer_data['rstd_cols'])
    self.num_cols = copy.copy(transfer_data['num_cols'])
    self.col_rank_dicts = copy.copy(transfer_data['col_rank_dicts'])

    self.best_k = best_k
    self.pca_reduced = pca_reduced
    self.model = transfer_data['models_rstd'][best_k]
    self.labels = transfer_data['models_rstd'][best_k].labels_
    self.cluster_centers = [self.get_cluster_center(cluster_i) for cluster_i in range(best_k)]
    default_center = [np.mean(df_income[col]) for col in self.rstd_cols]
    self.default_center = pd.DataFrame([default_center], columns=self.rstd_cols)
    self.col_stds = pd.DataFrame([[np.std(df_income[col]) for col in self.rstd_cols]], columns=self.rstd_cols)
    self.col_means = pd.DataFrame([[np.mean(df_income[col]) for col in self.rstd_cols]], columns=self.rstd_cols)
  
  def get_cluster_center(self, cluster_i):
    cluster_members = self.df_income[self.model.labels_ == cluster_i]
    cluster_members = cluster_members[self.rstd_cols]
    return [np.mean(cluster_members[col]) for col in self.rstd_cols]
  
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
        df_sample[col.replace('-display','')] = df_sample[col]
        display_cols.append(col)
    
    if len(display_cols)>0: df_sample = df_sample.drop(columns = display_cols)
    df_sample.columns = [underscore_formatter(col) for col in df_sample.columns]
    df_sample.replace({'electricity': {'Has electricity': 1, 'No electricity': 0}}, inplace=True)

    # Standardize
    num_feat_sample = [col for col in self.num_features if col in df_sample.columns]
    rstd_cat_cols_sample = [f'{col}_rstd' for col in self.cat_features if col in df_sample.columns]
    ranks_sample = []
    for col in rstd_cat_cols_sample:
      orig_cat_column = '_'.join(col.split('_')[:-1])
      category = underscore_formatter(df_sample.at[0, orig_cat_column])
      category_rank = None
      for i in range(len(self.col_rank_dicts)):
        try:
          category_rank = self.col_rank_dicts[i][category]
          if category_rank != None: break
        except:
          pass
      ranks_sample.append(category_rank)
    df_ranks = pd.DataFrame([ranks_sample], columns = rstd_cat_cols_sample)
    df_sample = df_sample.join(df_ranks)

    num_feat_std_sample = [f'{col}_std' for col in num_feat_sample]
    df_nums = copy.deepcopy(df_sample[num_feat_sample])
    df_nums.columns = num_feat_std_sample
    df_sample = df_sample.join(df_nums)

    sample_z_scores = []
    for col in rstd_cat_cols_sample + num_feat_std_sample:
      std = self.col_stds.at[0, col]
      mean = self.col_means.at[0, col]
      smp = int(df_sample.at[0, col])
      sample_z_scores.append((smp-mean)/std)

    rstd_cols_sample = rstd_cat_cols_sample + num_feat_std_sample
    df_final = pd.DataFrame([sample_z_scores], columns=rstd_cols_sample)
    df_model = copy.deepcopy(self.df_income[self.rstd_cols])

    # compare df_sample[std_cols] to df_income[std_cols]
    missing_cols = [col for col in df_model.columns if col not in df_final.columns]
    missing_cols_values = [self.default_center.at[0, col] for col in missing_cols]
    df_missing_cols = pd.DataFrame([missing_cols_values], columns=missing_cols)
    df_final = df_final.join(df_missing_cols)
    print(f'There are {len(missing_cols)} unsupplied columns filled with default value.')

    return df_final[df_model.columns]

  def predict(self, website_data):
    df_sample = self.get_df_sample(website_data)
    if self.pca_reduced==False: return self.model.predict(df_sample)+1
    min_index = 0
    for i in range(1, self.best_k):
      min_center = self.cluster_centers[min_index]
      next_center = self.cluster_centers[i]
      if self.get_distance(df_sample, min_center) > self.get_distance(df_sample, next_center): min_index = i
    print(f"Website sample belongs to cluster {min_index+1}.")
    return min_index+1
  
  def get_distance(self, df_sample, cluster_center):
    dist_squared = 0
    for i in range(self.best_k):
      x1 = df_sample.at[0, df_sample.columns[i]]
      x2 = cluster_center[i]
      dist_squared += (x1-x2)**2
    return dist_squared
  