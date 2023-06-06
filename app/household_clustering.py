import pandas as pd
import numpy as np
import pickle

class HouseholdClustering():
  def __init__(self, transfer_data_path, transfer_df_income_path, best_k):
    # Load transfer data from the pickle file
    with open(transfer_data_path, 'rb') as file:
      transfer_data = pickle.load(file)
    self.rstd_cols = transfer_data['rstd_cols']

    # Load df_income from the pickle file
    with open(transfer_df_income_path, 'rb') as file:
      df_income = pickle.load(file)

    self.best_k = best_k
    self.cluster_assignments = transfer_data['clusters_rstd'][best_k]
    default_center = [df_income[col].mean() for col in self.rstd_cols]
    self.default_center = pd.DataFrame([default_center], columns=self.rstd_cols)
    self.df_income = df_income
  
  def summarize_cluster(self, i):
    response = {
      'cluster': i,
      'number_of_households': len(self.df_income[self.cluster_assignments == i-1]),
      'income': self.describe_income(i),
      'region': self.describe_region(i),
      'expenditure': self.describe_expenditure(i),
      'household_head': self.describe_household_head(i),
      'household_family': self.describe_housedold_family(i),
      'household_building': self.describe_household_building(i),
      'household_utilities': self.describe_household_utilities(i),
      'properties': self.describe_properties(i)
    }
    print(response)
    return response
  
  def describe_income(self, i):
    cluster_members = self.df_income[self.cluster_assignments == i-1]
    average_income = np.mean(cluster_members['total_household_income'])
    median_income = np.median(cluster_members['total_household_income'])
    income_std = np.std(cluster_members['total_household_income'])
    income_lower = average_income - income_std
    income_upper = average_income + income_std
    income_min = np.min(cluster_members['total_household_income'])
    income_max = np.max(cluster_members['total_household_income'])

    ret = []

    ret.append(f"average income: {self.num_to_pesos(average_income)}")
    ret.append(f"median income: {self.num_to_pesos(median_income)}")
    ret.append(f"standard deviation: {self.num_to_pesos(income_std)}")
    ret.append(f"range (within 1 std): {self.num_to_pesos(income_lower)} - {self.num_to_pesos(income_upper)}")
    ret.append(f"range (full): {self.num_to_pesos(income_min)} - {self.num_to_pesos(income_max)}")
    ret.append(f"coefficient of variation: {income_std / average_income}")

    return ret
  
  def describe_region(self, i):
    total = len(self.df_income[self.cluster_assignments == i-1])
    df = self.df_income[self.cluster_assignments==i-1]
    df = df['region'].value_counts(sort=True)

    ret = []
    regs = df[:3].map(arg=lambda x: f'{round(x*100/total,2)}%').to_dict()
    ret = [f"{k}: {regs[k]}" for k in regs]

    return ret

  def describe_expenditure(self, i):
    expenditure_cols = ['total_food_expenditure', 'bread_and_cereals_expenditure', 'total_rice_expenditure',
       'meat_expenditure', 'total_fish_and_marine_products_expenditure',
       'fruit_expenditure', 'vegetables_expenditure',
       'restaurant_and_hotels_expenditure', 'alcoholic_beverages_expenditure',
       'tobacco_expenditure', 'clothing_footwear_and_other_wear_expenditure',
       'housing_and_water_expenditure', 'imputed_house_rental_value',
       'medical_care_expenditure', 'transportation_expenditure',
       'communication_expenditure', 'education_expenditure',
       'miscellaneous_goods_and_services_expenditure',
       'special_occasions_expenditure', 'crop_farming_and_gardening_expenses',]
    df_cluster_members = self.df_income[self.cluster_assignments == i-1]
    median_expenditure = [np.median(df_cluster_members[expd_col]) for expd_col in expenditure_cols[1:]]
    mean_expenditure = [np.mean(df_cluster_members[expd_col]) for expd_col in expenditure_cols[1:]]
    col_median = list(zip(expenditure_cols[1:], median_expenditure))
    col_median = sorted(col_median, key=lambda x: x[1], reverse=True)

    ret = {}

    ret['median_food_expenditure'] = (f"Median Food Expenditure = {self.num_to_pesos(np.median(df_cluster_members['total_food_expenditure']))}")
    ret['top_expenses'] = []
    for i in range(5):
      ret['top_expenses'].append(f'{col_median[i][0]} = {self.num_to_pesos(col_median[i][1])}')
    return ret

  def describe_household_head(self, i):
    total = len(self.df_income[self.cluster_assignments == i-1])
    df_cluster_members = self.df_income[self.cluster_assignments == i-1]
    df_male_female = df_cluster_members['household_head_sex'].value_counts(sort=True)
    df_marital_status = df_cluster_members['household_head_marital_status'].value_counts(sort=True)
    df_job_indicator = df_cluster_members['household_head_job_or_business_indicator'].value_counts(sort=True)

    ret = []
    ret.append(self.summarize_column(df_male_female, "Sex"))
    ret.append(f"Median Age: {np.median(df_cluster_members['household_head_age'])}")
    ret.append(self.summarize_column(df_marital_status, "Marital Status"))
    ret.append(self.summarize_column(df_job_indicator, "Job or Business Indicator"))
    return ret

  def describe_housedold_family(self, i):
    df_cluster_members = self.df_income[self.cluster_assignments == i-1]
    df_household_type = df_cluster_members['type_of_household'].value_counts(sort=True)
    
    ret = []
    ret.append(self.summarize_column(df_household_type, "Type of Household"))
    ret.append(f"Median Number of Members: {np.median(df_cluster_members['total_number_of_family_members'])}")
    ret.append(f"Median Number of Members with Age < 5 years: {np.median(df_cluster_members['members_with_age_less_than_5_year_old'])}")
    ret.append(f"Median Number of Members 5-17 years old: {np.median(df_cluster_members['members_with_age_5_17_years_old'])}")
    ret.append(f"Median Number of Employed Members: {np.median(df_cluster_members['total_number_of_family_members_employed'])}")
    return ret

  def describe_household_building(self, i):
    df_cluster_members = self.df_income[self.cluster_assignments == i-1]
    df_household_building = df_cluster_members['type_of_building_house'].value_counts(sort=True)
    df_roof = df_cluster_members['type_of_roof'].value_counts(sort=True)
    df_walls = df_cluster_members['type_of_walls'].value_counts(sort=True)
    df_tenure = df_cluster_members['tenure_status'].value_counts(sort=True)

    ret = []
    ret.append(self.summarize_column(df_household_building, "Type of Building"))
    ret.append(self.summarize_column(df_roof, "Type of Roof"))
    ret.append(self.summarize_column(df_walls, "Type of Walls"))
    ret.append(self.summarize_column(df_tenure, "Tenure Status"))
    ret.append(f"Median House Floor Area: {np.median(df_cluster_members['house_floor_area'])}")
    ret.append(f"Median House Age: {np.median(df_cluster_members['house_age'])}")
    ret.append(f"Median Number of Bedrooms: {np.median(df_cluster_members['number_of_bedrooms'])}")
    return ret

  def describe_household_utilities(self, i):
    df_cluster_members = self.df_income[self.cluster_assignments == i-1]
    df_toilet = df_cluster_members['toilet_facilities'].value_counts(sort=True)
    df_water = df_cluster_members['main_source_of_water_supply'].value_counts(sort=True)
    df_electricity = df_cluster_members['electricity'].value_counts(sort=True)
    df_electricity.index = ["Electricity" if item==1 else "No Electricity" for item in df_electricity.index]

    ret = []
    ret.append(self.summarize_column(df_toilet, "Toilet Facilities"))
    ret.append(self.summarize_column(df_electricity, "Electricity"))
    ret.append(self.summarize_column(df_water, "Water Supply"))
    return ret

  def describe_properties(self, i):
    pass


  def summarize_column(self, series, col_name, max_kinds=3):
    total = sum(series)
    series_percents = [f"{round(series.loc[index]*100/total,2)}%" for index in series.index]
    series_message = f"{col_name} ({len(series.index)}): "
    kinds_to_display = min(max_kinds, len(series.index))
    for i in range(kinds_to_display):
      series_message += f"{series_percents[i]} are {series.index[i]}"
      if i!=kinds_to_display-1: series_message += ", "
    return series_message

  def num_to_pesos(self, num):
      return '₱{:,.2f}'.format(num)