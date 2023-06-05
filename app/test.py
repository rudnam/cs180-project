import pandas as pd
import json

dataset_url = 'https://raw.githubusercontent.com/rudnam/CS-180-Project/main/Family%20Income%20and%20Expenditure.csv'
df_orig = pd.read_csv(dataset_url)

cat_features = ['region', 'main-source-of-income', 'household-head-sex', 'household-head-marital-status', 'household-head-job-or-business-indicator', 'household-head-class-of-worker', 'type-of-household', 'type-of-building-house', 'type-of-roof', 'type-of-walls', 'tenure-status', 'toilet-facilities', 'main-source-of-water-supply']

df_income = df_orig.drop(columns = ['Household Head Highest Grade Completed', 'Household Head Occupation'])
df_income.columns = [col.replace("/", " ").replace(",", "") for col in df_income.columns]
df_income.columns = [col.replace(" - ", " ") for col in df_income.columns]
df_income.columns = ["-".join(col.lower().split()) for col in df_income.columns]

cat_features = [col.replace("_", "-") for col in cat_features]
cat_choices = {}
for cat_feat in cat_features:
    cat_choices[cat_feat] = df_income[cat_feat].unique().tolist()

print(type(cat_choices))
json_object = json.dumps(cat_choices, indent = 4) 
print(json_object)