import pandas as pd

import pickle

class ModelWrapper:
  def __init__(self, model_path, clustering_path):
    self.model = self.load_model(model_path)
    self.clustering = self.load_clustering(clustering_path)
  
  def load_model(self, model_path):
    with open(model_path, 'rb') as f:
      model = pickle.load(f)
    return model

  def load_clustering(self, clustering_path):
    with open(clustering_path, 'rb') as f:
      clustering = pickle.load(f)
    return clustering

  def predict(self, data):
    df = pd.DataFrame([data])
    df_sample = df.drop(columns = [col for col in df.columns if '-display' in col])
    df_sample.columns = ["_".join(col.split('-')) for col in df_sample.columns]

    cat_features = ['region', 'main_source_of_income', 'household_head_sex', 'household_head_marital_status', 'household_head_job_or_business_indicator', 'household_head_class_of_worker', 'type_of_household', 'type_of_building_house', 'type_of_roof', 'type_of_walls', 'tenure_status', 'toilet_facilities', 'main_source_of_water_supply']
    num_features = [feat for feat in df_sample.columns if feat not in cat_features]

    df_sample.replace({'electricity': {'Has electricity': 1, 'No electricity': 0}}, inplace=True)

    # One hot encoding
    new_num_cols = []
    for feat in cat_features:
      df_sample[feat].fillna("nan", inplace=True)
      for uniq in df_sample[feat].unique():
        uniq0 = uniq
        uniq = "_".join(uniq.lower().replace(",", "").replace("/", " ").replace(" - ", " ").split())
        new_col = f"{feat}_{uniq}"
        new_num_cols.append(new_col)
        df_sample[new_col] = [int(uniq0==item) for item in df_sample[feat]]

    # Standardize
    from sklearn.preprocessing import StandardScaler
    import copy

    num_cols = new_num_cols + num_features
    std_cols = [f'{col}_std' for col in num_cols]
    df_prep = copy.deepcopy(df_sample[num_cols])
    scaler = StandardScaler()
    df_prep = pd.DataFrame(scaler.fit_transform(df_prep), columns=std_cols)

    df_sample = df_sample.join(df_prep)

    required_cols = ['region_car_std', 'region_caraga_std', 'region_vi_western_visayas_std', 'region_v_bicol_region_std', 'region_armm_std', 'region_iii_central_luzon_std', 'region_ii_cagayan_valley_std', 'region_iva_calabarzon_std', 'region_vii_central_visayas_std', 'region_x_northern_mindanao_std', 'region_xi_davao_region_std', 'region_viii_eastern_visayas_std', 'region_i_ilocos_region_std', 'region_ncr_std', 'region_ivb_mimaropa_std', 'region_xii_soccsksargen_std', 'region_ix_zasmboanga_peninsula_std', 'main_source_of_income_wage_salaries_std', 'main_source_of_income_other_sources_of_income_std', 'main_source_of_income_enterpreneurial_activities_std', 'household_head_sex_female_std', 'household_head_sex_male_std', 'household_head_marital_status_single_std', 'household_head_marital_status_married_std', 'household_head_marital_status_widowed_std', 'household_head_marital_status_divorced_separated_std', 'household_head_marital_status_annulled_std', 'household_head_marital_status_unknown_std', 'household_head_job_or_business_indicator_with_job_business_std', 'household_head_job_or_business_indicator_no_job_business_std', 'household_head_class_of_worker_worked_for_government_government_corporation_std', 'household_head_class_of_worker_worked_for_private_establishment_std', 'household_head_class_of_worker_employer_in_own_family-operated_farm_or_business_std', 'household_head_class_of_worker_self-employed_wihout_any_employee_std', 'household_head_class_of_worker_nan_std', 'household_head_class_of_worker_worked_without_pay_in_own_family-operated_farm_or_business_std', 'household_head_class_of_worker_worked_for_private_household_std', 'household_head_class_of_worker_worked_with_pay_in_own_family-operated_farm_or_business_std', 'type_of_household_extended_family_std', 'type_of_household_single_family_std', 'type_of_household_two_or_more_nonrelated_persons_members_std', 'type_of_building_house_single_house_std', 'type_of_building_house_duplex_std', 'type_of_building_house_commercial_industrial_agricultural_building_std', 'type_of_building_house_multi-unit_residential_std', 'type_of_building_house_institutional_living_quarter_std', 'type_of_building_house_other_building_unit_(e.g._cave_boat)_std', 'type_of_roof_strong_material(galvanizedironaltileconcretebrickstoneasbestos)_std', 'type_of_roof_light_material_(cogonnipaanahaw)_std', 'type_of_roof_mixed_but_predominantly_strong_materials_std', 'type_of_roof_mixed_but_predominantly_light_materials_std', 'type_of_roof_salvaged_makeshift_materials_std', 'type_of_roof_mixed_but_predominantly_salvaged_materials_std', 'type_of_roof_not_applicable_std', 'type_of_walls_strong_std', 'type_of_walls_light_std', 'type_of_walls_quite_strong_std', 'type_of_walls_very_light_std', 'type_of_walls_salvaged_std', 'type_of_walls_not_applicable_std', 'tenure_status_own_or_owner-like_possession_of_house_and_lot_std', 'tenure_status_rent-free_house_and_lot_with_consent_of_owner_std', 'tenure_status_own_house_rent-free_lot_with_consent_of_owner_std', 'tenure_status_own_house_rent-free_lot_without_consent_of_owner_std', 'tenure_status_own_house_rent_lot_std', 'tenure_status_rent_house_room_including_lot_std', 'tenure_status_not_applicable_std', 'tenure_status_rent-free_house_and_lot_without_consent_of_owner_std', 'toilet_facilities_water-sealed_sewer_septic_tank_used_exclusively_by_household_std', 'toilet_facilities_water-sealed_sewer_septic_tank_shared_with_other_household_std', 'toilet_facilities_closed_pit_std', 'toilet_facilities_water-sealed_other_depository_used_exclusively_by_household_std', 'toilet_facilities_open_pit_std', 'toilet_facilities_water-sealed_other_depository_shared_with_other_household_std', 'toilet_facilities_none_std', 'toilet_facilities_others_std', 'main_source_of_water_supply_own_use_faucet_community_water_system_std', 'main_source_of_water_supply_shared_faucet_community_water_system_std', 'main_source_of_water_supply_shared_tubed_piped_deep_well_std', 'main_source_of_water_supply_own_use_tubed_piped_deep_well_std', 'main_source_of_water_supply_protected_spring_river_stream_etc_std', 'main_source_of_water_supply_tubed_piped_shallow_well_std', 'main_source_of_water_supply_lake_river_rain_and_others_std', 'main_source_of_water_supply_unprotected_spring_river_stream_etc_std', 'main_source_of_water_supply_dug_well_std', 'main_source_of_water_supply_others_std', 'main_source_of_water_supply_peddler_std', 'total_household_income_std', 'total_food_expenditure_std', 'agricultural_household_indicator_std', 'bread_and_cereals_expenditure_std', 'total_rice_expenditure_std', 'meat_expenditure_std', 'total_fish_and_marine_products_expenditure_std', 'fruit_expenditure_std', 'vegetables_expenditure_std', 'restaurant_and_hotels_expenditure_std', 'alcoholic_beverages_expenditure_std', 'tobacco_expenditure_std', 'clothing_footwear_and_other_wear_expenditure_std', 'housing_and_water_expenditure_std', 'imputed_house_rental_value_std', 'medical_care_expenditure_std', 'transportation_expenditure_std', 'communication_expenditure_std', 'education_expenditure_std', 'miscellaneous_goods_and_services_expenditure_std', 'special_occasions_expenditure_std', 'crop_farming_and_gardening_expenses_std', 'total_income_from_entrepreneurial_acitivites_std', 'household_head_age_std', 'total_number_of_family_members_std', 'members_with_age_less_than_5_year_old_std', 'members_with_age_5_17_years_old_std', 'total_number_of_family_members_employed_std', 'house_floor_area_std', 'house_age_std', 'number_of_bedrooms_std', 'electricity_std', 'number_of_television_std', 'number_of_cd_vcd_dvd_std', 'number_of_component_stereo_set_std', 'number_of_refrigerator_freezer_std', 'number_of_washing_machine_std', 'number_of_airconditioner_std', 'number_of_car_jeep_van_std', 'number_of_landline_wireless_telephones_std', 'number_of_cellular_phone_std', 'number_of_personal_computer_std', 'number_of_stove_with_oven_gas_range_std', 'number_of_motorized_banca_std', 'number_of_motorcycle_tricycle_std']

    df_final = copy.deepcopy(df_sample[std_cols])

    # compare df_sample[std_cols] to df_income[std_cols]
    missing_cols = [col for col in required_cols if col not in df_final.columns]
    for col in missing_cols:
      df_final[col] = 0

    df_final = df_final[required_cols]

    return self.model.predict(df_final)[0]