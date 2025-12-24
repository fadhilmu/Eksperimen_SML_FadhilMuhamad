import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

OUTPUT_DIR = "preprocessing"
OUTPUT_FILE = "CaliforniaHousing_preprocessed.csv"

data = pd.read_csv("./CaliforniaHousing.csv")

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    df_copy = df.copy()


    df_copy['total_bedrooms'] = df_copy['total_bedrooms'].fillna(df_copy['total_bedrooms'].median())


    df_copy['rooms_per_house'] = df_copy['total_rooms'] / df_copy['households']
    df_copy['bedrooms_per_room'] = df_copy['total_bedrooms'] / df_copy['total_rooms']
    df_copy['pop_per_house'] = df_copy['population'] / df_copy['households']


    df_copy = pd.get_dummies(df_copy, columns=['ocean_proximity'], drop_first=True)


    skew_cols = ['total_rooms', 'total_bedrooms', 'population', 'households']
    for col in skew_cols:
        df_copy[col] = np.log1p(df_copy[col])


    features_to_scale = [
        'longitude', 'latitude', 'housing_median_age', 'median_income',
        'rooms_per_house', 'bedrooms_per_room', 'pop_per_house'
    ] + skew_cols

    scaler = StandardScaler()
    df_copy[features_to_scale] = scaler.fit_transform(df_copy[features_to_scale])

    return df_copy

processed_df = preprocess_data(data)

os.makedirs(OUTPUT_DIR, exist_ok=True)
output_path = os.path.join(OUTPUT_DIR, OUTPUT_FILE)
processed_df.to_csv(output_path, index=False)

print(f"Preprocessing selesai! File tersimpan di: {output_path}")