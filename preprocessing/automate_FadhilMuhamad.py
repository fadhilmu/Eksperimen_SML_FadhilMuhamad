import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def preprocess_housing(df: pd.DataFrame):
    df = df.copy()
    df['total_bedrooms'] = df['total_bedrooms'].fillna(df['total_bedrooms'].median())
    df['rooms_per_household'] = df['total_rooms'] / df['households']
    df['bedrooms_per_room'] = df['total_bedrooms'] / df['total_rooms']
    df['population_per_household'] = df['population'] / df['households']
    df = pd.get_dummies(df, columns=['ocean_proximity'], drop_first=True)
    skewed_features = ['total_rooms', 'total_bedrooms', 'population', 'households']
    for feature in skewed_features:
        df[feature] = np.log1p(df[feature])
    numerical_features = [
        'longitude', 'latitude', 'housing_median_age', 'median_income',
        'rooms_per_household', 'bedrooms_per_room', 'population_per_household'
    ] + skewed_features
    scaler = StandardScaler()
    df[numerical_features] = scaler.fit_transform(df[numerical_features])
    return df

script_dir = os.path.dirname(os.path.abspath(__file__))
dataset_path = os.path.abspath(os.path.join(script_dir, "..", "CaliforniaHousing.csv"))
output_folder = script_dir
os.makedirs(output_folder, exist_ok=True)

output_path = os.path.join(output_folder, "CaliforniaHousing_preprocessed.csv")

df = pd.read_csv(dataset_path)
df_preprocessed = preprocess_housing(df)
df_preprocessed.to_csv(output_path, index=False)
print(f"Preprocessing selesai. File tersimpan di: {output_path}")
