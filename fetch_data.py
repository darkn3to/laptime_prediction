import fastf1
import pandas as pd
import warnings
import os

warnings.filterwarnings("ignore")

event = 'Monaco Grand Prix'
driver_name = 'VER'

session = fastf1.get_session(2023, event, 'R')
session.load()

all_laps = session.laps.pick_drivers(driver_name).pick_quicklaps().pick_not_deleted().reset_index()
merged = all_laps[['LapNumber', 'LapTime', 'Compound', 'TyreLife']]
merged['LapTime'] = merged['LapTime'].dt.total_seconds()

output_directory = 'telemetry'
os.makedirs(output_directory, exist_ok=True)

for index, row in all_laps.iterrows():
    car_data = row.get_car_data()
    car_data['Brake'] = car_data['Brake'].replace({True: 1, False: 0})
    car_data = car_data.drop(['Date', 'Source', 'SessionTime', 'Time'], axis=1)
    file_name = os.path.join(output_directory, f"{driver_name}_{event}_{row['LapNumber']}_telemetry.csv")
    car_data.to_csv(file_name, index=False)

merged['Compound'] = merged['Compound'].replace({'SOFT': 1, 'MEDIUM': 2, 'HARD': 3, 'INTERMEDIATE': 4, 'WET': 5})
merged.to_csv(f'{driver_name}_{event}_lap&tyres.csv', index=False)

print("Telemetry & lap data saved.")