import pandas as pd
import os
import matplotlib.pyplot as plot
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import warnings
import math
warnings.filterwarnings("ignore")

event = 'Monaco Grand Prix'
driver_name = 'VER'

lap_tyres = pd.read_csv('VER_Monaco Grand Prix_lap&tyres.csv')

additional_train, additional_test = train_test_split(lap_tyres, test_size=0.2, random_state=42)
target_train = additional_train['LapTime']
target_test = additional_test['LapTime']
additional_train = additional_train.drop(columns=['LapTime'])
additional_test = additional_test.drop(columns=['LapTime'])
output_directory = 'telemetry'

# -------------------Functions-------------------
def fetch_telemetry(driver_name, event, lap_number):
    file_name = f'{driver_name}_{event}_{lap_number}_telemetry.csv'
    file_path = os.path.join(output_directory, file_name)
    if os.path.exists(file_path):
        return pd.read_csv(file_path)
    else:
        print(f"Telemetry file for lap {lap_number} does not exist.")
        return None

def average_data_points(telemetry_data):
    if not hasattr(average_data_points, 'total'):
        average_data_points.total = 0
    average_data_points.total += len(telemetry_data)

#-------------------Normalizing?-------------------
def race_effect(lap_scores, additional):
    for (index_lap, row_lap), (index_additional, row_additional) in zip(lap_scores.iterrows(), additional.iterrows()):
        row_lap['LapScore'] = row_lap['LapScore'] + (row_additional['LapNumber'] * 0.1) - (row_additional['TyreLife'] ** 0.001) - (row_additional['Compound'] ** 0.0001)
    return lap_scores

def compute_lap_score(telemetry_data, avg):
    weights = {'RPM': 0.3, 'Speed': 0.2, 'nGear': 0.2, 'Throttle': 0.2, 'Brake': 0.05, 'DRS': 0.1}
    total_lap_score = 0
    i=0
    for index, row in telemetry_data.iterrows():
        lap_score = sum(row[param] * weight for param, weight in weights.items())
        if i <= avg:
            i += 1
            total_lap_score += lap_score
        else:
            total_lap_score += (lap_score)/4
            

    average_lap_score = total_lap_score / len(telemetry_data)
    return average_lap_score
#-------------------Normalizing? end-------------------
#-------------------Functions end-------------------


#experimental
for lap_number in additional_test['LapNumber']:
    telemetry_data = fetch_telemetry(driver_name, event, lap_number)
    if telemetry_data is not None:
        data_points = average_data_points(telemetry_data)
avg = math.floor(average_data_points.total / len(additional_test))
#experimental end


lap_scores_train = pd.DataFrame(columns=['LapScore'])  
for lap_number in additional_train['LapNumber']:
    telemetry_data = fetch_telemetry(driver_name, event, lap_number)
    if telemetry_data is not None:
        lap_score = compute_lap_score(telemetry_data, avg)
        lap_scores_train = pd.concat([lap_scores_train, pd.DataFrame({'LapScore': [lap_score]})], ignore_index=True)

lap_scores_train = race_effect(lap_scores_train, additional_train)

additional_train.to_csv('additional_train.csv', index=False)
lap_scores_train.to_csv('lap_scores_train.csv', index=False)

lap_scores_test = pd.DataFrame(columns=['LapScore'])  

for lap_number in additional_test['LapNumber']:
    telemetry_data = fetch_telemetry(driver_name, event, lap_number)
    if telemetry_data is not None:
        lap_score = compute_lap_score(telemetry_data, avg)
        lap_scores_test = pd.concat([lap_scores_test, pd.DataFrame({'LapScore': [lap_score]})], ignore_index=True)

lap_scores_test = race_effect(lap_scores_test, additional_test)

linear_model = LinearRegression()
linear_model.fit(lap_scores_train, target_train)
linear_predictions = linear_model.predict(lap_scores_test)
linear_mse = mean_squared_error(target_test, linear_predictions)
print(f"Linear Regression Mean Squared Error: {linear_mse}")

plot.scatter(target_test, linear_predictions)
plot.xlabel('Actual Lap Time')
plot.ylabel('Predicted Lap Time')
plot.show()