import pandas as pd
import os
import matplotlib.pyplot as plot
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore")

event = 'Belgian Grand Prix'
driver_name = 'VER'

lap_tyres = pd.read_csv('VER_Belgian Grand Prix_lap&tyres.csv')

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

def compute_lap_score(telemetry_data):
    weights = {'RPM': 0.3, 'Speed': 0.2, 'nGear': 0.2, 'Throttle': 0.2, 'Brake': 0.05, 'DRS': 0.1}
    total_lap_score = 0
    for index, row in telemetry_data.iterrows():
        lap_score = sum(row[param] * weight for param, weight in weights.items())
        total_lap_score += lap_score
    
    average_lap_score = total_lap_score / len(telemetry_data)
    return average_lap_score
#-------------------Functions end-------------------


lap_scores_train = pd.DataFrame(columns=['LapScore'])  
for lap_number in additional_train['LapNumber']:
    telemetry_data = fetch_telemetry(driver_name, event, lap_number)
    if telemetry_data is not None:
        lap_score = compute_lap_score(telemetry_data)
        lap_scores_train = pd.concat([lap_scores_train, pd.DataFrame({'LapScore': [lap_score]})], ignore_index=True)

'''lap_scores_train.to_csv('lap_scores_train.csv', index=False)'''

lap_scores_test = pd.DataFrame(columns=['LapScore'])  
for lap_number in additional_test['LapNumber']:
    telemetry_data = fetch_telemetry(driver_name, event, lap_number)
    if telemetry_data is not None:
        lap_score = compute_lap_score(telemetry_data)
        lap_scores_test = pd.concat([lap_scores_test, pd.DataFrame({'LapScore': [lap_score]})], ignore_index=True)


linear_model = LinearRegression()
linear_model.fit(lap_scores_train, target_train)
linear_predictions = linear_model.predict(lap_scores_train)
linear_mse = mean_squared_error(target_train, linear_predictions)
print(f"Linear Regression Mean Squared Error: {linear_mse}")

plot.scatter(target_train, linear_predictions)
plot.xlabel('Actual Lap Time')
plot.ylabel('Predicted Lap Time')
plot.show()




    

