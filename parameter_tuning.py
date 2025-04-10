import pandas as pd
from tqdm import tqdm
from glob import glob
from datetime import datetime
from math import sqrt, atan, degrees
from re import match
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from time import time
from json import dump as json_dump


SCREEN_WIDTH, SCREEN_HEIGHT = 366, 229
# silence pandas warnings
pd.options.mode.chained_assignment = None  # default='warn'
data_dict = {"subject_id": [], "number_of_intrusive_saccades": [], "mean_srt": [], "cov_srt": [], "number_of_direction_errors": [], "is_patient": []}
counter = 0

def apply_zfill(x):
    fields = x.split("/")

    date_str = ""
    for field in fields:
        date_str += field.zfill(2) + "/"
    return date_str[:-1]

def find_timestamp(recording_timestamp, recording_start_date, recording_start_time):
    dt = recording_start_date + " " + recording_start_time
    #include milliseconds
    start_dt = datetime.strptime(dt, "%m/%d/%Y %H:%M:%S.%f")
    return start_dt.timestamp() * 1000 + recording_timestamp / 1000


def euclidean_distance_calculation(row):
	# Note the 'shifted_' prefix in the name.
    return sqrt(((row["shifted_Gaze point X (MCSnorm)"] - row["Gaze point X (MCSnorm)"]) * SCREEN_WIDTH) ** 2 + ((row["shifted_Gaze point Y (MCSnorm)"] - row["Gaze point Y (MCSnorm)"]) * SCREEN_HEIGHT) ** 2)

 # Then, we need a function to carry out the equation we defined
 # previously. For this, we use the math package that comes with 
 # Python.
def visual_angle_calculation(row, distance_to_screen = 600):

    distance = euclidean_distance_calculation(row)
    angle = 2 * atan((distance)/(2 * distance_to_screen))
    
    # Note that we convert the angle to degrees, as Python
    # math.atan function returns radians!
    return degrees(angle)

def shift_direction(row):
    return row["shifted_Gaze point X (MCSnorm)"] - row["Gaze point X (MCSnorm)"]

def extract_features(folders, y_value, angle_threshold, saccade_threshold):
    global counter

    
    for folder in folders:
        counter += 1
        eye_tracking_file = glob(f"{folder}/*.csv")[0]
        timing_log_file = glob(f"{folder}/*.txt")[0]


        eye_tracking_data = pd.read_csv(eye_tracking_file, usecols=["Recording date", "Recording timestamp", "Recording start time", "Gaze point X (MCSnorm)", "Gaze point Y (MCSnorm)", "Participant name", "Eye movement type", "Eye movement event duration", "Fixation point X (MCSnorm)", "Fixation point Y (MCSnorm)"])

        eye_tracking_data["Recording date"] = eye_tracking_data["Recording date"].apply(apply_zfill)
        eye_tracking_data["record_stamp"] = eye_tracking_data.apply(lambda x: find_timestamp(x["Recording timestamp"], x["Recording date"], x["Recording start time"]), axis=1).astype(int)
        eye_tracking_data = pd.concat([eye_tracking_data, eye_tracking_data[["Gaze point X (MCSnorm)", "Gaze point Y (MCSnorm)", "Fixation point X (MCSnorm)", "Fixation point Y (MCSnorm)"]].shift(-1).add_prefix('shifted_')], axis = 1).iloc[:-1, :]

        eye_tracking_data["angle"] = eye_tracking_data.apply(visual_angle_calculation, axis=1)
        eye_tracking_data["direction"] = eye_tracking_data.apply(shift_direction, axis=1)


        with open(timing_log_file, "r") as f:
            lines = f.readlines()

        pc_data_dict = {"task_id": [], "task_start": [], "task_end": [], "task_type": [], "location": []}
        fixation_start = match(".*Unix Time: (\d+)", lines[1]).group(1)
        fixation_end = match(".*Unix Time: (\d+)", lines[2]).group(1)
        pc_data_dict["task_id"].append(0)
        pc_data_dict["task_start"].append(int(fixation_start))
        pc_data_dict["task_end"].append(int(fixation_end))
        pc_data_dict["task_type"].append("F")
        pc_data_dict["location"].append("M")

        for i in range(7, 175, 6):
            saccade_task_start = match(".*Unix Time: (\d+)", lines[i]).group(1)
            location = match(".*([RL])", lines[i]).group(1)
            saccade_task_end = match(".*Unix Time: (\d+)", lines[i + 1]).group(1)
            saccade_task_type = match(".*([A-Z]).*saccade.*", lines[i]).group(1)
            
            pc_data_dict["task_id"].append(int((i-1)/6))
            pc_data_dict["task_start"].append(int(saccade_task_start))
            pc_data_dict["task_end"].append(int(saccade_task_end))
            pc_data_dict["task_type"].append(saccade_task_type)
            pc_data_dict["location"].append(location)

        pc_data = pd.DataFrame(pc_data_dict)
        # pc_data = pc_data[(pc_data["task_id"] == 0) | (pc_data["task_id"] <= trial_threshold)]


        parameters = {
            "number_of_intrusive_saccades": 0,
            "srt_values": [],
            "number_of_direction_errors": 0
        }

        for i in range(len(pc_data)):
            task_row = pc_data.iloc[i]
            task_data = eye_tracking_data[(eye_tracking_data["record_stamp"] > task_row["task_start"]) & (eye_tracking_data["record_stamp"] < task_row["task_end"])]
            if task_row["task_type"] == "F":
                parameters["number_of_intrusive_saccades"] += len(task_data[task_data["angle"] >= angle_threshold])
            else:
                task, location = task_row["task_type"], task_row["location"]
                location = 1 if location == "R" else -1
                if task == "A":
                    location *= -1

                task_data["record_stamp_norm"] = task_data["record_stamp"] - task_row["task_start"]
                task_data = task_data[task_data["record_stamp_norm"] > 90]
                task_data = task_data[task_data["Eye movement type"] == "Saccade"].reset_index()
                task_data = task_data[task_data["angle"] >= saccade_threshold]
                if task_data.empty :#or task_data["record_stamp_norm"].iloc[0] > 1000:
                    # parameters["srt_values"].append(1000)
                    # parameters["number_of_direction_errors"] += 1
                    continue

                direction_error = location * task_data["direction"].iloc[0]
                if direction_error < 0:
                    parameters["number_of_direction_errors"] += 1
                else:
                    srt = task_data["record_stamp_norm"].iloc[0]
                    parameters["srt_values"].append(srt)


        mean = np.mean(parameters["srt_values"])
        cov = np.std(parameters["srt_values"]) / mean

        data_dict["subject_id"].append(counter)
        data_dict["number_of_intrusive_saccades"].append(parameters["number_of_intrusive_saccades"])
        data_dict["mean_srt"].append(mean)
        data_dict["cov_srt"].append(cov)
        data_dict["number_of_direction_errors"].append(parameters["number_of_direction_errors"])
        data_dict["is_patient"].append(y_value)





def cv_model(predictor, X, y):
    scores = cross_val_score(predictor, X, y, cv=5, scoring="f1_micro")
    return scores.mean()


def model_training():
    df = pd.read_csv("features.csv")

    models = {
        "rfc20": RandomForestClassifier(n_estimators = 20),
        "rfc100": RandomForestClassifier(n_estimators = 100),
        "lr": LogisticRegression(solver="liblinear"),
        "svclinear": SVC(kernel='linear'),
        "svcrbf": SVC(kernel='rbf'),
        "svcrbf_c10": SVC(kernel='rbf', C=10),
        "svcpoly": SVC(kernel='poly')
    }

    scores = {model: [] for model in models}

    for _ in tqdm(range(100)):
        data = df.drop("subject_id", axis=1).to_numpy()
        np.random.shuffle(data)
        X, y = data[:, :-1], data[:, -1]

        for model in models:
            score = cv_model(models[model], X, y)
            scores[model].append(score)

    return {model: (np.mean(scores[model]), np.std(scores[model])) for model in scores}


if __name__ == "__main__":
    feature_extraction_parameters = {
        "intrusive_saccade_angle_threshold": [1.5, 2, 2.5],
        "saccade_angle_threshold": [0.5, 1, 1.5, 3, 5],
    }

    patient_folders = glob("final_data_pairs/patient_group/*")
    control_folders = glob("final_data_pairs/control_group/*")
    mean_table = {
        "Angle Threshold": [],
        "Saccade Threshold": [],
        "Model": [],
        "Mean": [],
        "Std": []
    }

    for angle_threshold in feature_extraction_parameters["intrusive_saccade_angle_threshold"]:
        with open("mean_table.json", "w") as f:
            json_dump(mean_table, f)
        for saccade_threshold in feature_extraction_parameters["saccade_angle_threshold"]:
            start = time()

            extract_features(patient_folders, 1, angle_threshold, saccade_threshold)
            extract_features(control_folders, 0, angle_threshold, saccade_threshold)

            data = pd.DataFrame(data_dict)
            data.dropna(inplace=True)
            data.to_csv("features.csv", index=False)

            results = model_training()

            for model in results:
                mean_table["Angle Threshold"].append(angle_threshold)
                mean_table["Saccade Threshold"].append(saccade_threshold)
                mean_table["Model"].append(model)
                mean_table["Mean"].append(results[model][0])
                mean_table["Std"].append(results[model][1])


            end = time()
            print(f"Angle Threshold: {angle_threshold}, Saccade Threshold: {saccade_threshold} done in", round(end - start, 2), "seconds")


    mean_table = pd.DataFrame(mean_table)
    mean_table.to_csv("mean_table.csv", index=False)