import pandas as pd
import os
import csv
import time
import logging
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import SGDClassifier
from thundersvm import SVC


os.environ['NUMEXPR_MAX_THREADS'] = '16'

# Set up logging to file and console
log_file = "ml_benchmark_log.txt"
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)

# Fetch the Higgs dataset
higgs = fetch_openml(data_id=44129, as_frame=True)

# Describe the dataset
X, y = higgs.data, higgs.target
logging.info(f"Number of attributes: {X.shape[1]}")
logging.info(f"Number of rows: {X.shape[0]}")
logging.info(f"Attributes: {', '.join(X.columns.tolist())}")
class_counts = y.value_counts()
is_balanced = abs(class_counts[0] - class_counts[1]) < 0.1 * sum(class_counts)
logging.info(f"Is the dataset balanced? {'Yes' if is_balanced else 'No'}")

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.75, random_state=42)

# Models to evaluate
models = {
    #"ThunderSVM": SVC(kernel='rbf'),
    #"KNeighborsClassifier tuned": KNeighborsClassifier(n_neighbors=19),
    #"RandomForestClassifier": RandomForestClassifier(random_state=42),
    "ExtraTreesClassifier": ExtraTreesClassifier(random_state=42),
    "SGDClassifier": SGDClassifier(max_iter=1000),
    "HistGradientBoostingClassifier_tuned": HistGradientBoostingClassifier(l2_regularization=1.0, learning_rate=0.2, max_depth=5, max_iter= 300, min_samples_leaf= 1),
    "HistGradientBoostingClassifier": HistGradientBoostingClassifier()

}

# CSV file setup
csv_file = "ml_benchmark_results_repeat.csv"
with open(csv_file, mode="w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["Model Name", "Cores", "using n_jobs parameter" ,"Train Time", "Prediction Time", "Accuracy", "Confusion Matrix", "F1"])


# Loop over cores
max_cores = os.cpu_count()
print(f"Max cores count is equal to: {max_cores}")


for m in range(150):
    print(f"Loop number {m}")
    
    for n_cores in range(max_cores, 0, -1):
        logging.info(f"Evaluating with {n_cores} cores...")

        for model_name, model in models.items():
            # Adjust the number of cores for applicable models
            
            if hasattr(model, "n_jobs"):
                model.set_params(n_jobs=n_cores)
                njob = "Yes"
                
            else:
                njob = "No"
                
            print(f"Calculating for model: {model}")

            # Measure training time
            start_time = time.time()
            model.fit(X_train, y_train)
            train_time = time.time() - start_time

            print(f"Model trained in: {train_time:.4f} seconds")

            # Measure prediction time
            start_time = time.time()
            y_pred = model.predict(X_test)
            prediction_time = time.time() - start_time

            print(f"Model predicted in: {prediction_time:.4f} seconds")

            # Adjust the prediction result to be the same type
            y_pred = y_pred.astype(int).astype(str)
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            conf_matrix = confusion_matrix(y_test, y_pred).tolist()
            f1 = f1_score(y_test, y_pred, pos_label='1')

            # Save results to CSV
            with open(csv_file, mode="a", newline="") as file:
                writer = csv.writer(file)
                writer.writerow([model_name, n_cores, njob, train_time, prediction_time, accuracy, conf_matrix, f1])

            logging.info(
                f"Finished {model_name} with {n_cores} cores: Train Time={train_time:.3f}s, "
                f"Prediction Time={prediction_time:.3f}s, Accuracy={accuracy:.4f}, "
                f"F1 {f1}"
                f"Algorithm used n_jobs parameter: {njob}"
            )

