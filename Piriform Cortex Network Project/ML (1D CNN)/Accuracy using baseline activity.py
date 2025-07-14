#!/usr/bin/env python
# coding: utf-8


import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, Dropout, Flatten, Dense, Input
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from mpi4py import MPI

# Initialize MPI environment
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Define paths
base_path = "/home/ankitk23/venv/graph_plot/Project_Data"
# Define output directory for saving results
output_dir = "/home/ankitk23/venv/graph_plot/results_baseline"

# Create output directory if it doesn't exist (only by root process)
if rank == 0:
    os.makedirs(output_dir, exist_ok=True)
    print(f"Results will be saved to: {output_dir}")

# Barrier to ensure directory is created before other processes might need it
comm.Barrier()


# List of main folders to process
main_folders = [
    "WOFF_FF_pcx_336"]

# Define folder label mappings
folder_mappings = {
    "A": {"A_2hz": 0, "A_20hz": 1},
    "B": {"B_2hz": 0, "B_20hz": 1},
    "Mix": {"Twenty_hz_Mix": 0, "Two_hz_Mix": 1}
}

# Define processing parameters
start_time = 0
end_time = 2000
bin_size = 20
num_bins = (end_time - start_time) // bin_size
num_neurons = 500
activation_value = 1  # Set activation presence value
num_trials = 144  # Set number of trials for each condition

def process_csv(file_path, activation_value):
    data = pd.read_csv(file_path).to_numpy()
    if data.shape[0] != num_neurons:
        raise ValueError(f"CSV file {file_path} does not have {num_neurons} rows.")

    binary_matrix = np.zeros((num_neurons, num_bins), dtype=float)

    for neuron_idx in range(num_neurons):
        activation_times = data[neuron_idx][~np.isnan(data[neuron_idx])]
        bin_indices = ((activation_times - start_time) // bin_size).astype(int)
        valid_indices = (bin_indices >= 0) & (bin_indices < num_bins)
        binary_matrix[neuron_idx, bin_indices[valid_indices]] = activation_value

    return binary_matrix

def shuffle_labels(labels, seed=None):
    if seed is not None:
        np.random.seed(seed)
    shuffled = labels.copy()
    np.random.shuffle(shuffled)
    return shuffled

def train_and_evaluate_model(data, real_labels, shuffled_labels, folder, mapping_name, trial_num, seed):
    results = []
    
    # Set random seed for reproducibility
    np.random.seed(seed)
    
    # Convert labels to categorical
    labels_categorical = to_categorical(real_labels, num_classes=2)
    shuffled_labels_categorical = to_categorical(shuffled_labels, num_classes=2)
    
    # Process with real labels
    X_train_real, X_test_real, y_train_real, y_test_real = train_test_split(
        data, labels_categorical, test_size=0.4, random_state=seed
    )
    
    # Create and train model with real labels
    model_real = Sequential([
        Input(shape=(100, num_neurons)),
        Conv1D(128, 3, activation='relu', padding='same'),
        Dropout(0.3),
        Conv1D(128, 3, activation='relu', padding='same'),
        Dropout(0.3),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.3),
        Dense(128, activation='relu'),
        Dropout(0.3),
        Dense(2, activation='softmax')
    ])
    
    model_real.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    
    model_real.fit(
        X_train_real, y_train_real, 
        validation_data=(X_test_real, y_test_real),
        epochs=75, batch_size=32, verbose=0,  # Changed to verbose=0 to reduce output for 144 trials
        callbacks=[early_stopping]
    )
    
    # Get predictions and calculate accuracy
    y_pred_real = np.argmax(model_real.predict(X_test_real, verbose=0), axis=1)
    y_test_actual = np.argmax(y_test_real, axis=1)
    acc_real = accuracy_score(y_test_actual, y_pred_real)
    results.append([folder, mapping_name, trial_num, acc_real, "real"])
    
    # Process with shuffled labels
    X_train_shuffled, X_test_shuffled, y_train_shuffled, y_test_shuffled = train_test_split(
        data, shuffled_labels_categorical, test_size=0.4, random_state=seed
    )
    
    # Create and train model with shuffled labels
    model_shuffled = Sequential([
        Input(shape=(100, num_neurons)),
        Conv1D(128, 3, activation='relu', padding='same'),
        Dropout(0.3),
        Conv1D(128, 3, activation='relu', padding='same'),
        Dropout(0.3),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.3),
        Dense(128, activation='relu'),
        Dropout(0.3),
        Dense(2, activation='softmax')
    ])
    
    model_shuffled.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    model_shuffled.fit(
        X_train_shuffled, y_train_shuffled, 
        validation_data=(X_test_shuffled, y_test_shuffled),
        epochs=75, batch_size=32, verbose=0,  # Changed to verbose=0 to reduce output for 144 trials
        callbacks=[early_stopping]
    )
    
    # Get predictions and calculate accuracy
    y_pred_shuffled = np.argmax(model_shuffled.predict(X_test_shuffled, verbose=0), axis=1)
    y_test_actual_shuffled = np.argmax(y_test_shuffled, axis=1)
    acc_shuffled = accuracy_score(y_test_actual_shuffled, y_pred_shuffled)
    results.append([folder, mapping_name, trial_num, acc_shuffled, "shuffled"])
    
    return results

# Create a list of tasks (combinations of main folders, mappings, and trials)
tasks = []
for main_folder in main_folders:
    for mapping_name in folder_mappings.keys():
        for trial in range(num_trials):
            tasks.append((main_folder, mapping_name, trial))

# Only the root process should print the intro
if rank == 0:
    print(f"Starting analysis with {size} MPI processes")
    print(f"Total tasks to process: {len(tasks)} ({num_trials} trials per condition)")

# Distribute tasks across processes
local_tasks = []
for i in range(len(tasks)):
    if i % size == rank:
        local_tasks.append(tasks[i])

if rank == 0:
    print(f"Tasks distribution: Each process will handle approximately {len(local_tasks)} tasks")

# Process local tasks
local_results = []

# Dictionary to store processed data for each folder-mapping combination
processed_data_cache = {}

for main_folder, mapping_name, trial_num in local_tasks:
    # Use a key to identify this folder-mapping combination
    folder_mapping_key = f"{main_folder}_{mapping_name}"
    
    # Log progress
    print(f"Process {rank}: Processing {main_folder} with mapping {mapping_name}, trial {trial_num}/{num_trials}")
    
    # Check if we've already processed data for this combination
    if folder_mapping_key not in processed_data_cache:
        all_data, all_labels = [], []
        for folder, label in folder_mappings[mapping_name].items():
            folder_path = os.path.join(base_path, main_folder, folder)
            if not os.path.exists(folder_path):
                print(f"Process {rank}: Folder {folder_path} does not exist, skipping...")
                continue

            for file in os.listdir(folder_path):
                if file.endswith(".csv"):
                    file_path = os.path.join(folder_path, file)
                    processed_data = process_csv(file_path, activation_value)
                    all_data.append(processed_data)
                    all_labels.append(label)

        if not all_data:
            print(f"Process {rank}: No data found for {main_folder} with mapping {mapping_name}. Skipping...")
            continue
            
        data = np.array(all_data)
        labels = np.array(all_labels)

        # Normalize data
        data = data / np.max(data)
        data = np.swapaxes(data, 1, 2)
        
        # Store processed data and labels in the cache
        processed_data_cache[folder_mapping_key] = (data, labels)
    else:
        # Retrieve from cache
        data, labels = processed_data_cache[folder_mapping_key]
    
    # Generate a unique seed for this trial
    seed = 1000 + trial_num
    
    # Generate shuffled labels for this specific trial
    shuffled_labels = shuffle_labels(labels.copy(), seed=seed)
    
    # Train and evaluate model for this trial
    task_results = train_and_evaluate_model(data, labels, shuffled_labels, main_folder, mapping_name, trial_num, seed)
    local_results.extend(task_results)
    
    # Report progress periodically (e.g., every 10 trials)
    if trial_num % 10 == 0:
        print(f"Process {rank}: Completed {main_folder}, {mapping_name}, trial {trial_num}")

# Gather all results to the root process
all_results = comm.gather(local_results, root=0)

# Only the root process should handle the final processing and plotting
# Only the root process should handle the final processing and plotting
if rank == 0:
    # Flatten the list of results
    flattened_results = [item for sublist in all_results if sublist for item in sublist]
    
    if not flattened_results:
        print("Error: No results collected. Please check the processing steps.")
        exit(1)
    
    print(f"Total collected results: {len(flattened_results)}")
    
    # Check which folders we have results for
    unique_folders = set(item[0] for item in flattened_results)
    print(f"Results collected for folders: {unique_folders}")
    
    # Separate real and shuffled results
    real_results = [r for r in flattened_results if r[4] == "real"]
    shuffled_results = [r for r in flattened_results if r[4] == "shuffled"]
    
    # Convert to dataframes
    df_real = pd.DataFrame(real_results, columns=["Folder", "Mapping", "Trial", "Real_Accuracy", "Type"])
    df_shuffled = pd.DataFrame(shuffled_results, columns=["Folder", "Mapping", "Trial", "Shuffled_Accuracy", "Type"])
    
    # Drop the Type column as it's no longer needed
    df_real = df_real.drop("Type", axis=1)
    df_shuffled = df_shuffled.drop("Type", axis=1)

    # Merge results for easier analysis
    df_results = pd.merge(df_real, df_shuffled, on=["Folder", "Mapping", "Trial"])
    
    # Print summary of collected results
    print("\nResults summary:")
    print(df_results.groupby(["Folder", "Mapping"]).size().reset_index(name='count'))
    
    # Save complete results to CSV
    csv_output_path = os.path.join(output_dir, "all_trials_results.csv")
    df_results.to_csv(csv_output_path, index=False)
    print(f"Complete results saved to {csv_output_path}")

    # Create summary dataframe with mean and std for each condition
    summary_df = df_results.groupby(["Folder", "Mapping"]).agg({
        "Real_Accuracy": ["mean", "std"],
        "Shuffled_Accuracy": ["mean", "std"]
    }).reset_index()
    
    # Flatten the multi-level columns
    summary_df.columns = ['_'.join(col).strip('_') for col in summary_df.columns.values]
    
    # Save summary to CSV
    summary_csv_path = os.path.join(output_dir, "summary_results.csv")
    summary_df.to_csv(summary_csv_path, index=False)
    print(f"Summary results saved to {summary_csv_path}")


