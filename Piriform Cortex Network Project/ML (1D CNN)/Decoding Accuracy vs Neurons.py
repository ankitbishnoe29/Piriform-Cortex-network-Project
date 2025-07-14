#!/usr/bin/env python
# coding: utf-8

import csv
from collections import Counter
from mpi4py import MPI
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

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Define paths
base_path = "/home/ankitk23/venv/graph_plot/Project_Data"

# List of main folders to process
main_folders = ["Control","Wo_FF_PYR_pcx_336",
    "WOFB_FB_pcx_336",
    "WOFB_PYR_pcx_336",
    "WOFF_FF_pcx_336",
    "WOFFandFB_pcx_336",
    "WOPYR_PYR_pcx_336"]

# Define folder label mappings
folder_mappings = {
    "A": {"A_2hz": 0, "A_20hz": 1},"B": {"B_2hz": 0, "B_20hz": 1},
    "Mix": {"Twenty_hz_Mix": 0, "Two_hz_Mix": 1}}

# Define processing parameters
start_time = 2000
end_time = 4000
bin_size = 20
num_bins = (end_time - start_time) // bin_size
num_neurons = 500
activation_value = 1  # Set activation presence value

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

def shuffle_labels(labels):
    np.random.shuffle(labels)
    return labels

# Distribute work across the processes
for main_folder in main_folders:
    for mapping_name, folders in folder_mappings.items():

        all_data, all_labels = [], []
        for folder, label in folders.items():
            folder_path = os.path.join(base_path, main_folder, folder)
            if not os.path.exists(folder_path):
                if rank == 0:
                    print(f"Path does not exist: {folder_path}")
                continue

            for file in os.listdir(folder_path):
                if file.endswith(".csv"):
                    file_path = os.path.join(folder_path, file)
                    processed_data = process_csv(file_path, activation_value)
                    all_data.append(processed_data)
                    all_labels.append(label)

        data = np.array(all_data)
        labels = np.array(all_labels)

        if data.size == 0:
            if rank == 0:
                print(f"No data found for {main_folder} - {mapping_name}")
            continue

        data = data / np.max(data)
        data = np.swapaxes(data, 1, 2)
        labels_categorical = to_categorical(labels, num_classes=2)

        # Generate shuffled labels
        shuffled_labels = shuffle_labels(labels.copy())
        shuffled_labels_categorical = to_categorical(shuffled_labels, num_classes=2)

        neuron_counts = [50, 100, 150, 200, 250, 300, 350, 400, 450, 500]
        num_trials = 144
        
        # Only split trials across processes, not neurons
        all_trials = list(range(num_trials))
        trials_per_process = len(all_trials) // size
        start_trial = rank * trials_per_process
        end_trial = start_trial + trials_per_process if rank < size - 1 else len(all_trials)
        my_trials = all_trials[start_trial:end_trial]
        
        if rank == 0:
            print(f"Process {rank}: Running trials {start_trial} to {end_trial-1}")
            
        # Initialize local results lists
        local_results_real = []
        local_results_shuffled = []

        # Each process handles all neuron counts but only its subset of trials
        for n_neurons in neuron_counts:
            for trial in my_trials:
                # Use consistent seed based on trial number for reproducibility
                np.random.seed(trial)
                selected_neurons = np.random.choice(data.shape[2], n_neurons, replace=False)

                # Train on real labels
                X_train, X_test, y_train, y_test = train_test_split(
                    data[:, :, selected_neurons], labels_categorical, test_size=0.4, random_state=trial
                )
                
                model = Sequential([
                    Input(shape=(100, n_neurons)),
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

                model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
                early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
                model.fit(X_train, y_train, validation_data=(X_test, y_test), 
                          epochs=75, batch_size=32, verbose=0, callbacks=[early_stopping])

                y_pred_real = np.argmax(model.predict(X_test), axis=1)
                y_test_actual = np.argmax(y_test, axis=1)
                acc_real = accuracy_score(y_test_actual, y_pred_real)
                
                # Store results for real labels
                local_results_real.append([main_folder, mapping_name, n_neurons, trial, acc_real])

                # Train on shuffled labels
                X_train, X_test, y_train, y_test = train_test_split(
                    data[:, :, selected_neurons], shuffled_labels_categorical, test_size=0.4, random_state=trial
                )
                
                model.fit(X_train, y_train, validation_data=(X_test, y_test), 
                          epochs=75, batch_size=32, verbose=0, callbacks=[early_stopping])

                y_pred_shuffled = np.argmax(model.predict(X_test), axis=1)
                y_test_actual_shuffled = np.argmax(y_test, axis=1)
                acc_shuffled = accuracy_score(y_test_actual_shuffled, y_pred_shuffled)
                
                # Store results for shuffled labels
                local_results_shuffled.append([main_folder, mapping_name, n_neurons, trial, acc_shuffled])
                
                if rank == 0 and (len(local_results_real) % 20 == 0):
                    print(f"Completed {len(local_results_real)} trials out of {len(neuron_counts) * len(my_trials)}")

        # Gather all results to rank 0
        all_results_real = comm.gather(local_results_real, root=0)
        all_results_shuffled = comm.gather(local_results_shuffled, root=0)

        # Rank 0 combines and saves the results
        if rank == 0:
            # Flatten the results from all ranks
            flattened_results_real = []
            flattened_results_shuffled = []
            
            for process_results in all_results_real:
                flattened_results_real.extend(process_results)
            
            for process_results in all_results_shuffled:
                flattened_results_shuffled.extend(process_results)
            
            print(f"Total real results: {len(flattened_results_real)}")
            print(f"Total shuffled results: {len(flattened_results_shuffled)}")

            # Save results as properly formatted CSV files
            df_real = pd.DataFrame(flattened_results_real, 
                                  columns=["Main Folder", "Mapping", "Neurons", "Trial", "Accuracy"])
            df_shuffled = pd.DataFrame(flattened_results_shuffled, 
                                      columns=["Main Folder", "Mapping", "Neurons", "Trial", "Accuracy"])
            
            output_dir = os.path.join(os.getcwd(), "results")
            os.makedirs(output_dir, exist_ok=True)
            
            real_csv_path = os.path.join(output_dir, f"real_results_{main_folder}_{mapping_name}.csv")
            shuffled_csv_path = os.path.join(output_dir, f"shuffled_results_{main_folder}_{mapping_name}.csv")
            
            df_real.to_csv(real_csv_path, index=False)
            df_shuffled.to_csv(shuffled_csv_path, index=False)
            
            print(f"Saved real results to {real_csv_path}")
            print(f"Saved shuffled results to {shuffled_csv_path}")

            # Plot accuracy graphs
            plt.figure(figsize=(10, 6))
            sns.set(style="whitegrid")
            
            # Calculate mean and std for each neuron count
            mean_acc_real = df_real.groupby("Neurons")["Accuracy"].mean()
            std_acc_real = df_real.groupby("Neurons")["Accuracy"].std()
            mean_acc_shuffled = df_shuffled.groupby("Neurons")["Accuracy"].mean()
            std_acc_shuffled = df_shuffled.groupby("Neurons")["Accuracy"].std()

            plt.plot(neuron_counts, mean_acc_real, marker='o', label="Real Labels", color='b')
            plt.fill_between(neuron_counts, mean_acc_real - std_acc_real, mean_acc_real + std_acc_real, alpha=0.2, color='b')

            plt.plot(neuron_counts, mean_acc_shuffled, marker='o', label="Shuffled Labels", color='r')
            plt.fill_between(neuron_counts, mean_acc_shuffled - std_acc_shuffled, mean_acc_shuffled + std_acc_shuffled, alpha=0.2, color='r')

            plt.xlabel("Number of Input Neurons")
            plt.ylabel("Accuracy")
            plt.legend()
            plt.title(f"Accuracy Comparison ({main_folder} - {mapping_name})")
            
            plot_path = os.path.join(output_dir, f"comparison_plot_{main_folder}_{mapping_name}.png")
            plt.savefig(plot_path)
            plt.close()
            
            print(f"Saved plot to {plot_path}")
