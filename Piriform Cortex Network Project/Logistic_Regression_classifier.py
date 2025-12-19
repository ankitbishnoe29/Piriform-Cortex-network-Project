import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import random
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier

# --- 1. CONFIGURATION ---
BASE_PATH = r"C:\Users\omcc\ml_env\Project_Data"
OUTPUT_DIR = os.path.join(BASE_PATH, "Results_CSVs")  # where CSVs will be saved
os.makedirs(OUTPUT_DIR, exist_ok=True)

MAIN_FOLDERS = [
    "CONTROL",
    "Wo_FF_PYR_pcx_336", 
    "Wo_FB_FB_pcx_336", 
    "Wo_FB_PYR_pcx_336",
    "Wo_FF_FF_pcx_336", 
    "Wo_FF_FB_pcx_336", 
    "Wo_PYR_PYR_pcx_336"
]

FOLDER_MAPPINGS = {
    "A": {"A_2hz": 0, "A_20hz": 1},
    "B": {"B_2hz": 0, "B_20hz": 1},
    "Mix": {"Twenty_hz_Mix": 0, "Two_hz_Mix": 1}
}

START_TIME = 2000
END_TIME = 4000
BIN_SIZE = 20
NUM_BINS = (END_TIME - START_TIME) // BIN_SIZE
NUM_NEURONS = 500
ACTIVATION_VALUE = 1.0

NEURON_COUNTS = list(range(25, NUM_NEURONS + 1, 25)) 
NUM_ITERATIONS = 144


# --- 2. DATA LOADING & PREPROCESSING ---
def process_csv(file_path: str, num_neurons: int, num_bins: int) -> np.ndarray:
    """Reads a single CSV and converts it into a binned binary matrix."""
    data = pd.read_csv(file_path, header=None, skiprows=1).to_numpy()
    if data.shape[0] != num_neurons:
        raise ValueError(f"CSV file {file_path} has {data.shape[0]} rows, expected {num_neurons}.")

    binary_matrix = np.zeros((num_neurons, num_bins), dtype=float)
    for neuron_idx in range(num_neurons):
        activation_times = data[neuron_idx][~np.isnan(data[neuron_idx])]
        bin_indices = ((activation_times - START_TIME) // BIN_SIZE).astype(int)
        valid_indices = (bin_indices >= 0) & (bin_indices < num_bins)
        binary_matrix[neuron_idx, bin_indices[valid_indices]] = ACTIVATION_VALUE
    return binary_matrix


def load_and_preprocess_data() -> dict:
    """Loads all data from disk, processes it, and organizes into a dictionary."""
    print("Starting data loading and preprocessing...")
    all_processed_data = {}

    for main_folder in MAIN_FOLDERS:
        for mapping_name, folders in FOLDER_MAPPINGS.items():
            print(f"  Processing {main_folder} -> {mapping_name}")
            data_key = (main_folder, mapping_name)
            
            all_data, all_labels = [], []
            for folder, label in folders.items():
                folder_path = os.path.join(BASE_PATH, main_folder, folder)
                if not os.path.isdir(folder_path):
                    continue

                for file_name in os.listdir(folder_path):
                    if file_name.endswith(".csv"):
                        file_path = os.path.join(folder_path, file_name)
                        try:
                            processed_matrix = process_csv(file_path, NUM_NEURONS, NUM_BINS)
                            all_data.append(processed_matrix)
                            all_labels.append(label)
                        except ValueError as e:
                            print(f"    Warning: Skipping file. {e}")
            
            if not all_data:
                print(f"    No data found for {main_folder} with mapping {mapping_name}. Skipping.")
                continue

            data_np = np.array(all_data)
            labels_np = np.array(all_labels)
            
            max_val = np.max(data_np)
            if max_val > 0:
                data_np = data_np / max_val
            
            data_np = np.swapaxes(data_np, 1, 2)
            all_processed_data[data_key] = {'data': data_np, 'labels': labels_np}
            
    print("Data preprocessing complete.\n")
    return all_processed_data


# --- 3. RUN PIPELINE ---
preprocessed_data = load_and_preprocess_data()


# --- 4. NEURON SCALING EXPERIMENT ---
results_summary = []

for dataset_key, dataset in preprocessed_data.items():
    X_full = dataset['data']    # shape: (samples, time, neurons)
    y_full = dataset['labels']

    X_train, X_test, y_train, y_test = train_test_split(
        X_full, y_full, test_size=0.4, random_state=None, stratify=y_full
    )

    acc_real_means, acc_shuff_means = [], []
    acc_real_stds, acc_shuff_stds = [], []

    print(f"\nRunning neuron scaling experiment for {dataset_key}...\n")

    for num_neurons_sel in tqdm(NEURON_COUNTS, desc=f"{dataset_key}"):
        acc_real_list, acc_shuff_list = [], []

        for _ in range(NUM_ITERATIONS):
            selected_neurons = random.sample(range(NUM_NEURONS), num_neurons_sel)
            X_train_sub = X_train[:, :, selected_neurons]
            X_test_sub = X_test[:, :, selected_neurons]

            X_train_flat = X_train_sub.reshape(X_train_sub.shape[0], -1)
            X_test_flat = X_test_sub.reshape(X_test_sub.shape[0], -1)

            # --- REAL TRAINING ---
            pipeline_real = Pipeline([
                ('scaler', StandardScaler()),
                ('model', SGDClassifier(loss='log_loss', random_state=None, max_iter=1000, tol=1e-3))
            ])
            pipeline_real.fit(X_train_flat, y_train)
            y_pred_real = pipeline_real.predict(X_test_flat)
            acc_real_list.append(accuracy_score(y_test, y_pred_real))

            # --- SHUFFLED BASELINE ---
            y_train_shuffled = shuffle(y_train, random_state=None)
            pipeline_shuff = Pipeline([
                ('scaler', StandardScaler()),
                ('model', SGDClassifier(loss='log_loss', random_state=None, max_iter=1000, tol=1e-3))
            ])
            pipeline_shuff.fit(X_train_flat, y_train_shuffled)
            y_pred_shuff = pipeline_shuff.predict(X_test_flat)
            acc_shuff_list.append(accuracy_score(y_test, y_pred_shuff))

        acc_real_means.append(np.mean(acc_real_list))
        acc_shuff_means.append(np.mean(acc_shuff_list))
        acc_real_stds.append(np.std(acc_real_list))
        acc_shuff_stds.append(np.std(acc_shuff_list))

        print(f"Neurons: {num_neurons_sel}, Real Mean Acc: {np.mean(acc_real_list):.3f}, Shuffled: {np.mean(acc_shuff_list):.3f}")

    # Store results
    dataset_name = f"{dataset_key[0]}_{dataset_key[1]}"
    df_results = pd.DataFrame({
        "Neuron_Count": NEURON_COUNTS,
        "Mean_Accuracy_Real": acc_real_means,
        "Std_Accuracy_Real": acc_real_stds,
        "Mean_Accuracy_Shuffled": acc_shuff_means,
        "Std_Accuracy_Shuffled": acc_shuff_stds
    })

    csv_path = os.path.join(OUTPUT_DIR, f"results_{dataset_name}.csv")
    df_results.to_csv(csv_path, index=False)
    print(f"✅ Saved results to {csv_path}\n")

    results_summary.append({
        "dataset": dataset_key,
        "neuron_counts": NEURON_COUNTS,
        "mean_acc_real": acc_real_means,
        "std_acc_real": acc_real_stds, 
        "mean_acc_shuff": acc_shuff_means,
        "std_acc_shuff": acc_shuff_stds
    })


# --- 5. PLOT RESULTS ---
plt.style.use('seaborn-v0_8-whitegrid')
fig, ax = plt.subplots(figsize=(12, 8)) 

for res in results_summary:
    neuron_counts = np.array(res["neuron_counts"])
    mean_real = np.array(res["mean_acc_real"])
    std_real = np.array(res["std_acc_real"])
    mean_shuff = np.array(res["mean_acc_shuff"])
    std_shuff = np.array(res["std_acc_shuff"])
    dataset_name = f"{res['dataset'][0]}_{res['dataset'][1]}"
    
    line, = ax.plot(neuron_counts, mean_real, marker='o', linestyle='-', label=f"{dataset_name} (Real)")
    ax.fill_between(neuron_counts, mean_real - std_real, mean_real + std_real, color=line.get_color(), alpha=0.2)

    line, = ax.plot(neuron_counts, mean_shuff, marker='x', linestyle='--', label=f"{dataset_name} (Shuffled)", color=line.get_color())
    ax.fill_between(neuron_counts, mean_shuff - std_shuff, mean_shuff + std_shuff, color=line.get_color(), alpha=0.15)

ax.set_xlabel("Number of Neurons Used", fontsize=12)
ax.set_ylabel(f"Mean Accuracy (over {NUM_ITERATIONS} iterations)", fontsize=12)
ax.set_title("Model Accuracy vs. Number of Sampled Neurons", fontsize=14, weight='bold')
ax.legend(title="Dataset & Condition")
ax.set_ylim(0.4, 1.05)
plt.tight_layout()
plt.show()


# --- 6. PRINT SUMMARY ---
print("\n========== FINAL SUMMARY ==========")
for res in results_summary:
    print(f"\n--- Dataset: {res['dataset']} ---")
    print(f"{'Neurons':<10} | {'Real Accuracy':<25} | {'Shuffled Accuracy':<25}")
    print("-" * 65)
    for n, acc_r, std_r, acc_s, std_s in zip(
        res['neuron_counts'], 
        res['mean_acc_real'], res['std_acc_real'],
        res['mean_acc_shuff'], res['std_acc_shuff']
    ):
        real_str = f"{acc_r:.3f} ± {std_r:.3f}"
        shuff_str = f"{acc_s:.3f} ± {std_s:.3f}"
        print(f"{n:<10} | {real_str:<25} | {shuff_str:<25}")
