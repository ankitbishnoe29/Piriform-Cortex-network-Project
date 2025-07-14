import os
import numpy as np
import pandas as pd
from tensorflow.keras import Input
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.losses import CategoricalCrossentropy

# ------------------- Constants -------------------
start_time_original = 2000
end_time_original = 4000
start_time_baseline = 0
end_time_baseline = 2000
bin_size = 20
num_bins = (end_time_baseline - start_time_baseline) // bin_size  # 100 bins
num_neurons = 500
activation_value = 1
iterations = 300
base_path = r"C:\Users\omcch\OneDrive\Desktop\Project_Data\CONTROL"

folder_mappings = {
    "A": {"A_2hz": 0, "A_20hz": 1},
    "B": {"B_2hz": 0, "B_20hz": 1},
    "Mix": {"Twenty_hz_Mix": 0, "Two_hz_Mix": 1}
}

# ------------------- Functions -------------------
def process_csv(file_path, start_time, end_time, activation_value):
    """Convert CSV spike times into binary matrix representation."""
    data = pd.read_csv(file_path).to_numpy()
    if data.shape[0] != num_neurons:
        raise ValueError(f"{file_path} does not have {num_neurons} rows.")
    num_bins_local = (end_time - start_time) // bin_size
    binary_matrix = np.zeros((num_neurons, num_bins_local), dtype=float)
    for neuron_idx in range(num_neurons):
        activation_times = data[neuron_idx][~np.isnan(data[neuron_idx])]
        bin_indices = ((activation_times - start_time) // bin_size).astype(int)
        valid_indices = (bin_indices >= 0) & (bin_indices < num_bins_local)
        binary_matrix[neuron_idx, bin_indices[valid_indices]] = activation_value
    return binary_matrix.T

def create_model(input_shape):
    model = Sequential([
        Input(shape=input_shape),  # Explicit Input layer
        Conv1D(filters=128, kernel_size=3, activation='relu', padding='same'),
        Dropout(0.3),
        Conv1D(filters=128, kernel_size=3, activation='relu', padding='same'),
        Dropout(0.3),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.3),
        Dense(128, activation='relu'),
        Dropout(0.3),
        Dense(2, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def compute_log_likelihood(model, X, y):
    y_pred = model.predict(X, verbose=0)
    return -CategoricalCrossentropy()(y, y_pred).numpy()

def load_dataset(folder_mapping):
    data_original, data_baseline, labels = [], [], []
    for folder, label in folder_mapping.items():
        folder_path = os.path.join(base_path, folder)
        files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith(".csv")]
        for file in files:
            data_original.append(process_csv(file, start_time_original, end_time_original, activation_value))
            data_baseline.append(process_csv(file, start_time_baseline, end_time_baseline, activation_value))
            labels.append(label)
    data_original = np.array(data_original)
    data_baseline = np.array(data_baseline)
    labels = to_categorical(np.array(labels), num_classes=2)
    data_original /= np.max(data_original)
    data_baseline /= np.max(data_baseline)
    return data_original, data_baseline, labels

# ------------------- Main Experiment -------------------
for exp_name, folder_mapping in folder_mappings.items():
    print(f"\n==== Running Experiment: {exp_name} ====")
    data_original, data_baseline, labels = load_dataset(folder_mapping)
    importance_matrix = np.zeros((iterations, num_neurons))

    for i in range(iterations):
        print(f"→ Iteration {i+1}/{iterations} for experiment {exp_name}")

        # Split data
        X_train_orig, X_test_orig, y_train, y_test = train_test_split(
            data_original, labels, test_size=0.3, random_state=i
        )
        X_train_base, X_test_base, _, _ = train_test_split(
            data_baseline, labels, test_size=0.3, random_state=i
        )

        # Train model
        model = create_model((num_bins, num_neurons))
        model.fit(X_train_orig, y_train, validation_data=(X_test_orig, y_test),
                  epochs=15, batch_size=32, verbose=0)
        print("   ✓ Model trained.")

        # Baseline log-likelihood
        baseline_ll = compute_log_likelihood(model, X_test_orig, y_test)

        # Neuron importance loop
        for neuron_idx in range(num_neurons):
            X_mod = X_test_orig.copy()
            X_mod[:, :, neuron_idx] = X_test_base[:, :, neuron_idx]
            mod_ll = compute_log_likelihood(model, X_mod, y_test)
            delta_ll = baseline_ll - mod_ll
            importance_matrix[i, neuron_idx] = delta_ll

        print(f"   ✓ Completed importance evaluation for iteration {i+1}")

    # Average and save
    mean_importance = np.mean(importance_matrix, axis=0)
    df = pd.DataFrame({
        "Neuron": np.arange(num_neurons),
        "MeanDeltaLogLikelihood": mean_importance
    }).sort_values(by="MeanDeltaLogLikelihood", ascending=False)

    df.to_csv(f"neuron_importance_ranked_{exp_name}_300_iters.csv", index=False)
    print(f"✅ Saved: neuron_importance_ranked_{exp_name}_300_iters.csv")

print("\n✅ All experiments completed!")
