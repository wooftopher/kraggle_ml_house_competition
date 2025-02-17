import lightgbm as lgb
import numpy as np

# Define parameters for GPU
params = {
    'device': 'gpu',
    'gpu_platform_id': 0,
    'gpu_device_id': 0,
}

# Convert the data to a NumPy array
data = np.array([[1, 2], [3, 4]])
label = np.array([0, 1])

# Create a LightGBM Dataset
dataset = lgb.Dataset(data, label=label)

# Train the model
model = lgb.train(params, dataset)

print("GPU is being used!")