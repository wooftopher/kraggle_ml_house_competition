#-----------------------------------------------------MLPRegressor------------------------------------------
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from sklearn.metrics import mean_absolute_error

# Define the PyTorch MLP model
class MLPModel(nn.Module):
    def __init__(self, input_dim, hidden_dim1, hidden_dim2):
        super(MLPModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim1)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.fc3 = nn.Linear(hidden_dim2, 1)  # Output layer with 1 node (regression)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x

# Convert to PyTorch tensors and move to GPU
X_train_tensor = torch.tensor(X_train_transformed.values, dtype=torch.float32).cuda()
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).cuda()

X_valid_tensor = torch.tensor(X_valid_transformed.values, dtype=torch.float32).cuda()
y_valid_tensor = torch.tensor(y_valid.values, dtype=torch.float32).cuda()

# Create DataLoader for batching
train_dataset = data.TensorDataset(X_train_tensor, y_train_tensor)
train_loader = data.DataLoader(train_dataset, batch_size=64, shuffle=True)

valid_dataset = data.TensorDataset(X_valid_tensor, y_valid_tensor)
valid_loader = data.DataLoader(valid_dataset, batch_size=64, shuffle=False)

# Initialize the MLP model, loss function, and optimizer
model = MLPModel(input_dim=X_train_tensor.shape[1], hidden_dim1=100, hidden_dim2=50).cuda()
criterion = nn.MSELoss()  # Mean Squared Error loss (regression)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 350
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for inputs, targets in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs.squeeze(), targets)  # Remove extra dimension from targets
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    
    # Validation loop
    model.eval()
    with torch.no_grad():
        valid_predictions = []
        valid_targets = []
        for inputs, targets in valid_loader:
            outputs = model(inputs)
            valid_predictions.append(outputs.squeeze().cpu().numpy())
            valid_targets.append(targets.cpu().numpy())
        
        valid_predictions = np.concatenate(valid_predictions)
        valid_targets = np.concatenate(valid_targets)
        
        valid_mae = mean_absolute_error(valid_targets, valid_predictions)
    
    print(f"Epoch {epoch + 1}/{num_epochs}, Training Loss: {running_loss / len(train_loader):.4f}, Validation MAE: {valid_mae:.4f}")

# Make final predictions (after training is done)
model.eval()
with torch.no_grad():
    final_predictions_train = model(X_train_tensor).squeeze().cpu().numpy()
    final_predictions_test = model(X_valid_tensor).squeeze().cpu().numpy()

# Calculate MAE on final predictions
train_mae = mean_absolute_error(y_train, final_predictions_train)
test_mae = mean_absolute_error(y_valid, final_predictions_test)

print(f"Final Training MAE: {train_mae}")
print(f"Final Test MAE: {test_mae}")

# sys.exit()