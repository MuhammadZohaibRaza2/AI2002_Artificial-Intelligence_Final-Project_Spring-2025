import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
import joblib
import os
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader

# Define the neural network with regression and classification outputs
class RaceController(nn.Module):
    def __init__(self, input_dim, num_gear_classes=8):
        super(RaceController, self).__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.2)
        )
        self.regression = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 3),  # Acceleration, Braking, Steering
            nn.Tanh()
        )
        self.classification = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, num_gear_classes)  # Gear_cmd (8 classes: -1 to 6)
        )
    
    def forward(self, x):
        shared = self.shared(x)
        reg_out = self.regression(shared)
        cls_out = self.classification(shared)
        return reg_out, cls_out

# Load and preprocess data with chunked loading
def load_data(csv_path='telemetry_log.csv', chunksize=100000):
    input_cols = [
        'Angle', 'CurrentLapTime', 'Damage', 'DistanceFromStart', 'DistanceCovered',
        'FuelLevel', 'Gear', 'LastLapTime', 'RacePosition', 'RPM', 'SpeedX', 'SpeedY', 'SpeedZ',
        'TrackPosition', 'WheelSpinVelocity_1', 'WheelSpinVelocity_2', 'WheelSpinVelocity_3',
        'WheelSpinVelocity_4', 'Z'
    ] + [f'Track_{i}' for i in range(1, 20)] + [f'Opponent_{i}' for i in [1, 9, 18, 27, 36]]
    
    output_cols = ['Acceleration', 'Braking', 'Steering', 'Gear_cmd']
    
    print(f"Expected input columns: {len(input_cols)}")
    print(f"Input columns: {input_cols}")
    print(f"Output columns: {output_cols}")
    
    df_sample = next(pd.read_csv(csv_path, chunksize=1000))
    df_sample.columns = df_sample.columns.str.strip()  # FIX: remove leading/trailing spaces
    missing_cols = [col for col in input_cols + output_cols if col not in df_sample.columns]
    if missing_cols:
        raise KeyError(f"Missing columns in telemetry_log.csv: {missing_cols}")
    
    print(f"Columns in telemetry_log.csv: {df_sample.columns.tolist()}")
    
    X_list, y_reg_list, y_cls_list = [], [], []
    scaler = StandardScaler()
    for chunk in pd.read_csv(csv_path, chunksize=chunksize):
        chunk.columns = chunk.columns.str.strip()  # FIX: strip spaces on all chunks
        chunk = chunk.dropna(subset=input_cols + output_cols)
        X_chunk = chunk[input_cols].values
        y_reg_chunk = chunk[output_cols[:3]].values
        y_cls_chunk = chunk['Gear_cmd'].values
        y_cls_chunk = np.array([int(g + 1) for g in y_cls_chunk], dtype=np.int64)  # Map -1 to 6 → 0 to 7
        X_list.append(scaler.partial_fit(X_chunk).transform(X_chunk))
        y_reg_list.append(y_reg_chunk)
        y_cls_list.append(y_cls_chunk)
    if not X_list:
        raise ValueError("No data remains after dropping rows with missing values")
    X = np.vstack(X_list)
    y_reg = np.vstack(y_reg_list)
    y_cls = np.concatenate(y_cls_list)
    print(f"Input data shape: {X.shape}")
    print(f"Regression output shape: {y_reg.shape}")
    print(f"Classification output shape: {y_cls.shape}")
    
    joblib.dump(scaler, 'scaler.pkl')
    return X, y_reg, y_cls, input_cols, output_cols

# Create PyTorch datasets and loaders
def create_dataloaders(X, y_reg, y_cls, batch_size=32):
    X_train, X_val, y_reg_train, y_reg_val, y_cls_train, y_cls_val = train_test_split(
        X, y_reg, y_cls, test_size=0.2, random_state=42)
    
    X_train_tensor = torch.FloatTensor(X_train)
    y_reg_train_tensor = torch.FloatTensor(y_reg_train)
    y_cls_train_tensor = torch.LongTensor(y_cls_train)
    X_val_tensor = torch.FloatTensor(X_val)
    y_reg_val_tensor = torch.FloatTensor(y_reg_val)
    y_cls_val_tensor = torch.LongTensor(y_cls_val)
    
    train_dataset = TensorDataset(X_train_tensor, y_reg_train_tensor, y_cls_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_reg_val_tensor, y_cls_val_tensor)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=0)
    
    return train_loader, val_loader

# Evaluate model with MAE, R², and accuracy
def evaluate_model(model, val_loader, device):
    model.eval()
    y_reg_true, y_reg_pred, y_cls_true, y_cls_pred = [], [], [], []
    with torch.no_grad():
        for X_batch, y_reg_batch, y_cls_batch in val_loader:
            X_batch = X_batch.to(device)
            y_reg_batch = y_reg_batch.to(device)
            y_cls_batch = y_cls_batch.to(device)
            reg_out, cls_out = model(X_batch)
            y_reg_true.extend(y_reg_batch.cpu().numpy())
            y_reg_pred.extend(reg_out.cpu().numpy())
            y_cls_true.extend(y_cls_batch.cpu().numpy())
            y_cls_pred.extend(torch.argmax(cls_out, dim=1).cpu().numpy())
    y_reg_true = np.array(y_reg_true)
    y_reg_pred = np.array(y_reg_pred)
    y_cls_true = np.array(y_cls_true)
    y_cls_pred = np.array(y_cls_pred)
    mae = mean_absolute_error(y_reg_true, y_reg_pred, multioutput='raw_values')
    r2 = r2_score(y_reg_true, y_reg_pred, multioutput='raw_values')
    accuracy = np.mean(y_cls_true == y_cls_pred)
    return mae, r2, accuracy

# Training function with early stopping
def train_model():
    X, y_reg, y_cls, input_cols, output_cols = load_data()
    train_loader, val_loader = create_dataloaders(X, y_reg, y_cls)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = RaceController(input_dim=len(input_cols)).to(device)
    reg_criterion = nn.MSELoss()
    cls_criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.003)
    
    epochs = 200
    patience = 10
    best_val_loss = float('inf')
    epochs_no_improve = 0
    train_losses = []
    val_losses = []
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for X_batch, y_reg_batch, y_cls_batch in train_loader:
            X_batch = X_batch.to(device)
            y_reg_batch = y_reg_batch.to(device)
            y_cls_batch = y_cls_batch.to(device)
            
            optimizer.zero_grad()
            reg_out, cls_out = model(X_batch)
            reg_loss = reg_criterion(reg_out, y_reg_batch)
            cls_loss = cls_criterion(cls_out, y_cls_batch)
            loss = reg_loss + cls_loss
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * X_batch.size(0)
        
        train_loss /= len(train_loader.dataset)
        train_losses.append(train_loss)
        
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_batch, y_reg_batch, y_cls_batch in val_loader:
                X_batch = X_batch.to(device)
                y_reg_batch = y_reg_batch.to(device)
                y_cls_batch = y_cls_batch.to(device)
                reg_out, cls_out = model(X_batch)
                reg_loss = reg_criterion(reg_out, y_reg_batch)
                cls_loss = cls_criterion(cls_out, y_cls_batch)
                loss = reg_loss + cls_loss
                val_loss += loss.item() * X_batch.size(0)
        
        val_loss /= len(val_loader.dataset)
        val_losses.append(val_loss)
        
        mae, r2, gear_accuracy = evaluate_model(model, val_loader, device)
        print(f'Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}')
        print(f'Validation MAE: Acceleration={mae[0]:.4f}, Braking={mae[1]:.4f}, Steering={mae[2]:.4f}')
        print(f'Validation R²: Acceleration={r2[0]:.4f}, Braking={r2[1]:.4f}, Steering={r2[2]:.4f}')
        print(f'Gear_cmd Accuracy: {gear_accuracy:.4f}')
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'race_controller.pth')
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
        if epochs_no_improve >= patience:
            print("Early stopping triggered")
            break
    
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('training_loss.png')
    plt.close()

if __name__ == "__main__":
    train_model()
