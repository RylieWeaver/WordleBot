# General
from tqdm import tqdm

# Torch
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Dataset, DataLoader, random_split

# Wordle
from reward_model import RewardNet


########################################################
# PyTorch Dataset from Tensors
########################################################
class WordleDataset(Dataset):
    def __init__(self, X, y):
        """
        X: [N, 286] or [N, 26, 11]  (state representations)
        y: [N]                      (entropy labels)
        """
        self.X = X
        self.y = y

    def __len__(self):
        return self.X.size(0)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


########################################################
# Training Utils
########################################################
def train(model, train_loader, optimizer, criterion, device):
    """
    Train the model for one epoch.
    """
    model.train()
    total_loss = 0.0
    for batch_x, batch_y in tqdm(train_loader, desc="Training", leave=False):
        # Move data to device
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        # Forward pass
        optimizer.zero_grad()
        preds = model(batch_x).squeeze()
        loss = criterion(preds, batch_y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)


def test(model, test_loader, criterion, device):
    """
    Evaluate the model on the test set.
    """
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for batch_x, batch_y in tqdm(test_loader, desc="Testing", leave=False):
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            preds = model(batch_x).squeeze(-1)
            loss = criterion(preds, batch_y)
            total_loss += loss.item()
    return total_loss / len(test_loader)


########################################################
# Main Training Function
##########################################################
def train_model(model, epochs, batch_size, lr, device, lr_patience=10, early_stop_patience=50, model_path="best_reward_net.pth"):
    # Prepare data
    X, y = torch.load("wordle_data.pt")  
    X = X.view(X.size(0), -1)  
    dataset = WordleDataset(X, y)
    dataset_size = len(dataset)
    val_size = int(0.1 * dataset_size)
    test_size = int(0.1 * dataset_size)
    train_size = dataset_size - val_size - test_size
    train_ds, val_ds, test_ds = random_split(dataset, [train_size, val_size, test_size])
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds,   batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds,  batch_size=batch_size, shuffle=False)

    # Optimizer, criterion, scheduler
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.MSELoss()
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.8, patience=lr_patience, verbose=True)

    best_val_loss = float('inf')
    no_improve_count = 0

    print("Starting training...")
    for epoch in range(1, epochs+1):
        # ——— Train phase ———
        train_loss = train(model, train_loader, optimizer, criterion, device)

        # ——— Eval phase ———
        val_loss = test(model, val_loader,   criterion, device)
        test_loss = test(model, test_loader,  criterion, device)

        # ——— Scheduler step ———
        scheduler.step(val_loss)

        # ——— Early-stopping logic ———
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            no_improve_count = 0
            torch.save(model.state_dict(), model_path)
            print(f"Epoch {epoch}: new best val loss {best_val_loss:.4f} — model saved.")
        else:
            no_improve_count += 1

        # ——— Report ———
        print(f"Epoch [{epoch}/{epochs}] "
              f"Train: {train_loss:.4f} | Val: {val_loss:.4f} | Test: {test_loss:.4f}")

        # ——— Check early stopping ———
        if no_improve_count >= early_stop_patience:
            print(f"No improvement in validation loss for {early_stop_patience} epochs. Stopping early.")
            break

    # ——— Final evaluation on best model ———
    model.load_state_dict(torch.load(model_path, map_location=device))
    final_test_loss = test(model, test_loader, criterion, device)
    print(f"Final Test MSE: {final_test_loss:.4f}")


if __name__ == "__main__":
    # Hyperparameters
    epochs = 100
    batch_size = 256
    lr = 3e-4
    device='mps'
    # Train Model
    model = RewardNet(state_size=26*11+6, hidden_dim=128, layers=3, dropout=0.1).to(device)
    train_model(model, epochs, batch_size, lr, device, model_path="best_small_reward_net.pth")
