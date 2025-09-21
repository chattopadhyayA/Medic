# This is medica_v1 for data preparation, dataset creation and training the model



import awkward as ak
import json
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import awkward as ak
import os
import pandas as pd
import matplotlib.pyplot as plt


# -------------------------
# Functions for reading data and inspecting branches
# -------------------------

def read_json_to_awkward(file_path):
    with open(file_path, "r") as f:
        json_str = f.read() 
    data = ak.from_json(json_str)
    return data


def branch_info(array):
    branches = array.fields
    print("The Branches in this array are:", branches)

    for br in branches:
        try:
            field = array[br]
            counts = ak.num(field)
            print(f"Number of entries in {br}:")
            print(f"  avg = {int(round(ak.mean(counts)))}")
            print(f"  min = {ak.min(counts)}")
            print(f"  max = {ak.max(counts)}\n")
        except Exception as e:
            print(f"branch {br} is not a list, but a scalar field")

# The following class is for converting awkward array to torch Dataset
class ColliderDataset(Dataset):
    def __init__(self, ak_array):
        """
        ak_array has branches: track [10,30,7], tower [10,30,8], missinget [10,1,3], output_prob [4]
        """
        self.track = ak_array["track"]
        self.tower = ak_array["tower"]
        self.missinget = ak_array["missinget"]
        self.output = ak_array["output_prob"]

    def __len__(self):
        return len(self.track)

    def __getitem__(self, idx):
        track = np.array(self.track[idx])       # [10,30,7]
        tower = np.array(self.tower[idx])       # [10,30,8]
        missinget = np.array(self.missinget[idx])  # [10,1,3]
        output = np.array(self.output[idx])     # [4]

        # Convert to torch
        track = torch.tensor(track, dtype=torch.float32)
        tower = torch.tensor(tower, dtype=torch.float32)
        missinget = torch.tensor(missinget, dtype=torch.float32)
        output = torch.tensor(output, dtype=torch.float32)

        return track, tower, missinget, output


# -------------------------
# Functions for data refinement and dataset creation
# -------------------------

# Function for data refinement
def refinement(array, track_min, tower_min):
    """
    Keep only events with at least track_min tracks and tower_min towers.
    
    Parameters:
        array: awkward array of events
        track_min: minimum number of tracks per event
        tower_min: minimum number of towers per event
        
    Returns:
        refined_array: filtered awkward array
    """
    n_tracks = ak.num(array["track"])
    n_towers = ak.num(array["tower"])
    
    mask = (n_tracks >= track_min) & (n_towers >= tower_min)
    refined_array = array[mask]
    
    print(f"Number of refined events: {len(refined_array)}")
    return refined_array




# The full dataset creation function
def Dataset_Creator(refined_data, window, event_tower, event_track, seed=None):
    """
    window: number of events per window (window size).
    event_tower: number of towers to pick/pad per event.
    event_track: number of tracks to pick/pad per event.

    Returns:
      X_tracks: list of length n_windows of arrays [window, event_track, n_track_features]
      X_towers: list of length n_windows of arrays [window, event_tower, n_tower_features]
      X_missinget: list of length n_windows of arrays [window, n_missinget_features]
      y: np.ndarray shape (n_windows, 4)
      ak_dataset: ak.Array length n_windows (one record per window)
    """
    if seed is not None:
        np.random.seed(seed)

    window_size = int(window)
    n_events = len(refined_data)
    if n_events < window_size:
        raise ValueError(f"Not enough events ({n_events}) for a window size of {window_size}")

    # how many full windows we can make (drop remainder)
    n_windows = n_events // window_size
    n_used = n_windows * window_size
    if n_used < n_events:
        dropped = n_events - n_used
    else:
        dropped = 0

    # shuffle and select first n_used indices
    perm = np.random.permutation(n_events)[:n_used]
    perm = perm.reshape(n_windows, window_size)  # shape (n_windows, window_size)

    # --- determine feature sizes from first non-empty entries ---
    # track features
    n_track_features = None
    for evt in refined_data:
        if len(evt["track"]) > 0:
            n_track_features = len(evt["track"][0].fields)
            break
    if n_track_features is None:
        raise ValueError("No event contains any track; cannot infer track feature dimension")

    # tower features
    n_tower_features = None
    for evt in refined_data:
        if len(evt["tower"]) > 0:
            n_tower_features = len(evt["tower"][0].fields)
            break
    if n_tower_features is None:
        raise ValueError("No event contains any tower; cannot infer tower feature dimension")

    # missinget features (take first non-empty missinget record)
    n_missinget_features = None
    for evt in refined_data:
        if len(evt["missinget"]) > 0:
            n_missinget_features = len(evt["missinget"][0].fields)
            break
    if n_missinget_features is None:
        raise ValueError("No event contains missinget; cannot infer missinget feature dimension")

    # containers
    X_tracks = []
    X_towers = []
    X_missinget = []
    y_list = []
    ak_dataset_list = []

    for w_idx in range(n_windows):
        idxs = perm[w_idx]            # indices for this window (length = window_size)
        set_tracks = []
        set_towers = []
        set_missinget = []
        set_blindfold = []

        for ev_i in idxs:
            event = refined_data[int(ev_i)]

            # ---------- tracks ----------
            n_tr = len(event["track"])
            if n_tr > 0: # Although refinement should ensure this, still added for safety
                pick_tr = np.random.choice(n_tr, min(event_track, n_tr), replace=False)
                track_records = event["track"][pick_tr]
                # column_stack fields -> shape (n_selected, n_track_features)
                track_vals = np.column_stack([ak.to_numpy(track_records[f]) for f in track_records.fields])
                track_array = np.zeros((event_track, n_track_features), dtype=np.float32)
                track_array[: track_vals.shape[0], :] = track_vals.astype(np.float32)
            else:
                track_array = np.zeros((event_track, n_track_features), dtype=np.float32)
            set_tracks.append(track_array)

            # ---------- towers ----------
            n_to = len(event["tower"])
            if n_to > 0:
                pick_to = np.random.choice(n_to, min(event_tower, n_to), replace=False)
                tower_records = event["tower"][pick_to]
                tower_vals = np.column_stack([ak.to_numpy(tower_records[f]) for f in tower_records.fields])
                tower_array = np.zeros((event_tower, n_tower_features), dtype=np.float32)
                tower_array[: tower_vals.shape[0], :] = tower_vals.astype(np.float32)
            else:
                tower_array = np.zeros((event_tower, n_tower_features), dtype=np.float32)
            set_towers.append(tower_array)

            # ---------- missinget ----------
            missing_vals = np.zeros((1, n_missinget_features), dtype=np.float32)

            if len(event["missinget"]) > 0:
                missing_records = event["missinget"]
                vals = np.column_stack([ak.to_numpy(missing_records[f]) for f in missing_records.fields])
                missing_vals[: vals.shape[0], :] = vals.astype(np.float32)

            set_missinget.append(missing_vals)

            # ---------- blindfold (labels) ----------
            set_blindfold.append(event["blindfold"])

        # stack event-level arrays into window array shapes
        X_tracks.append(np.stack(set_tracks))      # (window_size, event_track, n_track_features)
        X_towers.append(np.stack(set_towers))      # (window_size, event_tower, n_tower_features)
        X_missinget.append(np.stack(set_missinget))# (window_size, n_missinget_features)

        # compute output probabilities over the window
        blind_arr = ak.Array(set_blindfold)
        p_1 = float(ak.sum(blind_arr["unblind"]) / window_size)
        p_2 = float(ak.sum(blind_arr["blind_barrel"]) / window_size)
        p_3 = float(ak.sum(blind_arr["blind_endcap"]) / window_size)
        p_4 = float(ak.sum(blind_arr["blind_forward"]) / window_size)
        y_list.append([p_1, p_2, p_3, p_4])

        # build one record (one element) per window (store per-event arrays inside)
        ak_window_record = {
            "track": set_tracks,           # list length window_size of arrays (event_track, n_track_features)
            "tower": set_towers,           # list length window_size of arrays (event_tower, n_tower_features)
            "missinget": set_missinget,    # list length window_size of arrays (n_missinget_features,)
            "output_prob": np.array([p_1, p_2, p_3, p_4], dtype=np.float32)
        }
        ak_dataset_list.append(ak_window_record)

    y = np.asarray(y_list, dtype=np.float32)  # shape (n_windows, 4)
    ak_dataset = ak.Array(ak_dataset_list)    # length = n_windows

    # info print
    print(f"n_events = {n_events}, window_size = {window_size}, n_windows = {n_windows}, dropped = {dropped}")
    print(f"X_tracks windows: {len(X_tracks)}; one window shape: {X_tracks[0].shape}")
    print(f"X_towers windows: {len(X_towers)}; one window shape: {X_towers[0].shape}")
    print(f"X_missinget windows: {len(X_missinget)}; one window shape: {X_missinget[0].shape}")
    print(f"y shape: {y.shape}")

    return X_tracks, X_towers, X_missinget, y, ak_dataset




# The sliding window dataset creation function
def Sliding_Window_Dataset_Creator(refined_data, window, event_tower, event_track, seed=None, save_json_path=None):
    """
    Creates a sliding window dataset from events.

    Parameters
    ----------
    refined_data : list
        List of events, each with fields "track", "tower", "missinget", "blindfold".
    window : int
        Number of events per sliding window.
    event_tower : int
        Number of towers to pick/pad per event.
    event_track : int
        Number of tracks to pick/pad per event.
    seed : int, optional
        Random seed for reproducibility.
    save_json_path : str, optional
        If provided, saves the final awkward array as JSON to this path.

    Returns
    -------
    X_tracks : list of np.ndarray
        List of arrays (window, event_track, n_track_features)
    X_towers : list of np.ndarray
        List of arrays (window, event_tower, n_tower_features)
    X_missinget : list of np.ndarray
        List of arrays (window, n_missinget_features)
    y : np.ndarray
        Array of output probabilities (n_windows, 4)
    ak_dataset : ak.Array
        Full awkward array, one record per sliding window
    """
    if seed is not None:
        np.random.seed(seed)

    # --- shuffle events even though refined_data is already shuffled for safety---
    rand_indc = np.random.permutation(len(refined_data))
    refined_data = refined_data[rand_indc]

    n_events = len(refined_data)
    window_size = int(window)
    n_windows = n_events - window_size + 1  # sliding windows

    if n_windows <= 0:
        raise ValueError(f"Not enough events ({n_events}) for window size {window_size}")

    # --- determine feature sizes from first non-empty entries ---
    n_track_features = next(
        (len(evt["track"][0].fields) for evt in refined_data if len(evt["track"]) > 0), None)
    if n_track_features is None:
        raise ValueError("No event contains any track; cannot infer track feature dimension")

    n_tower_features = next(
        (len(evt["tower"][0].fields) for evt in refined_data if len(evt["tower"]) > 0), None)
    if n_tower_features is None:
        raise ValueError("No event contains any tower; cannot infer tower feature dimension")

    n_missinget_features = next(
        (len(evt["missinget"][0].fields) for evt in refined_data if len(evt["missinget"]) > 0), None)
    if n_missinget_features is None:
        raise ValueError("No event contains missinget; cannot infer missinget feature dimension")

    # containers
    X_tracks, X_towers, X_missinget, y_list, ak_dataset_list = [], [], [], [], []

    for w_start in range(n_windows):
        set_tracks, set_towers, set_missinget, set_blindfold = [], [], [], []

        # select window events
        for ev_i in range(w_start, w_start + window_size):
            event = refined_data[ev_i]

            # --- tracks ---
            n_tr = len(event["track"])
            if n_tr > 0:
                pick_tr = np.random.choice(n_tr, min(event_track, n_tr), replace=False)
                track_records = event["track"][pick_tr]
                track_vals = np.column_stack([ak.to_numpy(track_records[f]) for f in track_records.fields])
                track_array = np.zeros((event_track, n_track_features), dtype=np.float32)
                track_array[: track_vals.shape[0], :] = track_vals.astype(np.float32)
            else:
                track_array = np.zeros((event_track, n_track_features), dtype=np.float32)
            set_tracks.append(track_array)

            # --- towers ---
            n_to = len(event["tower"])
            if n_to > 0:
                pick_to = np.random.choice(n_to, min(event_tower, n_to), replace=False)
                tower_records = event["tower"][pick_to]
                tower_vals = np.column_stack([ak.to_numpy(tower_records[f]) for f in tower_records.fields])
                tower_array = np.zeros((event_tower, n_tower_features), dtype=np.float32)
                tower_array[: tower_vals.shape[0], :] = tower_vals.astype(np.float32)
            else:
                tower_array = np.zeros((event_tower, n_tower_features), dtype=np.float32)
            set_towers.append(tower_array)

            # --- missinget ---
            missing_vals = np.zeros((1, n_missinget_features), dtype=np.float32)
            if len(event["missinget"]) > 0:
                missing_records = event["missinget"]
                vals = np.column_stack([ak.to_numpy(missing_records[f]) for f in missing_records.fields])
                missing_vals[: vals.shape[0], :] = vals.astype(np.float32)
            set_missinget.append(missing_vals)

            # --- blindfold ---
            set_blindfold.append(event["blindfold"])

        # stack arrays
        X_tracks.append(np.stack(set_tracks))
        X_towers.append(np.stack(set_towers))
        X_missinget.append(np.stack(set_missinget))

        # compute output probabilities
        blind_arr = ak.Array(set_blindfold)
        p_1 = float(ak.sum(blind_arr["unblind"]) / window_size)
        p_2 = float(ak.sum(blind_arr["blind_barrel"]) / window_size)
        p_3 = float(ak.sum(blind_arr["blind_endcap"]) / window_size)
        p_4 = float(ak.sum(blind_arr["blind_forward"]) / window_size)
        y_list.append([p_1, p_2, p_3, p_4])

        # build awkward record
        ak_window_record = {
            "track": set_tracks,
            "tower": set_towers,
            "missinget": set_missinget,
            "output_prob": np.array([p_1, p_2, p_3, p_4], dtype=np.float32)
        }
        ak_dataset_list.append(ak_window_record)

    y = np.asarray(y_list, dtype=np.float32)
    ak_dataset = ak.Array(ak_dataset_list)

    # optionally save JSON
    if save_json_path is not None:
        # awkward -> list -> JSON
        with open(save_json_path, "w") as f:
            json.dump(ak.to_list(ak_dataset), f)

    # info print
    print(f"n_events = {n_events}, window_size = {window_size}, n_windows = {n_windows}")
    print(f"X_tracks windows: {len(X_tracks)}; one window shape: {X_tracks[0].shape}")
    print(f"X_towers windows: {len(X_towers)}; one window shape: {X_towers[0].shape}")
    print(f"X_missinget windows: {len(X_missinget)}; one window shape: {X_missinget[0].shape}")
    print(f"y shape: {y.shape}")

    return X_tracks, X_towers, X_missinget, y, ak_dataset



# ----------------------------
# Neural Network Model
# ----------------------------


class TransformerEncoder(nn.Module):
    def __init__(self, embed_dim, num_heads=4, ff_dim=128, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, embed_dim)
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        attn_out, _ = self.attn(x, x, x)
        x = self.norm1(x + self.dropout(attn_out))
        ff_out = self.ff(x)
        x = self.norm2(x + self.dropout(ff_out))
        return x
    


class AttentionPooling(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.query = nn.Linear(embed_dim, 1)

    def forward(self, x):
        # x: [B, P, embed]
        weights = torch.softmax(self.query(x), dim=1)  # [B, P, 1]
        pooled = (x * weights).sum(dim=1)              # [B, embed]
        return pooled


class MEDIC(nn.Module):
    def __init__(self, d_track=7, d_tower=8, d_met=3, embed_dim=128):
        super().__init__()

        # Track encoder
        self.track_proj = nn.Linear(d_track, embed_dim)
        self.track_transformer = TransformerEncoder(embed_dim)
        self.track_pool = AttentionPooling(embed_dim)

        # Tower encoder
        self.tower_proj = nn.Linear(d_tower, embed_dim)
        self.tower_transformer = TransformerEncoder(embed_dim)
        self.tower_pool = AttentionPooling(embed_dim)

        # MET encoder
        self.met_proj = nn.Sequential(
            nn.Linear(d_met, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)
        )

        # 2D CNN layers (treat [3,T,embed] as image-like input)
        self.cnn2d = nn.Sequential(
            nn.Conv2d(embed_dim, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # Fully connected classifier
        self.fc = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 4)
        )

    def forward(self, track, tower, met):
        B, T, P, _ = track.shape  # track: [B,10,30,d]

        # Track
        track = track.view(B*T, P, -1)      # [B*T, 30, d_track]
        track = self.track_proj(track)      # [B*T, 30, embed]
        track = self.track_transformer(track)
        track = self.track_pool(track)      # [B*T, embed]
        track = track.view(B, T, -1)        # [B, T, embed]

        # Tower
        tower = tower.view(B*T, P, -1)
        tower = self.tower_proj(tower)
        tower = self.tower_transformer(tower)
        tower = self.tower_pool(tower)      # [B*T, embed]
        tower = tower.view(B, T, -1)        # [B, T, embed]

        # MET
        met = met.view(B*T, -1)
        met = self.met_proj(met)            # [B*T, embed]
        met = met.view(B, T, -1)            # [B, T, embed]

        # Stack [B, 3, T, embed]
        x = torch.stack([track, tower, met], dim=1)

        # Reshape for CNN2d: [B, embed, 3, T]
        x = x.permute(0, 3, 1, 2)

        # Apply CNN2d
        x = self.cnn2d(x)                   # [B, 256, 3, T]
        x = self.avgpool(x).view(B, -1)     # [B, 256]

        # Fully connected classifier
        logits = self.fc(x)                 # [B, 4]
        probs = F.softmax(logits, dim=-1)   # probabilities
        log_probs = F.log_softmax(logits, dim=-1)  # log-probabilities

        return probs, log_probs


        
def train_model(model, train_loader, val_loader, optimizer, criterion, device, epochs=100, patience=10):
    os.makedirs("Analytics", exist_ok=True)

    log = {"epoch": [], "train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(epochs):
        # Training loop
        model.train()
        train_loss, correct, total = 0, 0, 0
        for track, tower, met, y in train_loader:
            track, tower, met, y = track.to(device), tower.to(device), met.to(device), y.to(device)
            optimizer.zero_grad()
            probs, log_probs = model(track, tower, met)
            loss = criterion(log_probs, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            pred_cls = probs.argmax(dim=1)
            true_cls = y.argmax(dim=1)
            correct += (pred_cls == true_cls).sum().item()
            total += y.size(0)
        train_loss /= len(train_loader)
        train_acc = correct / total

        # Validation loop
        model.eval()
        val_loss, correct, total = 0, 0, 0
        with torch.no_grad():
            for track, tower, met, y in val_loader:
                track, tower, met, y = track.to(device), tower.to(device), met.to(device), y.to(device)
                probs, log_probs = model(track, tower, met)
                loss = criterion(log_probs, y)
                val_loss += loss.item()
                pred_cls = probs.argmax(dim=1)
                true_cls = y.argmax(dim=1)
                correct += (pred_cls == true_cls).sum().item()
                total += y.size(0)
        val_loss /= len(val_loader)
        val_acc = correct / total

        log["epoch"].append(epoch+1)
        log["train_loss"].append(train_loss)
        log["val_loss"].append(val_loss)
        log["train_acc"].append(train_acc)
        log["val_acc"].append(val_acc)

        print(f"Epoch {epoch+1} | Train Loss {train_loss:.4f} Acc {train_acc:.4f} | "
              f"Val Loss {val_loss:.4f} Acc {val_acc:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), "Analytics/best_model.pt")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break

    # Save logs & plots
    df = pd.DataFrame(log)
    df.to_csv("Analytics/doctor_log.csv", index=False)

    plt.figure(figsize=(10,5))
    plt.subplot(1,2,1)
    plt.plot(df["epoch"], df["train_loss"], label="Train Loss")
    plt.plot(df["epoch"], df["val_loss"], label="Val Loss")
    plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.legend()
    plt.subplot(1,2,2)
    plt.plot(df["epoch"], df["train_acc"], label="Train Acc")
    plt.plot(df["epoch"], df["val_acc"], label="Val Acc")
    plt.xlabel("Epoch"); plt.ylabel("Accuracy"); plt.legend()
    plt.tight_layout()
    plt.savefig("Analytics/doctor_training.png")
    plt.close()

    return model


def test_model(model, test_loader, criterion, device):
    model.load_state_dict(torch.load("Analytics/best_model.pt"))
    model.eval()
    test_loss, correct, total = 0, 0, 0
    with torch.no_grad():
        for track, tower, met, y in test_loader:
            track, tower, met, y = track.to(device), tower.to(device), met.to(device), y.to(device)
            probs, log_probs = model(track, tower, met)
            loss = criterion(log_probs, y)
            test_loss += loss.item()
            pred_cls = probs.argmax(dim=1)
            true_cls = y.argmax(dim=1)
            correct += (pred_cls == true_cls).sum().item()
            total += y.size(0)
    test_loss /= len(test_loader)
    test_acc = correct / total
    print(f"Test Loss: {test_loss:.4f} | Test Accuracy: {test_acc:.4f}")
    return test_loss, test_acc

def predict_model(model, data_loader, device):
    """Return predictions (probabilities and predicted classes)."""
    model.load_state_dict(torch.load("Analytics/best_model.pt"))
    model.eval()
    all_probs, all_preds = [], []
    with torch.no_grad():
        for track, tower, met, _ in data_loader:
            track, tower, met = track.to(device), tower.to(device), met.to(device)
            probs, _ = model(track, tower, met)
            preds = probs.argmax(dim=1)
            all_probs.append(probs.cpu())
            all_preds.append(preds.cpu())
    all_probs = torch.cat(all_probs, dim=0)
    all_preds = torch.cat(all_preds, dim=0)
    return all_probs, all_preds
