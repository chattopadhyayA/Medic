# This is medica_v3 for data preparation, dataset creation and training the model


import uproot
import awkward as ak
import json
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset, random_split
import awkward as ak
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold 
from sklearn.metrics import (
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    accuracy_score,
    auc
)
from sklearn.preprocessing import label_binarize



# -------------------------
# Functions for reading data from the Delphes generated ROOT files
# -------------------------


# Function for creating individual data subtrees 
def extract_branches_from_csv(root_file, csv_file, label_name=None):
    """
    Extract selected subtrees from a Delphes ROOT file into an awkward array,
    based on a CSV file that lists subtree and branch names. 

    Parameters
    ----------
    root_file : str
        Path to the ROOT file.
    csv_file : str
        Path to CSV file with two columns: 'subtree' and 'branch'.
    label_name : str, optional
        Name of the constant label branch (e.g. "unblind").
        If None, no label subtree is added.

    Returns
    -------
    ak.Array
        Awkward array with structure:
        information[event_index][subtree][branch][measurement_index]
    """
    # read CSV
    df = pd.read_csv(csv_file)

    # group branches by subtree
    branches_by_subtree = {}
    for subtree, group in df.groupby("subtree"):
        branches = [b.split(".", 1)[-1] for b in group["branch"]]
        branches_by_subtree[subtree] = branches

    # open ROOT file + get Delphes tree
    output = uproot.open(root_file)
    tree = output["Delphes"]

    # collect awkward arrays for each subtree
    subtree_arrays = {}
    for subtree_name, branch_list in branches_by_subtree.items():
        zipped = ak.zip({
            branch: tree[f"{subtree_name}.{branch}"].array()
            for branch in branch_list
        })
        subtree_arrays[subtree_name.lower() + "s"] = zipped

    # if requested, add constant label branch (all 1s)
    if label_name is not None:
        n_events = tree.num_entries
        blindfold_branch = ak.zip({
            label_name: ak.Array(np.ones(n_events, dtype=np.int64))
        })
        subtree_arrays["blindfold"] = blindfold_branch

    # zip all subtrees together at event level
    information = ak.zip(subtree_arrays, depth_limit=1)

    return information

def combine_with_blindfold(arrays):
    """
    Combine multiple awkward arrays into one, harmonizing the blindfold subtree.

    Parameters
    ----------
    arrays : list of ak.Array
        Awkward arrays produced by extract_branches_from_csv, each with a 'blindfold' subtree contianing a unique branch. 

    Returns
    -------
    ak.Array
        Concatenated awkward array with consistent blindfold structure.
    """
    # collect all blindfold keys across input arrays
    all_keys = set()
    for arr in arrays:
        all_keys.update(arr["blindfold"].fields)
    blindfold_keys = sorted(all_keys)  # sorted for consistent ordering

    normalized = []
    for arr in arrays:
        n_events = len(arr)

        # figure out which key(s) are active for this array
        current_keys = list(arr["blindfold"].fields)
        if len(current_keys) != 1:
            raise ValueError(f"Expected exactly 1 blindfold key, got {current_keys}")
        active_key = current_keys[0]

        # build a unified blindfold subtree with all keys
        blindfold_dict = {}
        for key in blindfold_keys:
            if key == active_key:
                blindfold_dict[key] = ak.Array(np.ones(n_events, dtype=np.int64))
            else:
                blindfold_dict[key] = ak.Array(np.zeros(n_events, dtype=np.int64))

        # replace blindfold subtree
        new_arr = ak.zip({
            **{k: arr[k] for k in arr.fields if k != "blindfold"},
            "blindfold": ak.zip(blindfold_dict)
        }, depth_limit=1)

        normalized.append(new_arr)

    # concatenate along event axis
    return ak.concatenate(normalized, axis=0)


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
        # Handle singular/plural naming
        self.track = ak_array["tracks"] if "tracks" in ak_array.fields else ak_array["track"]
        self.tower = ak_array["towers"] if "towers" in ak_array.fields else ak_array["tower"]
        self.missinget = ak_array["missingets"] if "missingets" in ak_array.fields else ak_array["missinget"]
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
    n_tracks = ak.num(array["tracks"])
    n_towers = ak.num(array["towers"])
    
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
        if len(evt["tracks"]) > 0:
            n_track_features = len(evt["tracks"][0].fields)
            break
    if n_track_features is None:
        raise ValueError("No event contains any track; cannot infer track feature dimension")

    # tower features
    n_tower_features = None
    for evt in refined_data:
        if len(evt["towers"]) > 0:
            n_tower_features = len(evt["towers"][0].fields)
            break
    if n_tower_features is None:
        raise ValueError("No event contains any tower; cannot infer tower feature dimension")

    # missinget features (take first non-empty missinget record)
    n_missinget_features = None
    for evt in refined_data:
        if len(evt["missingets"]) > 0:
            n_missinget_features = len(evt["missingets"][0].fields)
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
            n_tr = len(event["tracks"])
            if n_tr > 0: # Although refinement should ensure this, still added for safety
                pick_tr = np.random.choice(n_tr, min(event_track, n_tr), replace=False)
                track_records = event["tracks"][pick_tr]
                # column_stack fields -> shape (n_selected, n_track_features)
                track_vals = np.column_stack([ak.to_numpy(track_records[f]) for f in track_records.fields])
                track_array = np.zeros((event_track, n_track_features), dtype=np.float32)
                track_array[: track_vals.shape[0], :] = track_vals.astype(np.float32)
            else:
                track_array = np.zeros((event_track, n_track_features), dtype=np.float32)
            set_tracks.append(track_array)

            # ---------- towers ----------
            n_to = len(event["towers"])
            if n_to > 0:
                pick_to = np.random.choice(n_to, min(event_tower, n_to), replace=False)
                tower_records = event["towers"][pick_to]
                tower_vals = np.column_stack([ak.to_numpy(tower_records[f]) for f in tower_records.fields])
                tower_array = np.zeros((event_tower, n_tower_features), dtype=np.float32)
                tower_array[: tower_vals.shape[0], :] = tower_vals.astype(np.float32)
            else:
                tower_array = np.zeros((event_tower, n_tower_features), dtype=np.float32)
            set_towers.append(tower_array)

            # ---------- missinget ----------
            missing_vals = np.zeros((1, n_missinget_features), dtype=np.float32)

            if len(event["missingets"]) > 0:
                missing_records = event["missingets"]
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
        List of events, each with fields "tracks", "towers", "missingets", "blindfold".
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
        (len(evt["tracks"][0].fields) for evt in refined_data if len(evt["tracks"]) > 0), None)
    if n_track_features is None:
        raise ValueError("No event contains any track; cannot infer track feature dimension")

    n_tower_features = next(
        (len(evt["towers"][0].fields) for evt in refined_data if len(evt["towers"]) > 0), None)
    if n_tower_features is None:
        raise ValueError("No event contains any tower; cannot infer tower feature dimension")

    n_missinget_features = next(
        (len(evt["missingets"][0].fields) for evt in refined_data if len(evt["missingets"]) > 0), None)
    if n_missinget_features is None:
        raise ValueError("No event contains missinget; cannot infer missinget feature dimension")

    # containers
    X_tracks, X_towers, X_missinget, y_list, ak_dataset_list = [], [], [], [], []

    for w_start in range(n_windows):
        set_tracks, set_towers, set_missinget, set_blindfold = [], [], [], []
        print(f"Processing window {w_start + 1} / {n_windows}", end="\r")

        # select window events
        for ev_i in range(w_start, w_start + window_size):
            event = refined_data[ev_i]

            # --- tracks ---
            n_tr = len(event["tracks"])
            if n_tr > 0:
                pick_tr = np.random.choice(n_tr, min(event_track, n_tr), replace=False)
                track_records = event["tracks"][pick_tr]
                track_vals = np.column_stack([ak.to_numpy(track_records[f]) for f in track_records.fields])
                track_array = np.zeros((event_track, n_track_features), dtype=np.float32)
                track_array[: track_vals.shape[0], :] = track_vals.astype(np.float32)
            else:
                track_array = np.zeros((event_track, n_track_features), dtype=np.float32)
            set_tracks.append(track_array)

            # --- towers ---
            n_to = len(event["towers"])
            if n_to > 0:
                pick_to = np.random.choice(n_to, min(event_tower, n_to), replace=False)
                tower_records = event["towers"][pick_to]
                tower_vals = np.column_stack([ak.to_numpy(tower_records[f]) for f in tower_records.fields])
                tower_array = np.zeros((event_tower, n_tower_features), dtype=np.float32)
                tower_array[: tower_vals.shape[0], :] = tower_vals.astype(np.float32)
            else:
                tower_array = np.zeros((event_tower, n_tower_features), dtype=np.float32)
            set_towers.append(tower_array)

            # --- missinget ---
            missing_vals = np.zeros((1, n_missinget_features), dtype=np.float32)
            if len(event["missingets"]) > 0:
                missing_records = event["missingets"]
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

# Brier score metric for model evaluation

def brier_score(probs, y_true):  # helper for Brier score
    return ((probs - y_true.float()) ** 2).sum(dim=1).mean().item()

# KL Divergence loss per sample
class KLDPerSample(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, log_probs, target_probs):
        # sum over classes (dim=1) but keep batch dimension
        return F.kl_div(log_probs, target_probs, reduction="none").sum(dim=1)

        
def train_model(model, train_loader, val_loader, optimizer, criterion, device, epochs=100, patience=10, fold_id=None):
    os.makedirs("Analytics", exist_ok=True)
    os.makedirs("Analytics/models", exist_ok=True) 

    labels = []
    for _, _, _, y in train_loader.dataset:
        labels.append(int(y.argmax(dim=0)))
    labels_tensor = torch.tensor(labels)
    unique, counts = torch.unique(labels_tensor, return_counts=True)
    class_weights = 1.0 / counts.float()
    class_weights = class_weights / class_weights.sum() * len(unique)
    class_weights = class_weights.to(device)
    print(f"Class weights applied: {[round(w, 2) for w in class_weights.tolist()]}") 

    log = {"epoch": [], "train_loss": [], "val_loss": [], "train_acc": [], "val_acc": [], "train_brier": [], "val_brier": []}
    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(epochs):
        # Training loop
        model.train()
        train_loss, correct, total, train_brier = 0, 0, 0, 0
        for track, tower, met, y in train_loader:
            track, tower, met, y = track.to(device), tower.to(device), met.to(device), y.to(device)
            optimizer.zero_grad()
            probs, log_probs = model(track, tower, met)
            loss = criterion(log_probs, y)

            true_cls = y.argmax(dim=1)
            sample_weights = class_weights[true_cls]  # [batch]
            loss_per_sample = criterion(log_probs, y)  # sum over classes
            loss = (loss_per_sample * sample_weights).mean()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            pred_cls = probs.argmax(dim=1)
            correct += (pred_cls == true_cls).sum().item()
            total += y.size(0)
            train_brier += brier_score(probs, y) * y.size(0)

        train_loss /= len(train_loader)
        train_acc = correct / total
        train_brier /= total

        # Validation loop
        model.eval()
        val_loss, correct, total, val_brier = 0, 0, 0, 0
        with torch.no_grad():
            for track, tower, met, y in val_loader:
                track, tower, met, y = track.to(device), tower.to(device), met.to(device), y.to(device)
                probs, log_probs = model(track, tower, met)


                true_cls = y.argmax(dim=1)
                sample_weights = class_weights[true_cls]
                loss_per_sample = criterion(log_probs, y)
                loss = (loss_per_sample * sample_weights).mean()
                val_loss += loss.item()
                pred_cls = probs.argmax(dim=1)

                correct += (pred_cls == true_cls).sum().item()
                total += y.size(0)
                val_brier += brier_score(probs, y) * y.size(0)

        val_loss /= len(val_loader)
        val_acc = correct / total
        val_brier /= total 

        log["epoch"].append(epoch+1)
        log["train_loss"].append(train_loss)
        log["val_loss"].append(val_loss)
        log["train_acc"].append(train_acc)
        log["val_acc"].append(val_acc)
        log["train_brier"].append(train_brier) 
        log["val_brier"].append(val_brier)  

        print(f"Fold {fold_id} | Epoch {epoch+1} | Train Loss {train_loss:.4f} Acc {train_acc:.4f} Brier {train_brier:.4f} | "
              f"Val Loss {val_loss:.4f} Acc {val_acc:.4f} Brier {val_brier:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), f"Analytics/models/best_model_fold_{fold_id}.pt")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered for Fold {fold_id} at epoch {epoch+1}.")
                break

    # Save logs & plots
    df = pd.DataFrame(log)
    df.to_csv(f"Analytics/model_logs_{fold_id}.csv", index=False)  

    # Save per-fold plots
    plt.figure(figsize=(6, 4))
    plt.plot(df["epoch"], df["train_acc"], label="Train Acc")
    plt.plot(df["epoch"], df["val_acc"], label="Val Acc")
    plt.axvline(x=epoch + 1 - patience_counter, color='red', linestyle='--', label="Early Stop")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.title(f"Model {fold_id} Accuracy")
    plt.tight_layout()
    plt.savefig(f"Analytics/training_fold_{fold_id}_accuracy.png")  
    plt.close()


    plt.figure(figsize=(6, 4))
    plt.plot(df["epoch"], df["train_brier"], label="Train Brier")
    plt.plot(df["epoch"], df["val_brier"], label="Val Brier")
    plt.axvline(x=epoch + 1 - patience_counter, color='red', linestyle='--', label="Early Stop")
    plt.xlabel("Epoch")
    plt.ylabel("Brier Score")
    plt.legend()
    plt.title(f"Model {fold_id} Brier Score")
    plt.tight_layout()
    plt.savefig(f"Analytics/training_fold_{fold_id}_brier.png") 
    plt.close()


    plt.figure(figsize=(6, 4))
    plt.plot(df["epoch"], df["train_loss"], label="Train Loss")
    plt.plot(df["epoch"], df["val_loss"], label="Val Loss")
    plt.axvline(x=epoch + 1 - patience_counter, color='red', linestyle='--', label="Early Stop")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title(f"Model {fold_id} Loss")
    plt.tight_layout()
    plt.savefig(f"Analytics/training_fold_{fold_id}_loss.png")
    plt.close()



    return df


def cross_validate_model(dataset, k, batch_size, model_class, model_kwargs, optimizer_class, optimizer_kwargs, criterion, device, epochs, patience):  
    
    # Handle both DataLoader and raw dataset
    if isinstance(dataset, DataLoader):
        print("DataLoader detected — using its dataset directly.")
        dataset = dataset.dataset

    print("\nRunning a quick analysis of the training data...")

    labels = []
    for i in range(len(dataset)):
        _, _, _, y = dataset[i]  # dataset returns (track, tower, met, y)
        labels.append(int(y.argmax(dim=0)))  # convert probabilities to class index

    labels_tensor = torch.tensor(labels)
    unique, counts = torch.unique(labels_tensor, return_counts=True)
    print("Label distribution in dataset:")
    for u, c in zip(unique.tolist(), counts.tolist()):
        print(f"  Class {u}: {c} samples")

    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
    all_logs = []

    for fold_id, (train_idx, val_idx) in enumerate(skf.split(range(len(dataset)), labels_tensor), start=1):
        print(f"\n--- Starting Fold {fold_id}/{k} Training ---")
        train_subset = Subset(dataset, train_idx)
        val_subset = Subset(dataset, val_idx)

        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=batch_size)

        model = model_class(**model_kwargs).to(device)
        optimizer = optimizer_class(model.parameters(), **optimizer_kwargs)

        df = train_model(model, train_loader, val_loader, optimizer, criterion, device, epochs, patience, fold_id=fold_id)
        all_logs.append(df)

        print(f"\n--- Completed Fold {fold_id}/{k} Training ---")


    # FINAL COMPARISON PLOTS

    plt.figure(figsize=(12, 5))
    for i, df in enumerate(all_logs, start=1):
        plt.plot(df["epoch"], df["val_acc"], label=f"Model {i}")
    plt.xlabel("Epoch"); plt.ylabel("Validation Accuracy")
    plt.title("Validation Accuracy for each model")
    plt.legend()
    plt.savefig("Analytics/validation_accuracy_all_folds.png")  
    plt.close()

    plt.figure(figsize=(12, 5))
    for i, df in enumerate(all_logs, start=1):
        plt.plot(df["epoch"], df["val_brier"], label=f"Model {i}")
    plt.xlabel("Epoch"); plt.ylabel("Validation Brier Score")
    plt.title("Validation Brier Score for each model")
    plt.legend()
    plt.savefig("Analytics/validation_brier_all_folds.png")  
    plt.close()

    plt.figure(figsize=(12, 5))
    for i, df in enumerate(all_logs, start=1):
        plt.plot(df["epoch"], df["val_loss"], label=f"Model {i}")
    plt.xlabel("Epoch"); plt.ylabel("Validation loss")
    plt.title("Validation loss for each model")
    plt.legend()
    plt.savefig("Analytics/validation_loss_all_folds.png")  
    plt.close()

    plt.figure(figsize=(12, 5))
    for i, df in enumerate(all_logs, start=1):
        plt.plot(df["epoch"], df["train_acc"], label=f"Model {i}")
    plt.xlabel("Epoch"); plt.ylabel("Training Accuracy")
    plt.title("Training Accuracy for each model")
    plt.legend()
    plt.savefig("Analytics/training_accuracy_all_folds.png")  
    plt.close()

    plt.figure(figsize=(12, 5))
    for i, df in enumerate(all_logs, start=1):
        plt.plot(df["epoch"], df["train_brier"], label=f"Model {i}")
    plt.xlabel("Epoch"); plt.ylabel("Training Brier Score")
    plt.title("Training Brier Score for each model")
    plt.legend()
    plt.savefig("Analytics/training_brier_all_folds.png")  
    plt.close()

    plt.figure(figsize=(12, 5))
    for i, df in enumerate(all_logs, start=1):
        plt.plot(df["epoch"], df["train_loss"], label=f"Model {i}")
    plt.xlabel("Epoch"); plt.ylabel("Training loss")
    plt.title("Training loss for each model")
    plt.legend()
    plt.savefig("Analytics/training_loss_all_folds.png")  
    plt.close()

def soft_confusion_matrix(y_true, y_probs, classes=None, normalize=True):
    """
    Compute a soft confusion matrix for probabilistic (soft) labels for specified classes.

    Parameters
    ----------
    y_true : np.ndarray of shape [n_samples, n_classes], true probabilities
    y_probs : np.ndarray of shape [n_samples, n_classes], predicted probabilities
    classes : list of int, optional
        Subset of class indices to compute the confusion matrix for. If None, use all classes.
    normalize : bool, optional (default=True)
        If True, rows are normalized to sum to 1.

    Returns
    -------
    soft_CM : np.ndarray of shape [len(classes), len(classes)]
        Soft confusion matrix (joint true-predicted mass) for selected classes.
    """
    if classes is None:
        classes = np.arange(y_true.shape[1])
    
    # Select only the columns corresponding to the specified classes
    y_true_sub = y_true[:, classes]   # [N, C_sub]
    y_probs_sub = y_probs[:, classes] # [N, C_sub]

    # Compute joint mass
    soft_CM = np.dot(y_true_sub.T, y_probs_sub)  # [C_sub, C_sub]

    if normalize:
        row_sums = soft_CM.sum(axis=1, keepdims=True)
        soft_CM = np.divide(soft_CM, row_sums, out=np.zeros_like(soft_CM), where=row_sums != 0)

    return soft_CM


def test_model(model_class, model_kwargs, test_loader, device, k=5, test_samples = 10000):
    print("\nRunning a quick analysis of the test data...")

    all_y = []
    for _, _, _, y in test_loader:
        all_y.append(y)
    all_y = torch.cat(all_y, dim=0)
    labels_tensor = all_y.argmax(dim=1)
    unique, counts = torch.unique(labels_tensor, return_counts=True)
    for u, c in zip(unique.tolist(), counts.tolist()):
        print(f"  Class {u}: {c} samples")

    min_class_samples = counts.min().item()
    if min_class_samples < test_samples:
        print(f"\n Not enough samples for test_samples={test_samples}. "
              f"Minimum available per class is {min_class_samples}. "
              f"Consider reducing 'test_samples' accordingly.\n")
        test_samples = min_class_samples  # adjust automatically

    selected_indices = []
    for cls in unique.tolist():
        cls_indices = (labels_tensor == cls).nonzero(as_tuple=True)[0]
        perm = torch.randperm(len(cls_indices))[:test_samples]
        selected_indices.extend(cls_indices[perm].tolist())

    selected_indices = torch.tensor(selected_indices)
    tracks, towers, mets, ys = [], [], [], []
    for i in range(len(test_loader.dataset)):
        track, tower, met, y = test_loader.dataset[i]
        tracks.append(track.unsqueeze(0))
        towers.append(tower.unsqueeze(0))
        mets.append(met.unsqueeze(0))
        ys.append(y.unsqueeze(0))

    tracks = torch.cat(tracks, dim=0)
    towers = torch.cat(towers, dim=0)
    mets = torch.cat(mets, dim=0)
    ys = torch.cat(ys, dim=0)

    # Select the balanced subset
    balanced_dataset = torch.utils.data.TensorDataset(
        tracks[selected_indices],
        towers[selected_indices],
        mets[selected_indices],
        ys[selected_indices]
    )
    test_loader = DataLoader(balanced_dataset, batch_size=test_loader.batch_size, shuffle=False)



    # Load all trained models
    models = []
    for i in range(1, k + 1):
        model_path = f"Analytics/models/best_model_fold_{i}.pt"
        model = model_class(**model_kwargs).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        models.append(model)
        print(f"Loaded model {i} from: {model_path}")

    all_probs_mean = []
    all_preds_majority = []
    all_targets = []

    with torch.no_grad():
        for track, tower, met, y in test_loader:
            track, tower, met, y = track.to(device), tower.to(device), met.to(device), y.to(device)

            # Collect outputs from all k models
            fold_probs = []
            fold_preds = []
            for model in models:
                probs, _ = model(track, tower, met)
                fold_probs.append(probs)
                fold_preds.append(probs.argmax(dim=1))

            # Stack predictions
            fold_probs = torch.stack(fold_probs)  # [k, batch, num_classes]
            fold_preds = torch.stack(fold_preds)  # [k, batch]

            # Majority Voting for Accuracy
            preds_majority = torch.mode(fold_preds, dim=0).values  # [batch]

            # Probability Averaging for Brier
            probs_mean = fold_probs.mean(dim=0)  # [batch, num_classes]

            all_probs_mean.append(probs_mean.cpu())
            all_preds_majority.append(preds_majority.cpu())
            all_targets.append(y.cpu())

    # Concatenate all batches
    all_probs_mean = torch.cat(all_probs_mean, dim=0)
    all_preds_majority = torch.cat(all_preds_majority, dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    # Convert to numpy
    all_probs = all_probs_mean.numpy()
    all_preds = all_preds_majority.numpy()
    all_targets_np = all_targets.numpy()
    y_true = np.argmax(all_targets_np, axis=1)
    y_true_onehot = label_binarize(y_true, classes=list(range(all_probs.shape[1])))

    # Compute metrics
    acc = accuracy_score(y_true, all_preds)
    brier = brier_score(torch.tensor(all_probs), all_targets)

    print(f"\nEnsemble Accuracy (Majority Vote): {acc:.4f}")
    print(f"Ensemble Brier Score (Averaged Probabilities): {brier:.4f}")


    # ============================================
    # Classification performance evaluation
    # ============================================

    acc_multi = accuracy_score(y_true, all_preds)
    auc_multi = roc_auc_score(y_true, all_probs, multi_class="ovr") 

    print(f"4-Class Accuracy: {acc_multi:.4f}, ROC-AUC: {auc_multi:.4f}")

    cm_multi = confusion_matrix(y_true, all_preds)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm_multi, annot=True, fmt='d', cmap='Blues',
                xticklabels=["Normal", "Barrel", "Endcap", "Forward"],
                yticklabels=["Normal", "Barrel", "Endcap", "Forward"])
    plt.title(f"Confusion Matrix (4-Class)\nAcc={acc_multi:.3f}")
    plt.xlabel("Predicted"); plt.ylabel("True")
    plt.tight_layout()
    plt.savefig("Analytics/confusion_matrix_multiclass.png")
    plt.close()

    cm_multi_soft = soft_confusion_matrix(all_targets_np, all_probs, classes=[0,1,2,3])
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm_multi_soft, annot=True, fmt='.1f', cmap='Blues',
                xticklabels=["Normal", "Barrel", "Endcap", "Forward"],
                yticklabels=["Normal", "Barrel", "Endcap", "Forward"])
    plt.title(f"Soft Confusion Matrix (4-Class)")
    plt.xlabel("Predicted"); plt.ylabel("True")
    plt.tight_layout()
    plt.savefig("Analytics/soft_confusion_matrix_multiclass.png")
    plt.close()

    

    # ROC for each class
    plt.figure(figsize=(7, 5))
    for i, label in enumerate(["Normal", "Barrel", "Endcap", "Forward"]):
        fpr, tpr, _ = roc_curve(y_true_onehot[:, i], all_probs[:, i])
        plt.plot(fpr, tpr, label=f"Class {label}")
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curves - 4-Class (AUC={auc_multi:.3f})")
    plt.legend()
    plt.tight_layout()
    plt.savefig("Analytics/roc_curve_multiclass.png")
    plt.close()

    # ============================================
    # Classification performance evaluation for Binary case
    # ============================================

    # --- Binary conversion (Normal=0, Anomalous=1) ---
    y_binary_true = np.where(np.argmax(all_targets_np, axis=1) == 0, 0, 1)  # class 0=Normal, others=Anomalous
    y_binary_pred = np.where(np.argmax(all_probs, axis=1) == 0, 0, 1)

    # --- Balance data ---
    normal_indices = np.where(y_binary_true == 0)[0] 
    anomalous_indices = np.where(y_binary_true == 1)[0] 
    n_samples_binary = min(test_samples, len(normal_indices), len(anomalous_indices)) 
    print(f"Using {n_samples_binary} samples per class for balanced evaluation of binary case.") 

    selected_indices = np.concatenate([
        np.random.choice(normal_indices, n_samples_binary, replace=False),
        np.random.choice(anomalous_indices, n_samples_binary, replace=False)
        ]) 

    y_binary_true_bal = y_binary_true[selected_indices] 
    y_binary_pred_bal = y_binary_pred[selected_indices]
    probs_binary_bal = all_probs[selected_indices]  

    p0_pred = all_probs[:,0]  # predicted probability for Normal
    probs_binary_bal = np.zeros((len(selected_indices), 2))  # shape [N, 2]
    probs_binary_bal[:,0] = p0_pred[selected_indices]       # Normal
    probs_binary_bal[:,1] = 1 - p0_pred[selected_indices]   # Anomalous

    p0_true = all_targets_np[:,0]  # true probability for Normal
    y_true_probs_bal = np.zeros((len(selected_indices), 2))
    y_true_probs_bal[:,0] = p0_true[selected_indices]       # Normal
    y_true_probs_bal[:,1] = 1 - p0_true[selected_indices] 

    # --- Soft confusion matrix  ---
    cm_binary = confusion_matrix(y_binary_true_bal, y_binary_pred_bal, labels=[0,1])
    cm_binary_soft = soft_confusion_matrix(y_true_probs_bal, probs_binary_bal, classes=[0,1])

    acc_binary = accuracy_score(y_binary_true_bal, y_binary_pred_bal)
    roc_auc_binary = roc_auc_score(y_binary_true_bal, 1 - probs_binary_bal[:, 0]) 

    print(f"Binary Accuracy: {acc_binary:.4f}, ROC-AUC: {roc_auc_binary:.4f}")

    # --- Plot confusion matrix  ---
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm_binary, annot=True, fmt='d', cmap='Greens',
                xticklabels=["Normal", "Anomalous"],
                yticklabels=["Normal", "Anomalous"])
    plt.title(f"Confusion Matrix binary\nAcc={acc_binary:.3f}")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig("Analytics/confusion_matrix_binary.png")
    plt.close()


    # --- Plot soft confusion matrix  ---
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm_binary_soft, annot=True, fmt='.1f', cmap='Greens',
                xticklabels=["Normal", "Anomalous"],
                yticklabels=["Normal", "Anomalous"])
    plt.title(f"Soft Confusion Matrix binary\nAcc={acc_binary:.3f}")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig("Analytics/soft_confusion_matrix_binary.png")
    plt.close()

    fpr, tpr, _ = roc_curve(y_binary_true_bal, 1 - probs_binary_bal[:, 0])

    plt.figure(figsize=(5, 4))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc_binary:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC: Normal vs Anomalous')
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig("Analytics/roc_binary.png")
    plt.close()


    # Save metrics summary
    metrics_summary = {
        "Ensemble_Accuracy_MajorityVote": acc,
        "Brier_Score_AvgProb": brier,
        "Multi_Accuracy": acc_multi,
        "Multi_ROC_AUC": auc_multi,
        "Binary_Accuracy": acc_binary,
        "Binary_ROC_AUC": roc_auc_binary
    }
    pd.DataFrame([metrics_summary]).to_csv("Analytics/ensemble_metrics.csv", index=False)

    print("\nAll evaluation plots and metrics saved under Analytics.\n")
    return metrics_summary




def predict_from_model(model_class, model_kwargs, test_loader, device, k=5):
    # Load all trained models and outputs the predictions only
    models = []
    for i in range(1, k + 1):
        model_path = f"Analytics/models/best_model_fold_{i}.pt"
        model = model_class(**model_kwargs).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        models.append(model)
        print(f"Loaded model {i} from: {model_path}")

    all_probs_mean = []
    all_preds_majority = []
    all_targets = []

    with torch.no_grad():
        for track, tower, met, y in test_loader:
            track, tower, met, y = track.to(device), tower.to(device), met.to(device), y.to(device)

            # Collect outputs from all k models
            fold_probs = []
            fold_preds = []
            for model in models:
                probs, _ = model(track, tower, met)
                fold_probs.append(probs)
                fold_preds.append(probs.argmax(dim=1))

            # Stack predictions
            fold_probs = torch.stack(fold_probs)  # [k, batch, num_classes]
            fold_preds = torch.stack(fold_preds)  # [k, batch]

            # Majority Voting for Accuracy
            preds_majority = torch.mode(fold_preds, dim=0).values  # [batch]

            # Probability Averaging for Brier
            probs_mean = fold_probs.mean(dim=0)  # [batch, num_classes]

            all_probs_mean.append(probs_mean.cpu())
            all_preds_majority.append(preds_majority.cpu())
            all_targets.append(y.cpu())

    # Concatenate all batches
    all_probs_mean = torch.cat(all_probs_mean, dim=0)
    all_preds_majority = torch.cat(all_preds_majority, dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    # Convert to numpy
    all_probs = all_probs_mean.numpy()
    all_preds = all_preds_majority.numpy()
    all_targets_np = all_targets.numpy()
    y_true = np.argmax(all_targets_np, axis=1)
    y_true_onehot = label_binarize(y_true, classes=list(range(all_probs.shape[1])))

    return all_probs, all_preds, y_true, y_true_onehot
