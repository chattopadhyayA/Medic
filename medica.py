import awkward as ak
import json
import numpy as np


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





        
