"""data_gen_function.py"""

# importing necessary packages 
import uproot
import matplotlib.pyplot as plt
import awkward as ak
import numpy as np
import pandas as pd


# the function that creates individual data subtrees 
def extract_branches_from_csv(root_file, csv_file, label_name=None):
    """
    Extract selected subtrees from a Delphes ROOT file into an awkward array,
    based on a CSV file that lists subtree and branch names. Optionally adds
    a constant 'label' branch (blindfold) filled with 1s.

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
