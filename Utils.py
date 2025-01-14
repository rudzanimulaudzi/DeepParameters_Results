import pickle
import os
from pathlib import Path
from datetime import datetime
import json


# Function to compute metrics, only KL divergence and SHD
def compute_metrics(data, dl_data, true_edges, learned_edges):
    divergences = bn_metrics.calculate_all_divergences(data, dl_data)
    shds = bn_metrics.calculate_shd(true_edges, learned_edges)
    return divergences, shds

def ensure_directory_exists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def save_object(obj, filename):
    with open(filename, 'wb') as f:
        pickle.dump(obj, f)

def save_data(data, filename):
    data.to_csv(filename, index=False)

def ensure_json_suffix(filename):
    """Ensure the filename ends with .json suffix."""
    base, ext = os.path.splitext(filename)
    if ext.lower() != ".json":
        filename = f"{base}.json"
    return filename

def is_valid_json(data):
    """Check if the data is serializable to JSON."""
    try:
        json.dumps(data)
        return True
    except (TypeError, ValueError):
        return False

import numpy as np

def numpy_converter(obj):
    """
    Convert numpy objects to serializable format.
    """
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    raise TypeError(f"Object of type '{obj.__class__.__name__}' is not JSON serializable")

def save_as_json(data, directory, filename):
    """
    Saves the given data as a JSON file in the specified directory with a timestamped filename.

    Parameters:
    - data: The data to be saved (must be serializable to JSON).
    - directory: The directory where the file will be saved.
    - filename_prefix: Prefix for the filename.
    - filename_suffix: Suffix for the filename (default is ".json").

    Returns:
    - The path to the saved file.
    """
    # Ensure the directory exists
    ensure_directory_exists(directory)
    # File is a valid JSON file
    filename = ensure_json_suffix(filename)
    
    #if not is_valid_json(data):
    #    raise ValueError("Provided data is not valid JSON serializable.")
    
    # Construct the filename with a filename
    filepath = os.path.join(directory, filename)

    try:
        with open(filepath, "w") as file:
            # Use the custom converter for numpy objects
            json.dump(data, file, default=numpy_converter)
            
        return filepath
    
    except Exception as e:
        main_logger.error(f"Error saving data as JSON: {e}")
        return None