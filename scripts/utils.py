import os
import json
import numpy as np

def get_feature_stats(storage_dir="../artwork_features"):
    """Get statistics about stored features"""
    metadata_file = os.path.join(storage_dir, "metadata.json")
    
    with open(metadata_file, 'r') as f:
        metadata = json.load(f)
    
    print(f"Total artworks: {len(metadata)}")
    
    if metadata:
        sample_key = next(iter(metadata.keys()))
        feature_path = metadata[sample_key]['feature_path']
        sample_features = np.load(feature_path)
        print(f"Feature dimension: {sample_features.shape[0]}")
    
    return metadata

if __name__ == "__main__":
    get_feature_stats()