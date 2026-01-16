import joblib

def save_model(model, path):
    """Save trained model to a .pkl file"""
    joblib.dump(model, path)

def load_model(path):
    """Load trained model from a .pkl file"""
    return joblib.load(path)
