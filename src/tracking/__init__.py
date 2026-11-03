import os

def resolve_tracker_config(config_name="tracker.yaml"):
    candidates = [
        config_name,
        os.path.join("configs", config_name),
        os.path.join("configs", "tracker.yaml"),
        "botsort_reid.yaml",
        os.path.join(os.path.dirname(__file__), "..", "..", "configs", "tracker.yaml"),
        os.path.join(os.path.dirname(__file__), "..", "..", "botsort_reid.yaml"),
    ]
    for c in candidates:
        if os.path.exists(c):
            return os.path.abspath(c)
    return "bytetrack.yaml"
