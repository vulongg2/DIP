# Backward-compatibility alias layer
from src.reid.custom_reid_net import CustomReIDNet, TripletLoss
from src.reid.reid_extractor import ReIDExtractor
from src.topology.camera_graph import CameraTopology
from src.association.global_manager import GlobalIdentity, GlobalIdentityManager

__all__ = [
    "CustomReIDNet", "TripletLoss", "ReIDExtractor",
    "CameraTopology", "GlobalIdentity", "GlobalIdentityManager"
]
