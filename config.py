import torch


class Config:

    home_path = "/home/philine/Documents/KLTN/AI2D/test_demo"
    # Model paths
    SAM_CHECKPOINT = f"{home_path}/weights/sam_vit_l_0b3195.pth"
    DETECTION_MODEL = f"{home_path}/weights/ai2d_detection_basic.pth"
    CLASSIFY_MODEL = f"{home_path}/weights/ai2d_classify_model.pt"

    # Data paths
    # IMAGE_DIR = "/content/drive/MyDrive/ai2d-all/ai2d/images_by_cat/lifeCycles"
    # RST_JSON = "/content/drive/MyDrive/ai2d-all/ai2d/categories_ai2d-rst.json"
    LABELS_JSON = {
        "lifeCycles": f"{home_path}/ovd_dict/lifeCycles_keepOnly.json",
        "foodChainsWebs": f"{home_path}/ovd_dict/foodChainsWebs_keepOnly.json",
        "circuits": f"{home_path}/ovd_dict/circuits_keepOnly.json",
        "photosynthesisRespiration": f"{home_path}/ovd_dict/photosynthesisRespiration_keepOnly.json",
        "rockCycle": f"{home_path}/ovd_dict/rockCycle_keepOnly.json",
        "waterCNPCycle": f"{home_path}/ovd_dict/waterCNPCycle_keepOnly.json",
    }
    OUTPUT_DIR = f"{home_path}/results"

    # Detection labels
    CLS_LABELS = ["arrow", "arrowHead", "blob", "text"]
    CANDIDATE_LABELS = ["Caption", "Label", "Misc", "Title"]

    # CLS for classify
    RST_CATEGORIES = [
        "illustration",
        "cross-section",
        "cycle",
        "network",
        "cut-out",
        "horizontal",
        "table",
        "mixed",
        "diagrammatic",
        "vertical",
        "photograph",
        "exploded",
    ]

    CATEGORIES = [
        "partsOfA",
        "typesOf",
        "foodChainsWebs",
        "lifeCycles",
        "moonPhaseEquinox",
        "circuits",
        "partsOfTheEarth",
        "rockCycle",
        "other",
        "rockStrata",
        "eclipses",
        "photosynthesisRespiration",
        "volcano",
        "faultsEarthquakes",
        "waterCNPCycle",
        "solarSystem",
        "atomStructure",
    ]

    RELATION_LABELS = {
        "partsOfA": "is_part_of",
        "typesOf": "is_a",
        "foodChainsWebs": "eaten_by",
        "lifeCycles": "transforms_to",
        "moonPhaseEquinox": "followed_by",
        "circuits": "connected_to",
        "partsOfTheEarth": "is_layer_of",
        "rockCycle": "turns_into",
        "other": "related_to",
        "rockStrata": "lies_above",
        "eclipses": "obscured_by",
        "photosynthesisRespiration": "produces",
        "volcano": "erupts_from",
        "faultsEarthquakes": "causes",
        "waterCNPCycle": "transitions_to",
        "solarSystem": "orbits",
        "atomStructure": "composed_of",
    }

    DETECTION_THRESHOLD = 0.5
    CLASSIFICATION_THRESHOLD = 0.5
    MAX_DIST_BLOB = 50
    MAX_DIST_ARROW_MID = 100
    MIN_TEXT_RATIO = 0.5
    TITLE_CENTER_FRACTION = 0.15
    CORNER_FRACTION = 0.2
    TITLE_BLOB_PROXIMITY = 20

    DEVICE = "cpu"


if __name__ == "__main__":
    config = Config()
    print(f"Using device: {config.DEVICE}")
