from rfdetr import RFDETRMedium
from ultralytics import YOLO
from transformers import CLIPProcessor, CLIPModel

from src.sam_processor import SamMaskProcessor
import os
import urllib.request
from tqdm import tqdm


class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


class ModelManager:
    """Manages all model loading and initialization."""

    def __init__(self, config):
        self.config = config
        self._detection_model = None
        self._classify_model = None
        self._sam_processor = None
        self._clip_model = None
        self._clip_processor = None

    def _download_if_missing(self, path, url, desc):
        if not os.path.exists(path):
            print(f"{desc} not found at {path}. Downloading...")
            os.makedirs(os.path.dirname(path), exist_ok=True)
            try:
                with DownloadProgressBar(
                    unit="B", unit_scale=True, miniters=1, desc=desc
                ) as t:
                    urllib.request.urlretrieve(url, path, reporthook=t.update_to)
                print(f"\n{desc} downloaded successfully!")
            except Exception as e:
                print(f"\nError downloading {desc}: {e}")
                if os.path.exists(path):
                    os.remove(path)
                raise e

    @property
    def detection_model(self):
        if self._detection_model is None:
            print("Loading RF-DETR detection model...")

            rfdetr_url = "https://huggingface.co/aria0081/AI2D-Relation-Models/resolve/main/ai2d_detection_basic.pth"
            self._download_if_missing(
                self.config.DETECTION_MODEL, rfdetr_url, "Downloading RF-DETR"
            )

            self._detection_model = RFDETRMedium(
                pretrain_weights=self.config.DETECTION_MODEL,
                device=self.config.DEVICE,
                num_classes=3,
            )
            self._detection_model.optimize_for_inference()
            print("Detection model loaded.")
        return self._detection_model

    @property
    def classify_model(self):
        if self._classify_model is None:
            print("Loading YOLO classification model...")

            yolo_url = "https://huggingface.co/aria0081/AI2D-Relation-Models/resolve/main/ai2d_classify_model.pt"
            self._download_if_missing(
                self.config.CLASSIFY_MODEL, yolo_url, "Downloading YOLO"
            )

            self._classify_model = YOLO(self.config.CLASSIFY_MODEL)
            print("Classification model loaded.")
        return self._classify_model

    @property
    def sam_processor(self):
        checkpoint_path = self.config.SAM_CHECKPOINT

        if self._sam_processor is None:
            sam_url = (
                "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth"
            )
            self._download_if_missing(
                self.config.SAM_CHECKPOINT, sam_url, "Downloading SAM"
            )

            self._sam_processor = SamMaskProcessor(
                self.config.SAM_CHECKPOINT,
                model_type="vit_l",
                device=self.config.DEVICE,
            )
        return self._sam_processor

    @property
    def clip_model(self):
        if self._clip_model is None:
            print("Loading CLIP model...")
            self._clip_model = CLIPModel.from_pretrained(
                "openai/clip-vit-large-patch14"
            ).to(self.config.DEVICE)
            print("CLIP model loaded.")
        return self._clip_model

    @property
    def clip_processor(self):
        if self._clip_processor is None:
            self._clip_processor = CLIPProcessor.from_pretrained(
                "openai/clip-vit-large-patch14"
            )
        return self._clip_processor
