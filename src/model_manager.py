from rfdetr import RFDETRMedium
from ultralytics import YOLO
from transformers import CLIPProcessor, CLIPModel

from src.sam_processor import SamMaskProcessor


class ModelManager:
    """Manages all model loading and initialization."""

    def __init__(self, config):
        self.config = config
        self._detection_model = None
        self._classify_model = None
        self._sam_processor = None
        self._clip_model = None
        self._clip_processor = None

    @property
    def detection_model(self):
        if self._detection_model is None:
            print("Loading RF-DETR detection model...")
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
            self._classify_model = YOLO(self.config.CLASSIFY_MODEL)
            print("Classification model loaded.")
        return self._classify_model

    @property
    def sam_processor(self):
        if self._sam_processor is None:
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
