import torch
import numpy as np
import cv2
from segment_anything import sam_model_registry, SamPredictor, SamAutomaticMaskGenerator


class SamMaskProcessor:
    """Handles SAM model for generating segmentation masks."""

    def __init__(self, checkpoint_path, model_type="vit_l", device="cuda"):
        print("Loading SAM model...")
        self.device = device
        self.sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
        self.sam.to(device=self.device)
        self.predictor = SamPredictor(self.sam)
        print("SAM loaded successfully.")

    def generate_masks(self, image_path, detections):
        """Generate masks for all detections in an image."""
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Cannot load image: {image_path}")

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.predictor.set_image(image_rgb)

        if not detections:
            return []

        input_boxes = torch.tensor(
            [d["bbox"] for d in detections], device=self.predictor.device
        )

        transformed_boxes = self.predictor.transform.apply_boxes_torch(
            input_boxes, image_rgb.shape[:2]
        )

        masks_tensor, _, _ = self.predictor.predict_torch(
            point_coords=None,
            point_labels=None,
            boxes=transformed_boxes,
            multimask_output=False,
        )

        binary_masks = masks_tensor.cpu().numpy().squeeze(1).astype(np.uint8)

        if len(binary_masks.shape) == 2:
            binary_masks = binary_masks[None, :, :]

        return list(binary_masks)

    def auto_masks_generate(self, image_path):
        image_bgr = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        auto_generator = SamAutomaticMaskGenerator(
            model=self.sam,
            points_per_side=64,
            pred_iou_thresh=0.88,
            stability_score_thresh=0.92,
            min_mask_region_area=120,
        )
        auto_masks = auto_generator.generate(image_rgb)
        print(f"\nSAM auto found {len(auto_masks)} segments total")
        return auto_masks
