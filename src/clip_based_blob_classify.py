from transformers import CLIPProcessor, CLIPModel
import torch
from pathlib import Path
from PIL import Image


class CLIPImageClassifier:
    def __init__(self, model, processor, device="cpu", categories=None):
        self.model = model
        self.processor = processor
        self.device = device

        self.categories = categories

    def classify(self, img):
        """Classify a blob crop using CLIP."""
        text_inputs = [
            f"a visual representation in the style of a {category}"
            for category in self.categories
        ]

        inputs = self.processor(
            text=text_inputs, images=img, return_tensors="pt", padding=True
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)

        logits_per_image = outputs.logits_per_image
        probs = logits_per_image.softmax(dim=1)

        best_idx = probs.argmax().item()
        best_label = self.categories[best_idx]
        best_score = probs[0, best_idx].item()

        result = {"label": best_label, "confidence": best_score}

        return result


class CLIPBlobClassifier:
    """Open-vocabulary blob classification using CLIP."""

    def __init__(self, model, processor, device="cpu"):
        self.model = model
        self.processor = processor
        self.device = device

    def classify_blob(self, crop_img, candidate_labels):
        """Classify a blob crop using CLIP."""
        text_inputs = [f"an image of {label}" for label in candidate_labels]

        inputs = self.processor(
            text=text_inputs, images=crop_img, return_tensors="pt", padding=True
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)

        logits_per_image = outputs.logits_per_image
        probs = logits_per_image.softmax(dim=1)

        best_idx = probs.argmax().item()
        best_label = candidate_labels[best_idx]
        best_score = probs[0, best_idx].item()

        return best_label, best_score

    def process_blobs(self, img_path, detections, candidate_labels, padding=40):
        """Process all blobs in an image."""
        image = Image.open(img_path).convert("RGB")
        img_w, img_h = image.size

        results = []

        for det in detections:
            if det["label"] != "blob":
                continue

            bbox = det["bbox"]
            x1, y1, x2, y2 = map(int, bbox)

            x1_pad = max(0, x1 - padding)
            y1_pad = max(0, y1 - padding)
            x2_pad = min(img_w, x2 + padding)
            y2_pad = min(img_h, y2 + padding)

            crop = image.crop((x1_pad, y1_pad, x2_pad, y2_pad))

            detected_type, clip_conf = self.classify_blob(crop, candidate_labels)

            results.append(
                {
                    "bbox": bbox,
                    "detr_conf": det["confidence"],
                    "clip_label": detected_type,
                    "clip_conf": clip_conf,
                }
            )

        return results
