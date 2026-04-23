import cv2
import re
import pytesseract
from PIL import Image

from config import Config


class DetectionUtils:
    """Utility functions for detection and OCR."""

    @staticmethod
    def detect_one_image(model, img_path, threshold=0.5, cls_labels=None):
        """Run detection on a single image."""
        if cls_labels is None:
            cls_labels = Config.CLS_LABELS

        id_prefixes = {"arrow": "A", "arrowHead": "H", "blob": "B", "text": "T"}

        counts = {label: 0 for label in id_prefixes.keys()}

        img = Image.open(img_path)
        detections = model.predict(img, threshold=threshold)

        results = []
        for detection in detections:
            bbox = detection[0]
            conf = detection[2]
            cls_id = detection[3]
            cls_label = cls_labels[cls_id]

            prefix = id_prefixes.get(cls_label, "obj")
            unique_id = f"{prefix}{counts[cls_label]}"

            counts[cls_label] += 1

            result = {
                "id": unique_id,
                "class_ID": int(cls_id),
                "label": cls_label,
                "confidence": round(float(conf), 2),
                "bbox": bbox.tolist(),
            }
            results.append(result)
        return results

    @staticmethod
    def classify_one_image(model, img_path, threshold=0.5, device="cpu"):
        """Classify an image using YOLO."""
        img = Image.open(img_path)
        results = model(img, conf=threshold, device=device)

        for result in results:
            probs = result.probs
            top1_index = probs.top1
            top1_conf = probs.top1conf.item()
            class_name = result.names[top1_index]

            return {
                "class_ID": int(top1_index),
                "label": class_name,
                "confidence": round(float(top1_conf), 2),
            }

    @staticmethod
    def clean_ocr_text(text):
        """Clean OCR output text."""
        if not text:
            return ""
        text = re.sub(r"-\n", "", text)
        text = text.replace("\n", " ")
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    @staticmethod
    def ocr_text_objects(detections, img_path):
        """Extract text from detected text regions using OCR."""
        img = cv2.imread(img_path)
        texts = [o for o in detections if o["label"] == "text"]

        for t in texts:
            x1, y1, x2, y2 = map(int, t["bbox"])
            crop = img[y1:y2, x1:x2]
            raw_text = pytesseract.image_to_string(crop, lang="eng").strip()
            t["text"] = DetectionUtils.clean_ocr_text(raw_text)

        return texts

    @staticmethod
    def filter_oversized_containers(detections, contain_threshold=0.85):
        indices_to_remove = set()

        for i in range(len(detections)):
            for j in range(len(detections)):
                if i == j:
                    continue

                det_large = detections[i]
                det_small = detections[j]

                if det_large["class_ID"] != det_small["class_ID"]:
                    continue

                x1_min, y1_min, x1_max, y1_max = det_large["bbox"]
                x2_min, y2_min, x2_max, y2_max = det_small["bbox"]

                area_large = (x1_max - x1_min) * (y1_max - y1_min)
                area_small = (x2_max - x2_min) * (y2_max - y2_min)

                if area_large > area_small:
                    inter_ymin = max(y1_min, y2_min)
                    inter_xmin = max(x1_min, x2_min)
                    inter_ymax = min(y1_max, y2_max)
                    inter_xmax = min(x1_max, x2_max)

                    inter_w = max(0, inter_xmax - inter_xmin)
                    inter_h = max(0, inter_ymax - inter_ymin)
                    inter_area = inter_w * inter_h

                    if area_small > 0:
                        containment_ratio = inter_area / float(area_small)

                        if containment_ratio > contain_threshold:
                            indices_to_remove.add(i)

        filtered_detections = [
            det for i, det in enumerate(detections) if i not in indices_to_remove
        ]

        print(f"Filtered out {len(indices_to_remove)} oversized container boxes.")
        return filtered_detections
