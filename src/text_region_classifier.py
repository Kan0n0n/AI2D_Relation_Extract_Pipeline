import cv2
import numpy as np
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment


class TextRegionClassifier:
    @staticmethod
    def iou(boxA, boxB):
        """Calculate Intersection over Union."""
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

        interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
        boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
        boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

        return interArea / float(boxAArea + boxBArea - interArea)

    @staticmethod
    def get_mask_distance(mask1, mask2, max_points=100, step=5):
        """Calculate minimum distance between two masks."""
        cnts1, _ = cv2.findContours(mask1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts2, _ = cv2.findContours(mask2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not cnts1 or not cnts2:
            return float("inf")

        pts1 = np.vstack(cnts1).squeeze()
        pts2 = np.vstack(cnts2).squeeze()

        if pts1.ndim == 1:
            pts1 = pts1[np.newaxis, :]
        if pts2.ndim == 1:
            pts2 = pts2[np.newaxis, :]

        if len(pts1) > max_points:
            pts1 = pts1[::step]
        if len(pts2) > max_points:
            pts2 = pts2[::step]

        return cdist(pts1, pts2).min()

    @staticmethod
    def get_arrow_midpoint(arrow_mask):
        """Calculate the midpoint of an arrow from its mask."""
        cnts, _ = cv2.findContours(
            arrow_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not cnts:
            return None

        M = cv2.moments(arrow_mask)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            return np.array([cx, cy])

        all_points = np.vstack(cnts).squeeze()
        if all_points.ndim == 1:
            all_points = all_points[np.newaxis, :]
        return np.mean(all_points, axis=0).astype(int)

    @staticmethod
    def get_text_center(text_bbox):
        """Get center point of text bounding box."""
        x1, y1, x2, y2 = text_bbox
        return np.array([(x1 + x2) / 2, (y1 + y2) / 2])

    @classmethod
    def match_elements_globally(
        cls, texts, blobs, arrows, image_shape, max_dist_blob=50, max_dist_arrow_mid=100
    ):
        """Match text elements to blobs and arrows. Unmatched texts go to misc."""
        if not texts:
            return {"blob_labels": [], "arrow_labels": [], "misc": []}

        h, w = image_shape[:2]
        blank = np.zeros((h, w), np.uint8)

        # Match texts to BLOBS
        blob_matches = []
        matched_to_blob = set()

        if blobs:
            cost_matrix_blob = np.full((len(texts), len(blobs)), float("inf"))

            for i, txt in enumerate(texts):
                x1, y1, x2, y2 = map(int, txt["bbox"])
                text_mask = blank.copy()
                cv2.rectangle(text_mask, (x1, y1), (x2, y2), 255, -1)

                for j, blob in enumerate(blobs):
                    cost_matrix_blob[i, j] = cls.get_mask_distance(
                        text_mask, blob["mask"]
                    )

            row_ind, col_ind = linear_sum_assignment(cost_matrix_blob)

            for r, c in zip(row_ind, col_ind):
                if cost_matrix_blob[r, c] < max_dist_blob:
                    matched_to_blob.add(r)
                    blob_matches.append((texts[r], blobs[c]))

        # Match remaining texts to ARROWS
        arrow_matches = []
        matched_to_arrow = set()
        unmatched_indices = [i for i in range(len(texts)) if i not in matched_to_blob]

        if arrows and unmatched_indices:
            cost_matrix_arrow = np.full(
                (len(unmatched_indices), len(arrows)), float("inf")
            )

            for i_idx, i in enumerate(unmatched_indices):
                text_center = cls.get_text_center(texts[i]["bbox"])

                for j, arrow in enumerate(arrows):
                    arrow_mid = cls.get_arrow_midpoint(arrow["mask"])
                    if arrow_mid is not None:
                        cost_matrix_arrow[i_idx, j] = np.linalg.norm(
                            text_center - arrow_mid
                        )

            row_ind, col_ind = linear_sum_assignment(cost_matrix_arrow)

            for r, c in zip(row_ind, col_ind):
                if cost_matrix_arrow[r, c] < max_dist_arrow_mid:
                    matched_to_arrow.add(unmatched_indices[r])
                    arrow_matches.append((texts[unmatched_indices[r]], arrows[c]))

        # Everything else is misc
        all_matched = matched_to_blob | matched_to_arrow
        misc = [txt for i, txt in enumerate(texts) if i not in all_matched]

        return {
            "blob_labels": blob_matches,
            "arrow_labels": arrow_matches,
            "misc": misc,
        }
