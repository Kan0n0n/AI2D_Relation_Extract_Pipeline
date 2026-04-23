import math
from skimage.morphology import skeletonize
from scipy.ndimage import convolve
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
from config import Config
import numpy as np
import cv2


class RelationshipExtractor:
    """Extracting relationships from diagrams."""

    def __init__(
        self,
        detections,
        image_name,
        category,
        classify_rst,
        masks=None,
        image_size=(1000, 1000),
        image_path=None,
        config=None,
    ):
        self.detections = detections
        self.image_name = image_name
        self.masks = masks
        self.H, self.W = image_size
        self.image_path = image_path

        self.config = config or Config()
        self.rst_category = classify_rst
        self.relation_label = self.config.RELATION_LABELS.get(category, "unknown")

        self.category = category

        self.blobs = []
        self.arrows = []
        self.texts = []
        self.arrowHeads = []

        for i, d in enumerate(detections):
            item = d.copy()
            item["index"] = i
            item["id"] = d["id"]

            if d["label"] == "blob":
                self.blobs.append(item)
            elif d["label"] == "arrow":
                self.arrows.append(item)
            elif d["label"] == "text":
                self.texts.append(item)
            elif d["label"] == "arrowHead":
                self.arrowHeads.append(item)

        self.debug_rays = []
        self.debug_skeletons = []

    def _get_center(self, bbox):
        return ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)

    def _rect_distance(self, box1, box2):
        """Calculate shortest distance between two rectangles."""
        x1a, y1a, x2a, y2a = box1
        x1b, y1b, x2b, y2b = box2

        if x2b < x1a:
            dist_x = x1a - x2b
        elif x1b > x2a:
            dist_x = x1b - x2a
        else:
            dist_x = 0

        if y2b < y1a:
            dist_y = y1a - y2b
        elif y1b > y2a:
            dist_y = y1b - y2a
        else:
            dist_y = 0

        return math.sqrt(dist_x**2 + dist_y**2)

    def _get_mask(self, item):
        if self.masks is not None and item["index"] < len(self.masks):
            return self.masks[item["index"]]
        return None

    def _find_skeleton_endpoints(self, skel):
        """Find endpoints of a skeleton."""
        kernel = np.array([[1, 1, 1], [1, 10, 1], [1, 1, 1]])
        neighbors = convolve(skel.astype(int), kernel, mode="constant", cval=0)
        ys, xs = np.where(neighbors == 11)
        points = list(zip(ys, xs))

        if len(points) > 2:
            max_dist = 0
            best_pair = (
                (points[0], points[1]) if len(points) >= 2 else (points[0], points[0])
            )
            for i in range(len(points)):
                for j in range(i + 1, len(points)):
                    p1, p2 = points[i], points[j]
                    d = (p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2
                    if d > max_dist:
                        max_dist = d
                        best_pair = (p1, p2)
            return list(best_pair)

        return points

    def _find_matching_arrowhead(self, arrow):
        """Find an arrowHead detection inside an arrow."""
        ax1, ay1, ax2, ay2 = arrow["bbox"]
        buffer = 15

        for head in self.arrowHeads:
            hx1, hy1, hx2, hy2 = head["bbox"]
            hcx, hcy = (hx1 + hx2) / 2, (hy1 + hy2) / 2

            if (ax1 - buffer < hcx < ax2 + buffer) and (
                ay1 - buffer < hcy < ay2 + buffer
            ):
                return head["bbox"]
        return None

    def _analyze_arrow_direction(self, arrow):
        """Analyze arrow mask to determine tail and head positions."""
        mask = self._get_mask(arrow)
        if mask is None:
            return None, None

        skel = skeletonize(mask)
        endpoints = self._find_skeleton_endpoints(skel)

        if len(endpoints) < 2:
            return None, None

        ys, xs = np.where(mask > 0)
        cy, cx = np.mean(ys), np.mean(xs)

        head_box = self._find_matching_arrowhead(arrow)

        p_tail = None
        p_head = None

        dbg = {
            "pixels": list(zip(xs, ys)),
            "candidates": endpoints,
            "head_box": head_box,
        }

        if head_box:
            hx1, hy1, hx2, hy2 = head_box
            hcx, hcy = (hx1 + hx2) / 2, (hy1 + hy2) / 2

            p_tail = max(endpoints, key=lambda p: (p[0] - hcy) ** 2 + (p[1] - hcx) ** 2)

            contours, _ = cv2.findContours(
                mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
            )
            if contours:
                cnt_points = []
                for cnt in contours:
                    for pt in cnt:
                        px, py = pt[0]
                        buffer = 5
                        if (hx1 - buffer < px < hx2 + buffer) and (
                            hy1 - buffer < py < hy2 + buffer
                        ):
                            cnt_points.append((py, px))

                if cnt_points:
                    p_head = max(
                        cnt_points,
                        key=lambda p: (p[0] - p_tail[0]) ** 2 + (p[1] - p_tail[1]) ** 2,
                    )
                else:
                    p_head = max(
                        endpoints,
                        key=lambda p: (p[0] - p_tail[0]) ** 2 + (p[1] - p_tail[1]) ** 2,
                    )
            else:
                p_head = max(
                    endpoints,
                    key=lambda p: (p[0] - p_tail[0]) ** 2 + (p[1] - p_tail[1]) ** 2,
                )
        else:
            p1 = endpoints[0]
            p2 = max(endpoints, key=lambda p: (p[0] - p1[0]) ** 2 + (p[1] - p1[1]) ** 2)
            p_tail = p1
            p_head = p2

        dbg["chosen_tail"] = p_tail
        dbg["chosen_head"] = p_head
        self.debug_skeletons.append(dbg)
        return p_tail, p_head

    def _get_closest_with_dist(self, point, nodes, exclude_id=None):
        """Get closest node to a point."""
        py, px = point
        best_node = None
        min_dist = float("inf")

        for n in nodes:
            if exclude_id and n["id"] == exclude_id:
                continue

            nx1, ny1, nx2, ny2 = n["bbox"]
            dx = max(nx1 - px, 0, px - nx2)
            dy = max(ny1 - py, 0, py - ny2)
            dist = math.sqrt(dx * dx + dy * dy)

            if dist < min_dist:
                min_dist = dist
                best_node = n

        return best_node, min_dist

    def _find_priority_node(self, point, blobs, texts, threshold=30, exclude_id=None):
        """Find nearest blob or text to a point."""
        closest_blob, dist_blob = self._get_closest_with_dist(point, blobs, exclude_id)
        if closest_blob and dist_blob < threshold:
            return closest_blob

        closest_text, dist_text = self._get_closest_with_dist(point, texts, exclude_id)
        if closest_text and dist_text < threshold:
            return closest_text

        return None

    def _determine_direction_clockwise(self, node_A, node_B, center):
        """Determine direction for clockwise ordering."""
        cx, cy = center
        ax, ay = self._get_center(node_A["bbox"])
        bx, by = self._get_center(node_B["bbox"])

        ang_A = math.atan2(ay - cy, ax - cx)
        ang_B = math.atan2(by - cy, bx - cx)

        if ang_A < 0:
            ang_A += 2 * math.pi
        if ang_B < 0:
            ang_B += 2 * math.pi

        diff = (ang_B - ang_A) % (2 * math.pi)

        if diff < math.pi:
            return node_A, node_B
        else:
            return node_B, node_A

    def _extract_wire_mask(self, arrow, dark_threshold=80):
        if self.image_path is None:
            return None
        image_bgr = cv2.imread(self.image_path)
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        h, w = image_rgb.shape[:2]

        x1, y1, x2, y2 = map(int, arrow["bbox"])
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        crop = image_rgb[y1:y2, x1:x2]
        if crop.size == 0:
            return None

        gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
        _, wire_mask = cv2.threshold(gray, dark_threshold, 255, cv2.THRESH_BINARY_INV)
        kernel = np.ones((2, 2), np.uint8)
        wire_mask = cv2.morphologyEx(wire_mask, cv2.MORPH_OPEN, kernel)

        full_mask = np.zeros((h, w), dtype=np.uint8)
        full_mask[y1:y2, x1:x2] = wire_mask
        return full_mask

    def process(self):
        """Main processing method - routes to appropriate solver."""
        print(f"[{self.image_name}] Strategy: {self.rst_category.upper()}")

        solver_mapping = {
            # APPROACH 1: Arrow-driven Flow
            "foodChainsWebs": self._solve_flow_approach,
            "lifeCycles": self._solve_flow_approach,
            "waterCNPCycle": self._solve_flow_approach,
            "rockCycle": self._solve_flow_approach,
            "photosynthesisRespiration": self._solve_flow_approach,
            # APPROACH 2: partsOfA approach
            "partsOfA": self._solve_parts_of_approach,
            # APPROACH 3: typesOf approach
            "typesOf": self._solve_typesOf_approach,
            # APPROACH 4: circuit approach
            "circuits": self._solve_circuit_topology,
        }

        solver_method = solver_mapping.get(self.category, self._solve_generic_fallback)
        return solver_method()

    def _merge_fragmented_arrows(self, arrows, merge_distance=30):
        if len(arrows) <= 1:
            return arrows

        def bbox_gap(a, b):
            ax1, ay1, ax2, ay2 = a["bbox"]
            bx1, by1, bx2, by2 = b["bbox"]
            dx = max(ax1 - bx2, bx1 - ax2, 0)
            dy = max(ay1 - by2, by1 - ay2, 0)
            return max(dx, dy)

        def same_orientation(a, b):
            ax1, ay1, ax2, ay2 = a["bbox"]
            bx1, by1, bx2, by2 = b["bbox"]
            return ((ax2 - ax1) > (ay2 - ay1)) == ((bx2 - bx1) > (by2 - by1))

        def overlaps_on_axis(a, b):
            ax1, ay1, ax2, ay2 = a["bbox"]
            bx1, by1, bx2, by2 = b["bbox"]
            a_horiz = (ax2 - ax1) > (ay2 - ay1)
            if a_horiz:
                return not (ay2 < by1 or by2 < ay1)
            else:
                return not (ax2 < bx1 or bx2 < ax1)

        def similar_size(a, b, ratio_threshold=3.0):
            """Reject merge if one bbox is much larger than the other."""
            ax1, ay1, ax2, ay2 = a["bbox"]
            bx1, by1, bx2, by2 = b["bbox"]
            area_a = (ax2 - ax1) * (ay2 - ay1)
            area_b = (bx2 - bx1) * (by2 - by1)
            if area_a == 0 or area_b == 0:
                return False
            ratio = max(area_a, area_b) / min(area_a, area_b)
            return ratio < ratio_threshold

        def neither_contains_other(a, b, threshold=0.7):
            """Don't merge if one bbox largely contains the other — that's overlap, not fragmentation."""

            def containment(inner, outer):
                ix1, iy1, ix2, iy2 = inner
                ox1, oy1, ox2, oy2 = outer
                inter_w = max(0, min(ix2, ox2) - max(ix1, ox1))
                inter_h = max(0, min(iy2, oy2) - max(iy1, oy1))
                inner_area = (ix2 - ix1) * (iy2 - iy1)
                return (inter_w * inter_h) / inner_area if inner_area > 0 else 0.0

            return (
                containment(a["bbox"], b["bbox"]) < threshold
                and containment(b["bbox"], a["bbox"]) < threshold
            )

        def _blob_between(a, b):
            """Return True if any blob bbox intersects the gap region between two arrows."""
            ax1, ay1, ax2, ay2 = a["bbox"]
            bx1, by1, bx2, by2 = b["bbox"]

            # gap region = union bbox of the two arrows
            gap_x1 = min(ax1, bx1)
            gap_y1 = min(ay1, by1)
            gap_x2 = max(ax2, bx2)
            gap_y2 = max(ay2, by2)

            for blob in self.blobs:
                blx1, bly1, blx2, bly2 = blob["bbox"]
                # check if blob overlaps the gap region
                if blx1 < gap_x2 and blx2 > gap_x1 and bly1 < gap_y2 and bly2 > gap_y1:
                    return True
            return False

        parent = list(range(len(arrows)))

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a, b):
            parent[find(a)] = find(b)

        for i in range(len(arrows)):
            for j in range(i + 1, len(arrows)):
                a, b = arrows[i], arrows[j]
                if (
                    same_orientation(a, b)
                    and overlaps_on_axis(a, b)
                    and similar_size(a, b)  # ← new
                    and neither_contains_other(a, b)
                    and not _blob_between(a, b)  # ← new
                    and bbox_gap(a, b) < merge_distance
                ):
                    union(i, j)
                    print(f"[{self.image_name}] Merging {a['id']} + {b['id']}")

        groups = {}
        for i in range(len(arrows)):
            groups.setdefault(find(i), []).append(arrows[i])

        merged = []
        for group in groups.values():
            if len(group) == 1:
                merged.append(group[0])
            else:
                x1 = min(a["bbox"][0] for a in group)
                y1 = min(a["bbox"][1] for a in group)
                x2 = max(a["bbox"][2] for a in group)
                y2 = max(a["bbox"][3] for a in group)
                merged.append(
                    {
                        "id": f"merged_{'_'.join(a['id'] for a in group)}",
                        "label": "arrow",
                        "class_ID": group[0].get("class_ID", 0),
                        "confidence": max(a.get("confidence", 0) for a in group),
                        "bbox": (x1, y1, x2, y2),
                        "merged_from": [a["id"] for a in group],
                    }
                )

        return merged

    def _solve_no_arrow_fallback(self, blobs):
        rels = []
        seen_pairs = set()

        if len(blobs) < 2:
            return rels

        if self.image_path:
            image_bgr = cv2.imread(self.image_path)
            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
            _, wire_bin = cv2.threshold(gray, 80, 255, cv2.THRESH_BINARY_INV)

            def are_connected_by_wire(b1, b2, samples=10):
                """Sample points along the line between two blobs, check for wire pixels."""
                cx1 = int((b1["bbox"][0] + b1["bbox"][2]) / 2)
                cy1 = int((b1["bbox"][1] + b1["bbox"][3]) / 2)
                cx2 = int((b2["bbox"][0] + b2["bbox"][2]) / 2)
                cy2 = int((b2["bbox"][1] + b2["bbox"][3]) / 2)
                hits = 0
                for t in np.linspace(0, 1, samples):
                    px = int(cx1 + t * (cx2 - cx1))
                    py = int(cy1 + t * (cy2 - cy1))
                    if 0 <= py < wire_bin.shape[0] and 0 <= px < wire_bin.shape[1]:
                        if wire_bin[py, px] > 0:
                            hits += 1
                return hits > samples * 0.3  # 30% of sampled points are wire

            for i, b1 in enumerate(blobs):
                for j, b2 in enumerate(blobs):
                    if i >= j:
                        continue
                    pair_key = tuple(sorted([b1["id"], b2["id"]]))
                    if pair_key in seen_pairs:
                        continue
                    if are_connected_by_wire(b1, b2):
                        seen_pairs.add(pair_key)
                        rels.append(
                            {
                                "source": b1["id"],
                                "target": b2["id"],
                                "relation": self.relation_label,
                                "via_arrow": "inferred",
                            }
                        )
                        print(
                            f"[{self.image_name}] Wire-inferred: "
                            f"{b1['id']} ↔ {b2['id']}"
                        )
        else:
            # pure proximity fallback — connect N nearest blob pairs
            PROX_THRESHOLD = min(self.W, self.H) * 0.4
            for i, b1 in enumerate(blobs):
                for j, b2 in enumerate(blobs):
                    if i >= j:
                        continue
                    dist = self._rect_distance(b1["bbox"], b2["bbox"])
                    if dist < PROX_THRESHOLD:
                        pair_key = tuple(sorted([b1["id"], b2["id"]]))
                        seen_pairs.add(pair_key)
                        rels.append(
                            {
                                "source": b1["id"],
                                "target": b2["id"],
                                "relation": self.relation_label,
                                "via_arrow": "inferred_proximity",
                            }
                        )

        return rels

    def _solve_circuit_topology(self):
        rels = []

        # merge fragmented arrows before clustering
        merged_arrows = self._merge_fragmented_arrows(self.arrows)
        if len(merged_arrows) != len(self.arrows):
            print(
                f"[{self.image_name}] Arrows after merge: "
                f"{len(self.arrows)} → {len(merged_arrows)}"
            )

        # temporarily swap in merged arrows for clustering
        original_arrows = self.arrows
        self.arrows = merged_arrows

        groups = self._cluster_circuit_groups()
        print(f"[{self.image_name}] Found {len(groups)} circuit group(s)")

        for g_idx, group in enumerate(groups):
            group_blobs = group["blobs"]
            group_arrows = group["arrows"]
            print(
                f"[{self.image_name}] Group {g_idx}: "
                f"{len(group_blobs)} blob(s), {len(group_arrows)} arrow(s)"
            )

            if not group_arrows:
                print(
                    f"[{self.image_name}] Group {g_idx}: "
                    f"no arrows — using wire inference fallback"
                )
                group_rels = self._solve_no_arrow_fallback(group_blobs)
            else:
                group_rels = self._solve_single_circuit(group_blobs, group_arrows)

            rels.extend(group_rels)

        self.arrows = original_arrows  # restore
        print(f"[{self.image_name}] circuit_topology total: {len(rels)} connection(s).")
        return rels

    def _cluster_circuit_groups(self, expansion=40):
        """
        Group blobs and arrows into separate circuits using bbox overlap/proximity.
        Expands each bbox by `expansion` px before checking overlap so nearby
        components that don't literally touch still get merged.
        """
        all_nodes = [{"item": b, "type": "blob"} for b in self.blobs] + [
            {"item": a, "type": "arrow"} for a in self.arrows
        ]

        if not all_nodes:
            return []

        parent = list(range(len(all_nodes)))

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a, b):
            parent[find(a)] = find(b)

        def expand(bbox):
            x1, y1, x2, y2 = bbox
            return (x1 - expansion, y1 - expansion, x2 + expansion, y2 + expansion)

        def overlaps(b1, b2):
            ax1, ay1, ax2, ay2 = expand(b1)
            bx1, by1, bx2, by2 = expand(b2)
            return ax1 < bx2 and ax2 > bx1 and ay1 < by2 and ay2 > by1

        for i in range(len(all_nodes)):
            for j in range(i + 1, len(all_nodes)):
                if overlaps(all_nodes[i]["item"]["bbox"], all_nodes[j]["item"]["bbox"]):
                    union(i, j)

        # collect groups
        groups_map = {}
        for i, node in enumerate(all_nodes):
            root = find(i)
            if root not in groups_map:
                groups_map[root] = {"blobs": [], "arrows": []}
            if node["type"] == "blob":
                groups_map[root]["blobs"].append(node["item"])
            else:
                groups_map[root]["arrows"].append(node["item"])

        # only keep groups that have at least 1 blob and 1 arrow
        valid = [
            g
            for g in groups_map.values()
            if len(g["blobs"]) >= 1 and len(g["arrows"]) >= 1
        ]
        return valid

    def _solve_single_circuit(self, blobs, arrows, endpoint_threshold=80):
        """Solve one isolated circuit group."""
        rels = []
        seen_pairs = set()

        def nearest_blob(point, blobs, arrow_bbox, exclude_id=None):
            py, px = point
            ax1, ay1, ax2, ay2 = arrow_bbox
            is_vertical = (ay2 - ay1) > (ax2 - ax1)
            best, best_score = None, float("inf")
            for b in blobs:
                if exclude_id and b["id"] == exclude_id:
                    continue
                bx1, by1, bx2, by2 = b["bbox"]
                dx = max(bx1 - px, 0, px - bx2)
                dy = max(by1 - py, 0, py - by2)
                dist = math.sqrt(dx * dx + dy * dy)
                if dist > endpoint_threshold:
                    continue
                bcx, bcy = (bx1 + bx2) / 2, (by1 + by2) / 2
                penalty = abs(bcx - px) * 0.5 if is_vertical else abs(bcy - py) * 0.5
                score = dist + penalty
                if score < best_score:
                    best_score, best = score, b
            return best, best_score

        for arrow in arrows:
            wire_mask = self._extract_wire_mask(arrow)
            if wire_mask is None:
                wire_mask = self._get_mask(arrow)
            if wire_mask is None:
                continue

            skel = skeletonize(wire_mask > 0)
            endpoints = self._find_skeleton_endpoints(skel)

            blob_a, blob_b = None, None

            if len(endpoints) >= 2:
                ep1 = endpoints[0]
                ep2 = max(
                    endpoints[1:],
                    key=lambda p: (p[0] - ep1[0]) ** 2 + (p[1] - ep1[1]) ** 2,
                )

                blob_a, _ = nearest_blob(ep1, blobs, arrow["bbox"])
                blob_b, _ = nearest_blob(
                    ep2,
                    blobs,
                    arrow["bbox"],
                    exclude_id=blob_a["id"] if blob_a else None,
                )

            # bbox center fallback — only within this group's blobs
            if not (blob_a and blob_b):
                ax1, ay1, ax2, ay2 = arrow["bbox"]
                acx, acy = (ax1 + ax2) / 2, (ay1 + ay2) / 2
                sorted_blobs = sorted(
                    blobs,
                    key=lambda b: math.dist(
                        (
                            (b["bbox"][0] + b["bbox"][2]) / 2,
                            (b["bbox"][1] + b["bbox"][3]) / 2,
                        ),
                        (acx, acy),
                    ),
                )
                if len(sorted_blobs) >= 2:
                    blob_a, blob_b = sorted_blobs[0], sorted_blobs[1]
                    print(f"[{self.image_name}] Arrow {arrow['id']} used bbox fallback")

            if not (blob_a and blob_b) or blob_a["id"] == blob_b["id"]:
                print(f"[{self.image_name}] Arrow {arrow['id']} skipped")
                continue

            pair_key = tuple(sorted([blob_a["id"], blob_b["id"]]))
            if pair_key in seen_pairs:
                continue
            seen_pairs.add(pair_key)

            rels.append(
                {
                    "source": blob_a["id"],
                    "target": blob_b["id"],
                    "relation": self.relation_label,
                    "via_arrow": arrow["id"],
                }
            )
            print(
                f"[{self.image_name}] {blob_a['id']} ↔ {blob_b['id']} "
                f"via {arrow['id']}"
            )

        return rels

    def _solve_generic_fallback(self):
        """Fallback method triggered when a category has no mapped solver approach."""
        return [
            {
                "error": f"Unknown category '{self.category}'",
                "source": None,
                "target": None,
            }
        ]

    def _match_blobs_to_labels(self, exclude_text_ids=None):
        exclude_text_ids = exclude_text_ids or set()
        candidates = [t for t in self.texts if t["id"] not in exclude_text_ids]

        if not candidates or not self.blobs:
            return {b["id"]: [] for b in self.blobs}

        LOCAL_THRESHOLD = min(self.W, self.H) * 0.15
        blob_map = {b["id"]: [] for b in self.blobs}
        unassigned_texts = list(candidates)

        def center_dist(text, blob):
            tcx, tcy = self._get_center(text["bbox"])
            bcx, bcy = self._get_center(blob["bbox"])
            return math.sqrt((tcx - bcx) ** 2 + (tcy - bcy) ** 2)

        for text in list(unassigned_texts):
            nearest = min(self.blobs, key=lambda b: center_dist(text, b))
            d = center_dist(text, nearest)
            if d < LOCAL_THRESHOLD:
                blob_map[nearest["id"]].append(text)
                unassigned_texts.remove(text)
                print(
                    f"[{self.image_name}] {text['id']} → {nearest['id']} "
                    f"(center dist={d:.1f})"
                )
            else:
                print(
                    f"[{self.image_name}] {text['id']} unassigned "
                    f"(nearest center {d:.1f}px > threshold)"
                )

        unmatched_blobs = [b for b in self.blobs if not blob_map[b["id"]]]
        if unmatched_blobs:
            print(
                f"[{self.image_name}] Coverage repair for: "
                f"{[b['id'] for b in unmatched_blobs]}"
            )

        for blob in unmatched_blobs:
            if unassigned_texts:
                best_text = min(unassigned_texts, key=lambda t: center_dist(t, blob))
                blob_map[blob["id"]].append(best_text)
                unassigned_texts.remove(best_text)
                print(f"[{self.image_name}] Repair: {best_text['id']} → {blob['id']}")
            else:
                best_text, donor = self._find_steal_candidate(blob, blob_map)
                if best_text and donor:
                    blob_map[donor["id"]].remove(best_text)
                    blob_map[blob["id"]].append(best_text)
                    print(
                        f"[{self.image_name}] Stole {best_text['id']}: "
                        f"{donor['id']} → {blob['id']}"
                    )
                    if not blob_map[donor["id"]]:
                        unmatched_blobs.append(donor)

        blob_map = self._normalize_blob_text_counts(blob_map)

        for blob_id, texts in blob_map.items():
            print(
                f"[{self.image_name}] {blob_id} → "
                f"{[t.get('value', t['id']) for t in texts]}"
            )

        return blob_map

    def _normalize_blob_text_counts(self, blob_map):
        from collections import Counter

        counts = [len(texts) for texts in blob_map.values()]
        if not counts:
            return blob_map

        mode_count = Counter(counts).most_common(1)[0][0]
        print(f"[{self.image_name}] Target text count per blob: {mode_count}")

        for blob in self.blobs:
            texts = blob_map[blob["id"]]
            if len(texts) <= mode_count:
                continue

            bx1, by1, bx2, by2 = blob["bbox"]

            def _sort_key(text):
                tx1, ty1, tx2, ty2 = text["bbox"]
                tcx, tcy = (tx1 + tx2) / 2, (ty1 + ty2) / 2

                # Primary: center inside blob bbox (0) beats touching edge (1)
                is_inside = int(not (bx1 <= tcx <= bx2 and by1 <= tcy <= by2))

                # Secondary: rect distance (closer is better)
                dist = self._rect_distance(blob["bbox"], text["bbox"])

                return (is_inside, dist)

            texts_sorted = sorted(texts, key=_sort_key)
            removed = texts_sorted[mode_count:]
            blob_map[blob["id"]] = texts_sorted[:mode_count]

            print(
                f"[{self.image_name}] Trimmed {blob['id']}: "
                f"removed {[t.get('value', t['id']) for t in removed]}, "
                f"kept {[t.get('value', t['id']) for t in blob_map[blob['id']]]}"
            )

        return blob_map

    def _find_steal_candidate(self, target_blob, blob_map):
        best_text = None
        best_donor = None
        best_dist = float("inf")

        for blob in self.blobs:
            if blob["id"] == target_blob["id"]:
                continue
            if len(blob_map[blob["id"]]) <= 1:
                continue  # don't leave donor empty

            for text in blob_map[blob["id"]]:
                dist = self._rect_distance(target_blob["bbox"], text["bbox"])
                if dist < best_dist:
                    best_dist = dist
                    best_text = text
                    best_donor = blob

        return best_text, best_donor

    def _get_text_groups(self, texts, merge_threshold=10):
        if not texts:
            return []

        # Union-Find for grouping
        parent = {t["id"]: t["id"] for t in texts}

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(x, y):
            parent[find(x)] = find(y)

        # Merge texts that are within threshold of each other
        for i, t1 in enumerate(texts):
            for j, t2 in enumerate(texts):
                if i >= j:
                    continue
                if self._rect_distance(t1["bbox"], t2["bbox"]) < merge_threshold:
                    union(t1["id"], t2["id"])

        # Group texts by root
        groups = {}
        for t in texts:
            root = find(t["id"])
            if root not in groups:
                groups[root] = []
            groups[root].append(t)

        # Build merged bboxes for each group
        result = []
        for root, group_texts in groups.items():
            x1 = min(t["bbox"][0] for t in group_texts)
            y1 = min(t["bbox"][1] for t in group_texts)
            x2 = max(t["bbox"][2] for t in group_texts)
            y2 = max(t["bbox"][3] for t in group_texts)
            result.append(
                {
                    "ids": {t["id"] for t in group_texts},
                    "bbox": (x1, y1, x2, y2),
                    "texts": group_texts,
                }
            )

        return result

    def _is_likely_title(self, text):
        import re

        tx1, ty1, tx2, ty2 = text["bbox"]
        tcx = (tx1 + tx2) / 2
        tcy = (ty1 + ty2) / 2
        value = text.get("value", "").strip()

        if re.match(r"^[a-zA-Z0-9]\)", value) or re.match(r"^\d+\.", value):
            return True

        all_groups = self._get_text_groups(self.texts)
        my_group = next((g for g in all_groups if text["id"] in g["ids"]), None)

        check_bbox = my_group["bbox"] if my_group else (tx1, ty1, tx2, ty2)
        cx1, cy1, cx2, cy2 = check_bbox
        merged_width = cx2 - cx1
        merged_tcx = (cx1 + cx2) / 2
        merged_ty1 = cy1

        if merged_width > self.W * 0.4:
            return True

        in_top_zone = (cy1 + cy2) / 2 < self.H * config.TITLE_CENTER_FRACTION
        is_centered = self.W * 0.3 < merged_tcx < self.W * 0.7

        if in_top_zone and is_centered:
            # Blob proximity still overrides positional rules
            for blob in self.blobs:
                bx1, by1, bx2, by2 = blob["bbox"]
                if bx1 <= tcx <= bx2 and by1 <= tcy <= by2:
                    return False
                if (
                    self._rect_distance(text["bbox"], blob["bbox"])
                    < config.TITLE_BLOB_PROXIMITY
                ):
                    return False
            return True

        return False

    def _solve_typesOf_approach(self):
        rels = []

        if not self.blobs:
            print(f"[{self.image_name}] No blobs detected, skipping.")
            return rels

        if not self.texts:
            print(f"[{self.image_name}] No texts detected, skipping.")
            return rels

        title_ids = {t["id"] for t in self.texts if self._is_likely_title(t)}
        if title_ids:
            print(
                f"[{self.image_name}] Excluded {len(title_ids)} title text(s): "
                f"{title_ids}"
            )

        blob_map = self._match_blobs_to_labels(exclude_text_ids=title_ids)

        for blob in self.blobs:
            matched_texts = blob_map.get(blob["id"], [])

            if not matched_texts:
                print(
                    f"[{self.image_name}] WARNING: blob {blob['id']} "
                    f"has no label after repair — skipping."
                )
                continue

            for text in matched_texts:
                rels.append(
                    {
                        "source": blob["id"],
                        "target": text["id"],
                        "relation": self.relation_label,
                        "via_arrow": "none",
                    }
                )

        print(f"[{self.image_name}] typesOf: {len(rels)} relation(s) extracted.")
        return rels

    def _solve_parts_of_approach(self):
        rels = []

        if not self.blobs or not self.texts:
            return rels

        def _nearest_node(source, candidates, limit=None):
            sc = self._get_center(source["bbox"])
            best, best_dist = None, float("inf")
            for c in candidates:
                d = math.dist(sc, self._get_center(c["bbox"]))
                if d < best_dist:
                    best_dist = d
                    best = c
            if limit is not None and best_dist > limit:
                return None
            return best

        # ── No-arrow shortcut ───────────────────────────────────────────────────
        if not self.arrows:
            print(
                f"[{self.image_name}] No arrows found — linking every text directly to nearest blob."
            )
            for text in self.texts:
                blob = _nearest_node(text, self.blobs)
                if blob is None:
                    continue
                rels.append(
                    {
                        "source": text["id"],
                        "target": blob["id"],
                        "relation": self.relation_label,
                        "via_arrow": None,
                    }
                )
            return rels

        # ── Step 1: assign each text to its nearest arrow ──────────────────────
        text_to_arrow = {}
        for text in self.texts:
            arrow = _nearest_node(text, self.arrows, limit=200)
            if arrow is None:
                print(
                    f"[{self.image_name}] Text '{text['id']}' has no arrow within range — skipped."
                )
                continue
            text_to_arrow[text["id"]] = arrow

        # ── Step 2: assign each arrow to its nearest blob ───────────────────────
        arrow_to_blob = {}
        for arrow in self.arrows:
            blob = _nearest_node(arrow, self.blobs)
            arrow_to_blob[arrow["id"]] = blob

        # ── Step 3: chain text → arrow → blob and emit relations ────────────────
        for text in self.texts:
            arrow = text_to_arrow.get(text["id"])
            if arrow is None:
                continue

            blob = arrow_to_blob.get(arrow["id"])
            if blob is None:
                print(
                    f"[{self.image_name}] Arrow '{arrow['id']}' could not be matched to any blob — skipped."
                )
                continue

            rels.append(
                {
                    "source": text["id"],
                    "target": blob["id"],
                    "relation": self.relation_label,
                    "via_arrow": arrow["id"],
                }
            )

        return rels

    def _solve_flow_approach(self):
        """Approach 1: Solves diagrams using arrows, falling back to layout topology."""
        rels = self._solve_directed_flow()
        if rels:
            return rels

        if self.rst_category == "cycle":
            return self._solve_cycle()
        elif self.rst_category == "network":
            return self._solve_network()
        elif self.rst_category == "horizontal":
            return self._solve_linear(axis="x")
        elif self.rst_category == "vertical":
            return self._solve_linear(axis="y")

        raise ValueError(
            f"Extraction Error: No arrows detected and unknown layout strategy "
            f"'{self.rst_category}' for flow diagram '{self.image_name}'."
        )

    def _solve_directed_flow(self):
        """Solve using arrow direction analysis with raycast fallback."""
        relations = []
        nodes = self.blobs + self.texts

        if not self.arrows or not self.arrowHeads:
            return None

        for arrow in self.arrows:
            p_tail, p_head = self._analyze_arrow_direction(arrow)
            if p_tail is None or p_head is None:
                continue

            source_node = self._find_priority_node(
                p_tail, self.blobs, self.texts, threshold=30
            )

            if source_node is None:
                direction = self._get_arrow_tip_direction(arrow, p_tail, is_tail=True)
                if direction:
                    source_node = self._find_node_by_raycast(
                        p_tail, direction, self.blobs, dist_limit=400, cone_angle_deg=60
                    )
                    if source_node is None:
                        source_node = self._find_node_by_raycast(
                            p_tail,
                            direction,
                            self.texts,
                            dist_limit=400,
                            cone_angle_deg=60,
                        )

            ex_id = source_node["id"] if source_node else None
            target_node = self._find_priority_node(
                p_head, self.blobs, self.texts, threshold=30, exclude_id=ex_id
            )

            if target_node is None:
                direction = self._get_arrow_tip_direction(arrow, p_head)
                if direction:
                    target_node = self._find_node_by_raycast(
                        p_head, direction, self.blobs, dist_limit=300, cone_angle_deg=45
                    )
                    if target_node is None:
                        target_node = self._find_node_by_raycast(
                            p_head,
                            direction,
                            self.texts,
                            dist_limit=300,
                            cone_angle_deg=45,
                        )

            if target_node is None:
                fallback_node, dist = self._get_closest_with_dist(
                    p_head, self.blobs, exclude_id=ex_id
                )
                if fallback_node and dist < 100:
                    target_node = fallback_node

            if source_node and target_node and source_node["id"] != target_node["id"]:
                exists = any(
                    r["source"] == source_node["id"]
                    and r["target"] == target_node["id"]
                    for r in relations
                )
                if not exists:
                    relations.append(
                        {
                            "source": source_node["id"],
                            "target": target_node["id"],
                            "relation": self.relation_label,
                            "via_arrow": arrow["id"],
                        }
                    )

        return relations if relations else None

    def _get_arrow_tip_direction(self, arrow, tip_point, is_tail=False, depth=15):
        """Calculate direction vector at arrow tip/tail using skeleton analysis."""
        mask = self._get_mask(arrow)
        if mask is None:
            return None

        skel = skeletonize(mask)

        ys, xs = np.where(skel > 0)

        if len(ys) < 2:
            return None

        self.debug_skeletons.append(
            {
                "pixels": list(zip(xs, ys)),
                "candidates": [tip_point],
                "chosen_head": tip_point,
            }
        )

        ty, tx = tip_point

        dists_sq = (ys - ty) ** 2 + (xs - tx) ** 2

        min_dist_sq = 10**2
        max_dist_sq = 40**2

        selection = (dists_sq > min_dist_sq) & (dists_sq < max_dist_sq)

        selected_ys = ys[selection]
        selected_xs = xs[selection]

        if len(selected_ys) == 0:
            selected_ys = ys
            selected_xs = xs

        mean_y = np.mean(selected_ys)
        mean_x = np.mean(selected_xs)

        dy = ty - mean_y
        dx = tx - mean_x

        mag = math.sqrt(dx * dx + dy * dy)
        if mag == 0:
            return None

        vx, vy = dx / mag, dy / mag

        return (vx, vy)

    def _find_node_by_raycast(
        self, origin, direction, nodes, dist_limit=300, cone_angle_deg=45
    ):
        """Find node by raycasting from origin in given direction within a cone."""
        oy, ox = origin
        dx, dy = direction

        best_node = None
        min_dist = dist_limit

        end_x = ox + dx * dist_limit
        end_y = oy + dy * dist_limit
        ray_segment = ((ox, oy), (end_x, end_y))

        ray_debug = {
            "origin": (int(ox), int(oy)),
            "vec": (dx, dy),
            "dist": dist_limit,
            "angle": cone_angle_deg,
            "hit_id": None,
        }

        threshold_cos = math.cos(math.radians(cone_angle_deg))

        for n in nodes:
            nx1, ny1, nx2, ny2 = n["bbox"]
            cx, cy = self._get_center(n["bbox"])

            vec_x = cx - ox
            vec_y = cy - oy
            dist_center = math.sqrt(vec_x**2 + vec_y**2)

            is_in_cone = False
            if 0 < dist_center <= dist_limit:
                dot_prod = (dx * (vec_x / dist_center)) + (dy * (vec_y / dist_center))
                if dot_prod > threshold_cos:
                    is_in_cone = True

            intersects, dist_intersect = self._line_intersects_box(
                ray_segment, (nx1, ny1, nx2, ny2)
            )

            if is_in_cone or intersects:
                valid_dist = dist_intersect if intersects else dist_center

                if valid_dist < min_dist:
                    min_dist = valid_dist
                    best_node = n

        if best_node:
            ray_debug["hit_id"] = best_node["id"]

        self.debug_rays.append(ray_debug)
        return best_node

    def _line_intersects_box(self, segment, box):
        """
        Checks if line segment (p1, p2) intersects rectangle (x1, y1, x2, y2).
        Returns (True/False, distance_to_intersection)
        """
        (x1, y1), (x2, y2) = segment
        bx1, by1, bx2, by2 = box

        if (
            max(x1, x2) < bx1
            or min(x1, x2) > bx2
            or max(y1, y2) < by1
            or min(y1, y2) > by2
        ):
            return False, float("inf")

        borders = [
            ((bx1, by1), (bx2, by1)),
            ((bx1, by2), (bx2, by2)),
            ((bx1, by1), (bx1, by2)),
            ((bx2, by1), (bx2, by2)),
        ]

        closest_dist = float("inf")
        hit = False

        for b_start, b_end in borders:
            pt = self._line_intersection((x1, y1), (x2, y2), b_start, b_end)
            if pt:
                d = math.sqrt((pt[0] - x1) ** 2 + (pt[1] - y1) ** 2)
                if d < closest_dist:
                    closest_dist = d
                    hit = True

        if (bx1 <= x1 <= bx2) and (by1 <= y1 <= by2):
            return True, 0

        return hit, closest_dist

    def _line_intersection(self, line1, line2, line3, line4):
        """
        Finds intersection between Line((x1,y1)->(x2,y2)) and Segment((x3,y3)->(x4,y4)).
        """
        x1, y1 = line1
        x2, y2 = line2
        x3, y3 = line3
        x4, y4 = line4

        denom = (y4 - y3) * (x2 - x1) - (x4 - x3) * (y2 - y1)
        if denom == 0:
            return None

        ua = ((x4 - x3) * (y1 - y3) - (y4 - y3) * (x1 - x3)) / denom
        ub = ((x2 - x1) * (y1 - y3) - (y2 - y1) * (x1 - x3)) / denom

        if 0 <= ua <= 1 and 0 <= ub <= 1:
            return (x1 + ua * (x2 - x1), y1 + ua * (y2 - y1))
        return None

    def _solve_cycle(self):
        """Solve cycle diagrams using topology."""
        nodes = self.blobs if self.blobs else self.texts
        if len(nodes) < 2:
            return []

        all_centers = [self._get_center(n["bbox"]) for n in nodes]
        cx = np.mean([c[0] for c in all_centers])
        cy = np.mean([c[1] for c in all_centers])

        def get_angle(n):
            nx, ny = self._get_center(n["bbox"])
            return math.atan2(ny - cy, nx - cx)

        sorted_nodes = sorted(nodes, key=get_angle)

        rels = []
        count = len(sorted_nodes)
        for i in range(count):
            rels.append(
                {
                    "source": sorted_nodes[i]["id"],
                    "target": sorted_nodes[(i + 1) % count]["id"],
                    "relation": self.relation_label,
                    "via_arrow": "unknown",
                }
            )

        return rels

    def _solve_linear(self, axis="x"):
        """Solve linear (horizontal/vertical) diagrams."""
        if len(self.blobs) < 2:
            return []

        idx = 0 if axis == "x" else 1
        threshold = self.W * 0.4 if axis == "x" else self.H * 0.4

        sorted_blobs = sorted(
            self.blobs, key=lambda b: self._get_center(b["bbox"])[idx]
        )

        rels = []
        for i in range(len(sorted_blobs) - 1):
            u, v = sorted_blobs[i], sorted_blobs[i + 1]
            dist = math.sqrt(
                sum(
                    (a - b) ** 2
                    for a, b in zip(
                        self._get_center(u["bbox"]), self._get_center(v["bbox"])
                    )
                )
            )
            if dist < threshold:
                rels.append(
                    {
                        "source": u["id"],
                        "target": v["id"],
                        "relation": self.relation_label,
                        "via_arrow": "unknown",
                    }
                )

        return rels

    def _solve_network(self):
        """Solve network diagrams using skeleton analysis."""
        rels = []
        if self.masks is None:
            return []

        for arrow in self.arrows:
            mask = self._get_mask(arrow)
            if mask is None:
                continue

            skel = skeletonize(mask)
            endpoints = self._find_skeleton_endpoints(skel)

            if len(endpoints) < 2:
                continue

            touched_blobs = []
            for pt in [endpoints[0], endpoints[-1]]:
                py, px = pt
                for b in self.blobs:
                    b_mask = self._get_mask(b)
                    if b_mask is None:
                        continue

                    H, W = b_mask.shape
                    radius = 15
                    y1, y2 = max(0, py - radius), min(H, py + radius)
                    x1, x2 = max(0, px - radius), min(W, px + radius)

                    if np.any(b_mask[y1:y2, x1:x2]):
                        touched_blobs.append(b["id"])
                        break

            if len(touched_blobs) == 2 and touched_blobs[0] != touched_blobs[1]:
                u = next(b for b in self.blobs if b["id"] == touched_blobs[0])
                v = next(b for b in self.blobs if b["id"] == touched_blobs[1])

                if self._get_center(u["bbox"])[1] > self._get_center(v["bbox"])[1]:
                    rels.append(
                        {
                            "source": u["id"],
                            "target": v["id"],
                            "relation": self.relation_label,
                            "via_arrow": "unknown",
                        }
                    )
                else:
                    rels.append(
                        {
                            "source": v["id"],
                            "target": u["id"],
                            "relation": self.relation_label,
                            "via_arrow": "unknown",
                        }
                    )

        return rels
