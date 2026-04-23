import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import networkx as nx
from PIL import Image
import numpy as np
import math
from scipy.spatial.distance import cdist
from src.text_region_classifier import TextRegionClassifier


class Visualizer:
    @staticmethod
    def visualize_relationships(image_path, pipeline, relations, save_path=None):
        img = cv2.imread(image_path)
        if img is None:
            print(f"Error: Cannot load image {image_path}")
            return

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        vis_img = img.copy()

        colors = {
            "blob": (0, 255, 0),
            "arrow": (255, 0, 0),
            "arrowHead": (255, 182, 193),
            "text": (0, 0, 255),
        }

        all_items = pipeline.blobs + pipeline.arrows + pipeline.texts
        id_to_center = {}

        for item in all_items:
            x1, y1, x2, y2 = map(int, item["bbox"])
            label = item["label"]
            obj_id = item.get("id", "N/A")
            color = colors.get(label, (255, 255, 255))

            cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
            id_to_center[obj_id] = (cx, cy)

            cv2.rectangle(vis_img, (x1, y1), (x2, y2), color, 2)

            text_caption = f"{label} {obj_id}"
            (w, h), _ = cv2.getTextSize(text_caption, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(vis_img, (x1, y1 - 20), (x1 + w, y1), color, -1)
            cv2.putText(
                vis_img,
                text_caption,
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
            )

        # Draw relations
        for rel in relations:
            src_id = rel["source"]
            tgt_id = rel["target"]

            if src_id in id_to_center and tgt_id in id_to_center:
                start_pt = id_to_center[src_id]
                end_pt = id_to_center[tgt_id]
                cv2.arrowedLine(
                    vis_img, start_pt, end_pt, (255, 215, 0), 3, tipLength=0.05
                )

        plt.figure(figsize=(12, 12))
        plt.imshow(vis_img)
        plt.axis("off")
        plt.title(f"RST: {pipeline.rst_category.upper()} | Relations: {len(relations)}")
        plt.show()

        if save_path:
            plt.imsave(save_path, vis_img)

    @staticmethod
    def visualize_text_matching(image_path, results, save_path=None):
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]
        blank_mask = np.zeros((h, w), dtype=np.uint8)

        for text_obj, target_obj in results.get("blob_labels", []):
            t_mask = blank_mask.copy()
            x1, y1, x2, y2 = map(int, text_obj["bbox"])
            cv2.rectangle(t_mask, (x1, y1), (x2, y2), 255, -1)

            target_mask = target_obj["mask"]

            cnts1, _ = cv2.findContours(
                t_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            cnts2, _ = cv2.findContours(
                target_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            if cnts1 and cnts2:
                pts1 = np.vstack(cnts1).squeeze()
                pts2 = np.vstack(cnts2).squeeze()

                if pts1.ndim == 1:
                    pts1 = pts1[np.newaxis, :]
                if pts2.ndim == 1:
                    pts2 = pts2[np.newaxis, :]

                dists = cdist(pts1, pts2)
                min_idx = np.unravel_index(np.argmin(dists), dists.shape)
                pt1 = tuple(map(int, pts1[min_idx[0]]))
                pt2 = tuple(map(int, pts2[min_idx[1]]))

                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                contours, _ = cv2.findContours(
                    target_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )
                cv2.drawContours(img, contours, -1, (0, 255, 0), 2)
                cv2.line(img, pt1, pt2, (255, 0, 255), 3)

        for text_obj, arrow_obj in results.get("arrow_labels", []):
            x1, y1, x2, y2 = map(int, text_obj["bbox"])
            text_center = TextRegionClassifier.get_text_center(text_obj["bbox"])
            arrow_mid = TextRegionClassifier.get_arrow_midpoint(arrow_obj["mask"])

            if arrow_mid is not None:
                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 255, 0), 2)
                contours, _ = cv2.findContours(
                    arrow_obj["mask"], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )
                cv2.drawContours(img, contours, -1, (255, 165, 0), 2)

                mid_pt = tuple(map(int, arrow_mid))
                text_pt = tuple(map(int, text_center))

                cv2.circle(img, mid_pt, 10, (0, 255, 255), -1)
                cv2.line(img, text_pt, mid_pt, (0, 255, 255), 3)

        colors = {
            "titles": (255, 0, 0),
            "captions": (128, 0, 128),
            "misc": (128, 128, 128),
        }
        for category, color in colors.items():
            for txt in results.get(category, []):
                x1, y1, x2, y2 = map(int, txt["bbox"])
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                cv2.putText(
                    img,
                    category.upper(),
                    (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    color,
                    1,
                )

        plt.figure(figsize=(15, 15))
        plt.imshow(img)
        plt.axis("off")
        plt.title("GREEN+MAGENTA=Blob | YELLOW+CYAN=Arrow")

        if save_path:
            plt.imsave(save_path, img)
        else:
            plt.show()
        plt.close()

    @staticmethod
    def visualize_clip_results(image_path, results, save_path=None, max_crops=12):
        img = Image.open(image_path).convert("RGB")

        display_results = results[:max_crops]
        n_crops = len(display_results)
        n_cols = 4
        n_rows = (n_crops + n_cols - 1) // n_cols + 1

        fig = plt.figure(figsize=(16, 4 * n_rows))
        gs = fig.add_gridspec(n_rows, n_cols, hspace=0.4, wspace=0.3)

        ax_main = fig.add_subplot(gs[0, :])
        ax_main.imshow(img)
        ax_main.axis("off")
        ax_main.set_title("Original Image with Detections", fontsize=14, weight="bold")

        for i, r in enumerate(display_results):
            bbox = r["bbox"]
            x1, y1, x2, y2 = bbox
            color = "green" if r["clip_conf"] > 0.5 else "orange"

            rect = patches.Rectangle(
                (x1, y1),
                x2 - x1,
                y2 - y1,
                linewidth=2,
                edgecolor=color,
                facecolor="none",
            )
            ax_main.add_patch(rect)
            ax_main.text(
                x1 + 5,
                y1 + 15,
                str(i + 1),
                color="white",
                fontsize=10,
                weight="bold",
                bbox=dict(boxstyle="circle", facecolor=color, alpha=0.8),
            )

        for i, r in enumerate(display_results):
            row = (i // n_cols) + 1
            col = i % n_cols
            ax = fig.add_subplot(gs[row, col])

            bbox = r["bbox"]
            x1, y1, x2, y2 = map(int, bbox)
            crop = img.crop((x1, y1, x2, y2))

            ax.imshow(crop)
            ax.axis("off")

            title = f"#{i+1}: {r['clip_label']}\n"
            title += f"CLIP: {r['clip_conf']:.2f} | DETR: {r['detr_conf']:.2f}"

            color = "green" if r["clip_conf"] > 0.5 else "orange"
            ax.set_title(title, fontsize=9, weight="bold", color=color)

        plt.suptitle(
            f"Blob Classification Results - {n_crops}/{len(results)} blobs",
            fontsize=16,
            weight="bold",
            y=0.995,
        )
        print(f"Visualizing CLIP results for {n_crops} blobs...")

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"Saved CLIP visualization to {save_path}")
        else:
            plt.show()

        plt.close(fig)

    @staticmethod
    def visualize_debug_rays(image_path, pipeline, relations, save_path=None):
        """Visualize raycast debug information."""
        img = cv2.imread(image_path)
        if img is None:
            return
        vis_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        for item in pipeline.blobs + pipeline.texts + pipeline.arrows:
            x1, y1, x2, y2 = map(int, item["bbox"])
            cv2.rectangle(vis_img, (x1, y1), (x2, y2), (200, 200, 200), 1)

        for ray in pipeline.debug_rays:
            print(ray)
            ox, oy = ray["origin"]
            dx, dy = ray["vec"]
            dist = ray["dist"]
            angle_rad = math.radians(ray["angle"])

            end_x = int(ox + dx * dist)
            end_y = int(oy + dy * dist)
            cv2.line(vis_img, (ox, oy), (end_x, end_y), (0, 255, 255), 2)
            cv2.circle(vis_img, (ox, oy), 5, (0, 255, 255), -1)

            dx_p = dx * math.cos(angle_rad) - dy * math.sin(angle_rad)
            dy_p = dx * math.sin(angle_rad) + dy * math.cos(angle_rad)
            cv2.line(
                vis_img,
                (ox, oy),
                (int(ox + dx_p * dist), int(oy + dy_p * dist)),
                (0, 200, 200),
                1,
                cv2.LINE_AA,
            )

            dx_n = dx * math.cos(-angle_rad) - dy * math.sin(-angle_rad)
            dy_n = dx * math.sin(-angle_rad) + dy * math.cos(-angle_rad)
            cv2.line(
                vis_img,
                (ox, oy),
                (int(ox + dx_n * dist), int(oy + dy_n * dist)),
                (0, 200, 200),
                1,
                cv2.LINE_AA,
            )

            if ray["hit_id"]:
                hit_obj = next(
                    (
                        x
                        for x in pipeline.blobs + pipeline.texts
                        if x["id"] == ray["hit_id"]
                    ),
                    None,
                )
                if hit_obj:
                    hx1, hy1, hx2, hy2 = map(int, hit_obj["bbox"])
                    cv2.rectangle(vis_img, (hx1, hy1), (hx2, hy2), (255, 0, 255), 3)

        id_to_center = {
            x["id"]: pipeline._get_center(x["bbox"])
            for x in pipeline.blobs + pipeline.texts
        }
        for rel in relations:
            s, t = rel["source"], rel["target"]
            if s in id_to_center and t in id_to_center:
                s_pt = tuple(map(int, id_to_center[s]))
                e_pt = tuple(map(int, id_to_center[t]))
                cv2.arrowedLine(vis_img, s_pt, e_pt, (255, 215, 0), 4, tipLength=0.05)

        plt.figure(figsize=(12, 12))
        plt.imshow(vis_img)
        plt.title("Yellow: Relations | Cyan: RayCast Search | Magenta: RayCast Hits")
        plt.axis("off")

        if save_path:
            plt.imsave(save_path, vis_img)
        else:
            plt.show()

        plt.close()

    @staticmethod
    def visualize_debug_skeleton(image_path, pipeline, save_path=None):
        """Visualize skeleton debug information."""
        img = cv2.imread(image_path)
        if img is None:
            return
        vis_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        for dbg in pipeline.debug_skeletons:
            if dbg.get("head_box"):
                hx1, hy1, hx2, hy2 = map(int, dbg["head_box"])
                cv2.rectangle(vis_img, (hx1, hy1), (hx2, hy2), (0, 255, 0), 1)

            for sx, sy in dbg.get("pixels", []):
                cv2.circle(vis_img, (int(sx), int(sy)), 1, (255, 255, 255), -1)

            for ey, ex in dbg.get("candidates", []):
                cv2.circle(vis_img, (int(ex), int(ey)), 3, (0, 100, 255), -1)

            if dbg.get("chosen_tail"):
                ty, tx = dbg["chosen_tail"]
                cv2.rectangle(
                    vis_img,
                    (int(tx) - 4, int(ty) - 4),
                    (int(tx) + 4, int(ty) + 4),
                    (255, 0, 0),
                    -1,
                )

            if dbg.get("chosen_head"):
                hy, hx = dbg["chosen_head"]
                cv2.circle(vis_img, (int(hx), int(hy)), 5, (0, 255, 0), -1)
                cv2.circle(vis_img, (int(hx), int(hy)), 7, (0, 0, 0), 1)

        plt.figure(figsize=(15, 15))
        plt.imshow(vis_img)
        plt.title("White=Skeleton | Orange=Endpoints | Green=Head | Red=Tail")
        plt.axis("off")
        plt.show()

        if save_path:
            plt.imsave(save_path, vis_img)

    @staticmethod
    def visualize_knowledge_graph(G, save_path=None):
        plt.figure(figsize=(15, 12))
        pos = nx.kamada_kawai_layout(G)

        node_types = nx.get_node_attributes(G, "type")
        node_labels = nx.get_node_attributes(G, "label")

        color_map = {
            "root": "#FFD700",
            "super_node": "#FF4500",
            "blob": "#90EE90",
            "text": "#87CEEB",
        }

        colors = [color_map.get(node_types.get(node), "#333333") for node in G.nodes()]

        nx.draw_networkx_nodes(G, pos, node_size=3000, node_color=colors, alpha=0.85)
        nx.draw_networkx_edges(
            G, pos, width=1.5, alpha=0.4, edge_color="gray", arrows=True
        )
        nx.draw_networkx_labels(
            G, pos, labels=node_labels, font_size=8, font_weight="bold"
        )

        edge_labels = nx.get_edge_attributes(G, "relation")
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=7)

        plt.title("Hierarchical Life Cycle Knowledge Graph")
        plt.axis("off")
        print("Visualizing knowledge graph...")

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        else:
            plt.show()
        plt.close()
