from config import Config
from src.model_manager import ModelManager
from src.clip_based_blob_classify import CLIPImageClassifier, CLIPBlobClassifier
from src.relation_extractor import RelationshipExtractor
from src.text_region_classifier import TextRegionClassifier
from src.detection_utils import DetectionUtils
from src.knowledge_graph_generator import KnowledgeGraphGenerator
from src.visualizer import Visualizer
import cv2
import os
import json


class CombinedPipeline:
    def __init__(self, config=None):
        self.config = config or Config()
        self.models = ModelManager(self.config)
        self._clip_blob_classifier = None
        self._clip_category_classifier = None
        self._clip_rst_classifier = None

    @property
    def clip_classifier(self):
        if self._clip_blob_classifier is None:
            self._clip_classifier = CLIPBlobClassifier(
                self.models.clip_model, self.models.clip_processor, self.config.DEVICE
            )
        return self._clip_classifier

    @property
    def clip_category_classifier(self):
        if self._clip_category_classifier is None:
            self._clip_category_classifier = CLIPImageClassifier(
                self.models.clip_model,
                self.models.clip_processor,
                self.config.DEVICE,
                self.config.CATEGORIES,
            )
        return self._clip_category_classifier

    @property
    def clip_rst_classifier(self):
        if self._clip_rst_classifier is None:
            self._clip_rst_classifier = CLIPImageClassifier(
                self.models.clip_model,
                self.models.clip_processor,
                self.config.DEVICE,
                self.config.RST_CATEGORIES,
            )
        return self._clip_rst_classifier

    def filter_arrows_inside_blobs(self, detections, containment_threshold=0.6):
        blobs = [d for d in detections if d["label"] == "blob"]
        arrows = [d for d in detections if d["label"] == "arrow"]
        others = [d for d in detections if d["label"] not in ("blob", "arrow")]

        def containment(inner, outer):
            ix1, iy1, ix2, iy2 = inner
            ox1, oy1, ox2, oy2 = outer
            inter_w = max(0, min(ix2, ox2) - max(ix1, ox1))
            inter_h = max(0, min(iy2, oy2) - max(iy1, oy1))
            inner_area = (ix2 - ix1) * (iy2 - iy1)
            return (inter_w * inter_h) / inner_area if inner_area > 0 else 0.0

        def center_inside(arrow_bbox, blob_bbox):
            ax1, ay1, ax2, ay2 = arrow_bbox
            acx, acy = (ax1 + ax2) / 2, (ay1 + ay2) / 2
            bx1, by1, bx2, by2 = blob_bbox
            return bx1 <= acx <= bx2 and by1 <= acy <= by2

        kept, removed = [], []
        for arrow in arrows:
            inside_any = any(
                center_inside(arrow["bbox"], b["bbox"])
                or containment(arrow["bbox"], b["bbox"]) > containment_threshold
                for b in blobs
            )
            if inside_any:
                removed.append(arrow["id"])
            else:
                kept.append(arrow)

        if removed:
            print(f"Filtered {len(removed)} arrow(s) inside blobs: {removed}")

        return blobs + kept + others

    def process_image(
        self,
        image_path,
        run_relationships=True,
        run_clip=True,
        run_graph=True,
    ):
        """Process a single image through the complete pipeline."""
        print(f"Processing: {image_path}")

        results = {
            "image_path": image_path,
            "detections": None,
            "masks": None,
            "relationships": None,
            "text_matching": None,
            "clip_results": None,
            "classify_category": None,
            "classify_rst": None,
            "knowledge_graph": None,
            "config": self.config,
        }

        print("Running classify...")
        print(f"Classifying category...")

        classify_results = DetectionUtils.classify_one_image(
            self.models.classify_model,
            image_path,
            threshold=self.config.CLASSIFICATION_THRESHOLD,
            device=self.config.DEVICE,
        )
        results["classify_category"] = classify_results["label"]
        print(
            f"The category is: {classify_results["label"]} + confident is: {classify_results["confidence"]}"
        )

        blob_labels = None
        dict_path = self.config.LABELS_JSON.get(results["classify_category"])
        if dict_path and os.path.exists(dict_path):
            with open(dict_path, "r") as f:
                blob_labels = json.load(f).get("keep", [])
            print(
                f"Loaded {len(blob_labels)} blob labels for category '{results['classify_category']}'"
            )
        else:
            print(
                f"No blob labels found for category '{results['classify_category']}' at {dict_path}"
            )

        print(f"Classifying RST...")
        classify_results = self.clip_rst_classifier.classify(image_path)
        results["classify_rst"] = classify_results["label"]
        print(
            f"The RST is: {classify_results["label"]} + confident is: {classify_results["confidence"]}"
        )

        print("Running detection...")
        detections = DetectionUtils.detect_one_image(
            self.models.detection_model,
            image_path,
            self.config.DETECTION_THRESHOLD,
        )
        detections = self.filter_arrows_inside_blobs(detections)
        results["detections"] = detections
        print(f"Found {len(detections)} objects")
        print(f"{len([d for d in detections if d['label'] == 'blob'])} blobs")

        print("Generating SAM masks...")
        masks = self.models.sam_processor.generate_masks(image_path, detections)
        results["masks"] = masks

        blobs_list = []
        arrows_list = []

        for i, det in enumerate(detections):
            det["mask"] = masks[i]
            if det["label"] == "arrow":
                arrows_list.append(det)
            elif det["label"] == "blob":
                blobs_list.append(det)

        print("Running OCR on text regions...")
        ocr_results = DetectionUtils.ocr_text_objects(detections, image_path)
        print(ocr_results)
        formatted_texts = [
            {"id": t["id"], "bbox": t["bbox"], "content": t.get("text", "")}
            for t in ocr_results
        ]
        print(f"Extracted text from {len(formatted_texts)} regions")
        for t in formatted_texts:
            print(f"{t}")

        if run_relationships:
            print("Extracting relationships...")
            img = cv2.imread(image_path)
            h, w = img.shape[:2]
            image_name = os.path.basename(image_path)

            pipeline = RelationshipExtractor(
                detections=detections,
                image_name=image_name,
                category=results["classify_category"],
                classify_rst=results["classify_rst"],
                masks=masks,
                image_size=(h, w),
                image_path=image_path,
            )

            relationships = pipeline.process()
            results["relationships"] = relationships
            results["rst_pipeline"] = pipeline
            print(f"Found {len(relationships)} relationships")

        run_text_matching = True
        if (
            results["classify_category"] == "aPartOfs"
            or results["classify_category"] == "typesOf"
        ):
            run_text_matching = False
            print(
                f"Category is {results['classify_category']}, skipping text matching."
            )
        else:
            blobs_num = len(blobs_list)
            text_num = len(formatted_texts)
            arrow_num = len(arrows_list)
            total_num = blobs_num + arrow_num
            if blobs_num == 0:
                print("No blobs found, skipping text matching")
                run_text_matching = False
            elif total_num == 0 or text_num == 0:
                print("No objects or texts found, skipping text matching.")
                run_text_matching = False
            elif text_num / total_num < self.config.MIN_TEXT_RATIO:
                print(
                    f"Text ratio is {text_num / total_num:.2f}, skipping text matching."
                )
                run_text_matching = False
            else:
                print(f"running text matching.")

        if run_text_matching:
            print("Matching text to regions...")
            img = cv2.imread(image_path)

            text_matching = TextRegionClassifier.match_elements_globally(
                texts=formatted_texts,
                blobs=blobs_list,
                arrows=arrows_list,
                image_shape=img.shape,
                max_dist_blob=self.config.MAX_DIST_BLOB,
                max_dist_arrow_mid=self.config.MAX_DIST_ARROW_MID,
            )
            results["text_matching"] = text_matching
            print(f"    Blob labels: {len(text_matching['blob_labels'])}")
            print(f"    Arrow labels: {len(text_matching['arrow_labels'])}")
        else:
            results["text_matching"] = {}

        if run_clip and blob_labels:
            print("Running CLIP blob classification...")
            clip_results = self.clip_classifier.process_blobs(
                image_path, detections, blob_labels
            )
            results["clip_results"] = clip_results
            print(f"    Classified {len(clip_results)} blobs")

        if run_graph:
            print("Generating Knowledge Graph...")
            kg_gen = KnowledgeGraphGenerator()
            results["knowledge_graph"] = kg_gen.generate_graph(results)

        if results["knowledge_graph"]:
            node_mapping = {}
            nodes_data = results["knowledge_graph"].nodes(data=True)

            for node_id, data in nodes_data:
                if data.get("type") == "super_node":
                    components = data.get("components", node_id.split("+"))
                    node_label = components[0]
                    for comp in components:
                        node_mapping[comp] = node_label

            for relation in results["relationships"]:
                relation["source"] = node_mapping.get(
                    relation["source"], relation["source"]
                )
                relation["target"] = node_mapping.get(
                    relation["target"], relation["target"]
                )
        return results

    def visualize_results(self, results, save_dir=None):
        image_path = results["image_path"]

        if results.get("relationships") and results.get("rst_pipeline"):
            print("Visualizing relationships...")
            save_path = None
            if save_dir:
                save_path = os.path.join(
                    save_dir,
                    f"{os.path.basename(image_path).split('.')[0]}_relations.png",
                )
            Visualizer.visualize_relationships(
                image_path, results["rst_pipeline"], results["relationships"], save_path
            )

        if results.get("text_matching"):
            print("Visualizing text matching...")
            save_path = None
            if save_dir:
                save_path = os.path.join(
                    save_dir, f"{os.path.basename(image_path).split('.')[0]}_text.png"
                )
            Visualizer.visualize_text_matching(
                image_path, results["text_matching"], save_path
            )

        if results.get("clip_results"):
            print("Visualizing CLIP results...")
            save_path = None
            if save_dir:
                save_path = os.path.join(
                    save_dir, f"{os.path.basename(image_path).split('.')[0]}_clip.png"
                )
            Visualizer.visualize_clip_results(
                image_path, results["clip_results"], save_path
            )

        if results.get("knowledge_graph"):
            print("Visualizing knowledge graph...")
            save_path = None
            if save_dir:
                save_path = os.path.join(
                    save_dir, f"{os.path.basename(image_path).split('.')[0]}_graph.png"
                )
            Visualizer.visualize_knowledge_graph(results["knowledge_graph"], save_path)

    def visualize_debug_rays(self, results, save_path=None):
        image_path = results["image_path"]
        relations = results.get("relationships", [])
        rst_pipeline = results.get("rst_pipeline")

        if rst_pipeline is None:
            print("No pipeline found for debug visualization.")
            return

        Visualizer.visualize_debug_rays(image_path, rst_pipeline, relations, save_path)

    def visualize_debug_skeleton(self, results, save_path=None):
        image_path = results["image_path"]
        rst_pipeline = results.get("rst_pipeline")

        if rst_pipeline is None:
            print("No pipeline found for debug visualization.")
            return

        Visualizer.visualize_debug_skeleton(image_path, rst_pipeline, save_path)
