import networkx as nx
import os


class KnowledgeGraphGenerator:
    @staticmethod
    def generate_graph(results):
        G = nx.DiGraph()
        image_name = os.path.basename(results["image_path"])
        image_id = image_name.split(".")[0]

        root_id = "Image_Root"
        G.add_node(root_id, label=f"I{image_id}", type="root")

        detections = results.get("detections", [])
        text_matching = results.get("text_matching", {})
        blob_matches = text_matching.get("blob_labels", [])
        arrow_matches = text_matching.get("arrow_labels", [])
        relationships = results.get("relationships", [])
        image_type = results["classify_category"]
        edge_label = results["config"].RELATION_LABELS.get(image_type, "unknown")

        processed_blob_ids = set()
        processed_text_ids = set()

        for i, det in enumerate(detections):
            label_type = det.get("label")
            det_id = str(det.get("id", "None ID"))

            if label_type == "blob":
                G.add_node(f"{det_id}", label=f"{det_id}", type="blob")
            elif label_type == "text":
                content = det.get("text", "")
                G.add_node(f"{det_id}", label=f"{det_id}", type="text")

        id_to_super = {}

        for text_obj, blob_obj in blob_matches:
            print(text_obj)
            b_id = str(blob_obj.get("id"))
            t_id = str(text_obj.get("id"))
            super_id = f"{b_id}+{t_id}"

            id_to_super[b_id] = super_id
            id_to_super[t_id] = super_id

            G.add_node(super_id, label=f"{super_id}", type="super_node")

            G.add_edge(root_id, super_id, relation="has a")

            G.add_edge(super_id, f"{b_id}", relation="")
            G.add_edge(super_id, f"{t_id}", relation="")

            processed_blob_ids.add(b_id)
            processed_text_ids.add(t_id)

        for i, det in enumerate(detections):
            det_id = str(det.get("id", i))
            if det["label"] == "blob" and det_id not in processed_blob_ids:
                G.add_edge(root_id, f"{det_id}", relation="has a")

            if det["label"] == "text" and det_id not in processed_text_ids:
                is_arrow_label = any(
                    str(arrow_obj.get("id")) == det_id for _, arrow_obj in arrow_matches
                )
                if not is_arrow_label:
                    G.add_edge(root_id, f"{det_id}", relation="has a")

        arrow_id_to_text = {
            str(a.get("id")): t.get("text", "") for t, a in arrow_matches
        }

        for rel in relationships:
            src_node = f"{rel['source']}"
            tgt_node = f"{rel['target']}"
            if src_node in id_to_super:
                src_node = id_to_super[src_node]
            if tgt_node in id_to_super:
                tgt_node = id_to_super[tgt_node]
            arrow_id = str(rel.get("via_arrow", ""))

            if G.has_node(src_node) and G.has_node(tgt_node):
                print(f"Adding edge between {src_node} and {tgt_node}")
                edge_desc = arrow_id_to_text.get(arrow_id, edge_label)
                G.add_edge(src_node, tgt_node, relation=edge_desc)
        return G
