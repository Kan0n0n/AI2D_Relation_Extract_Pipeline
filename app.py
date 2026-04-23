import os
import json
import time
from flask import Flask, request, jsonify, url_for
from flask_cors import CORS
from src.pipeline import CombinedPipeline

app = Flask(__name__)
CORS(app)

print("Loading AI Models... Please wait.")
pipeline = CombinedPipeline()
os.makedirs("uploads", exist_ok=True)
os.makedirs("static/outputs", exist_ok=True)


def format_results_as_json(results):
    json_setting = {
        "image_id": os.path.basename(results["image_path"]),
        "image_type": results["classify_category"],
        "blobs": [],
        "texts": [],
        "arrows": [],
        "arrowHeads": [],
        "relationships": [],
        "clip_results": [],
    }
    blob_objects = []
    text_objects = []
    arrow_objects = []
    arrowHead_objects = []

    relation_label = "unknown"
    config = results.get("config", None)
    if not config:
        print("Warning: Config not provided, using default relation label 'unknown'")
    else:
        relation_label = config.RELATION_LABELS.get(
            results["classify_category"], "unknown"
        )

    for obj in results["detections"]:
        if obj["label"] == "blob":
            blob_objects.append(
                {
                    "id": obj["id"],
                    "bbox": obj["bbox"],
                    "label": obj["label"],
                    "score": obj["confidence"],
                }
            )
        if obj["label"] == "text":
            text_objects.append(
                {
                    "id": obj["id"],
                    "bbox": obj["bbox"],
                    "label": obj["label"],
                    "score": obj["confidence"],
                    "text": obj.get("text", ""),
                }
            )
        if obj["label"] == "arrow":
            arrow_objects.append(
                {
                    "id": obj["id"],
                    "bbox": obj["bbox"],
                    "label": obj["label"],
                    "score": obj["confidence"],
                }
            )
        if obj["label"] == "arrowHead":
            arrowHead_objects.append(
                {
                    "id": obj["id"],
                    "bbox": obj["bbox"],
                    "label": obj["label"],
                    "score": obj["confidence"],
                }
            )
    json_setting["blobs"] = blob_objects
    json_setting["texts"] = text_objects
    json_setting["arrows"] = arrow_objects
    json_setting["arrowHeads"] = arrowHead_objects
    relationship_combines = []
    for rel in results["relationships"]:
        print(rel)
        rel_id = f"{rel['source']}+{rel['target']}"
        relationship_combines.append(
            {
                "id": rel_id,
                "source": rel["source"],
                "target": rel["target"],
                "via_arrow": rel["via_arrow"],
                "relation_type:": relation_label,
            }
        )
    if results["text_matching"]:
        tm = results["text_matching"]
        if tm["blob_labels"]:
            for blob_matching in tm["blob_labels"]:
                text = blob_matching[0]
                blob = blob_matching[1]
                matching_id = f"{blob['id']}+{text['id']}"
                relationship_combines.append(
                    {
                        "id": matching_id,
                        "source": text["id"],
                        "target": blob["id"],
                        "via_arrow": "None",
                        "relation_type:": "blob_label",
                    }
                )
        if tm["arrow_labels"]:
            relationship_combines.append(
                {"id": "not handle yet", "relation_type": "an arrow label rel"}
            )
    json_setting["relationships"] = relationship_combines
    json_setting["clip_results"] = results["clip_results"]
    return json_setting


@app.route("/api/analyze", methods=["POST"])
def analyze_image():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    image_path = os.path.join("uploads", file.filename)
    file.save(image_path)

    print(f"Processing {image_path}...")
    results = pipeline.process_image(
        image_path,
        run_relationships=True,
        run_clip=True,
        run_graph=True,
    )

    base_filename = os.path.splitext(file.filename)[0]
    pipeline.visualize_results(results, save_dir="static/outputs")

    final_json_data = format_results_as_json(results)

    return jsonify(
        {
            "message": "Success",
            "json_data": final_json_data,
            "images": {
                "relationships": url_for(
                    "static",
                    filename=f"outputs/{base_filename}_relations.png",
                    _external=True,
                ),
                "text_matching": url_for(
                    "static",
                    filename=f"outputs/{base_filename}_text.png",
                    _external=True,
                ),
                "clip_results": url_for(
                    "static",
                    filename=f"outputs/{base_filename}_clip.png",
                    _external=True,
                ),
                "knowledge_graph": url_for(
                    "static",
                    filename=f"outputs/{base_filename}_graph.png",
                    _external=True,
                ),
            },
        }
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
