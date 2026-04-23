import os
import json
import argparse
from src.pipeline import CombinedPipeline


def save_results_as_json(results, output_path):
    """Saves the pipeline results to a formatted JSON file."""
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

    blob_objects, text_objects, arrow_objects, arrowHead_objects = [], [], [], []

    # We default relation_label to unknown if config is not globally accessible here
    relation_label = "unknown"

    for obj in results.get("detections", []):
        formatted_obj = {
            "id": obj["id"],
            "bbox": obj["bbox"],
            "label": obj["label"],
            "score": obj["confidence"],
        }

        if obj["label"] == "blob":
            blob_objects.append(formatted_obj)
        elif obj["label"] == "text":
            formatted_obj["text"] = obj.get("text", "")
            text_objects.append(formatted_obj)
        elif obj["label"] == "arrow":
            arrow_objects.append(formatted_obj)
        elif obj["label"] == "arrowHead":
            arrowHead_objects.append(formatted_obj)

    json_setting["blobs"] = blob_objects
    json_setting["texts"] = text_objects
    json_setting["arrows"] = arrow_objects
    json_setting["arrowHeads"] = arrowHead_objects

    relationship_combines = []
    for rel in results.get("relationships", []):
        rel_id = f"{rel['source']}+{rel['target']}"
        relationship_combines.append(
            {
                "id": rel_id,
                "source": rel["source"],
                "target": rel["target"],
                "via_arrow": rel.get("via_arrow", ""),
                "relation_type": relation_label,
            }
        )

    if results.get("text_matching"):
        tm = results["text_matching"]
        if tm.get("blob_labels"):
            for text_blob in tm["blob_labels"]:
                text, blob = text_blob[0], text_blob[1]
                matching_id = f"{blob['id']}+{text['id']}"
                relationship_combines.append(
                    {
                        "id": matching_id,
                        "source": text["id"],
                        "target": blob["id"],
                        "via_arrow": "None",
                        "relation_type": "blob_label",
                    }
                )

    json_setting["relationships"] = relationship_combines
    json_setting["clip_results"] = results.get("clip_results", [])

    with open(output_path, "w") as f:
        json.dump(json_setting, f, indent=4)
    print(f"Results saved to {output_path}")


def main():
    # parser = argparse.ArgumentParser(
    #     description="Run the Combined Diagram Analysis Pipeline"
    # )
    # parser.add_argument(
    #     "--image", type=str, required=True, help="Path to the input image"
    # )
    # parser.add_argument(
    #     "--labels",
    #     type=str,
    #     default=None,
    #     help="Path to labels JSON (e.g. ovd_dict/lifeCycles_keepOnly.json)",
    # )
    # parser.add_argument(
    #     "--outdir",
    #     type=str,
    #     default="results",
    #     help="Directory to save output JSON and visualziations",
    # )
    # args = parser.parse_args()

    img_path = "/home/philine/Documents/KLTN/AI2D/Images/ai2d-all/ai2d/images/37.png"

    # 1. Initialize Pipeline
    print("Initializing CombinedPipeline...")
    pipeline = CombinedPipeline()

    # 2. Load custom blob labels if provided
    labels = "ovd_dict/lifeCycles_keepOnly.json"
    blob_labels = []
    if labels and os.path.exists(labels):
        with open(labels, "r") as f:
            labels_data = json.load(f)
            blob_labels = labels_data.get("keep", [])

    # 3. Create output directory if it doesn't exist
    os.makedirs("results", exist_ok=True)

    # 4. Process the Image
    results = pipeline.process_image(
        img_path,
        blob_labels=blob_labels,
        run_relationships=True,
        run_text_matching=True,
        run_clip=True,
        run_graph=True,
    )

    # 5. Output and Save
    # json_path = os.path.join(args.outdir, f"{os.path.basename(args.image)}.json")
    # save_results_as_json(results, json_path)

    # 6. Visualize (Saves images to the outdir)
    pipeline.visualize_results(results)

    print("\n--- PIPELINE COMPLETED ---")
    if results.get("relationships"):
        print(f"Relationships ({len(results['relationships'])}):")
        for rel in results["relationships"]:
            print(f"  {rel['source']} -> {rel['target']}")


if __name__ == "__main__":
    main()
