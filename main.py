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
    parser = argparse.ArgumentParser(
        description="Run the Combined Diagram Analysis Pipeline"
    )
    parser.add_argument(
        "--image", type=str, required=True, help="Path to the input image"
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Whether to save the output JSON",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default="results",
        help="Directory to save output JSON",
    )
    args = parser.parse_args()
    if not args.image or not os.path.exists(args.image):
        print("Error: Valid image path is required.")
        return
    img_path = args.image

    # 1. Initialize Pipeline
    print("Initializing CombinedPipeline...")
    pipeline = CombinedPipeline()

    # 2. Process the Image
    results = pipeline.process_image(
        img_path,
        run_relationships=True,
        run_clip=True,
        run_graph=True,
    )

    # 3. Output and Save , Visualize Results
    if args.save:
        save_dir = args.outdir if args.outdir else "results"
        os.makedirs(save_dir, exist_ok=True)
        save_dir_image = os.path.join(save_dir, os.path.basename(args.image)[:-4])
        os.makedirs(save_dir_image, exist_ok=True)
        if not any(fname.endswith(".json") for fname in os.listdir(save_dir_image)):
            json_path = os.path.join(
                save_dir_image, f"{os.path.basename(args.image)}.json"
            )
            save_results_as_json(results, json_path)
        if not any(fname.endswith(".png") for fname in os.listdir(save_dir_image)):
            pipeline.visualize_results(results, save_dir=save_dir_image)
        else:
            pipeline.visualize_results(results, save_dir=None)
    else:
        pipeline.visualize_results(results)

    print("\n--- PIPELINE COMPLETED ---")
    if results.get("relationships"):
        print(f"Relationships ({len(results['relationships'])}):")
        for rel in results["relationships"]:
            print(f"  {rel['source']} -> {rel['target']}")


if __name__ == "__main__":
    main()
