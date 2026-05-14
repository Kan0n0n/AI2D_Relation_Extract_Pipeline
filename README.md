# AI2D Visualizer & Relation Extractor

A complete pipeline for extracting blobs, text, arrows, and their semantic relationships from scientific diagrams. This project combines SAM (Segment Anything), CLIP, YOLO, RF-DETR, and Graph extraction into a unified tool with both a CLI and a Web interface.

## Features
* **Element Detection:** Detects blobs, arrows, text, and arrowheads using [YOLO / RF-DETR].
* **Semantic Segmentation:** Extracts precise masks using SAM (Segment Anything).
* **Relationship Extraction:** Extract Relationship based on it geometry features depend on different categories.
* **Scene Graph Generation:** Visualizes the extracted data as a node-based network.
* **Dual Interface:** Run it quickly in the terminal or launch the interactive web UI.

## Relation and JSON Structure
The pipeline classifies scientific diagrams into **8 distinct categories** and applies specific extraction logic based on the diagram's structural topology (Directed vs. Taxonomic). 

### 1. Directed Diagrams (Arrow-Based Group)
**Categories:** `lifeCycle`, `rockCycle`, `photosynthesisRespiration`, `foodChainsWebs`, `waterCNPCycle`, `circuits`

For diagrams depicting processes, flows, or cycles, the model extracts three primary relationship types:
* **Semantic Transitions (Blob ➔ Arrow ➔ Blob):** The core relationship mapping the flow between two visual entities. The relation's label is dynamically assigned based on the diagram's category (e.g., `transform_to` for life cycles, `eaten_by` for food chains). 
  * *Note: Transition names are fully customizable via the `RELATION_LABELS` configuration.*
* **Entity Labeling (Text ➔ Blob):** Links a text component to its corresponding visual entity, categorized as a `blob_label`.
* **Action Labeling (Text ➔ Arrow):** Associates an explanatory text component with a specific directional process, categorized as an `arrow_label`.

### 2. Taxonomic Diagrams (Non-Arrow-Based Group)
**Categories:** `partsOfA`, `typesOf`

For structural or hierarchical diagrams, the model bypasses arrow detection and focuses on direct semantic associations:
* **Structural Mapping (Text ➔ Blob):** Links textual descriptors directly to their corresponding spatial or categorical entities. The output relation is classified as `a_part_of` (for `partsOfA` diagrams) or `type_of` (for `typesOf` diagrams). 

---

## JSON Output Schema

The pipeline generates a structured JSON payload that closely adheres to the standard AI2D dataset format, making it easy to integrate with existing parsers and evaluation scripts.

```json
{
  "image_id": "string",            // The unique identifier/filename of the processed image
  "image_type": "string",          // The classified diagram category (e.g., "lifeCycle")
  "blobs": [                       // Array of detected visual entities
    {
      "id": "string",
      "bbox": [x1, y1, x2, y2],
      "label": "string",
      "score": 0.98
    }
  ],
  "arrows": [],                    // Array of detected arrow bodies
  "arrowHeads": [],                // Array of detected arrowheads
  "texts": [],                     // Array of detected text regions
  "relations": [ // Array of extracted source-to-target relationships
     {
        "source": "string_id",
        "target": "string_id",
        "via_arrow": "string_id",
        "relation_type": "string" // e.g., "eaten_by", "blob_label"
     }
  ],
  "clip_results": [] // Zero-shot semantic classification labels for blobs
}
```

---

## Installation & Setup

### 1. Prerequisites
Ensure you have Python 3.8+ installed. 

### 2. Clone and Install
* This is a usuable version but there might be update in the future for this project so you probably want to clone the project instead!
```bash
git clone https://github.com/Kan0n0n/AI2D_Relation_Extract_Pipeline.git
cd AI2D_Relation_Extract_Pipeline-main

# Create a virtual environment 
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`

# Install dependencies
pip install -r requirements.txt
```

### 3. Using the model 
* CLI Command:
```bash
python3 main.py --image [Path to image] --save (If you want to save the result) --outdir [Path for output directory]
```
* Web version:
  * The web version design as api so you can call it in your web to use however you like for the return structure please look in app.py as explain this sound like a pain uwu.
  * However you can use my demo by doing these step
    1. ```python3 app.py ```
    2. Open the ultilities.html inside web folder using live server or whatever you choice.
    3. Use it to your heart content!

### 4. Project Structure
```

├── app.py                  # Flask web server entry point
├── config.py               # The config 
├── main.py                 # CLI entry point
├── src/
│   ├── pipeline.py         # Main combined logic
│   ├── model_manager.py    # Auto-downloads and caches AI models
│   ├── visualizer.py       # Matplotlib and OpenCV drawing tools
│   └── ...                 # Other pipeline modules
├── weights/                # Directory for local model checkpoints
└── web/                    # Frontend HTML/CSS/JS for the Web Demo
```