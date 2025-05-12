import json
import zipfile
import requests
from pathlib import Path


DATA_DIR = Path("data")
ANN_DIR = DATA_DIR / "annotations"
IMAGE_DIR = DATA_DIR / "images"
OUTPUT_JSON = DATA_DIR / "annotations.json"

ZIPS = {
    "objects_v1.json.zip": "objects.json",
    "relationships_v1.json.zip": "relationships.json",
    "image_data.json.zip": "image_data.json",
}

SUBSET_SIZE  = 500  # how many images to include

def unzip_if_needed(zip_name: str, json_name: str):
    zip_path  = ANN_DIR / zip_name
    json_path = ANN_DIR / json_name
    if zip_path.exists() and not json_path.exists():
        print(f"Unzipping {zip_name}")
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(ANN_DIR)
        print(f"Extracted -> {json_name}")

def main():
    ANN_DIR.mkdir(parents=True, exist_ok=True)
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    for zip_name, json_name in ZIPS.items():
        unzip_if_needed(zip_name, json_name)

    print("Loading image_data.json …")
    with open(ANN_DIR / "image_data.json", "r", encoding="utf-8") as f:
        img_data_list = json.load(f)
    id2url = {
        int(item["image_id"]): item["url"]
        for item in img_data_list
        if "image_id" in item and "url" in item
    }
    print(f"Loaded URLs for {len(id2url)} images")

    print(f"Reading first {SUBSET_SIZE} entries from objects.json ...")
    with open(ANN_DIR / "objects.json", "r", encoding="utf-8") as f:
        objects_entries = json.load(f)  

    selected_entries = objects_entries[:SUBSET_SIZE]
    subset_ids = [entry["id"] for entry in selected_entries]
    print(f"Selected {len(subset_ids)} image IDs: {subset_ids[0]} to {subset_ids[-1]}")

    objs_for = {eid: entry["objects"] for eid, entry in zip(subset_ids, selected_entries)}

    print("Loading relationships.json …")
    with open(ANN_DIR / "relationships.json", "r", encoding="utf-8") as f:
        rel_entries = json.load(f)  

    rels_for = {eid: [] for eid in subset_ids}
    i_r = 0
    for entry in rel_entries:
        print(f'iteration {i_r}')
        img_id = entry["id"]
        if img_id in rels_for:
            rels_for[img_id] = entry.get("relationships", [])
        i_r+=1

    print("Building annotations.json ...")
    annotations  = []
    download_info = []

    i_a = 0
    for img_id in subset_ids:
        print(f'iteration {i_a}')
        obj_list = objs_for.get(img_id, [])
        rel_list = rels_for.get(img_id, [])

        boxes, labels, idx_map = [], [], {}
        for idx, o in enumerate(obj_list):
            x = o["x"]
            y = o["y"]
            w = o["w"]
            h = o["h"]
            boxes.append([x, y, x + w, y + h])
            label = o["names"][0] if o.get("names") else "__UNK__"
            labels.append(label)
            idx_map[o["id"]] = idx

        relations = []
        for r in rel_list:
            s_id, o_id, pred = r['subject']["id"], r["object"]['id'], r["predicate"]
            if s_id in idx_map and o_id in idx_map:
                relations.append([idx_map[s_id], idx_map[o_id], pred])

        img_path = IMAGE_DIR / f"{img_id}.jpg"
        annotations.append({
            "file_name": str(img_path.resolve()),
            "boxes": boxes,
            "labels": labels,
            "relations": relations
        })

        url = id2url.get(img_id)
        if url:
            download_info.append((img_id, url))
        i_a+=1

    print("Downloading subset images ...")
    for img_id, url in download_info:
        dest = IMAGE_DIR / f"{img_id}.jpg"
        if dest.exists():
            continue
        try:
            resp = requests.get(url, stream=True, timeout=10)
            resp.raise_for_status()
            with open(dest, "wb") as fp:
                for chunk in resp.iter_content(1024):
                    fp.write(chunk)
        except Exception as e:
            print(f"Failed to download {img_id}: {e}")

    print("Pruning annotations for missing images ...")
    valid_annotations = []
    for ann in annotations:
        img_path = Path(ann["file_name"])
        if img_path.exists():
            valid_annotations.append(ann)
        else:
            print(f"Skipping annotation for missing {img_path.name}")
    annotations = valid_annotations

    print("Writing", OUTPUT_JSON)
    with open(OUTPUT_JSON, "w", encoding="utf-8") as out:
        json.dump(annotations, out, indent=2)

    print("Done. Images in:", IMAGE_DIR)
    print("Annotations in:", OUTPUT_JSON)

if __name__ == "__main__":
    main()
