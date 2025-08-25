import os
import xml.etree.ElementTree as ET
from sklearn.model_selection import train_test_split

# paths
annotations_path = "dataset/annotations"
images_path = "dataset/images"
labels_path = "dataset/labels"

# make sure output folders exist
os.makedirs(f"{labels_path}/train", exist_ok=True)
os.makedirs(f"{labels_path}/val", exist_ok=True)

# get all xml files
xml_files = [f for f in os.listdir(annotations_path) if f.endswith(".xml")]

train_files, val_files = train_test_split(xml_files, test_size=0.2, random_state=42)

# class map
class_map = {"licence": 0}  # only one class

def convert_bbox(size, box):
    dw = 1.0 / size[0]
    dh = 1.0 / size[1]
    x = (box[0] + box[1]) / 2.0 - 1
    y = (box[2] + box[3]) / 2.0 - 1
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return (x, y, w, h)

def convert(xml_file, split):
    in_file = open(os.path.join(annotations_path, xml_file))
    tree = ET.parse(in_file)
    root = tree.getroot()

    size = root.find("size")
    w = int(size.find("width").text)
    h = int(size.find("height").text)

    label_file = xml_file.replace(".xml", ".txt")
    out_file = open(os.path.join(f"{labels_path}/{split}", label_file), "w")

    for obj in root.iter("object"):
        cls = obj.find("name").text
        if cls not in class_map:
            continue
        cls_id = class_map[cls]
        xmlbox = obj.find("bndbox")
        b = (float(xmlbox.find("xmin").text), float(xmlbox.find("xmax").text),
             float(xmlbox.find("ymin").text), float(xmlbox.find("ymax").text))
        bb = convert_bbox((w, h), b)
        out_file.write(f"{cls_id} {' '.join([str(a) for a in bb])}\n")
    out_file.close()

# convert files
for f in train_files:
    convert(f, "train")
for f in val_files:
    convert(f, "val")

print("✅ Conversion finished! Check dataset/labels/train and dataset/labels/val")
