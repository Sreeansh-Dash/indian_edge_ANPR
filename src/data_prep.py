import os
import shutil
import random
import xml.etree.ElementTree as ET

def convert_xml_to_yolo(xml_path, img_width, img_height):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    yolo_labels = []
    
    for obj in root.findall('object'):
        bndbox = obj.find('bndbox')
        if bndbox is not None:
            xmin = float(bndbox.find('xmin').text)
            ymin = float(bndbox.find('ymin').text)
            xmax = float(bndbox.find('xmax').text)
            ymax = float(bndbox.find('ymax').text)
            
            # YOLO format: class x_center y_center width height (normalized)
            x_center = ((xmin + xmax) / 2) / img_width
            y_center = ((ymin + ymax) / 2) / img_height
            width = (xmax - xmin) / img_width
            height = (ymax - ymin) / img_height
            
            # Assuming class 0 for license plate
            yolo_labels.append(f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
            
    return yolo_labels

def main():
    base_src_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'license_plates')
    base_dest_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
    
    dirs_to_create = [
        os.path.join(base_dest_dir, 'images', 'train'),
        os.path.join(base_dest_dir, 'images', 'val'),
        os.path.join(base_dest_dir, 'labels', 'train'),
        os.path.join(base_dest_dir, 'labels', 'val')
    ]
    
    for d in dirs_to_create:
        os.makedirs(d, exist_ok=True)
        
    all_data = []
    
    # Traverse directory to find image and xml pairs
    for root, dirs, files in os.walk(base_src_dir):
        for file in files:
            if file.endswith('.xml'):
                xml_path = os.path.join(root, file)
                # Check for corresponding image file (.jpg, .png, .jpeg)
                base_name = os.path.splitext(file)[0]
                img_path = None
                for ext in ['.jpg', '.jpeg', '.png']:
                    potential_img = os.path.join(root, f"{base_name}{ext}")
                    if os.path.exists(potential_img):
                        img_path = potential_img
                        break
                
                if img_path:
                    # Parse image size from XML
                    tree = ET.parse(xml_path)
                    xml_root = tree.getroot()
                    size = xml_root.find('size')
                    if size is not None:
                        width = float(size.find('width').text)
                        height = float(size.find('height').text)
                        all_data.append((img_path, xml_path, width, height))

    print(f"Found {len(all_data)} valid image-label pairs.")
    
    # Shuffle and split 80/20
    random.seed(42)
    random.shuffle(all_data)
    split_idx = int(len(all_data) * 0.8)
    train_data = all_data[:split_idx]
    val_data = all_data[split_idx:]
    
    def process_split(data, split_name):
        for img_path, xml_path, w, h in data:
            base_name = os.path.splitext(os.path.basename(img_path))[0]
            
            # Copy image
            dest_img = os.path.join(base_dest_dir, 'images', split_name, os.path.basename(img_path))
            shutil.copy2(img_path, dest_img)
            
            # Generate and save YOLO label
            yolo_labels = convert_xml_to_yolo(xml_path, w, h)
            dest_label = os.path.join(base_dest_dir, 'labels', split_name, f"{base_name}.txt")
            with open(dest_label, 'w') as f:
                f.write("\n".join(yolo_labels))
                
    print("Processing training data...")
    process_split(train_data, 'train')
    print("Processing validation data...")
    process_split(val_data, 'val')
    
    # Generate data.yaml
    yaml_path = os.path.join(base_dest_dir, 'data.yaml')
    with open(yaml_path, 'w') as f:
        f.write(f"train: {os.path.join(base_dest_dir, 'images', 'train')}\n")
        f.write(f"val: {os.path.join(base_dest_dir, 'images', 'val')}\n\n")
        f.write("nc: 1\n")
        f.write("names: ['license_plate']\n")
        
    print(f"Data preparation complete. YAML saved to {yaml_path}")

if __name__ == '__main__':
    main()
