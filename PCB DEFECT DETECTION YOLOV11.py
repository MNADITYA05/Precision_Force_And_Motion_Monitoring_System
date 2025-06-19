!pip install ultralytics
!pip install roboflow

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO
import yaml
from pathlib import Path
import shutil
from sklearn.model_selection import train_test_split
import xml.etree.ElementTree as ET

dataset_path = "/kaggle/input/pcb-defects/PCB_DATASET"
annotations_path = os.path.join(dataset_path, "Annotations")
images_path = os.path.join(dataset_path, "images")
rotation_path = os.path.join(dataset_path, "rotation")

def collect_all_images():
    all_images = []
    
    defect_types = ['Missing_hole', 'Mouse_bite', 'Open_circuit', 'Short', 'Spur', 'Spurious_copper']
    
    for defect_type in defect_types:
        annotation_folder = os.path.join(annotations_path, defect_type)
        image_folder = os.path.join(images_path, defect_type)
        rotation_folder = os.path.join(rotation_path, f"{defect_type}_rotation")
        
        if os.path.exists(annotation_folder) and os.path.exists(image_folder):
            annotations = [f for f in os.listdir(annotation_folder) if f.endswith('.xml')]
            for ann_file in annotations:
                img_name = ann_file.replace('.xml', '.jpg')
                img_path = os.path.join(image_folder, img_name)
                ann_path = os.path.join(annotation_folder, ann_file)
                
                if os.path.exists(img_path):
                    all_images.append({
                        'image_path': img_path,
                        'annotation_path': ann_path,
                        'defect_type': defect_type.lower(),
                        'source': 'original'
                    })
        
        if os.path.exists(rotation_folder):
            rotation_images = [f for f in os.listdir(rotation_folder) if f.endswith('.jpg')]
            for img_file in rotation_images:
                img_path = os.path.join(rotation_folder, img_file)
                all_images.append({
                    'image_path': img_path,
                    'annotation_path': None,
                    'defect_type': defect_type.lower(),
                    'source': 'rotation'
                })
    
    print(f"Total images collected: {len(all_images)}")
    return all_images

all_images = collect_all_images()

def convert_xml_to_yolo(xml_path, img_width, img_height):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    yolo_annotations = []
    
    class_map = {
        'missing_hole': 0,
        'mouse_bite': 1,
        'open_circuit': 2,
        'short': 3,
        'spur': 4,
        'spurious_copper': 5
    }
    
    for obj in root.findall('object'):
        class_name = obj.find('name').text.lower()
        
        if class_name not in class_map:
            continue
            
        class_id = class_map[class_name]
        
        bbox = obj.find('bndbox')
        xmin = float(bbox.find('xmin').text)
        ymin = float(bbox.find('ymin').text)
        xmax = float(bbox.find('xmax').text)
        ymax = float(bbox.find('ymax').text)
        
        x_center = (xmin + xmax) / 2.0 / img_width
        y_center = (ymin + ymax) / 2.0 / img_height
        width = (xmax - xmin) / img_width
        height = (ymax - ymin) / img_height
        
        yolo_annotations.append(f"{class_id} {x_center} {y_center} {width} {height}")
    
    return yolo_annotations

def prepare_yolo_dataset():
    os.makedirs("/kaggle/working/dataset/images/train", exist_ok=True)
    os.makedirs("/kaggle/working/dataset/images/val", exist_ok=True)
    os.makedirs("/kaggle/working/dataset/labels/train", exist_ok=True)
    os.makedirs("/kaggle/working/dataset/labels/val", exist_ok=True)
    
    images_with_annotations = [img for img in all_images if img['annotation_path'] is not None]
    
    train_images, val_images = train_test_split(images_with_annotations, test_size=0.2, random_state=42, 
                                               stratify=[img['defect_type'] for img in images_with_annotations])
    
    def process_split(images, split_name):
        for i, img_data in enumerate(images):
            img_path = img_data['image_path']
            ann_path = img_data['annotation_path']
            
            img_name = f"{split_name}_{i}_{os.path.basename(img_path)}"
            txt_name = img_name.replace('.jpg', '.txt')
            
            dst_img = f"/kaggle/working/dataset/images/{split_name}/{img_name}"
            dst_txt = f"/kaggle/working/dataset/labels/{split_name}/{txt_name}"
            
            shutil.copy2(img_path, dst_img)
            
            if ann_path and os.path.exists(ann_path):
                img = cv2.imread(img_path)
                img_height, img_width = img.shape[:2]
                
                yolo_annotations = convert_xml_to_yolo(ann_path, img_width, img_height)
                
                with open(dst_txt, 'w') as f:
                    f.write('\n'.join(yolo_annotations))
    
    process_split(train_images, 'train')
    process_split(val_images, 'val')
    
    rotation_images = [img for img in all_images if img['source'] == 'rotation']
    for i, img_data in enumerate(rotation_images[:200]):
        img_path = img_data['image_path']
        img_name = f"rotation_{i}_{os.path.basename(img_path)}"
        dst_img = f"/kaggle/working/dataset/images/train/{img_name}"
        shutil.copy2(img_path, dst_img)
        
        txt_name = img_name.replace('.jpg', '.txt')
        dst_txt = f"/kaggle/working/dataset/labels/train/{txt_name}"
        with open(dst_txt, 'w') as f:
            f.write('')
    
    print(f"Dataset prepared: {len(train_images)} training, {len(val_images)} validation images")
    print(f"Added {min(200, len(rotation_images))} rotation images for augmentation")

prepare_yolo_dataset()

dataset_config = {
    'path': '/kaggle/working/dataset',
    'train': 'images/train',
    'val': 'images/val',
    'nc': 6,
    'names': [
        'missing_hole',
        'mouse_bite', 
        'open_circuit',
        'short',
        'spur',
        'spurious_copper'
    ]
}

with open('/kaggle/working/dataset.yaml', 'w') as f:
    yaml.dump(dataset_config, f)

print("Dataset configuration created!")

model = YOLO('yolo11n.pt')

results = model.train(
    data='/kaggle/working/dataset.yaml',
    epochs=100,
    imgsz=640,
    batch=16,
    name='pcb_defect_detection',
    cache=True,
    device=0,
    workers=2,
    patience=20,
    save=True,
    plots=True
)

print("Training completed!")

metrics = model.val()

print(f"mAP50: {metrics.box.map50}")
print(f"mAP50-95: {metrics.box.map}")

def display_results():
    results_path = "/kaggle/working/runs/detect/pcb_defect_detection"
    
    confusion_matrix_path = os.path.join(results_path, "confusion_matrix.png")
    if os.path.exists(confusion_matrix_path):
        img = cv2.imread(confusion_matrix_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.figure(figsize=(10, 8))
        plt.imshow(img_rgb)
        plt.title("Confusion Matrix")
        plt.axis('off')
        plt.show()
    
    results_img_path = os.path.join(results_path, "results.png")
    if os.path.exists(results_img_path):
        img = cv2.imread(results_img_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.figure(figsize=(15, 10))
        plt.imshow(img_rgb)
        plt.title("Training Results")
        plt.axis('off')
        plt.show()

display_results()

best_model = YOLO('/kaggle/working/runs/detect/pcb_defect_detection/weights/best.pt')

def test_inference(image_path, conf_threshold=0.25):
    results = best_model(image_path, conf=conf_threshold)
    
    annotated_img = results[0].plot()
    
    plt.figure(figsize=(12, 8))
    plt.imshow(cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB))
    plt.title(f"PCB Defect Detection - {os.path.basename(image_path)}")
    plt.axis('off')
    plt.show()
    
    boxes = results[0].boxes
    if boxes is not None:
        for i, box in enumerate(boxes):
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])
            class_name = dataset_config['names'][class_id]
            print(f"Detection {i+1}: {class_name} (confidence: {confidence:.2f})")
    else:
        print("No defects detected!")

sample_images = [f"/kaggle/working/dataset/images/val/{img}" 
                for img in os.listdir("/kaggle/working/dataset/images/val")[:3]]

for img_path in sample_images:
    test_inference(img_path)

def batch_predict(input_folder, output_folder, conf_threshold=0.25):
    os.makedirs(output_folder, exist_ok=True)
    
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
    image_files = [f for f in os.listdir(input_folder) 
                   if f.lower().endswith(image_extensions)]
    
    results_summary = []
    
    for img_file in image_files:
        img_path = os.path.join(input_folder, img_file)
        results = best_model(img_path, conf=conf_threshold)
        
        annotated_img = results[0].plot()
        output_path = os.path.join(output_folder, f"detected_{img_file}")
        cv2.imwrite(output_path, annotated_img)
        
        boxes = results[0].boxes
        defect_count = len(boxes) if boxes is not None else 0
        results_summary.append({
            'image': img_file,
            'defect_count': defect_count,
            'defects': []
        })
        
        if boxes is not None:
            for box in boxes:
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])
                class_name = dataset_config['names'][class_id]
                results_summary[-1]['defects'].append({
                    'type': class_name,
                    'confidence': confidence
                })
    
    return results_summary

best_model.export(format='onnx')
best_model.export(format='torchscript')

print("Model exported successfully!")
