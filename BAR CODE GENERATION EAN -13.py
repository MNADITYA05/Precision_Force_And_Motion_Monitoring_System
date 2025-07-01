import os
import json
import barcode
from barcode import EAN13
from barcode.writer import ImageWriter
from datetime import datetime
import cv2
from PIL import Image, ImageDraw, ImageFont
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import random

class KagglePCBEmbeddedBarcodeGenerator:
    def __init__(self):
        self.dataset_base = "/kaggle/input"
        self.output_base = "/kaggle/working"
        self.dataset_path = self.find_pcb_dataset()
        self.batch_id = "BATCH001"
        self.defect_categories = [
            "Missing_hole",
            "Mouse_bite", 
            "Open_circuit",
            "Short",
            "Spur",
            "Spurious_copper"
        ]
        
        os.makedirs(f"{self.output_base}/temp_barcodes", exist_ok=True)
        os.makedirs(f"{self.output_base}/embedded_pcb_images", exist_ok=True)
        
    def find_pcb_dataset(self):
        if os.path.exists(self.dataset_base):
            available_datasets = os.listdir(self.dataset_base)
            
            for dataset_name in available_datasets:
                dataset_path = os.path.join(self.dataset_base, dataset_name)
                if os.path.isdir(dataset_path):
                    pcb_dataset_path = os.path.join(dataset_path, "PCB_DATASET")
                    if os.path.exists(pcb_dataset_path):
                        return pcb_dataset_path
                    
                    if any(folder in os.listdir(dataset_path) for folder in ["PCB_USED", "images", "Annotations"]):
                        return dataset_path
        return None
    
    def find_images_in_dataset(self):
        image_data = []
        
        if not self.dataset_path:
            return image_data
        
        pcb_used_path = os.path.join(self.dataset_path, "PCB_USED")
        if os.path.exists(pcb_used_path):
            image_files = [f for f in os.listdir(pcb_used_path) 
                          if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            
            for image_file in image_files:
                defect_category = random.choice(self.defect_categories)
                
                image_data.append({
                    'path': os.path.join(pcb_used_path, image_file),
                    'filename': image_file,
                    'defect_category': defect_category,
                    'source_folder': 'PCB_USED'
                })
        
        images_path = os.path.join(self.dataset_path, "images")
        if os.path.exists(images_path):
            for defect_folder in os.listdir(images_path):
                defect_path = os.path.join(images_path, defect_folder)
                if os.path.isdir(defect_path):
                    image_files = [f for f in os.listdir(defect_path) 
                                  if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                    
                    defect_category = defect_folder.replace('_rotation', '')
                    if defect_category not in self.defect_categories:
                        defect_category = 'Unknown'
                    
                    for image_file in image_files:
                        image_data.append({
                            'path': os.path.join(defect_path, image_file),
                            'filename': image_file,
                            'defect_category': defect_category,
                            'source_folder': f'images/{defect_folder}'
                        })
        
        return image_data
    
    def generate_device_metadata(self, image_name, defect_category, source_folder):
        clean_name = image_name.split('.')[0]
        device_id = f"PCB_{self.batch_id}_{clean_name}"
        
        metadata = {
            "device_id": device_id,
            "batch_id": self.batch_id,
            "manufacturing_date": datetime.now().strftime("%Y-%m-%d"),
            "defect_category": defect_category,
            "source_folder": source_folder,
            "rohs_compliance": True,
            "production_line": "LINE_A",
            "timestamp": datetime.now().isoformat(),
            "quality_status": "DEFECTIVE" if defect_category != "GOOD" else "PASS"
        }
        
        return device_id, metadata
    
    def generate_numeric_id(self, device_id, defect_category):
        defect_map = {
            "Missing_hole": "01",
            "Mouse_bite": "02", 
            "Open_circuit": "03",
            "Short": "04",
            "Spur": "05",
            "Spurious_copper": "06"
        }
        
        numbers = re.findall(r'\d+', device_id)
        device_num = numbers[-1] if numbers else "001"
        device_num = device_num.zfill(3)
        defect_code = defect_map.get(defect_category, "00")
        
        return device_num, defect_code
    
    def generate_ean13_barcode(self, device_id, defect_category):
        try:
            device_num, defect_code = self.generate_numeric_id(device_id, defect_category)
            country_code = "123"
            manufacturer_code = "4567"
            product_code = f"{defect_code}{device_num}"
            ean13_code_base = f"{country_code}{manufacturer_code}{product_code}"
            
            ean13 = EAN13(ean13_code_base, writer=ImageWriter())
            barcode_filename = f"{self.output_base}/temp_barcodes/{device_id}_ean13"
            ean13.save(barcode_filename)
            
            full_ean13_code = ean13.get_fullcode()
            
            return f"{barcode_filename}.png", full_ean13_code
        except Exception as e:
            return None, None
    
    def embed_barcode_in_pcb(self, pcb_image_path, ean13_path, metadata):
        try:
            pcb_img = cv2.imread(pcb_image_path)
            if pcb_img is None:
                return None
                
            pcb_img = cv2.cvtColor(pcb_img, cv2.COLOR_BGR2RGB)
            pcb_pil = Image.fromarray(pcb_img)
            
            pcb_width, pcb_height = pcb_pil.size
            
            if ean13_path and os.path.exists(ean13_path):
                ean13_img = Image.open(ean13_path).convert('RGB')
                
                barcode_width = int(pcb_width * 0.8)
                barcode_height = int(barcode_width * 0.4)
                ean13_resized = ean13_img.resize((barcode_width, barcode_height), Image.Resampling.LANCZOS)
                
                total_height = pcb_height + barcode_height + 20
                total_width = max(pcb_width, barcode_width)
                
                composite = Image.new('RGB', (total_width, total_height), 'white')
                
                pcb_x = (total_width - pcb_width) // 2
                composite.paste(pcb_pil, (pcb_x, 0))
                
                barcode_x = (total_width - barcode_width) // 2
                barcode_y = pcb_height + 10
                composite.paste(ean13_resized, (barcode_x, barcode_y))
                
                output_filename = f"{self.output_base}/embedded_pcb_images/{metadata['device_id']}_embedded.png"
                composite.save(output_filename, quality=95)
                
                return output_filename
            
            return None
            
        except Exception as e:
            return None
    
    def process_dataset(self, max_images=50):
        results = []
        
        image_data = self.find_images_in_dataset()
        
        if not image_data:
            return results
        
        if len(image_data) > max_images:
            image_data = image_data[:max_images]
        
        for i, img_data in enumerate(image_data):
            device_id, metadata = self.generate_device_metadata(
                img_data['filename'], 
                img_data['defect_category'],
                img_data['source_folder']
            )
            
            ean13_path, ean13_code = self.generate_ean13_barcode(device_id, img_data['defect_category'])
            
            embedded_image_path = self.embed_barcode_in_pcb(
                img_data['path'], ean13_path, metadata
            )
            
            result = {
                "original_image": img_data['path'],
                "device_id": device_id,
                "defect_category": img_data['defect_category'],
                "source_folder": img_data['source_folder'],
                "embedded_image": embedded_image_path,
                "ean13_code": ean13_code,
                "metadata": metadata
            }
            results.append(result)
        
        self.cleanup_temp_files()
        
        return results
    
    def cleanup_temp_files(self):
        import shutil
        temp_dir = f"{self.output_base}/temp_barcodes"
        
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
    
    def create_database(self, results):
        for result in results:
            if result['ean13_code']:
                data = [{
                    'Device_ID': result['device_id'],
                    'Original_Image': result['original_image'],
                    'Embedded_Image': result['embedded_image'],
                    'Defect_Category': result['defect_category'],
                    'Source_Folder': result['source_folder'],
                    'Quality_Status': result['metadata']['quality_status'],
                    'Manufacturing_Date': result['metadata']['manufacturing_date'],
                    'Batch_ID': result['metadata']['batch_id'],
                    'RoHS_Compliance': result['metadata']['rohs_compliance']
                }]
                
                df = pd.DataFrame(data)
                csv_path = f"{self.output_base}/{result['ean13_code']}.csv"
                df.to_csv(csv_path, index=False)
    
    def display_sample_results(self, results, num_samples=5):
        if not results:
            return
        
        fig, axes = plt.subplots(num_samples, 1, figsize=(10, 8*num_samples))
        if num_samples == 1:
            axes = [axes]
        
        for i in range(min(num_samples, len(results))):
            result = results[i]
            
            if result['embedded_image'] and os.path.exists(result['embedded_image']):
                embedded_img = Image.open(result['embedded_image'])
                axes[i].imshow(embedded_img)
                axes[i].set_title(f"PCB with EAN-13 Barcode")
                axes[i].axis('off')
        
        plt.tight_layout()
        plt.show()

def main():
    generator = KagglePCBEmbeddedBarcodeGenerator()
    
    if not generator.dataset_path:
        print("PCB dataset not found!")
        return
    
    results = generator.process_dataset(max_images=50)
    
    if results:
        generator.create_database(results)
        generator.display_sample_results(results, num_samples=5)
        print(f"Processed {len(results)} images successfully")
    else:
        print("No images were processed")

if __name__ == "__main__":
    main()
