import os
import json
import qrcode
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

class KagglePCBBarcodeGenerator:
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
        
        os.makedirs(f"{self.output_base}/generated_barcodes", exist_ok=True)
        os.makedirs(f"{self.output_base}/generated_qrcodes", exist_ok=True)
        os.makedirs(f"{self.output_base}/labeled_images", exist_ok=True)
        
    def find_pcb_dataset(self):
        print("Searching for PCB dataset in Kaggle...")
        
        possible_names = [
            "pcb-defects",
            "pcb-dataset", 
            "pcbdefects",
            "pcb-defect-detection"
        ]
        
        if os.path.exists(self.dataset_base):
            available_datasets = os.listdir(self.dataset_base)
            print(f"Available datasets: {available_datasets}")
            
            for dataset_name in available_datasets:
                dataset_path = os.path.join(self.dataset_base, dataset_name)
                if os.path.isdir(dataset_path):
                    pcb_dataset_path = os.path.join(dataset_path, "PCB_DATASET")
                    if os.path.exists(pcb_dataset_path):
                        print(f"Found PCB dataset at: {pcb_dataset_path}")
                        return pcb_dataset_path
                    
                    if any(folder in os.listdir(dataset_path) for folder in ["PCB_USED", "images", "Annotations"]):
                        print(f"Found PCB dataset at: {dataset_path}")
                        return dataset_path
        
        print("PCB dataset not found!")
        return None
    
    def find_images_in_dataset(self):
        image_data = []
        
        if not self.dataset_path:
            print("No dataset path found!")
            return image_data
        
        print(f"Exploring dataset structure in: {self.dataset_path}")
        
        for item in os.listdir(self.dataset_path):
            item_path = os.path.join(self.dataset_path, item)
            if os.path.isdir(item_path):
                print(f"Found folder: {item}")
        
        pcb_used_path = os.path.join(self.dataset_path, "PCB_USED")
        if os.path.exists(pcb_used_path):
            print(f"Processing PCB_USED folder...")
            image_files = [f for f in os.listdir(pcb_used_path) 
                          if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            print(f"Found {len(image_files)} images in PCB_USED")
            
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
            print(f"Processing images folder...")
            
            for defect_folder in os.listdir(images_path):
                defect_path = os.path.join(images_path, defect_folder)
                if os.path.isdir(defect_path):
                    print(f"  Processing {defect_folder} folder...")
                    image_files = [f for f in os.listdir(defect_path) 
                                  if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                    print(f"    Found {len(image_files)} images")
                    
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
        
        print(f"Total images found: {len(image_data)}")
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
            ean13_code = f"{country_code}{manufacturer_code}{product_code}"
            
            ean13 = EAN13(ean13_code, writer=ImageWriter())
            barcode_filename = f"{self.output_base}/generated_barcodes/{device_id}_ean13"
            ean13.save(barcode_filename)
            
            return f"{barcode_filename}.png"
        except Exception as e:
            print(f"Error generating EAN-13 for {device_id}: {e}")
            return None
    
    def generate_qr_code(self, device_id, metadata):
        try:
            data_string = json.dumps(metadata, indent=2)
            
            qr = qrcode.QRCode(
                version=1,
                error_correction=qrcode.constants.ERROR_CORRECT_M,
                box_size=8,
                border=4,
            )
            qr.add_data(data_string)
            qr.make(fit=True)
            
            qr_img = qr.make_image(fill_color="black", back_color="white")
            qr_filename = f"{self.output_base}/generated_qrcodes/{device_id}_qr.png"
            qr_img.save(qr_filename)
            
            return qr_filename
        except Exception as e:
            print(f"Error generating QR code for {device_id}: {e}")
            return None
    
    def create_labeled_image(self, pcb_image_path, ean13_path, qr_path, metadata):
        try:
            pcb_img = cv2.imread(pcb_image_path)
            if pcb_img is None:
                print(f"Could not load: {pcb_image_path}")
                return None
                
            pcb_img = cv2.cvtColor(pcb_img, cv2.COLOR_BGR2RGB)
            pcb_pil = Image.fromarray(pcb_img)
            
            max_width = 500
            if pcb_pil.width > max_width:
                ratio = max_width / pcb_pil.width
                new_height = int(pcb_pil.height * ratio)
                pcb_pil = pcb_pil.resize((max_width, new_height))
            
            pcb_width, pcb_height = pcb_pil.size
            label_height = 200
            total_width = max(pcb_width, 700)
            total_height = pcb_height + label_height
            
            composite = Image.new('RGB', (total_width, total_height), 'white')
            composite.paste(pcb_pil, (0, 0))
            
            y_start = pcb_height + 20
            
            if ean13_path and os.path.exists(ean13_path):
                ean13_img = Image.open(ean13_path)
                ean13_resized = ean13_img.resize((300, 120))
                composite.paste(ean13_resized, (20, y_start))
                
                draw = ImageDraw.Draw(composite)
                draw.text((20, y_start - 15), "EAN-13 Barcode", fill='black')
            
            draw = ImageDraw.Draw(composite)
            try:
                font = ImageFont.load_default()
            except:
                font = None
            
            text_x = 350
            text_y = y_start + 10
            text_info = [
                f"Device ID: {metadata['device_id']}",
                f"Batch: {metadata['batch_id']}",
                f"Defect: {metadata['defect_category']}",
                f"Date: {metadata['manufacturing_date']}",
                f"Status: {metadata['quality_status']}",
                f"RoHS: {metadata['rohs_compliance']}",
                "",
                "EAN-13: Global Standard",
                "13-digit retail format",
                "Universal scanner support"
            ]
            
            for i, text in enumerate(text_info):
                draw.text((text_x, text_y + i*15), text, fill='black', font=font)
            
            output_filename = f"{self.output_base}/labeled_images/{metadata['device_id']}_labeled.png"
            composite.save(output_filename)
            
            return output_filename
            
        except Exception as e:
            print(f"Error creating labeled image: {e}")
            return None
    
    def process_dataset(self, max_images=50):
        results = []
        
        image_data = self.find_images_in_dataset()
        
        if not image_data:
            print("No images found!")
            return results
        
        if len(image_data) > max_images:
            print(f"Processing first {max_images} images out of {len(image_data)}")
            image_data = image_data[:max_images]
        
        print(f"Processing {len(image_data)} images...")
        
        for i, img_data in enumerate(image_data):
            if i % 10 == 0:
                print(f"Processing {i+1}/{len(image_data)}: {img_data['filename']}")
            
            device_id, metadata = self.generate_device_metadata(
                img_data['filename'], 
                img_data['defect_category'],
                img_data['source_folder']
            )
            
            ean13_path = self.generate_ean13_barcode(device_id, img_data['defect_category'])
            qr_path = self.generate_qr_code(device_id, metadata)
            
            labeled_image_path = self.create_labeled_image(
                img_data['path'], ean13_path, qr_path, metadata
            )
            
            result = {
                "original_image": img_data['path'],
                "device_id": device_id,
                "defect_category": img_data['defect_category'],
                "source_folder": img_data['source_folder'],
                "ean13_barcode": ean13_path,
                "qr_code": qr_path,
                "labeled_image": labeled_image_path,
                "metadata": metadata
            }
            results.append(result)
        
        return results
    
    def create_database(self, results):
        data = []
        for result in results:
            data.append({
                'Device_ID': result['device_id'],
                'Batch_ID': result['metadata']['batch_id'],
                'Manufacturing_Date': result['metadata']['manufacturing_date'],
                'Defect_Category': result['defect_category'],
                'Source_Folder': result['source_folder'],
                'Quality_Status': result['metadata']['quality_status'],
                'RoHS_Compliance': result['metadata']['rohs_compliance'],
                'EAN13_Barcode': result['ean13_barcode'],
                'QR_Code': result['qr_code'],
                'Labeled_Image': result['labeled_image']
            })
        
        df = pd.DataFrame(data)
        csv_path = f"{self.output_base}/pcb_barcode_database.csv"
        df.to_csv(csv_path, index=False)
        print(f"Database saved: {csv_path}")
        return df
    
    def display_sample_results(self, results, num_samples=5):
        if not results:
            return
        
        fig, axes = plt.subplots(num_samples, 3, figsize=(15, 5*num_samples))
        if num_samples == 1:
            axes = axes.reshape(1, -1)
        
        for i in range(min(num_samples, len(results))):
            result = results[i]
            
            if os.path.exists(result['original_image']):
                pcb_img = cv2.imread(result['original_image'])
                pcb_img = cv2.cvtColor(pcb_img, cv2.COLOR_BGR2RGB)
                axes[i, 0].imshow(pcb_img)
                axes[i, 0].set_title(f"PCB Image\n{result['device_id']}")
                axes[i, 0].axis('off')
            
            if result['ean13_barcode'] and os.path.exists(result['ean13_barcode']):
                ean13_img = Image.open(result['ean13_barcode'])
                axes[i, 1].imshow(ean13_img)
                axes[i, 1].set_title(f"EAN-13 Barcode\nDefect: {result['defect_category']}")
                axes[i, 1].axis('off')
            
            if result['labeled_image'] and os.path.exists(result['labeled_image']):
                labeled_img = Image.open(result['labeled_image'])
                axes[i, 2].imshow(labeled_img)
                axes[i, 2].set_title(f"Complete Label\nStatus: {result['metadata']['quality_status']}")
                axes[i, 2].axis('off')
        
        plt.tight_layout()
        plt.show()

def main():
    print("🔧 EAN-13 PCB Barcode Generator")
    print("=" * 50)
    
    generator = KagglePCBBarcodeGenerator()
    
    if not generator.dataset_path:
        print("❌ PCB dataset not found!")
        print("Please add a PCB defects dataset to your Kaggle notebook.")
        return
    
    print("✅ Dataset found!")
    
    print("\n🔄 Processing dataset...")
    results = generator.process_dataset(max_images=50)
    
    if results:
        print(f"\n✅ Successfully processed {len(results)} PCB images")
        
        df = generator.create_database(results)
        
        print(f"\n📊 Processing Statistics:")
        print(f"Total images processed: {len(results)}")
        print(f"Defect categories found: {df['Defect_Category'].nunique()}")
        print(f"Defect distribution:")
        print(df['Defect_Category'].value_counts())
        
        print(f"\n🖼️ Sample Results (5 samples):")
        generator.display_sample_results(results, num_samples=5)
        
        print(f"\n📁 Generated Files:")
        print(f"- EAN-13 Barcodes: /kaggle/working/generated_barcodes/")
        print(f"- QR codes: /kaggle/working/generated_qrcodes/")
        print(f"- Labeled images: /kaggle/working/labeled_images/")
        print(f"- Database: /kaggle/working/pcb_barcode_database.csv")
        
        print(f"\n✅ EAN-13 Format Details:")
        print(f"- Country Code: 123 (Custom PCB identifier)")
        print(f"- Manufacturer: 4567 (Your company code)")
        print(f"- Product Code: [Defect Code][Device Number]")
        print(f"- Check Digit: Auto-calculated")
        
    else:
        print("❌ No images were processed.")

if __name__ == "__main__":
    main()
