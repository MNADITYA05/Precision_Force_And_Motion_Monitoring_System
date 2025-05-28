import os
import random
import cv2
import numpy as np
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from PIL import Image, ImageEnhance
import matplotlib.pyplot as plt
import torch
import warnings
import gc
warnings.filterwarnings('ignore')

# Memory optimization settings
torch.cuda.empty_cache()
gc.collect()

# Model name for Qwen2-VL-2B-Instruct
model_name = "Qwen/Qwen2-VL-2B-Instruct"

print(f"Loading {model_name} model...")

# Load model with memory optimizations
model = Qwen2VLForConditionalGeneration.from_pretrained(
    model_name,
    torch_dtype=torch.float16,  # Use half precision to save ~50% memory
    device_map="auto",
    low_cpu_mem_usage=True,  # Reduce CPU memory usage during loading
    trust_remote_code=True,  # Required for Qwen models
)

processor = AutoProcessor.from_pretrained(
    model_name,
    trust_remote_code=True  # Required for Qwen models
)

print("Model loaded successfully!")

# Base path for the dataset
base_path = "/kaggle/input/intelocr/OCR_DataValidation_Noise"
categories = [
    "Too Bright false",
    "Too Bright pass", 
    "Too Dark false",
    "Too Dark pass",
    "Vibration false",
    "Vibration pass"
]

class AdaptiveImageProcessor:
    def analyze_image_quality(self, image):
        """Analyze various quality metrics of the image"""
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if len(image.shape) == 3 else image
        
        # Basic quality metrics
        brightness = np.mean(gray)
        contrast = np.std(gray)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        # Entropy calculation
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        hist_norm = hist.flatten() / hist.sum()
        entropy = -np.sum(hist_norm * np.log2(hist_norm + 1e-10))
        
        # Edge analysis
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size
        
        # Gradient analysis
        sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
        gradient_mean = np.mean(gradient_magnitude)
        
        return {
            'brightness': brightness,
            'contrast': contrast,
            'sharpness': laplacian_var,
            'entropy': entropy,
            'edge_density': edge_density,
            'gradient_strength': gradient_mean
        }
    
    def detect_noise_type(self, image):
        """Detect different types of noise in the image"""
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if len(image.shape) == 3 else image
        
        # Gaussian noise detection
        gaussian_filtered = cv2.GaussianBlur(gray, (5, 5), 0)
        noise_level = np.mean(np.abs(gray.astype(float) - gaussian_filtered.astype(float)))
        
        # Motion blur detection
        motion_kernel = np.ones((1, 15), np.float32) / 15
        motion_filtered = cv2.filter2D(gray, -1, motion_kernel)
        motion_blur_score = np.mean(np.abs(gray.astype(float) - motion_filtered.astype(float)))
        
        # Salt and pepper noise detection
        median_filtered = cv2.medianBlur(gray, 5)
        salt_pepper_score = np.mean(np.abs(gray.astype(float) - median_filtered.astype(float)))
        
        return {
            'noise_level': noise_level,
            'motion_blur': motion_blur_score,
            'salt_pepper': salt_pepper_score
        }
    
    def adaptive_enhance_image(self, image):
        """Apply adaptive enhancement based on image analysis"""
        if isinstance(image, Image.Image):
            image = np.array(image)
        
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if len(image.shape) == 3 else image
        enhanced = gray.copy().astype(np.float64)
        
        # Get quality and noise metrics
        quality_metrics = self.analyze_image_quality(image)
        noise_metrics = self.detect_noise_type(image)
        
        brightness = quality_metrics['brightness']
        contrast = quality_metrics['contrast']
        sharpness = quality_metrics['sharpness']
        noise_level = noise_metrics['noise_level']
        
        # Brightness correction
        if brightness < 80:  # Too dark
            gamma = 1.5 + (80 - brightness) / 80 * 0.8
            enhanced = np.power(enhanced / 255.0, 1/gamma) * 255.0
            enhanced = np.clip(enhanced, 0, 255)
        elif brightness > 180:  # Too bright
            gamma = 0.6 - (brightness - 180) / 75 * 0.3
            gamma = max(gamma, 0.3)
            enhanced = np.power(enhanced / 255.0, 1/gamma) * 255.0
            enhanced = np.clip(enhanced, 0, 255)
        
        # Contrast enhancement
        if contrast < 30:  # Low contrast
            alpha = 1.5 + (30 - contrast) / 30 * 1.0
            beta = -enhanced.mean() * (alpha - 1)
            enhanced = alpha * enhanced + beta
            enhanced = np.clip(enhanced, 0, 255)
        elif contrast > 80:  # High contrast
            enhanced = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(8,8)).apply(enhanced.astype(np.uint8))
        else:  # Normal contrast
            enhanced = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)).apply(enhanced.astype(np.uint8))
        
        # Noise reduction
        if noise_level > 15:
            if noise_metrics['salt_pepper'] > noise_metrics['motion_blur']:
                enhanced = cv2.medianBlur(enhanced.astype(np.uint8), 3)
            else:
                enhanced = cv2.bilateralFilter(enhanced.astype(np.uint8), 9, 75, 75)
        
        # Sharpening for blurry images
        if sharpness < 100:
            kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
            sharpening_strength = (100 - sharpness) / 100 * 0.5
            kernel = kernel * sharpening_strength
            kernel[1,1] = 1 + kernel[1,1] - 1
            enhanced = cv2.filter2D(enhanced.astype(np.uint8), -1, kernel)
            enhanced = np.clip(enhanced, 0, 255)
        
        # Final post-processing
        enhanced = enhanced.astype(np.uint8)
        enhanced = cv2.morphologyEx(enhanced, cv2.MORPH_CLOSE, np.ones((2,2), np.uint8))
        enhanced_eq = cv2.equalizeHist(enhanced)
        enhanced = cv2.addWeighted(enhanced, 0.7, enhanced_eq, 0.3, 0)
        
        return enhanced

def get_random_image_from_category(category):
    """Get a random image from the specified category"""
    category_path = os.path.join(base_path, category)
    if os.path.exists(category_path):
        images = [f for f in os.listdir(category_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        if images:
            random_image = random.choice(images)
            return os.path.join(category_path, random_image)
    return None

def process_single_image_ocr(image, max_tokens=256):
    """Process a single image with OCR using Qwen2-VL-2B-Instruct"""
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": "Extract all the text in this image. Provide only the text content without any additional description."}
            ]
        }
    ]
    
    # Process with memory management
    with torch.no_grad():  # Disable gradient computation
        # Apply chat template and process inputs
        text_prompt = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        
        inputs = processor(
            text=[text_prompt],
            images=[image],
            padding=True,
            return_tensors="pt"
        )
        inputs = inputs.to("cuda" if torch.cuda.is_available() else "cpu")
        
        # Generate with reduced token count for 2B model
        output_ids = model.generate(
            **inputs, 
            max_new_tokens=max_tokens,
            do_sample=False,  # Use greedy decoding to save memory
            pad_token_id=processor.tokenizer.eos_token_id
        )
        
        generated_ids = [
            output_ids[len(input_ids):] 
            for input_ids, output_ids in zip(inputs.input_ids, output_ids)
        ]
        
        response = processor.batch_decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        
        # Clean up GPU memory
        del inputs, output_ids, generated_ids
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return response.strip()

def process_image_with_adaptive_enhancement(image_path):
    """Process image with adaptive enhancement and OCR"""
    print(f"Loading image: {image_path}")
    image = Image.open(image_path)
    
    # Convert to RGB if necessary
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Resize image if too large to save memory
    max_size = 1024
    if max(image.size) > max_size:
        ratio = max_size / max(image.size)
        new_size = tuple(int(dim * ratio) for dim in image.size)
        image = image.resize(new_size, Image.Resampling.LANCZOS)
        print(f"Image resized to: {new_size}")
    
    original_image = image.copy()
    
    # Initialize adaptive processor
    adaptive_processor = AdaptiveImageProcessor()
    
    # Analyze image quality
    image_np = np.array(image)
    quality_metrics = adaptive_processor.analyze_image_quality(image_np)
    noise_metrics = adaptive_processor.detect_noise_type(image_np)
    
    print("Image analysis completed. Applying adaptive enhancement...")
    
    # Apply adaptive enhancement
    enhanced_gray = adaptive_processor.adaptive_enhance_image(image)
    enhanced_rgb = cv2.cvtColor(enhanced_gray, cv2.COLOR_GRAY2RGB)
    enhanced_pil = Image.fromarray(enhanced_rgb)
    
    print("Processing original image with OCR...")
    original_text = process_single_image_ocr(original_image)
    
    # Force garbage collection between inferences
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    print("Processing enhanced image with OCR...")
    enhanced_text = process_single_image_ocr(enhanced_pil)
    
    # Final cleanup
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return {
        'original_image': original_image,
        'enhanced_image': enhanced_rgb,
        'original_text': original_text,
        'enhanced_text': enhanced_text,
        'quality_metrics': quality_metrics,
        'noise_metrics': noise_metrics
    }

def visualize_adaptive_ocr_results(results, category, image_path):
    """Visualize the OCR results with before/after comparison"""
    fig, axes = plt.subplots(2, 2, figsize=(20, 16))
    
    # Original image
    axes[0, 0].imshow(results['original_image'])
    axes[0, 0].set_title(f"Original Image - {category}", fontsize=14, fontweight='bold')
    axes[0, 0].axis('off')
    
    # Enhanced image
    axes[0, 1].imshow(results['enhanced_image'])
    axes[0, 1].set_title("Adaptive Enhanced Image", fontsize=14, fontweight='bold')
    axes[0, 1].axis('off')
    
    # Original OCR output
    axes[1, 0].text(0.05, 0.95, "Original OCR Output:", fontsize=16, fontweight='bold', 
                   transform=axes[1, 0].transAxes, verticalalignment='top')
    axes[1, 0].text(0.05, 0.80, results['original_text'], fontsize=11, 
                   transform=axes[1, 0].transAxes, verticalalignment='top', 
                   wrap=True, fontfamily='monospace')
    axes[1, 0].axis('off')
    
    # Enhanced OCR output
    axes[1, 1].text(0.05, 0.95, "Enhanced OCR Output:", fontsize=16, fontweight='bold', 
                   transform=axes[1, 1].transAxes, verticalalignment='top')
    axes[1, 1].text(0.05, 0.80, results['enhanced_text'], fontsize=11, 
                   transform=axes[1, 1].transAxes, verticalalignment='top', 
                   wrap=True, fontfamily='monospace')
    axes[1, 1].axis('off')
    
    plt.suptitle(f"Adaptive OCR Analysis - {os.path.basename(image_path)}", fontsize=18, fontweight='bold')
    plt.tight_layout()
    plt.show()
    
    # Print detailed analysis
    print(f"\nCategory: {category}")
    print(f"Image path: {image_path}")
    print(f"\nImage Quality Metrics:")
    for key, value in results['quality_metrics'].items():
        print(f"  {key.capitalize()}: {value:.3f}")
    
    print(f"\nNoise Analysis:")
    for key, value in results['noise_metrics'].items():
        print(f"  {key.replace('_', ' ').title()}: {value:.3f}")
    
    print(f"\nOriginal OCR Length: {len(results['original_text'])} characters")
    print(f"Enhanced OCR Length: {len(results['enhanced_text'])} characters")
    print(f"\nOriginal OCR: {results['original_text']}")
    print(f"Enhanced OCR: {results['enhanced_text']}")
    print("=" * 80)

# Clear memory before processing
if torch.cuda.is_available():
    torch.cuda.empty_cache()
gc.collect()

# Select random category and process image
selected_category = random.choice(categories)
image_path = get_random_image_from_category(selected_category)

if image_path:
    print(f"Processing image from category: {selected_category}")
    try:
        results = process_image_with_adaptive_enhancement(image_path)
        visualize_adaptive_ocr_results(results, selected_category, image_path)
    except torch.cuda.OutOfMemoryError:
        print("GPU memory exhausted. Try restarting the kernel or using CPU inference.")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
    except Exception as e:
        print(f"Error processing image: {str(e)}")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
else:
    print(f"No images found in category: {selected_category}")
    print("Available categories and their paths:")
    for cat in categories:
        cat_path = os.path.join(base_path, cat)
        exists = os.path.exists(cat_path)
        print(f"  {cat}: {cat_path} - {'EXISTS' if exists else 'NOT FOUND'}")

# Final cleanup
if torch.cuda.is_available():
    torch.cuda.empty_cache()
gc.collect()

print("\nAdaptive OCR processing completed!")
