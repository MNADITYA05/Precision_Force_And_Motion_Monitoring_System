import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import random
import os

class ImageProcessor:
    def __init__(self):
        pass
    
    def enhance_overexposed_image(self, image):
        """Advanced overexposed image enhancement optimized for text readability"""
        if len(image.shape) == 3:
            # Convert to LAB for better processing
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
        else:
            l = image.copy()
        
        # Apply CLAHE for better local contrast
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8,8))
        l_enhanced = clahe.apply(l)
        
        # Apply gamma correction to reduce brightness while maintaining details
        gamma = 1.5  # Values > 1 darken the image
        lookupTable = np.empty((1,256), np.uint8)
        for i in range(256):
            lookupTable[0,i] = np.clip(pow(i / 255.0, gamma) * 255.0, 0, 255)
        l_enhanced = cv2.LUT(l_enhanced, lookupTable)
        
        # Conservative contrast adjustment to improve text visibility
        alpha = 1.2  # Contrast factor
        beta = -20   # Brightness factor (negative for darker)
        l_enhanced = cv2.convertScaleAbs(l_enhanced, alpha=alpha, beta=beta)
        
        # Edge enhancement for text boundaries
        kernel = np.array([[-1,-1,-1], 
                          [-1, 9,-1],
                          [-1,-1,-1]])
        l_enhanced = cv2.filter2D(l_enhanced, -1, kernel)
        
        if len(image.shape) == 3:
            enhanced_lab = cv2.merge([l_enhanced, a, b])
            enhanced = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
        else:
            enhanced = l_enhanced
        
        return enhanced
    
    def enhance_dark_image(self, image):
        """Advanced dark image enhancement optimized for text readability"""
        if len(image.shape) == 3:
            # Convert to HSV color space for better separation of brightness and color
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            h, s, v = cv2.split(hsv)
            gray = v.copy()  # Value channel contains brightness information
        else:
            gray = image.copy()
        
        mean_brightness = np.mean(gray)
        
        # Step 1: Initial brightness boost with adaptive parameters
        if mean_brightness < 20:
            alpha = 3.0
            beta = 30
        elif mean_brightness < 50:
            alpha = 2.5
            beta = 25
        else:
            alpha = 2.0
            beta = 15
        
        # Apply initial brightness correction
        brightened = cv2.convertScaleAbs(gray, alpha=alpha, beta=beta)
        
        # Step 2: Apply adaptive CLAHE with parameters based on image characteristics
        if mean_brightness < 30:
            # For very dark images, use more aggressive CLAHE
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        else:
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        
        clahe_applied = clahe.apply(brightened)
        
        # Step 3: Apply bilateral filtering to reduce noise while preserving edges
        denoised = cv2.bilateralFilter(clahe_applied, 5, 75, 75)
        
        # Step 4: Sharpening to enhance text edges
        kernel = np.array([[-1,-1,-1], 
                          [-1, 9,-1],
                          [-1,-1,-1]])
        sharpened = cv2.filter2D(denoised, -1, kernel)
        
        # Step 5: Local contrast enhancement
        # Create a slightly blurred version for local contrast
        blurred = cv2.GaussianBlur(sharpened, (0, 0), 3)
        local_contrast = cv2.addWeighted(sharpened, 1.5, blurred, -0.5, 0)
        
        # Step 6: Normalize histogram to spread out pixel values more evenly
        normalized = cv2.equalizeHist(local_contrast)
        
        # Step 7: Blend the results for optimal readability
        enhanced_gray = cv2.addWeighted(local_contrast, 0.7, normalized, 0.3, 0)
        
        # If original is color, convert back to color
        if len(image.shape) == 3:
            # Replace value channel with enhanced version
            hsv_enhanced = cv2.merge([h, s, enhanced_gray])
            enhanced = cv2.cvtColor(hsv_enhanced, cv2.COLOR_HSV2BGR)
        else:
            enhanced = enhanced_gray
        
        return enhanced
    
    def enhance_blurry_image(self, image):
        """Advanced blur correction specifically optimized for text readability in vibration-affected images"""
        if len(image.shape) == 3:
            # Convert to grayscale for processing if it's a color image
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            is_color = True
        else:
            gray = image.copy()
            is_color = False
        
        # Step 1: Apply motion blur correction
        # Use a deconvolution technique to address motion blur
        # Create a custom kernel for horizontal motion blur (most common in OCR scenarios)
        kernel_size = 9
        motion_kernel = np.zeros((kernel_size, kernel_size))
        motion_kernel[kernel_size//2, :] = 1.0 / kernel_size
        
        # Apply deconvolution using Wiener filter (simplified approximation)
        # First apply the blur in the opposite direction
        deconvolved = cv2.filter2D(gray, -1, motion_kernel)
        # Then apply a sharpening kernel to restore edges
        kernel_sharp = np.array([[-1, -1, -1],
                                [-1, 9, -1],
                                [-1, -1, -1]])
        deconvolved = cv2.filter2D(deconvolved, -1, kernel_sharp)
        
        # Step 2: Apply bilateral filtering for edge-preserving noise reduction
        # Use larger diameter and smaller color/space sigmas for text
        bilateral = cv2.bilateralFilter(gray, 7, 25, 25)
        
        # Step 3: Apply advanced edge enhancement
        # Use a Laplacian filter to detect edges
        laplacian = cv2.Laplacian(bilateral, cv2.CV_64F)
        # Normalize Laplacian to 0-255 range
        laplacian_norm = np.uint8(np.absolute(laplacian)/np.max(np.absolute(laplacian))*255)
        # Enhance edges
        edge_enhanced = cv2.addWeighted(bilateral, 1.0, laplacian_norm, 0.5, 0)
        
        # Step 4: Combine the deconvolved and edge-enhanced images
        combined = cv2.addWeighted(deconvolved, 0.6, edge_enhanced, 0.4, 0)
        
        # Step 5: Apply an adaptive contrast enhancement 
        # Create a CLAHE object with optimal parameters for text
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        contrast_enhanced = clahe.apply(combined)
        
        # Step 6: Use a strong text-specific sharpening kernel
        text_kernel = np.array([[-1, -1, -1, -1, -1],
                               [-1, -1, -1, -1, -1],
                               [-1, -1, 25, -1, -1],
                               [-1, -1, -1, -1, -1],
                               [-1, -1, -1, -1, -1]])
        text_sharpened = cv2.filter2D(contrast_enhanced, -1, text_kernel)
        
        # Step 7: Blend the results for optimal text clarity
        result = cv2.addWeighted(contrast_enhanced, 0.6, text_sharpened, 0.4, 0)
        
        # If original was color, convert back to color
        if is_color:
            # Preserve original color but use enhanced luminance
            # Convert original to HSV
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            h, s, v = cv2.split(hsv)
            # Replace V channel with our enhanced image
            enhanced_hsv = cv2.merge([h, s, result])
            # Convert back to BGR
            return cv2.cvtColor(enhanced_hsv, cv2.COLOR_HSV2BGR)
        else:
            return result
    
    def preprocess_image(self, image):
        """Advanced preprocessing for better text recognition"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Scale up for better quality
        height, width = gray.shape
        scale_factor = max(4, 200 // width) if width < 200 else 4
        
        # Use INTER_CUBIC for better quality upscaling
        resized = cv2.resize(gray, None, fx=scale_factor, fy=scale_factor, 
                           interpolation=cv2.INTER_CUBIC)
        
        # Apply edge-preserving denoising based on image characteristics
        if np.std(resized) > 30:  # Only denoise if there's significant noise
            denoised = cv2.bilateralFilter(resized, 5, 50, 50)
        else:
            denoised = resized
        
        # Create multiple versions for different purposes
        versions = []
        
        # Version 1: Enhanced for general use
        # Apply CLAHE for better contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(denoised)
        versions.append(enhanced)
        
        # Version 2: Text-optimized with edge enhancement
        kernel = np.array([[-1,-1,-1], 
                          [-1, 9,-1],
                          [-1,-1,-1]])
        sharpened = cv2.filter2D(enhanced, -1, kernel)
        versions.append(sharpened)
        
        # Version 3: Inverted (in case background is dark)
        inverted = cv2.bitwise_not(enhanced)
        versions.append(inverted)
        
        return versions
    
    def get_random_image(self, dataset_path):
        """Get random image from dataset"""
        dataset_path = Path(dataset_path)
        categories = ['Too Bright false', 'Too Bright pass', 'Too Dark false', 
                     'Too Dark pass', 'Vibration false', 'Vibration pass']
        
        all_images = []
        for category in categories:
            category_path = dataset_path / category
            if category_path.exists():
                image_files = list(category_path.glob('*'))
                valid_images = [f for f in image_files if f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']]
                for img in valid_images:
                    all_images.append((img, category))
        
        if not all_images:
            raise ValueError("No valid images found in dataset")
        
        return random.choice(all_images)
    
    def process_random_image(self, dataset_path, output_dir="image_processing_output"):
        """Process random image with enhancement pipeline"""
        plt.close('all')
        
        random_image_path, category = self.get_random_image(dataset_path)
        
        print(f"Selected image: {random_image_path.name}")
        print(f"Category: {category}")
        
        original = cv2.imread(str(random_image_path))
        if original is None:
            raise ValueError(f"Could not load image: {random_image_path}")
        
        # Apply appropriate enhancement
        if 'Too Bright' in category:
            enhanced = self.enhance_overexposed_image(original)
        elif 'Too Dark' in category:
            enhanced = self.enhance_dark_image(original)
        elif 'Vibration' in category:
            enhanced = self.enhance_blurry_image(original)
        else:
            enhanced = original.copy()
        
        # Convert for display
        original_rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
        enhanced_rgb = cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB)
        
        # Create visualization
        fig, axes = plt.subplots(1, 2, figsize=(16, 8))
        fig.suptitle(f'Image Enhancement Pipeline - {category}', fontsize=16, fontweight='bold')
        
        axes[0].imshow(original_rgb)
        axes[0].set_title(f'Original ({category})', fontsize=12)
        axes[0].axis('off')
        
        axes[1].imshow(enhanced_rgb)
        axes[1].set_title('Enhanced Image', fontsize=12)
        axes[1].axis('off')
        
        plt.tight_layout()
        plt.show()
        
        # Create histogram comparison
        fig2, axes2 = plt.subplots(1, 1, figsize=(10, 6))
        fig2.suptitle('Histogram Comparison', fontsize=16, fontweight='bold')
        
        original_gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
        enhanced_gray = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)
        
        axes2.hist(original_gray.ravel(), bins=256, alpha=0.6, label='Original', color='red', density=True)
        axes2.hist(enhanced_gray.ravel(), bins=256, alpha=0.6, label='Enhanced', color='blue', density=True)
        axes2.set_xlabel('Pixel Intensity')
        axes2.set_ylabel('Density')
        axes2.legend()
        axes2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        # Save outputs
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        filename = random_image_path.stem
        
        cv2.imwrite(str(output_path / f"{filename}_original.jpg"), original)
        cv2.imwrite(str(output_path / f"{filename}_enhanced.jpg"), enhanced)
        fig.savefig(str(output_path / f"{filename}_comparison.png"), dpi=300, bbox_inches='tight')
        fig2.savefig(str(output_path / f"{filename}_histogram.png"), dpi=300, bbox_inches='tight')
        
        print(f"\nFiles saved to {output_path}:")
        print(f"- {filename}_original.jpg")
        print(f"- {filename}_enhanced.jpg") 
        print(f"- {filename}_comparison.png")
        print(f"- {filename}_histogram.png")
        
        return original, enhanced
    
    def process_single_image(self, image_path, enhancement_type="auto", output_dir="single_image_output"):
        """Process a single image with specified enhancement"""
        image_path = Path(image_path)
        
        if not image_path.exists():
            raise ValueError(f"Image not found: {image_path}")
        
        original = cv2.imread(str(image_path))
        if original is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        # Apply enhancement based on type
        if enhancement_type == "bright" or enhancement_type == "overexposed":
            enhanced = self.enhance_overexposed_image(original)
            enhancement_name = "Overexposed"
        elif enhancement_type == "dark":
            enhanced = self.enhance_dark_image(original)
            enhancement_name = "Dark"
        elif enhancement_type == "blur" or enhancement_type == "vibration":
            enhanced = self.enhance_blurry_image(original)
            enhancement_name = "Vibration/Blur"
        elif enhancement_type == "auto":
            # Auto-detect enhancement needed
            gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
            mean_brightness = np.mean(gray)
            
            # Check for blur using Laplacian variance
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            
            if mean_brightness > 180:
                enhanced = self.enhance_overexposed_image(original)
                enhancement_name = "Auto-detected: Overexposed"
                print("Auto-detected: Overexposed image - applying brightness correction")
            elif mean_brightness < 60:
                enhanced = self.enhance_dark_image(original)
                enhancement_name = "Auto-detected: Dark"
                print("Auto-detected: Dark image - applying brightness enhancement")
            elif laplacian_var < 100:
                enhanced = self.enhance_blurry_image(original)
                enhancement_name = "Auto-detected: Vibration/Blur"
                print("Auto-detected: Blurry image - applying sharpening")
            else:
                enhanced = original.copy()
                enhancement_name = "Auto-detected: No enhancement needed"
                print("Auto-detected: Image quality acceptable - no enhancement applied")
        else:
            enhanced = original.copy()
            enhancement_name = "None"
        
        # Create visualization
        plt.figure(figsize=(15, 10))
        
        plt.subplot(2, 2, 1)
        plt.imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
        plt.title('Original Image')
        plt.axis('off')
        
        plt.subplot(2, 2, 2)
        plt.imshow(cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB))
        plt.title(f'Enhanced Image ({enhancement_name})')
        plt.axis('off')
        
        # Histograms
        plt.subplot(2, 2, 3)
        original_gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
        plt.hist(original_gray.ravel(), bins=256, alpha=0.7, color='red', density=True)
        plt.title('Original Histogram')
        plt.xlabel('Pixel Intensity')
        plt.ylabel('Density')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(2, 2, 4)
        enhanced_gray = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)
        plt.hist(enhanced_gray.ravel(), bins=256, alpha=0.7, color='blue', density=True)
        plt.title('Enhanced Histogram')
        plt.xlabel('Pixel Intensity')
        plt.ylabel('Density')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        # Save outputs
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        filename = image_path.stem
        
        cv2.imwrite(str(output_path / f"{filename}_original.jpg"), original)
        cv2.imwrite(str(output_path / f"{filename}_enhanced.jpg"), enhanced)
        plt.savefig(str(output_path / f"{filename}_analysis.png"), dpi=300, bbox_inches='tight')
        
        print(f"\nProcessed: {image_path.name}")
        print(f"Enhancement applied: {enhancement_name}")
        print(f"Files saved to {output_path}:")
        print(f"- {filename}_original.jpg")
        print(f"- {filename}_enhanced.jpg") 
        print(f"- {filename}_analysis.png")
        
        return original, enhanced

# Usage examples
if __name__ == "__main__":
    processor = ImageProcessor()

    # Example 1: Process random image from dataset
    dataset_path = "/kaggle/input/intelocr/OCR_DataValidation_Noise"

    try:
        # Process a random image from dataset
        original, enhanced = processor.process_random_image(dataset_path)
        
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure the dataset path exists and contains valid images")

    # Example 2: Process a specific image
    # processor.process_single_image("path/to/your/image.jpg", enhancement_type="auto")
