import easyocr
import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import glob
import random
import warnings
import re
from collections import Counter
warnings.filterwarnings('ignore')

class ImprovedAdaptiveOCRProcessor:
    def __init__(self, languages=['en']):
        self.reader = easyocr.Reader(languages, gpu=True)
        
    def assess_image_quality(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if len(image.shape) == 3 else image
        
        mean_brightness = np.mean(gray)
        std_brightness = np.std(gray)
        
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        hist_normalized = hist.ravel() / hist.sum()
        
        dark_pixels = np.sum(gray < 60) / gray.size
        bright_pixels = np.sum(gray > 200) / gray.size
        mid_pixels = np.sum((gray >= 80) & (gray <= 180)) / gray.size
        
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        edges = cv2.Canny(gray, 30, 100)
        edge_density = np.sum(edges > 0) / edges.size
        
        noise_level = np.std(gray - cv2.medianBlur(gray, 5))
        
        gray_norm = gray.astype(np.float64) / 255.0
        glcm_contrast = np.var(gray_norm)
        
        assessment = {
            'mean_brightness': mean_brightness,
            'std_brightness': std_brightness,
            'dark_pixel_ratio': dark_pixels,
            'bright_pixel_ratio': bright_pixels,
            'mid_pixel_ratio': mid_pixels,
            'contrast_score': laplacian_var,
            'edge_density': edge_density,
            'noise_level': noise_level,
            'texture_contrast': glcm_contrast,
            'histogram': hist_normalized
        }
        
        issues = []
        severity_scores = {}
        
        if mean_brightness < 80:
            issues.append('too_dark')
            severity_scores['too_dark'] = (80 - mean_brightness) / 80
        elif mean_brightness > 180:
            issues.append('too_bright')
            severity_scores['too_bright'] = (mean_brightness - 180) / 75
            
        if std_brightness < 35:
            issues.append('low_contrast')
            severity_scores['low_contrast'] = (35 - std_brightness) / 35
            
        if laplacian_var < 100:
            issues.append('blurry')
            severity_scores['blurry'] = (100 - laplacian_var) / 100
            
        if noise_level > 15:
            issues.append('noisy')
            severity_scores['noisy'] = min(noise_level / 30, 1.0)
            
        if dark_pixels > 0.4:
            issues.append('underexposed')
            severity_scores['underexposed'] = min(dark_pixels, 1.0)
        elif bright_pixels > 0.4:
            issues.append('overexposed')
            severity_scores['overexposed'] = min(bright_pixels, 1.0)
        
        assessment['issues'] = issues
        assessment['severity_scores'] = severity_scores
        assessment['needs_enhancement'] = len(issues) > 0
        assessment['total_severity'] = sum(severity_scores.values())
        
        return assessment
    
    def get_adaptive_confidence_threshold(self, assessment):
        total_severity = assessment.get('total_severity', 0)
        if total_severity > 4.5:
            return 0.01
        elif total_severity > 3.5:
            return 0.05
        elif total_severity > 2.5:
            return 0.1
        elif total_severity > 1.5:
            return 0.2
        else:
            return 0.3
    
    def get_optimal_ocr_parameters(self, assessment):
        base_params = {
            'detail': 1,
            'paragraph': False,
        }
        
        parameter_sets = []
        
        total_severity = assessment.get('total_severity', 0)
        has_blur = 'blurry' in assessment['issues']
        has_noise = 'noisy' in assessment['issues']
        has_contrast_issues = 'low_contrast' in assessment['issues']
        has_darkness_issues = 'too_dark' in assessment['issues'] or 'underexposed' in assessment['issues']
        
        if total_severity > 4.0:
            parameter_sets.extend([
                {**base_params, 'width_ths': 0.1, 'height_ths': 0.1, 'text_threshold': 0.1, 'low_text': 0.05, 'link_threshold': 0.05, 'canvas_size': 5120, 'mag_ratio': 3.0},
                {**base_params, 'width_ths': 0.15, 'height_ths': 0.15, 'text_threshold': 0.15, 'low_text': 0.08, 'link_threshold': 0.08, 'canvas_size': 4096, 'mag_ratio': 2.5},
                {**base_params, 'width_ths': 0.2, 'height_ths': 0.2, 'text_threshold': 0.2, 'low_text': 0.1, 'link_threshold': 0.1, 'canvas_size': 3840, 'mag_ratio': 2.2},
            ])
        
        if total_severity > 2.5:
            parameter_sets.extend([
                {**base_params, 'width_ths': 0.25, 'height_ths': 0.25, 'text_threshold': 0.25, 'low_text': 0.12, 'link_threshold': 0.12, 'canvas_size': 3840, 'mag_ratio': 2.0},
                {**base_params, 'width_ths': 0.3, 'height_ths': 0.3, 'text_threshold': 0.3, 'low_text': 0.15, 'link_threshold': 0.15, 'canvas_size': 3200, 'mag_ratio': 1.8},
                {**base_params, 'width_ths': 0.35, 'height_ths': 0.35, 'text_threshold': 0.35, 'low_text': 0.18, 'link_threshold': 0.18, 'canvas_size': 3200, 'mag_ratio': 1.7},
            ])
        
        if has_contrast_issues or has_darkness_issues:
            parameter_sets.extend([
                {**base_params, 'width_ths': 0.4, 'height_ths': 0.3, 'text_threshold': 0.3, 'low_text': 0.15, 'link_threshold': 0.15, 'canvas_size': 3200, 'mag_ratio': 1.8},
                {**base_params, 'width_ths': 0.5, 'height_ths': 0.4, 'text_threshold': 0.4, 'low_text': 0.2, 'link_threshold': 0.2, 'canvas_size': 2880, 'mag_ratio': 1.6},
            ])
        
        parameter_sets.extend([
            {**base_params, 'width_ths': 0.6, 'height_ths': 0.6, 'text_threshold': 0.5, 'low_text': 0.25, 'link_threshold': 0.25, 'canvas_size': 2560, 'mag_ratio': 1.5},
            {**base_params, 'width_ths': 0.7, 'height_ths': 0.7, 'text_threshold': 0.6, 'low_text': 0.3, 'link_threshold': 0.3, 'canvas_size': 2560, 'mag_ratio': 1.4},
            {**base_params, 'width_ths': 0.8, 'height_ths': 0.6, 'text_threshold': 0.65, 'low_text': 0.35, 'link_threshold': 0.35, 'canvas_size': 2240, 'mag_ratio': 1.3},
        ])
        
        return parameter_sets
    
    def is_valid_text_adaptive(self, text, assessment, min_length=1):
        total_severity = assessment.get('total_severity', 0)
        
        if len(text.strip()) < min_length:
            return False
        
        if total_severity > 4.0:
            if re.match(r'^[^\w\s]*$', text):
                return False
            return len(text.strip()) >= 1
        elif total_severity > 2.0:
            if re.match(r'^[^\w\s]*$', text):
                return False
            alpha_ratio = sum(c.isalpha() for c in text) / len(text)
            if alpha_ratio < 0.05 and len(text) > 2:
                return False
            return True
        else:
            if re.match(r'^[^\w\s]*$', text):
                return False
            alpha_ratio = sum(c.isalpha() for c in text) / len(text)
            if alpha_ratio < 0.1 and len(text) > 3:
                return False
            return True
    
    def filter_and_merge_results(self, all_results, assessment):
        if not all_results:
            return []
        
        confidence_threshold = self.get_adaptive_confidence_threshold(assessment)
        
        valid_detections = []
        for results in all_results:
            for detection in results:
                bbox, text, confidence = detection
                if self.is_valid_text_adaptive(text, assessment) and confidence > confidence_threshold:
                    valid_detections.append(detection)
        
        if not valid_detections:
            return []
        
        def boxes_overlap(box1, box2, threshold=0.6):
            x1_min, y1_min = np.min(box1, axis=0)
            x1_max, y1_max = np.max(box1, axis=0)
            x2_min, y2_min = np.min(box2, axis=0)
            x2_max, y2_max = np.max(box2, axis=0)
            
            intersection_area = max(0, min(x1_max, x2_max) - max(x1_min, x2_min)) * max(0, min(y1_max, y2_max) - max(y1_min, y2_min))
            box1_area = (x1_max - x1_min) * (y1_max - y1_min)
            box2_area = (x2_max - x2_min) * (y2_max - y2_min)
            
            if box1_area == 0 or box2_area == 0:
                return False
            
            overlap_ratio1 = intersection_area / box1_area
            overlap_ratio2 = intersection_area / box2_area
            
            return overlap_ratio1 > threshold or overlap_ratio2 > threshold
        
        merged_results = []
        used_indices = set()
        
        for i, detection in enumerate(valid_detections):
            if i in used_indices:
                continue
            
            bbox, text, confidence = detection
            overlapping_detections = [detection]
            used_indices.add(i)
            
            for j, other_detection in enumerate(valid_detections[i+1:], i+1):
                if j in used_indices:
                    continue
                
                other_bbox, other_text, other_confidence = other_detection
                if boxes_overlap(bbox, other_bbox):
                    overlapping_detections.append(other_detection)
                    used_indices.add(j)
            
            if len(overlapping_detections) == 1:
                merged_results.append(detection)
            else:
                best_detection = max(overlapping_detections, key=lambda x: x[2])
                
                texts = [det[1] for det in overlapping_detections]
                text_counts = Counter(texts)
                most_common_text = text_counts.most_common(1)[0][0]
                
                confidences = [det[2] for det in overlapping_detections if det[1] == most_common_text]
                avg_confidence = np.mean(confidences)
                
                merged_results.append((best_detection[0], most_common_text, avg_confidence))
        
        merged_results.sort(key=lambda x: x[2], reverse=True)
        return merged_results
    
    def post_process_text(self, text):
        text = re.sub(r'[^\w\s\.\,\!\?\;\:\-\(\)\[\]\{\}\"\'\/\\\@\#\$\%\^\&\*\+\=\<\>\|]', '', text)
        
        text = re.sub(r'(\w)[Il1|]{1,2}(\w)', r'\1\2', text)
        text = re.sub(r'\b[Il1|]{1,2}(\w)', r'\1', text)
        text = re.sub(r'(\w)[Il1|]{1,2}\b', r'\1', text)
        
        text = re.sub(r'[O0o]{2,}', 'O', text)
        text = re.sub(r'[Il1|]{2,}', 'I', text)
        
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
    
    def multi_pass_ocr(self, image_path, assessment):
        parameter_sets = self.get_optimal_ocr_parameters(assessment)
        all_results = []
        
        print(f"Running {len(parameter_sets)} OCR passes with different parameters...")
        
        for i, params in enumerate(parameter_sets):
            try:
                results = self.reader.readtext(image_path, **params)
                all_results.append(results)
                print(f"  Pass {i+1}: {len(results)} detections, avg confidence: {np.mean([r[2] for r in results]):.3f}" if results else f"  Pass {i+1}: 0 detections")
            except Exception as e:
                print(f"  Pass {i+1}: Failed - {str(e)}")
                continue
        
        if not all_results:
            return []
        
        merged_results = self.filter_and_merge_results(all_results, assessment)
        
        final_results = []
        for bbox, text, confidence in merged_results:
            processed_text = self.post_process_text(text)
            if self.is_valid_text_adaptive(processed_text, assessment):
                final_results.append((bbox, processed_text, confidence))
        
        final_results.sort(key=lambda x: (x[0][0][1], x[0][0][0]))
        
        print(f"Final merged results: {len(final_results)} detections")
        return final_results
    
    def debug_zero_detections(self, image_path, assessment):
        print("Running debug analysis for zero detections...")
        try:
            raw_results = self.reader.readtext(image_path, 
                                             width_ths=0.05, height_ths=0.05, 
                                             text_threshold=0.05, low_text=0.01,
                                             link_threshold=0.01, detail=1)
            
            print(f"Raw debug detections (ultra-low threshold): {len(raw_results)}")
            for i, (bbox, text, conf) in enumerate(raw_results[:10]):
                print(f"  [{i+1}] '{text}' (conf: {conf:.4f})")
            
            if len(raw_results) > 10:
                print(f"  ... and {len(raw_results)-10} more detections")
                
        except Exception as e:
            print(f"Debug analysis failed: {str(e)}")
    
    def create_enhancement_variants(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if len(image.shape) == 3 else image
        variants = {'original': gray}
        
        clahe_gentle = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        variants['gentle_clahe'] = clahe_gentle.apply(gray)
        
        clahe_moderate = cv2.createCLAHE(clipLimit=3.5, tileGridSize=(6,6))
        variants['moderate_clahe'] = clahe_moderate.apply(gray)
        
        clahe_aggressive = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(4,4))
        variants['aggressive_clahe'] = clahe_aggressive.apply(gray)
        
        gamma_bright = 0.6
        gamma_enhanced = np.power(gray / 255.0, gamma_bright) * 255.0
        variants['gamma_bright'] = gamma_enhanced.astype(np.uint8)
        
        gamma_very_bright = 0.4
        gamma_very_enhanced = np.power(gray / 255.0, gamma_very_bright) * 255.0
        variants['gamma_very_bright'] = gamma_very_enhanced.astype(np.uint8)
        
        gamma_dark = 1.3
        gamma_reduced = np.power(gray / 255.0, gamma_dark) * 255.0
        variants['gamma_dark'] = gamma_reduced.astype(np.uint8)
        
        gaussian = cv2.GaussianBlur(gray, (0, 0), 1.0)
        unsharp = cv2.addWeighted(gray, 1.5, gaussian, -0.5, 0)
        variants['unsharp'] = np.clip(unsharp, 0, 255).astype(np.uint8)
        
        denoised = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
        clahe_denoised = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(6,6))
        variants['denoise_contrast'] = clahe_denoised.apply(denoised)
        
        bilateral = cv2.bilateralFilter(gray, 9, 80, 80)
        variants['bilateral'] = bilateral
        
        alpha = 1.3
        beta = 20
        contrast_bright = cv2.convertScaleAbs(gray, alpha=alpha, beta=beta)
        variants['contrast_bright'] = contrast_bright
        
        alpha_aggressive = 1.8
        beta_aggressive = 40
        contrast_aggressive = cv2.convertScaleAbs(gray, alpha=alpha_aggressive, beta=beta_aggressive)
        variants['contrast_aggressive'] = contrast_aggressive
        
        combined_gamma = np.power(gray / 255.0, 0.7) * 255.0
        combined_gamma = combined_gamma.astype(np.uint8)
        clahe_combined = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(6,6))
        variants['gamma_clahe_combo'] = clahe_combined.apply(combined_gamma)
        
        extreme_gamma = np.power(gray / 255.0, 0.5) * 255.0
        extreme_gamma = extreme_gamma.astype(np.uint8)
        extreme_enhanced = cv2.convertScaleAbs(extreme_gamma, alpha=1.5, beta=30)
        clahe_extreme = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(4,4))
        variants['extreme_enhancement'] = clahe_extreme.apply(extreme_enhanced)
        
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        morph_enhanced = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)
        morph_combined = cv2.addWeighted(gray, 0.8, morph_enhanced, 0.2, 0)
        variants['morphological'] = morph_combined.astype(np.uint8)
        
        return variants
    
    def quick_ocr_test(self, image_array, test_params=None):
        if test_params is None:
            test_params = {
                'detail': 1,
                'paragraph': False,
                'width_ths': 0.3,
                'height_ths': 0.3,
                'text_threshold': 0.3,
                'low_text': 0.15,
                'link_threshold': 0.15,
                'canvas_size': 3200,
                'mag_ratio': 1.8
            }
        
        temp_path = 'temp_quick_test.jpg'
        cv2.imwrite(temp_path, image_array)
        
        try:
            results = self.reader.readtext(temp_path, **test_params)
            os.remove(temp_path)
            
            if results:
                confidences = [r[2] for r in results]
                avg_confidence = np.mean(confidences)
                detection_count = len(results)
                text_length = sum(len(r[1]) for r in results)
                valid_detections = sum(1 for r in results if len(r[1].strip()) > 0)
                
                score = (detection_count * 0.25) + (avg_confidence * 0.5) + (text_length * 0.01) + (valid_detections * 0.25)
            else:
                score = 0
                avg_confidence = 0
                detection_count = 0
                text_length = 0
            
            return {
                'score': score,
                'confidence': avg_confidence,
                'detections': detection_count,
                'text_length': text_length,
                'results': results
            }
        except Exception as e:
            if os.path.exists(temp_path):
                os.remove(temp_path)
            return {'score': 0, 'confidence': 0, 'detections': 0, 'text_length': 0, 'results': []}
    
    def adaptive_enhance_selection(self, image, assessment):
        variants = self.create_enhancement_variants(image)
        
        variant_scores = {}
        
        for variant_name, variant_image in variants.items():
            score_data = self.quick_ocr_test(variant_image)
            variant_scores[variant_name] = score_data
        
        best_variant = max(variant_scores.keys(), key=lambda k: variant_scores[k]['score'])
        best_score = variant_scores[best_variant]['score']
        best_confidence = variant_scores[best_variant]['confidence']
        
        print(f"Selected best variant: {best_variant} (Score: {best_score:.3f})")
        
        original_score = variant_scores['original']['score']
        original_confidence = variant_scores['original']['confidence']
        
        total_severity = assessment.get('total_severity', 0)
        severe_issues = len([s for s in assessment['severity_scores'].values() if s > 0.7])
        
        if total_severity > 4.0:
            improvement_threshold = 1.01
            confidence_threshold = 0.9
            print(f"Extreme image issues detected (severity: {total_severity:.2f}), using minimal threshold")
        elif total_severity > 2.5:
            improvement_threshold = 1.02
            confidence_threshold = 0.95
            print(f"Severe image issues detected (severity: {total_severity:.2f}), using aggressive threshold")
        elif total_severity > 1.0:
            improvement_threshold = 1.05
            confidence_threshold = 1.0
            print(f"Moderate image issues detected (severity: {total_severity:.2f}), using moderate threshold")
        else:
            improvement_threshold = 1.1
            confidence_threshold = 1.05
        
        score_improved = best_score > original_score * improvement_threshold
        confidence_acceptable = (best_confidence >= original_confidence * confidence_threshold) or (original_confidence < 0.05)
        
        if score_improved and confidence_acceptable:
            print(f"Enhancement accepted: {(best_score/original_score-1)*100:.1f}% score improvement, confidence: {original_confidence:.3f} → {best_confidence:.3f}")
        else:
            best_variant = 'original'
            if not score_improved:
                print(f"Score improvement {(best_score/original_score-1)*100:.1f}% below threshold, using original image")
            else:
                print(f"Confidence declined too much ({original_confidence:.3f} → {best_confidence:.3f}), using original image")
        
        return variants[best_variant], variant_scores, best_variant
    
    def targeted_enhancement(self, image, assessment, selected_variant):
        if selected_variant == 'original':
            return image
        
        enhanced = image.copy()
        total_severity = assessment.get('total_severity', 0)
        
        if total_severity > 4.0:
            for issue, severity in assessment['severity_scores'].items():
                if severity > 0.8:
                    if issue == 'too_dark' and 'bright' not in selected_variant:
                        gamma = max(0.3, 0.8 - (severity * 0.3))
                        enhanced = np.power(enhanced / 255.0, gamma) * 255.0
                        enhanced = enhanced.astype(np.uint8)
                    
                    elif issue == 'low_contrast':
                        clahe = cv2.createCLAHE(clipLimit=min(6.0, 2.0 + severity * 2), tileGridSize=(4,4))
                        enhanced = clahe.apply(enhanced)
                    
                    elif issue == 'blurry':
                        kernel = np.array([[-1,-1,-1], [-1,10,-1], [-1,-1,-1]])
                        enhanced = cv2.filter2D(enhanced, -1, kernel)
                        enhanced = np.clip(enhanced, 0, 255).astype(np.uint8)
        else:
            for issue, severity in assessment['severity_scores'].items():
                if severity > 0.7:
                    if issue == 'too_dark' and 'bright' not in selected_variant:
                        gamma = 0.7 - (severity * 0.2)
                        enhanced = np.power(enhanced / 255.0, gamma) * 255.0
                        enhanced = enhanced.astype(np.uint8)
                    
                    elif issue == 'too_bright' and 'dark' not in selected_variant:
                        enhanced = cv2.convertScaleAbs(enhanced, alpha=0.9, beta=-10)
                    
                    elif issue == 'low_contrast':
                        clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8,8))
                        enhanced = clahe.apply(enhanced)
                    
                    elif issue == 'blurry':
                        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
                        enhanced = cv2.filter2D(enhanced, -1, kernel)
                        enhanced = np.clip(enhanced, 0, 255).astype(np.uint8)
        
        return enhanced
    
    def process_image_adaptive(self, image_path):
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        assessment = self.assess_image_quality(image_rgb)
        
        print(f"Image assessment: {assessment['issues']}")
        print(f"Severity scores: {assessment['severity_scores']}")
        print(f"Total severity: {assessment['total_severity']:.2f}")
        
        if not assessment['needs_enhancement']:
            print("No enhancement needed")
            enhanced_image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
            enhanced_results = self.multi_pass_ocr(image_path, assessment)
            selected_variant = 'original'
            variant_scores = {}
        else:
            enhanced_image, variant_scores, selected_variant = self.adaptive_enhance_selection(image_rgb, assessment)
            
            enhanced_image = self.targeted_enhancement(enhanced_image, assessment, selected_variant)
            
            temp_enhanced_path = 'temp_enhanced_adaptive.jpg'
            cv2.imwrite(temp_enhanced_path, enhanced_image)
            
            enhanced_results = self.multi_pass_ocr(temp_enhanced_path, assessment)
            
            if os.path.exists(temp_enhanced_path):
                os.remove(temp_enhanced_path)
        
        original_results = self.multi_pass_ocr(image_path, assessment)
        
        if len(original_results) == 0 and len(enhanced_results) == 0:
            print("\nZero detections found - running debug analysis...")
            self.debug_zero_detections(image_path, assessment)
        
        return {
            'original_image': image_rgb,
            'enhanced_image': enhanced_image,
            'original_results': original_results,
            'enhanced_results': enhanced_results,
            'quality_assessment': assessment,
            'selected_variant': selected_variant,
            'variant_scores': variant_scores
        }
    
    def visualize_adaptive_results(self, results, title="Improved Adaptive OCR Enhancement"):
        fig, axes = plt.subplots(2, 3, figsize=(24, 16))
        
        axes[0,0].imshow(results['original_image'])
        axes[0,0].set_title("Original Image")
        axes[0,0].axis('off')
        
        if len(results['enhanced_image'].shape) == 2:
            axes[0,1].imshow(results['enhanced_image'], cmap='gray')
        else:
            axes[0,1].imshow(results['enhanced_image'])
        axes[0,1].set_title(f"Enhanced Image\n(Method: {results['selected_variant']})")
        axes[0,1].axis('off')
        
        if results['variant_scores']:
            variant_names = list(results['variant_scores'].keys())
            scores = [results['variant_scores'][name]['score'] for name in variant_names]
            
            axes[0,2].bar(range(len(variant_names)), scores)
            axes[0,2].set_xticks(range(len(variant_names)))
            axes[0,2].set_xticklabels(variant_names, rotation=45, ha='right')
            axes[0,2].set_title("Enhancement Variant Scores")
            axes[0,2].set_ylabel("OCR Performance Score")
        else:
            axes[0,2].text(0.5, 0.5, "No Enhancement\nNeeded", ha='center', va='center', transform=axes[0,2].transAxes, fontsize=16)
            axes[0,2].axis('off')
        
        orig_with_boxes = results['original_image'].copy()
        for (bbox, text, confidence) in results['original_results']:
            top_left = tuple([int(val) for val in bbox[0]])
            bottom_right = tuple([int(val) for val in bbox[2]])
            cv2.rectangle(orig_with_boxes, top_left, bottom_right, (255, 0, 0), 3)
            text_pos = (top_left[0], max(top_left[1] - 15, 25))
            cv2.putText(orig_with_boxes, f"{text} ({confidence:.2f})", text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        
        axes[1,0].imshow(orig_with_boxes)
        axes[1,0].set_title(f"Original OCR\n{len(results['original_results'])} detections")
        axes[1,0].axis('off')
        
        enh_with_boxes = results['enhanced_image'].copy()
        if len(enh_with_boxes.shape) == 2:
            enh_with_boxes = cv2.cvtColor(enh_with_boxes, cv2.COLOR_GRAY2RGB)
        
        for (bbox, text, confidence) in results['enhanced_results']:
            top_left = tuple([int(val) for val in bbox[0]])
            bottom_right = tuple([int(val) for val in bbox[2]])
            cv2.rectangle(enh_with_boxes, top_left, bottom_right, (0, 255, 0), 3)
            text_pos = (top_left[0], max(top_left[1] - 15, 25))
            cv2.putText(enh_with_boxes, f"{text} ({confidence:.2f})", text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        axes[1,1].imshow(enh_with_boxes)
        axes[1,1].set_title(f"Enhanced OCR\n{len(results['enhanced_results'])} detections")
        axes[1,1].axis('off')
        
        orig_conf = np.mean([r[2] for r in results['original_results']]) if results['original_results'] else 0
        enh_conf = np.mean([r[2] for r in results['enhanced_results']]) if results['enhanced_results'] else 0
        
        metrics = ['Detections', 'Avg Confidence']
        original_vals = [len(results['original_results']), orig_conf]
        enhanced_vals = [len(results['enhanced_results']), enh_conf]
        
        x = np.arange(len(metrics))
        width = 0.35
        
        axes[1,2].bar(x - width/2, original_vals, width, label='Original', color='red', alpha=0.7)
        axes[1,2].bar(x + width/2, enhanced_vals, width, label='Enhanced', color='green', alpha=0.7)
        
        axes[1,2].set_xlabel('Metrics')
        axes[1,2].set_ylabel('Values')
        axes[1,2].set_title('Performance Comparison')
        axes[1,2].set_xticks(x)
        axes[1,2].set_xticklabels(metrics)
        axes[1,2].legend()
        
        plt.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.show()
    
    def print_adaptive_analysis(self, results, image_path):
        print(f"\n{'='*90}")
        print(f"IMPROVED ADAPTIVE ENHANCEMENT ANALYSIS: {os.path.basename(image_path)}")
        print(f"{'='*90}")
        
        qa = results['quality_assessment']
        print(f"\nIMAGE QUALITY ASSESSMENT:")
        print(f"  Mean Brightness: {qa['mean_brightness']:.1f}")
        print(f"  Brightness Std: {qa['std_brightness']:.1f}")
        print(f"  Dark Pixels: {qa['dark_pixel_ratio']:.1%}")
        print(f"  Bright Pixels: {qa['bright_pixel_ratio']:.1%}")
        print(f"  Contrast Score: {qa['contrast_score']:.1f}")
        print(f"  Edge Density: {qa['edge_density']:.3f}")
        print(f"  Noise Level: {qa['noise_level']:.1f}")
        print(f"  Total Severity: {qa['total_severity']:.2f}")
        print(f"  Detected Issues: {', '.join(qa['issues']) if qa['issues'] else 'None'}")
        print(f"  Severity Scores: {qa['severity_scores']}")
        
        print(f"\nADAPTIVE ENHANCEMENT:")
        print(f"  Selected Method: {results['selected_variant']}")
        print(f"  Confidence Threshold Used: {self.get_adaptive_confidence_threshold(qa):.3f}")
        
        if results['variant_scores']:
            print(f"  Variant Performance:")
            for variant, scores in results['variant_scores'].items():
                print(f"    {variant}: Score={scores['score']:.3f}, Detections={scores['detections']}, Confidence={scores['confidence']:.3f}")
        
        orig_results = results['original_results']
        enh_results = results['enhanced_results']
        
        print(f"\nOCR PERFORMANCE COMPARISON:")
        print(f"  Original - Detections: {len(orig_results)}")
        print(f"  Enhanced - Detections: {len(enh_results)}")
        
        if orig_results:
            orig_avg_conf = np.mean([r[2] for r in orig_results])
            orig_text = ' '.join([r[1] for r in orig_results])
            print(f"  Original - Avg Confidence: {orig_avg_conf:.3f}")
            print(f"  Original - Text: '{orig_text}'")
        
        if enh_results:
            enh_avg_conf = np.mean([r[2] for r in enh_results])
            enh_text = ' '.join([r[1] for r in enh_results])
            print(f"  Enhanced - Avg Confidence: {enh_avg_conf:.3f}")
            print(f"  Enhanced - Text: '{enh_text}'")
        
        detection_improvement = len(enh_results) - len(orig_results)
        print(f"\nIMPROVEMENT METRICS:")
        print(f"  Detection Change: {detection_improvement:+d}")
        
        if orig_results and enh_results:
            conf_improvement = np.mean([r[2] for r in enh_results]) - np.mean([r[2] for r in orig_results])
            print(f"  Confidence Change: {conf_improvement:+.3f}")
            
            if len(orig_results) > 0:
                detection_improvement_pct = (detection_improvement / len(orig_results)) * 100
                print(f"  Detection Improvement: {detection_improvement_pct:+.1f}%")
        
        if len(orig_results) == 0 and len(enh_results) == 0:
            print(f"\n  WARNING: No text detected in either original or enhanced image")
            print(f"  This may indicate extremely poor image quality or text that is unreadable")
            print(f"  Consider manual preprocessing or different imaging conditions")
        
        print(f"{'='*90}")

def get_random_image_from_dataset():
    dataset_folders = [
        "/kaggle/input/intelocr/OCR_DataValidation_Noise/Too Bright false",
        "/kaggle/input/intelocr/OCR_DataValidation_Noise/Too Bright pass", 
        "/kaggle/input/intelocr/OCR_DataValidation_Noise/Too Dark false",
        "/kaggle/input/intelocr/OCR_DataValidation_Noise/Too Dark pass",
        "/kaggle/input/intelocr/OCR_DataValidation_Noise/Vibration false",
        "/kaggle/input/intelocr/OCR_DataValidation_Noise/Vibration pass"
    ]
    
    extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff', '*.tif']
    
    all_images = []
    for folder_path in dataset_folders:
        if os.path.exists(folder_path):
            for ext in extensions:
                all_images.extend(glob.glob(os.path.join(folder_path, ext)))
                all_images.extend(glob.glob(os.path.join(folder_path, ext.upper())))
    
    if not all_images:
        raise ValueError("No images found in dataset folders")
    
    return random.choice(all_images)

def process_improved_adaptive_random_image():
    try:
        random_image_path = get_random_image_from_dataset()
        
        processor = ImprovedAdaptiveOCRProcessor(languages=['en'])
        
        results = processor.process_image_adaptive(random_image_path)
        
        processor.print_adaptive_analysis(results, random_image_path)
        
        folder_name = os.path.basename(os.path.dirname(random_image_path))
        processor.visualize_adaptive_results(results, title=f"{folder_name} - {os.path.basename(random_image_path)}")
        
        return results, random_image_path
        
    except Exception as e:
        print(f"Error processing improved adaptive image: {str(e)}")
        return None, None

if __name__ == "__main__":
    process_improved_adaptive_random_image()
