"""
YOLO Model Performance Optimizer

This module implements advanced YOLO model optimization techniques including:
- Model quantization for faster inference
- Ensemble methods combining multiple YOLO models
- Performance benchmarking and comparison
"""

import torch
import time
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from ultralytics import YOLO
import logging


class YOLOOptimizer:
    """
    Optimizes YOLO models for better performance in proctoring scenarios.
    
    Provides model quantization, ensemble methods, and performance benchmarking
    to improve detection speed and accuracy.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the YOLO optimizer.
        
        Args:
            config: Optional configuration dictionary
        """
        self.config = config or {}
        self.logger = logging.getLogger(f"{__name__}.YOLOOptimizer")
        
        # Performance tracking
        self.benchmark_results = {}
        self.optimized_models = {}
        
        self.logger.info("YOLOOptimizer initialized")
    
    def quantize_model(self, model_path: str, output_path: str) -> bool:
        """
        Quantize a YOLO model for faster inference.
        
        Args:
            model_path: Path to the original model
            output_path: Path to save the quantized model
            
        Returns:
            True if quantization successful, False otherwise
        """
        try:
            model = YOLO(model_path)
            
            # Export to TensorRT or ONNX for optimization
            model.export(format='onnx', optimize=True)
            
            self.logger.info(f"Model quantized and saved to {output_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Model quantization failed: {e}")
            return False
    
    def benchmark_model(self, model_path: str, test_images: List[np.ndarray]) -> Dict[str, float]:
        """
        Benchmark model performance on test images.
        
        Args:
            model_path: Path to the model to benchmark
            test_images: List of test images
            
        Returns:
            Dictionary containing benchmark metrics
        """
        try:
            model = YOLO(model_path)
            
            inference_times = []
            total_detections = 0
            
            for image in test_images:
                start_time = time.time()
                results = model(image, verbose=False)
                inference_time = (time.time() - start_time) * 1000  # Convert to ms
                
                inference_times.append(inference_time)
                
                # Count detections
                for result in results:
                    if result.boxes is not None:
                        total_detections += len(result.boxes)
            
            metrics = {
                'avg_inference_time_ms': np.mean(inference_times),
                'min_inference_time_ms': np.min(inference_times),
                'max_inference_time_ms': np.max(inference_times),
                'total_detections': total_detections,
                'avg_detections_per_image': total_detections / len(test_images) if test_images else 0,
                'fps': 1000 / np.mean(inference_times) if inference_times else 0
            }
            
            self.benchmark_results[model_path] = metrics
            self.logger.info(f"Benchmark completed for {model_path}: {metrics}")
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Benchmarking failed for {model_path}: {e}")
            return {}
    
    def create_ensemble(self, model_paths: List[str], weights: Optional[List[float]] = None) -> 'EnsembleYOLO':
        """
        Create an ensemble of YOLO models.
        
        Args:
            model_paths: List of paths to YOLO models
            weights: Optional weights for each model
            
        Returns:
            EnsembleYOLO instance
        """
        if weights is None:
            weights = [1.0] * len(model_paths)
        
        return EnsembleYOLO(model_paths, weights)
    
    def optimize_for_deployment(self, model_path: str, target_device: str = 'cpu') -> str:
        """
        Optimize model for specific deployment target.
        
        Args:
            model_path: Path to the model to optimize
            target_device: Target device ('cpu', 'gpu', 'edge')
            
        Returns:
            Path to optimized model
        """
        try:
            model = YOLO(model_path)
            
            if target_device == 'cpu':
                # Optimize for CPU inference
                optimized_path = model_path.replace('.pt', '_cpu_optimized.onnx')
                model.export(format='onnx', optimize=True, half=False)
                
            elif target_device == 'gpu':
                # Optimize for GPU inference
                optimized_path = model_path.replace('.pt', '_gpu_optimized.engine')
                model.export(format='engine', optimize=True, half=True)
                
            elif target_device == 'edge':
                # Optimize for edge devices
                optimized_path = model_path.replace('.pt', '_edge_optimized.tflite')
                model.export(format='tflite', optimize=True, int8=True)
            
            else:
                raise ValueError(f"Unsupported target device: {target_device}")
            
            self.optimized_models[target_device] = optimized_path
            self.logger.info(f"Model optimized for {target_device}: {optimized_path}")
            
            return optimized_path
            
        except Exception as e:
            self.logger.error(f"Optimization failed for {target_device}: {e}")
            return model_path


class EnsembleYOLO:
    """
    Ensemble of YOLO models for improved detection accuracy.
    """
    
    def __init__(self, model_paths: List[str], weights: List[float]):
        """
        Initialize ensemble with multiple YOLO models.
        
        Args:
            model_paths: List of paths to YOLO models
            weights: Weights for each model in ensemble
        """
        self.models = [YOLO(path) for path in model_paths]
        self.weights = np.array(weights) / np.sum(weights)  # Normalize weights
        self.logger = logging.getLogger(f"{__name__}.EnsembleYOLO")
        
        self.logger.info(f"Ensemble created with {len(self.models)} models")
    
    def predict(self, image: np.ndarray, confidence_threshold: float = 0.5) -> List[Dict[str, Any]]:
        """
        Run ensemble prediction on image.
        
        Args:
            image: Input image
            confidence_threshold: Minimum confidence threshold
            
        Returns:
            List of ensemble detection results
        """
        all_detections = []
        
        # Get predictions from all models
        for i, model in enumerate(self.models):
            try:
                results = model(image, verbose=False)
                weight = self.weights[i]
                
                for result in results:
                    if result.boxes is not None:
                        for box in result.boxes:
                            detection = {
                                'bbox': box.xyxy[0].cpu().numpy(),
                                'confidence': box.conf[0].item() * weight,
                                'class_id': int(box.cls[0]),
                                'model_index': i
                            }
                            all_detections.append(detection)
                            
            except Exception as e:
                self.logger.error(f"Error in model {i}: {e}")
                continue
        
        # Apply Non-Maximum Suppression across ensemble
        final_detections = self._ensemble_nms(all_detections, confidence_threshold)
        
        return final_detections
    
    def _ensemble_nms(self, detections: List[Dict[str, Any]], threshold: float) -> List[Dict[str, Any]]:
        """
        Apply Non-Maximum Suppression across ensemble detections.
        
        Args:
            detections: List of all detections from ensemble
            threshold: Confidence threshold
            
        Returns:
            List of filtered detections
        """
        if not detections:
            return []
        
        # Filter by confidence threshold
        filtered = [d for d in detections if d['confidence'] >= threshold]
        
        if not filtered:
            return []
        
        # Group by class
        class_groups = {}
        for detection in filtered:
            class_id = detection['class_id']
            if class_id not in class_groups:
                class_groups[class_id] = []
            class_groups[class_id].append(detection)
        
        final_detections = []
        
        # Apply NMS within each class
        for class_id, class_detections in class_groups.items():
            if len(class_detections) == 1:
                final_detections.extend(class_detections)
                continue
            
            # Sort by confidence
            class_detections.sort(key=lambda x: x['confidence'], reverse=True)
            
            # Apply NMS
            keep = []
            while class_detections:
                best = class_detections.pop(0)
                keep.append(best)
                
                # Remove overlapping detections
                class_detections = [
                    det for det in class_detections
                    if self._calculate_iou(best['bbox'], det['bbox']) < 0.5
                ]
            
            final_detections.extend(keep)
        
        return final_detections
    
    def _calculate_iou(self, box1: np.ndarray, box2: np.ndarray) -> float:
        """
        Calculate Intersection over Union (IoU) between two bounding boxes.
        
        Args:
            box1: First bounding box [x1, y1, x2, y2]
            box2: Second bounding box [x1, y1, x2, y2]
            
        Returns:
            IoU value between 0 and 1
        """
        # Calculate intersection area
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        if x2 <= x1 or y2 <= y1:
            return 0.0
        
        intersection = (x2 - x1) * (y2 - y1)
        
        # Calculate union area
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0