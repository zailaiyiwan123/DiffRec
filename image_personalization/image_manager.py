"""
Smart image management system
Automatically save, update, and clean generated images during training to avoid excessive memory usage
"""

import os
import shutil
import time
from pathlib import Path
from typing import Optional, Dict, Any, List
from datetime import datetime
import PIL.Image


class TrainingImageManager:
    """Training image manager"""
    
    def __init__(
        self, 
        base_dir: str = "training_images",
        max_images: int = 50,  # Maximum 50 images
        max_recent_images: int = 10,  # Recent 10 images saved separately
        cleanup_interval: int = 100,  # Clean up after every 100 saves
    ):
        self.base_dir = Path(base_dir)
        self.max_images = max_images
        self.max_recent_images = max_recent_images
        self.cleanup_interval = cleanup_interval
        self.save_count = 0
        
        # Create directory structure
        self.setup_directories()
        
    def setup_directories(self):
        """Create directory structure"""
        self.all_images_dir = self.base_dir / "all_images"
        self.recent_images_dir = self.base_dir / "recent_images"
        self.best_images_dir = self.base_dir / "best_images"
        self.metadata_dir = self.base_dir / "metadata"
        
        for dir_path in [self.all_images_dir, self.recent_images_dir, 
                        self.best_images_dir, self.metadata_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
            
        print(f"📁 Image management directory created: {self.base_dir}")
        
    def save_training_image(
        self, 
        image: PIL.Image.Image,
        step: int,
        epoch: int,
        loss: float,
        metrics: Dict[str, float],
        sample_info: Dict[str, Any],
        force_save: bool = False
    ) -> Dict[str, str]:
        """
        Save images generated during training
        
        Args:
            image: PIL image object
            step: Training step
            epoch: Training epoch
            loss: Current loss
            metrics: Evaluation metrics
            sample_info: Sample information
            force_save: Force save (even if quality is low)
            
        Returns:
            Dictionary of saved file paths
        """
        self.save_count += 1
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Generate filename
        filename = f"step_{step:06d}_epoch_{epoch:03d}_{timestamp}.png"
        
        # Calculate average quality score
        avg_score = self._calculate_average_score(metrics)
        quality_label = self._get_quality_label(avg_score)
        
        saved_paths = {}
        
        # 1. Save to recent images directory (always save)
        recent_path = self.recent_images_dir / filename
        image.save(recent_path)
        saved_paths["recent"] = str(recent_path)
        
        # 2. Decide whether to save to all images directory based on quality
        if avg_score >= 3.5 or force_save:  # Good quality or force save
            all_path = self.all_images_dir / filename
            image.save(all_path)
            saved_paths["all"] = str(all_path)
            
        # 3. Save high quality images to best images directory
        if avg_score >= 4.0:
            best_filename = f"best_{quality_label}_{filename}"
            best_path = self.best_images_dir / best_filename
            image.save(best_path)
            saved_paths["best"] = str(best_path)
            
        # 4. Save metadata
        metadata = {
            "step": step,
            "epoch": epoch,
            "loss": loss,
            "metrics": metrics,
            "avg_score": avg_score,
            "quality_label": quality_label,
            "timestamp": timestamp,
            "sample_info": {
                "instruction": sample_info.get("instruction", "")[:100],  # Truncate long text
                "title": sample_info.get("title", ""),
                "adaptive_weight": sample_info.get("adaptive_weight", 0.0)
            }
        }
        
        metadata_file = self.metadata_dir / f"{filename}.json"
        import json
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
            
        # 5. Periodic cleanup
        if self.save_count % self.cleanup_interval == 0:
            self._cleanup_old_images()
            
        # Print save information
        print(f"🖼️ Image saved [Step{step}|Epoch{epoch}] Quality:{quality_label}({avg_score:.2f}) -> {len(saved_paths)} locations")
        
        return saved_paths
        
    def _calculate_average_score(self, metrics: Dict[str, float]) -> float:
        """Calculate average quality score"""
        score_keys = ['consistency', 'accuracy', 'integrity', 'quality']
        valid_scores = []
        
        for key in score_keys:
            if key in metrics and metrics[key] is not None:
                try:
                    score = float(metrics[key])
                    if 0 <= score <= 5:  # Valid score range
                        valid_scores.append(score)
                except (ValueError, TypeError):
                    continue
                    
        return sum(valid_scores) / len(valid_scores) if valid_scores else 3.0
        
    def _get_quality_label(self, avg_score: float) -> str:
        """Get quality label based on score"""
        if avg_score >= 4.5:
            return "excellent"
        elif avg_score >= 4.0:
            return "good"
        elif avg_score >= 3.5:
            return "fair"
        elif avg_score >= 3.0:
            return "poor"
        else:
            return "bad"
            
    def _cleanup_old_images(self):
        """Clean up old images, maintain quantity limits"""
        print("🧹 Starting to clean up old images...")
        
        # 1. Clean recent images directory (keep newest N images)
        self._cleanup_directory(self.recent_images_dir, self.max_recent_images)
        
        # 2. Clean all images directory (keep newest N images)
        self._cleanup_directory(self.all_images_dir, self.max_images)
        
        # 3. Don't clean best images directory (keep all high quality images)
        
        print("✅ Image cleanup completed")
        
    def _cleanup_directory(self, directory: Path, max_files: int):
        """Clean specified directory, maintain file quantity limits"""
        if not directory.exists():
            return
            
        # Get all image files, sorted by modification time
        image_files = list(directory.glob("*.png"))
        image_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        
        # Delete old files that exceed limits
        files_to_delete = image_files[max_files:]
        
        for file_path in files_to_delete:
            try:
                file_path.unlink()
                # Also delete corresponding metadata files
                metadata_file = self.metadata_dir / f"{file_path.name}.json"
                if metadata_file.exists():
                    metadata_file.unlink()
            except Exception as e:
                print(f"⚠️ Failed to delete file {file_path}: {e}")
                
        if files_to_delete:
            print(f"🗑️ Cleaned {len(files_to_delete)} old image files from {directory.name}")
            
    def get_recent_images(self, n: int = 5) -> List[Dict[str, Any]]:
        """Get information of the most recent N images"""
        image_files = list(self.recent_images_dir.glob("*.png"))
        image_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        
        recent_images = []
        for img_file in image_files[:n]:
            metadata_file = self.metadata_dir / f"{img_file.name}.json"
            
            image_info = {
                "path": str(img_file),
                "filename": img_file.name,
                "size_mb": img_file.stat().st_size / (1024 * 1024)
            }
            
            # Read metadata
            if metadata_file.exists():
                try:
                    import json
                    with open(metadata_file, 'r', encoding='utf-8') as f:
                        metadata = json.load(f)
                    image_info.update(metadata)
                except Exception:
                    pass
                    
            recent_images.append(image_info)
            
        return recent_images
        
    def get_best_images(self, n: int = 10) -> List[Dict[str, Any]]:
        """Get information of the best N images"""
        return self._get_images_from_directory(self.best_images_dir, n)
        
    def _get_images_from_directory(self, directory: Path, n: int) -> List[Dict[str, Any]]:
        """Get image information from specified directory"""
        if not directory.exists():
            return []
            
        image_files = list(directory.glob("*.png"))
        image_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        
        images = []
        for img_file in image_files[:n]:
            metadata_file = self.metadata_dir / f"{img_file.name}.json"
            
            image_info = {
                "path": str(img_file),
                "filename": img_file.name,
                "size_mb": img_file.stat().st_size / (1024 * 1024)
            }
            
            # Read metadata
            if metadata_file.exists():
                try:
                    import json
                    with open(metadata_file, 'r', encoding='utf-8') as f:
                        metadata = json.load(f)
                    image_info.update(metadata)
                except Exception:
                    pass
                    
            images.append(image_info)
            
        return images
        
    def get_stats(self) -> Dict[str, Any]:
        """Get image management statistics"""
        stats = {
            "total_saved": self.save_count,
            "recent_count": len(list(self.recent_images_dir.glob("*.png"))),
            "all_count": len(list(self.all_images_dir.glob("*.png"))),
            "best_count": len(list(self.best_images_dir.glob("*.png"))),
            "total_size_mb": 0
        }
        
        # Calculate total file size
        for directory in [self.recent_images_dir, self.all_images_dir, self.best_images_dir]:
            if directory.exists():
                for img_file in directory.glob("*.png"):
                    stats["total_size_mb"] += img_file.stat().st_size / (1024 * 1024)
                    
        stats["total_size_mb"] = round(stats["total_size_mb"], 2)
        
        return stats
