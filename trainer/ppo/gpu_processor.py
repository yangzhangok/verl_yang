"""
GPU-based multi-modal data processor for PPO training.
This module handles GPU processing of pixel values to embeddings in the training loop,
avoiding GPU usage in the dataset for better performance and stability.
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional, List
import logging

logger = logging.getLogger(__name__)


class MultiModalGPUProcessor:
    """
    Handles GPU processing of multi-modal data during training.
    This should be used in the trainer instead of processing in the dataset.
    """
    
    def __init__(self, model: nn.Module, device: torch.device):
        """
        Initialize the GPU processor.
        
        Args:
            model: The model with visual processing capabilities
            device: The device to process data on
        """
        self.model = model
        self.device = device
        self.model.to(device)
        
        logger.info(f"MultiModalGPUProcessor initialized on device: {device}")
    
    def process_batch(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a batch of multi-modal data on GPU.
        
        Args:
            batch: Batch containing multi_modal_inputs with pixel_values
            
        Returns:
            Batch with processed image embeddings
        """
        if "multi_modal_inputs" not in batch:
            return batch
        
        multi_modal_inputs = batch["multi_modal_inputs"]
        
        # Check if we have pixel_values to process
        if "pixel_values" not in multi_modal_inputs:
            return batch
        
        pixel_values = multi_modal_inputs["pixel_values"]
        image_grid_thw = multi_modal_inputs.get("image_grid_thw")
        
        logger.debug(f"Processing batch with pixel_values shape: {pixel_values.shape}")
        
        # Move data to GPU
        pixel_values = pixel_values.to(self.device)
        if image_grid_thw is not None:
            image_grid_thw = image_grid_thw.to(self.device)
        
        # Process images to embeddings
        with torch.no_grad():
            if image_grid_thw is not None:
                image_embeddings = self.model.visual(pixel_values, image_grid_thw)
            else:
                image_embeddings = self.model.visual(pixel_values)
        
        logger.debug(f"Generated image embeddings shape: {image_embeddings.shape}")
        
        # Update batch with embeddings
        batch["multi_modal_inputs"]["image_embeddings"] = image_embeddings
        
        # Remove pixel_values to save memory
        batch["multi_modal_inputs"].pop("pixel_values", None)
        batch["multi_modal_inputs"].pop("image_grid_thw", None)
        
        return batch
    
    def process_single_item(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a single item's multi-modal data on GPU.
        
        Args:
            item: Single item containing multi_modal_inputs
            
        Returns:
            Item with processed image embeddings
        """
        if "multi_modal_inputs" not in item:
            return item
        
        multi_modal_inputs = item["multi_modal_inputs"]
        
        if "pixel_values" not in multi_modal_inputs:
            return item
        
        pixel_values = multi_modal_inputs["pixel_values"]
        image_grid_thw = multi_modal_inputs.get("image_grid_thw")
        
        # Ensure batch dimension
        if pixel_values.dim() == 3:
            pixel_values = pixel_values.unsqueeze(0)
        
        # Move to GPU
        pixel_values = pixel_values.to(self.device)
        if image_grid_thw is not None:
            image_grid_thw = image_grid_thw.to(self.device)
        
        # Process
        with torch.no_grad():
            if image_grid_thw is not None:
                image_embeddings = self.model.visual(pixel_values, image_grid_thw)
            else:
                image_embeddings = self.model.visual(pixel_values)
        
        # Remove batch dimension if it was added
        if image_embeddings.dim() > 2:
            image_embeddings = image_embeddings.squeeze(0)
        
        # Update item
        item["multi_modal_inputs"]["image_embeddings"] = image_embeddings
        item["multi_modal_inputs"].pop("pixel_values", None)
        
        return item


class BatchProcessor:
    """
    Utility class for batch processing of multi-modal data.
    """
    
    @staticmethod
    def collect_pixel_values(batch: List[Dict[str, Any]]) -> Optional[torch.Tensor]:
        """
        Collect pixel_values from a batch of items.
        
        Args:
            batch: List of items
            
        Returns:
            Batched pixel_values tensor or None
        """
        pixel_values_list = []
        image_grid_thw_list = []
        
        for item in batch:
            if ("multi_modal_inputs" in item and 
                "pixel_values" in item["multi_modal_inputs"]):
                pixel_values_list.append(item["multi_modal_inputs"]["pixel_values"])
                if "image_grid_thw" in item["multi_modal_inputs"]:
                    image_grid_thw_list.append(item["multi_modal_inputs"]["image_grid_thw"])
        
        if not pixel_values_list:
            return None
        
        # Stack pixel_values
        batched_pixel_values = torch.stack(pixel_values_list)
        
        # Stack image_grid_thw if available
        batched_image_grid_thw = None
        if image_grid_thw_list and len(image_grid_thw_list) == len(pixel_values_list):
            batched_image_grid_thw = torch.stack(image_grid_thw_list)
        
        return batched_pixel_values, batched_image_grid_thw
    
    @staticmethod
    def distribute_embeddings(batch: List[Dict[str, Any]], 
                            embeddings: torch.Tensor) -> List[Dict[str, Any]]:
        """
        Distribute computed embeddings back to batch items.
        
        Args:
            batch: List of items
            embeddings: Computed embeddings tensor
            
        Returns:
            Updated batch with embeddings
        """
        embedding_idx = 0
        
        for item in batch:
            if ("multi_modal_inputs" in item and 
                "pixel_values" in item["multi_modal_inputs"]):
                
                # Assign embedding to this item
                item["multi_modal_inputs"]["image_embeddings"] = embeddings[embedding_idx]
                
                # Remove pixel_values
                item["multi_modal_inputs"].pop("pixel_values", None)
                item["multi_modal_inputs"].pop("image_grid_thw", None)
                
                embedding_idx += 1
        
        return batch
