#!/usr/bin/env python3
# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Offline preprocessing script for converting pixel_values to image embeddings.

This script processes parquet files containing images and generates image embeddings
using Qwen2.5-VL's visual encoder. The processed embeddings are stored back in the
parquet files to avoid real-time processing during training.

Usage:
    python preprocess_embeddings.py \
        --input_parquet /path/to/input.parquet \
        --output_parquet /path/to/output.parquet \
        --model_path /path/to/qwen2.5-vl-model \
        --batch_size 32 \
        --max_workers 4
"""

import argparse
import logging
import os
import pickle
import shutil
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from qwen_vl_utils import process_vision_info

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from verl.utils.fs import copy_to_local

logger = logging.getLogger(__name__)


class ImageEmbeddingDataset(Dataset):
    """Dataset for processing images to embeddings."""
    
    def __init__(self, df: pd.DataFrame, image_key: str = "images"):
        self.df = df
        self.image_key = image_key
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # 添加数据验证
        if self.image_key not in row:
            raise KeyError(f"Image key '{self.image_key}' not found in row {idx}")
        
        image_data = row[self.image_key]
        
        # 更严格的图像数据验证
        if image_data is None:
            raise ValueError(f"Image data is None at index {idx}")
        
        # 处理多图像情况
        if isinstance(image_data, (list, tuple)):
            if len(image_data) == 0:
                raise ValueError(f"Empty image list at index {idx}")
            image_data = image_data[0]
        
        # 验证图像数据格式
        if isinstance(image_data, np.ndarray):
            if image_data.size == 0:
                raise ValueError(f"Empty image array at index {idx}")
            image_data = torch.from_numpy(image_data)
        elif not isinstance(image_data, torch.Tensor):
            try:
                image_data = torch.tensor(image_data)
            except Exception as e:
                raise ValueError(f"Invalid image data format at index {idx}: {e}") from e
        
        return {
            'index': idx,
            'image_data': image_data,
            'row_data': row.to_dict()
        }


class ImageEmbeddingProcessor:
    """Processor for converting images to embeddings using Qwen2.5-VL."""
    
    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        batch_size: int = 32,
        max_workers: int = 4,
        cache_dir: Optional[str] = None
    ):
        self.model_path = model_path
        self.device = device
        self.batch_size = batch_size
        self.max_workers = max_workers
        self.cache_dir = cache_dir or tempfile.mkdtemp()
        
        # Initialize model and processor
        self._load_model()
        
    def _load_model(self):
        """Load the Qwen2.5-VL model and processor."""
        logger.info(f"Loading model from {self.model_path}")
        
        # Copy model to local if needed
        local_model_path = copy_to_local(self.model_path, verbose=True)
        
        # Load processor
        self.processor = AutoProcessor.from_pretrained(
            local_model_path,
            trust_remote_code=True
        )
        
        # Load model
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            local_model_path,
            torch_dtype=torch.float16,
            device_map=self.device,
            trust_remote_code=True
        )
        
        # Set to eval mode
        self.model.eval()
        
        logger.info("Model loaded successfully")
    
    @torch.inference_mode()
    def process_batch(
        self,
        batch_data: List[Dict],
        image_key: str = "images",
    ) -> Tuple[List[int], List[torch.Tensor]]:
        """
        返回:
          indices: 与 embeddings 一一对应的原始行索引列表
          embeddings: 每张图的原始token序列 (变长)
        """
        if not batch_data:
            return [], []

        # 1) 过滤 None，并抽取每条样本的"第一张图片"
        msgs, indices = [], []
        for item in batch_data:
            if item is None:
                continue
            row = item.get("row_data", {})
            img_field = row.get(image_key, item.get("image_data"))
            if img_field is None:
                continue
            # 多图只取第一张
            img = img_field[0] if isinstance(img_field, (list, tuple)) and len(img_field) > 0 else img_field

            # 组装一条"只含图片"的消息
            msgs.append({"role": "user", "content": [{"type": "image", "image": img}]})
            indices.append(item["index"])  # 使用原始索引

        if not msgs:
            return [], []

        # 2) Processor 统一预处理
        texts = self.processor.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=False
        )
        images, _ = process_vision_info(msgs)
        inputs = self.processor(
            text=texts,
            images=images,
            padding=True,
            return_tensors="pt",
        ).to(self.model.device)

        # 3) 视觉编码
        image_embeds = self.model.visual(
            inputs.pixel_values, grid_thw=inputs.image_grid_thw
        )   # [sum_tokens, hidden]

        # 4) 按每张图的 token 数切分
        grid = inputs.image_grid_thw  # [num_images, 3] -> (T,H,W)
        token_counts = (grid[:, 0] * grid[:, 1] * grid[:, 2]).tolist()
        
        # 确保token_counts与图像数量匹配
        if len(token_counts) != len(msgs):
            logger.error(f"Token count mismatch: {len(token_counts)} vs {len(msgs)}")
            return [], []
        
        per_image_tokens = list(torch.split(image_embeds, token_counts, dim=0))

        # 5) 保留原始token序列（不池化）
        per_image = per_image_tokens  # 直接使用原始token序列

        # 6) 转回 CPU（保持tensor格式，不转numpy）
        per_image = [e.detach().float().cpu() for e in per_image]
        
        # 验证结果
        if len(per_image) != len(indices):
            logger.error(f"Length mismatch: {len(per_image)} embeddings vs {len(indices)} indices")
            return [], []
        
        # 检查token序列形状
        if per_image and len(per_image) > 0:
            for i, emb in enumerate(per_image):
                if len(emb.shape) != 2:
                    logger.error(f"Invalid token sequence shape at index {i}: {emb.shape}, expected [num_tokens, hidden_dim]")
                    return [], []

        return indices, per_image
    
    def process_dataframe(
        self,
        df: pd.DataFrame,
        image_key: str = "images",
        progress_bar: bool = True
    ) -> pd.DataFrame:
        """Process entire dataframe to generate embeddings."""
        logger.info(f"Processing {len(df)} images")
        
        # Create dataset
        dataset = ImageEmbeddingDataset(df, image_key)
        
        # Create dataloader
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.max_workers,
            collate_fn=lambda x: x  # Keep as list of dicts
        )
        
        # Process batches
        all_embeddings = []
        processed_indices = []
        failed_count = 0
        
        iterator = tqdm(dataloader, desc="Processing images") if progress_bar else dataloader
        
        for batch_idx, batch in enumerate(iterator):
            try:
                # 过滤掉None值
                valid_batch = [item for item in batch if item is not None]
                if not valid_batch:
                    continue
                    
                indices, embeddings = self.process_batch(valid_batch, image_key)
                
                # 存储嵌入向量 - 使用正确的索引映射
                for idx, embedding in zip(indices, embeddings):
                    all_embeddings.append(embedding)
                    processed_indices.append(idx)
                    
            except Exception as e:
                logger.error(f"Failed to process batch {batch_idx}: {e}")
                failed_count += len(batch)
                continue
        
        # Create new dataframe with embeddings
        result_df = df.copy()
        result_df['image_embeddings'] = None
        
        # Fill in embeddings - 使用正确的索引映射
        for idx, embedding in zip(processed_indices, all_embeddings):
            result_df.at[idx, 'image_embeddings'] = embedding
        
        logger.info(f"Processed {len(all_embeddings)} embeddings successfully, {failed_count} failed")
        
        return result_df


def process_parquet_file(
    input_path: str,
    output_path: str,
    model_path: str,
    image_key: str = "images",
    batch_size: int = 32,
    max_workers: int = 4,
    device: str = "cuda",
    overwrite: bool = False
) -> bool:
    """Process a single parquet file to generate embeddings."""
    
    # Check if output exists
    if os.path.exists(output_path) and not overwrite:
        logger.info(f"Output file {output_path} already exists, skipping")
        return True
    
    # Create output directory
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    try:
        # Load input data
        logger.info(f"Loading data from {input_path}")
        df = pd.read_parquet(input_path)
        
        # Check if images column exists
        if image_key not in df.columns:
            logger.error(f"Image key '{image_key}' not found in dataframe columns: {df.columns.tolist()}")
            return False
        
        # Check if embeddings already exist
        if 'image_embeddings' in df.columns:
            logger.info("Image embeddings already exist in the data")
            # Copy to output
            df.to_parquet(output_path)
            return True
        
        # Initialize processor
        processor = ImageEmbeddingProcessor(
            model_path=model_path,
            device=device,
            batch_size=batch_size,
            max_workers=max_workers
        )
        
        # Process dataframe
        result_df = processor.process_dataframe(df, image_key)
        
        # Save result
        logger.info(f"Saving processed data to {output_path}")
        result_df.to_parquet(output_path)
        
        logger.info("Processing completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"Error processing file {input_path}: {e}")
        return False


def main():
    """Main function for command line usage."""
    parser = argparse.ArgumentParser(description="Preprocess images to embeddings")
    parser.add_argument("--input_parquet", required=True, help="Input parquet file path")
    parser.add_argument("--output_parquet", required=True, help="Output parquet file path")
    parser.add_argument("--model_path", required=True, help="Path to Qwen2.5-VL model")
    parser.add_argument("--image_key", default="images", help="Key for image data in parquet")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for processing")
    parser.add_argument("--max_workers", type=int, default=4, help="Number of workers for data loading")
    parser.add_argument("--device", default="cuda", help="Device to use (cuda/cpu)")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output file")
    parser.add_argument("--log_level", default="INFO", help="Logging level")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Process file
    success = process_parquet_file(
        input_path=args.input_parquet,
        output_path=args.output_parquet,
        model_path=args.model_path,
        image_key=args.image_key,
        batch_size=args.batch_size,
        max_workers=args.max_workers,
        device=args.device,
        overwrite=args.overwrite
    )
    
    if success:
        logger.info("Processing completed successfully")
        exit(0)
    else:
        logger.error("Processing failed")
        exit(1)


if __name__ == "__main__":
    main()
