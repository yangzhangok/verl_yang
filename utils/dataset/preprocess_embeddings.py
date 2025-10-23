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

import datasets
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from verl.utils.fs import copy_to_local

logger = logging.getLogger(__name__)


from verl.utils.dataset.vision_utils import process_image


def collate_fn(data_list: list[dict]) -> dict:
    """
    Collate a batch of sample dicts into batched tensors and arrays.
    参考 rl_dataset.py 的 collate_fn 实现

    Args:
        data_list: List of dicts mapping feature names to torch.Tensor or other values.

    Returns:
        Dict where tensor entries are stacked into a torch.Tensor of shape
        (batch_size, \*dims) and non-tensor entries are converted to
        np.ndarray of dtype object with shape (batch_size,).
    """
    from collections import defaultdict
    
    tensors = defaultdict(list)
    non_tensors = defaultdict(list)

    for data in data_list:
        for key, val in data.items():
            if isinstance(val, torch.Tensor):
                tensors[key].append(val)
            else:
                non_tensors[key].append(val)

    for key, val in tensors.items():
        tensors[key] = torch.stack(val, dim=0)

    for key, val in non_tensors.items():
        non_tensors[key] = np.fromiter(val, dtype=object, count=len(val))

    return {**tensors, **non_tensors}


class ImageEmbeddingDataset(Dataset):
    """Dataset for processing images to embeddings using HuggingFace datasets format."""
    
    def __init__(self, dataset: datasets.Dataset, image_key: str = "images"):
        self.dataset = dataset
        self.image_key = image_key
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        try:
            row = self.dataset[idx]
        except Exception as e:
            raise IndexError(f"Failed to access dataset at index {idx}: {e}") from e
        
        # 添加数据验证
        if self.image_key not in row:
            raise KeyError(f"Image key '{self.image_key}' not found in row {idx}")
        
        image_data = row[self.image_key]
        
        # 更严格的图像数据验证
        if image_data is None:
            raise ValueError(f"Image data is None at index {idx}")
        
        # 处理多图像情况 - 参考 rl_dataset.py 的处理方式
        if isinstance(image_data, (list, tuple)):
            if len(image_data) == 0:
                raise ValueError(f"Empty image list at index {idx}")
            # 对于多图像，我们取第一张图像进行处理
            image_data = image_data[0]
        
        # 使用统一的图像处理函数
        try:
            image_data = process_image(image_data)
        except Exception as e:
            raise ValueError(f"Failed to process image at index {idx}: {e}") from e
        
        return {
            'index': idx,
            'image_data': image_data,
            'row_data': row
        }
def unbatch_dict_of_arrays(batch: dict) -> list[dict]:
    # 推断 batch 大小
    bs = None
    for v in batch.values():
        if isinstance(v, (list, tuple, np.ndarray, torch.Tensor)):
            bs = len(v)
            break
    if bs is None:
        return []

    rows = []
    for i in range(bs):
        item = {}
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                item[k] = v[i]
            elif isinstance(v, np.ndarray):
                item[k] = v[i]
            elif isinstance(v, (list, tuple)):
                item[k] = v[i]
            else:
                # 标量/不可索引对象，原样放入
                item[k] = v
        rows.append(item)
    return rows


class ImageEmbeddingProcessor:
    """Processor for converting images to embeddings using Qwen2.5-VL."""
    
    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        batch_size: int = 32,
        max_workers: int = 4,
        cache_dir: Optional[str] = None,
        filter_invalid_images: bool = True
    ):
        self.model_path = model_path
        self.device = device
        self.batch_size = batch_size
        self.max_workers = max_workers
        self.cache_dir = cache_dir or tempfile.mkdtemp()
        self.filter_invalid_images = filter_invalid_images
        
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
    
    def filter_invalid_data(self, dataset: datasets.Dataset, image_key: str = "images") -> datasets.Dataset:
        """Filter out invalid image data from the dataset."""
        if not self.filter_invalid_images:
            return dataset
            
        def is_valid_image(example):
            """Check if the image data is valid."""
            try:
                image_data = example.get(image_key)
                if image_data is None:
                    return False
                
                # 处理多图像情况
                if isinstance(image_data, (list, tuple)):
                    if len(image_data) == 0:
                        return False
                    image_data = image_data[0]
                
                # 验证图像数据格式
                if image_data is not None:
                    return True
                        
            except Exception:
                return False
        
        logger.info(f"Filtering invalid images from dataset of size {len(dataset)}")
        filtered_dataset = dataset.filter(is_valid_image, desc="Filtering invalid images")
        logger.info(f"Filtered dataset size: {len(filtered_dataset)}")
        
        return filtered_dataset
    
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

        # 4) 用第一列作为每张图的 token 个数（已 merge 后）
        grid = inputs.image_grid_thw.to(torch.long).view(-1, 3)
        token_counts = grid[:, 0].tolist()

        # 保险：做一次一致性校验，必要时回退
        total_tokens = image_embeds.size(0)
        if sum(token_counts) != total_tokens:
            # 兼容老版本/非常规 pipeline：用 T*H*W / factor 回退
            prod = (grid[:, 0] * grid[:, 1] * grid[:, 2]).tolist()
            s_prod = sum(prod)
            if s_prod % total_tokens == 0:
                factor = s_prod // total_tokens   # 通常是 4
                # 确保可整除
                if all(p % factor == 0 for p in prod):
                    token_counts = [p // factor for p in prod]
                else:
                    raise RuntimeError(
                        f"Cannot reconcile token counts: sum(prod)={s_prod}, "
                        f"embeds={total_tokens}, factor={factor} not dividing all."
                    )
            else:
                raise RuntimeError(
                    f"Token count mismatch: sum(grid[:,0])={sum(token_counts)} vs embeds={total_tokens}"
                )

        # 5) split
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
    
    def process_dataset(
        self,
        dataset: datasets.Dataset,
        image_key: str = "images",
        progress_bar: bool = True
    ) -> datasets.Dataset:
        """Process entire dataset to generate embeddings."""
        logger.info(f"Processing {len(dataset)} images")
        
        # Filter invalid data first
        filtered_dataset = self.filter_invalid_data(dataset, image_key)
        
        # Create dataset wrapper
        image_dataset = ImageEmbeddingDataset(filtered_dataset, image_key)
        
        # Create dataloader with improved collate function
        dataloader = DataLoader(
            image_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.max_workers,
            collate_fn=collate_fn  # Use improved collate function
        )
        
        # Process batches with improved error handling
        all_embeddings = []
        processed_indices = []
        failed_count = 0
        total_batches = len(dataloader)
        
        iterator = tqdm(dataloader, desc="Processing images", total=total_batches) if progress_bar else dataloader
        
        for batch_idx, batch in enumerate(iterator):
            try:
                # 验证批次数据
                if not batch:
                    logger.warning(f"Empty batch at index {batch_idx}")
                    continue
                
                # 过滤掉None值
                # 新的（正确）：
                samples = unbatch_dict_of_arrays(batch)

                valid_batch = []
                for item in samples:
                    if item is None:
                        continue
                    row = item.get("row_data")
                    img = None
                    if isinstance(row, dict):
                        img = row.get(image_key)
                    if img is None:
                        img = item.get("image_data")
                    if img is not None:
                        valid_batch.append(item)

                if not valid_batch:
                    failed_count += len(samples)
                    continue

                indices, embeddings = self.process_batch(valid_batch, image_key)
                if not valid_batch:
                    logger.warning(f"No valid items in batch {batch_idx}")
                    failed_count += len(batch)
                    continue
                    
                # 处理批次
                indices, embeddings = self.process_batch(valid_batch, image_key)
                
                # 验证处理结果
                if len(indices) != len(embeddings):
                    logger.error(f"Index-embedding mismatch in batch {batch_idx}: {len(indices)} indices vs {len(embeddings)} embeddings")
                    failed_count += len(valid_batch)
                    continue
                
                # 存储嵌入向量 - 使用正确的索引映射
                for idx, embedding in zip(indices, embeddings):
                    if embedding is not None:
                        all_embeddings.append(embedding)
                        processed_indices.append(idx)
                    else:
                        logger.warning(f"None embedding for index {idx}")
                        failed_count += 1
                    
            except Exception as e:
                logger.error(f"Failed to process batch {batch_idx}: {e}", exc_info=True)
                failed_count += len(batch) if batch else 1
                continue
        
        # Create new dataset with embeddings
        # 首先创建一个字典来存储嵌入向量
        embeddings_dict = {}
        for idx, embedding in zip(processed_indices, all_embeddings):
            embeddings_dict[idx] = embedding
        
        # 为数据集添加嵌入向量列
        def add_embeddings(example, idx):
            example['image_embeddings'] = embeddings_dict.get(idx, None)
            return example
        
        import ipdb;ipdb.set_trace()
        # 使用 map 函数添加嵌入向量
        result_dataset = dataset.map(
            add_embeddings,
            with_indices=True,
            desc="Adding image embeddings"
        )
        
        # 添加详细的统计信息
        success_rate = len(all_embeddings) / (len(all_embeddings) + failed_count) * 100 if (len(all_embeddings) + failed_count) > 0 else 0
        logger.info(f"Processing completed:")
        logger.info(f"  - Successfully processed: {len(all_embeddings)} embeddings")
        logger.info(f"  - Failed: {failed_count} items")
        logger.info(f"  - Success rate: {success_rate:.2f}%")
        logger.info(f"  - Total batches processed: {total_batches}")
        
        return result_dataset


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
        # Load input data using HuggingFace datasets - 参考 rl_dataset.py 的方式
        logger.info(f"Loading data from {input_path}")
        dataset = datasets.load_dataset("parquet", data_files=input_path)["train"]
        
        # Check if images column exists
        if image_key not in dataset.column_names:
            logger.error(f"Image key '{image_key}' not found in dataset columns: {dataset.column_names}")
            return False
        
        # Check if embeddings already exist
        if 'image_embeddings' in dataset.column_names:
            logger.info("Image embeddings already exist in the data")
            # Convert to pandas and save
            df = dataset.to_pandas()
            df.to_parquet(output_path)
            return True
        
        # Initialize processor
        processor = ImageEmbeddingProcessor(
            model_path=model_path,
            device=device,
            batch_size=batch_size,
            max_workers=max_workers
        )
        
        # Process dataset
        result_dataset = processor.process_dataset(dataset, image_key)
        
        # Convert to pandas and save result
        logger.info(f"Saving processed data to {output_path}")
        result_df = result_dataset.to_pandas()
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