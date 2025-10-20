# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
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
Single Process Actor
"""

import logging
import os
import numpy as np
import torch
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.tensor import DTensor
from copy import deepcopy

import verl.utils.torch_functional as verl_F
from verl import DataProto
from verl.trainer.ppo.core_algos import agg_loss, get_policy_loss_fn, kl_penalty
from verl.utils.attention_utils import index_first_axis, pad_input, rearrange, unpad_input
from verl.utils.device import get_device_id, get_device_name
from verl.utils.fsdp_utils import FSDPModule, fsdp2_clip_grad_norm_
from verl.utils.profiler import GPUMemoryLogger
from verl.utils.py_functional import append_to_dict
from verl.utils.seqlen_balancing import prepare_dynamic_batch, restore_dynamic_batch
from verl.utils.torch_functional import logprobs_from_logits
from verl.utils.ulysses import gather_outputs_and_unpad, ulysses_pad, ulysses_pad_and_slice_inputs
from verl.workers.actor import BasePPOActor
from verl.workers.config import ActorConfig

__all__ = ["DataParallelPPOActor"]

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class DataParallelPPOActor(BasePPOActor):
    """FSDP DataParallel PPO Actor or Ref worker

    Args:
        config (ActorConfig): Actor config
        actor_module (nn.Module): Actor or ref module
        actor_optimizer (torch.optim.Optimizer, optional): Actor optimizer. Defaults to None.
    """

    def __init__(self, config: ActorConfig, actor_module: nn.Module, actor_optimizer: torch.optim.Optimizer = None):
        """When optimizer is None, it is Reference Policy"""
        super().__init__(config)
        self.actor_module = actor_module
        self.actor_optimizer = actor_optimizer
        role = "Ref" if actor_optimizer is None else "Actor"

        self.use_remove_padding = self.config.get("use_remove_padding", False)
        if torch.distributed.get_rank() == 0:
            print(f"{role} use_remove_padding={self.use_remove_padding}")
        self.use_fused_kernels = self.config.get("use_fused_kernels", False)
        if torch.distributed.get_rank() == 0:
            print(f"{role} use_fused_kernels={self.use_fused_kernels}")

        self.ulysses_sequence_parallel_size = self.config.ulysses_sequence_parallel_size
        self.use_ulysses_sp = self.ulysses_sequence_parallel_size > 1
        
        # Debug: Print Ulysses configuration
        if torch.distributed.get_rank() == 0:
            print(f"Ulysses sequence parallel size: {self.ulysses_sequence_parallel_size}")
            print(f"Use Ulysses SP: {self.use_ulysses_sp}")

        if self.config.entropy_from_logits_with_chunking:
            entropy_from_logits = verl_F.entropy_from_logits_with_chunking
        else:
            entropy_from_logits = verl_F.entropy_from_logits

        self.compute_entropy_from_logits = (
            torch.compile(entropy_from_logits, dynamic=True)
            if self.config.get("use_torch_compile", True)  #  use torch compile by default
            else entropy_from_logits
        )
        self.device_name = get_device_name()

        # 注意：视觉模块的 FSDP 配置应该在 fsdp_workers.py 中处理
        # 这里不再进行额外的 FSDP 包装，避免双重包装问题
        if torch.distributed.get_rank() == 0:
            print(f"Visual module type: {type(self.actor_module.visual)}")
            print(f"Visual module is FSDP wrapped: {hasattr(self.actor_module.visual, '_fsdp_wrapped_module')}")

    def process_pixel_values_to_embeddings(self, data: DataProto) -> DataProto:
        """
        Process pixel_values to image embeddings using the actor module's visual encoder.
        
        This function converts pixel_values to image embeddings using the model's visual component,
        which is already loaded on GPU. This avoids the need to load a separate visual model
        and leverages the existing actor_module for processing.
        
        Args:
            data (DataProto): DataProto containing multi_modal_inputs with pixel_values
            
        Returns:
            DataProto: Updated DataProto with image_embeddings instead of pixel_values
        """
        # Extract multi_modal_inputs from non_tensor_batch
        if "multi_modal_inputs" not in data.non_tensor_batch:
            return data
            
        multi_modal_inputs_list = data.non_tensor_batch["multi_modal_inputs"]
        
        # Process each multi_modal_input in the batch
        processed_inputs = []
        for multi_modal_input in multi_modal_inputs_list:
            # Convert numpy arrays back to torch tensors
            converted_dict = {}
            for key, val in multi_modal_input.items():
                if isinstance(val, np.ndarray):
                    # Convert numpy array back to torch tensor
                    converted_dict[key] = torch.from_numpy(val)
                else:
                    converted_dict[key] = val
            
            processed_input = self._process_single_multi_modal_input(converted_dict)
            processed_inputs.append(processed_input)
        
        # Update the data with processed inputs
        updated_data = deepcopy(data)
        updated_data.non_tensor_batch["multi_modal_inputs"] = processed_inputs
        
        return updated_data
    
    def _process_single_multi_modal_input(self, multi_modal_inputs: dict) -> dict:
        """
        Process a single multi-modal input to convert pixel_values to embeddings.
        
        Args:
            multi_modal_inputs (dict): Single multi-modal input dictionary
            
        Returns:
            dict: Updated multi-modal input with image_embeddings
        """

        if "pixel_values" not in multi_modal_inputs:
            return multi_modal_inputs
        
        pixel_values = multi_modal_inputs["pixel_values"]
        image_grid_thw = multi_modal_inputs.get("image_grid_thw")
        
        # Ensure pixel_values is on the correct device
        if pixel_values.device != self.actor_module.device:
            pixel_values = pixel_values.to(self.actor_module.device)
        
        if image_grid_thw is not None and image_grid_thw.device != self.actor_module.device:
            image_grid_thw = image_grid_thw.to(self.actor_module.device)
        
        # Set model to eval mode for inference
        was_training = self.actor_module.training
        self.actor_module.eval()
        
        try:
            with torch.no_grad():
                # Debug: Print pixel_values shape and dtype
                print(f"DEBUG: pixel_values shape: {pixel_values.shape}, dtype: {pixel_values.dtype}")
                if image_grid_thw is not None:
                    print(f"DEBUG: image_grid_thw shape: {image_grid_thw.shape}, dtype: {image_grid_thw.dtype}")
                
                # Use the model's visual encoder to process pixel_values
                if hasattr(self.actor_module, 'visual'):
                    # Direct access to visual encoder (FSDP wrapped)
                    # Convert pixel_values to the correct dtype for visual encoder
                    pixel_values = pixel_values.type(self.actor_module.visual.dtype)
                    print(f"DEBUG: After dtype conversion - pixel_values shape: {pixel_values.shape}, dtype: {pixel_values.dtype}")
                    
                    if image_grid_thw is not None:
                        #image_grid_thw = image_grid_thw.unsqueeze(0)
                        image_embeddings = self.actor_module.visual(pixel_values, grid_thw=image_grid_thw)
                    else:
                        image_embeddings = self.actor_module.visual(pixel_values)
                elif hasattr(self.actor_module, 'module') and hasattr(self.actor_module.module, 'visual'):
                    # FSDP wrapped model
                    # Convert pixel_values to the correct dtype for visual encoder
                    pixel_values = pixel_values.type(self.actor_module.module.visual.dtype)
                    
                    if image_grid_thw is not None:
                        image_embeddings = self.actor_module.module.visual(pixel_values, grid_thw=image_grid_thw)
                    else:
                        image_embeddings = self.actor_module.module.visual(pixel_values)
                else:
                    # Try to find visual encoder in model structure
                    model = getattr(self.actor_module, 'module', self.actor_module)
                    if hasattr(model, 'model') and hasattr(model.model, 'visual'):
                        # Convert pixel_values to the correct dtype for visual encoder
                        pixel_values = pixel_values.type(model.model.visual.dtype)
                        
                        if image_grid_thw is not None:
                            image_embeddings = model.model.visual(pixel_values, grid_thw=image_grid_thw)
                        else:
                            image_embeddings = model.model.visual(pixel_values)
                    else:
                        raise AttributeError("Could not find visual encoder in actor_module")
                
                logger.debug(f"Processed pixel_values shape {pixel_values.shape} to embeddings shape {image_embeddings.shape}")
                
                # Update multi_modal_inputs with embeddings
                updated_inputs = multi_modal_inputs.copy()
                updated_inputs["image_embeddings"] = image_embeddings
                
                # Remove pixel_values and image_grid_thw to save memory
                updated_inputs.pop("pixel_values", None)
                #updated_inputs.pop("image_grid_thw", None)
                
                return updated_inputs
                
        finally:
            # Restore original training mode
            if was_training:
                self.actor_module.train()

    def process_batch_pixel_values_to_embeddings(self, batch_multi_modal_inputs: list) -> list:
        """
        Process a batch of multi-modal inputs to convert pixel_values to embeddings.
        
        This function efficiently processes multiple samples in batch, which is more efficient
        than processing them one by one. It collects all pixel_values, processes them together,
        and then distributes the results back to individual samples.
        
        Args:
            batch_multi_modal_inputs (list): List of multi-modal input dictionaries
            
        Returns:
            list: Updated list of multi-modal inputs with image_embeddings
        """
        if not batch_multi_modal_inputs:
            return batch_multi_modal_inputs
        
        # Collect all pixel_values and their metadata
        pixel_values_list = []
        image_grid_thw_list = []
        valid_indices = []
        
        for i, multi_modal_inputs in enumerate(batch_multi_modal_inputs):
            if "pixel_values" in multi_modal_inputs:
                pixel_values_list.append(multi_modal_inputs["pixel_values"])
                image_grid_thw_list.append(multi_modal_inputs.get("image_grid_thw"))
                valid_indices.append(i)
        
        if not pixel_values_list:
            return batch_multi_modal_inputs
        
        # Batch process all pixel_values together
        try:
            # Stack pixel_values for batch processing
            batched_pixel_values = torch.stack(pixel_values_list)
            
            # Stack image_grid_thw if available
            batched_image_grid_thw = None
            if image_grid_thw_list and all(thw is not None for thw in image_grid_thw_list):
                batched_image_grid_thw = torch.stack(image_grid_thw_list)
            
            # Process the batch
            processed_batch = self.process_pixel_values_to_embeddings({
                "pixel_values": batched_pixel_values,
                "image_grid_thw": batched_image_grid_thw
            })
            
            # Extract embeddings
            batched_embeddings = processed_batch["image_embeddings"]
            
            # Distribute embeddings back to individual samples
            result = batch_multi_modal_inputs.copy()
            for i, valid_idx in enumerate(valid_indices):
                # Get the embedding for this sample
                sample_embedding = batched_embeddings[i]
                
                # Update the sample
                updated_inputs = result[valid_idx].copy()
                updated_inputs["image_embeddings"] = sample_embedding
                updated_inputs.pop("pixel_values", None)
                updated_inputs.pop("image_grid_thw", None)
                result[valid_idx] = updated_inputs
            
            logger.debug(f"Batch processed {len(pixel_values_list)} pixel_values to embeddings")
            return result
            
        except Exception as e:
            logger.error(f"Error in batch processing pixel_values: {e}")
            # Fallback to individual processing
            logger.info("Falling back to individual processing")
            result = []
            for multi_modal_inputs in batch_multi_modal_inputs:
                result.append(self.process_pixel_values_to_embeddings(multi_modal_inputs))
            return result

    def _forward_micro_batch(
        self, micro_batch, temperature, calculate_entropy=False
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            entropy: # (bs, response_len)
            log_probs: # (bs, response_len)
        """
        # If only partial multimodal embeds are provided, construct full inputs_embeds from input_ids

        if "inputs_embeds" not in micro_batch and "multi_modal_inputs" in micro_batch:
            mm = micro_batch["multi_modal_inputs"] or {}
            has_img = "image_embeddings" in mm and mm["image_embeddings"] is not None
            has_vid = "video_embeddings" in mm and mm["video_embeddings"] is not None
            if has_img or has_vid:
                input_ids = micro_batch["input_ids"]
                # 1) get token embeddings
                get_emb_module = None
                if hasattr(self.actor_module, "get_input_embeddings"):
                    get_emb_module = self.actor_module.get_input_embeddings()
                elif hasattr(self.actor_module, "model") and hasattr(self.actor_module.model, "get_input_embeddings"):
                    get_emb_module = self.actor_module.model.get_input_embeddings()
                if get_emb_module is None:
                    raise RuntimeError("Model does not expose get_input_embeddings() to build inputs_embeds")
                token_embeds = get_emb_module(input_ids)

                # 2) scatter image/video embeddings into placeholder token positions
                model_cfg = getattr(self.actor_module, "config", getattr(getattr(self.actor_module, "model", None), "config", None))
                image_token_id = getattr(model_cfg, "image_token_id", None)
                video_token_id = getattr(model_cfg, "video_token_id", None)

                inputs_embeds = token_embeds
                if has_img:
                    if image_token_id is None:
                        raise RuntimeError("image_token_id not found in model config; cannot place image embeddings")
                    img_embeds = mm["image_embeddings"].to(inputs_embeds.device, inputs_embeds.dtype)
                    mask = (input_ids == image_token_id)
                    num_tokens = int(mask.sum().item())
                    if img_embeds.dim() == 3:
                        img_embeds = img_embeds.reshape(-1, img_embeds.size(-1))
                    if img_embeds.size(0) != num_tokens:
                        raise ValueError(f"Image features ({img_embeds.size(0)}) do not match image tokens ({num_tokens})")
                    mask_expanded = mask.unsqueeze(-1).expand_as(inputs_embeds)
                    inputs_embeds = inputs_embeds.masked_scatter(mask_expanded, img_embeds)

                if has_vid:
                    if video_token_id is None:
                        raise RuntimeError("video_token_id not found in model config; cannot place video embeddings")
                    vid_embeds = mm["video_embeddings"].to(inputs_embeds.device, inputs_embeds.dtype)
                    mask = (input_ids == video_token_id)
                    num_tokens = int(mask.sum().item())
                    if vid_embeds.dim() == 3:
                        vid_embeds = vid_embeds.reshape(-1, vid_embeds.size(-1))
                    if vid_embeds.size(0) != num_tokens:
                        raise ValueError(f"Video features ({vid_embeds.size(0)}) do not match video tokens ({num_tokens})")
                    mask_expanded = mask.unsqueeze(-1).expand_as(inputs_embeds)
                    inputs_embeds = inputs_embeds.masked_scatter(mask_expanded, vid_embeds)

                micro_batch["inputs_embeds"] = inputs_embeds

        # Dispatch to inputs_embeds path if provided (either precomputed or just constructed above)
        if "inputs_embeds" in micro_batch:
            return self._forward_micro_batch_with_input_embeds(
                micro_batch, temperature=temperature, calculate_entropy=calculate_entropy
            )
        response_length = micro_batch["responses"].size(-1)
        multi_modal_inputs = {}
        if "multi_modal_inputs" in micro_batch.keys():
            from verl.utils.model import extract_multi_modal_inputs

            multi_modal_inputs = extract_multi_modal_inputs(micro_batch["multi_modal_inputs"])

        with torch.autocast(device_type=self.device_name, dtype=torch.bfloat16):
            input_ids = micro_batch["input_ids"]
            batch_size, seqlen = input_ids.shape
            attention_mask = micro_batch["attention_mask"]
            position_ids = micro_batch["position_ids"]
            entropy = None
            if position_ids.dim() == 3:  # qwen2vl mrope
                position_ids = position_ids.transpose(0, 1)  # (bsz, 4, seqlen) -> (4, bsz, seqlen)

            if self.use_remove_padding:
                input_ids_rmpad, indices, cu_seqlens, *_ = unpad_input(
                    input_ids.unsqueeze(-1), attention_mask
                )  # input_ids_rmpad (total_nnz, ...)
                input_ids_rmpad = input_ids_rmpad.transpose(0, 1)  # (1, total_nnz)

                # unpad the position_ids to align the rotary
                if position_ids.dim() == 3:
                    position_ids_rmpad = (
                        index_first_axis(rearrange(position_ids, "c b s ... -> (b s) c ..."), indices)
                        .transpose(0, 1)
                        .unsqueeze(1)
                    )  # (4, bsz, seqlen) -> (4, 1, bsz * seqlen)
                else:
                    position_ids_rmpad = index_first_axis(
                        rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."), indices
                    ).transpose(0, 1)

                if "image_bound" in multi_modal_inputs:
                    from verl.utils.dataset.vision_utils import process_multi_modal_inputs_for_minicpmo

                    multi_modal_inputs = process_multi_modal_inputs_for_minicpmo(
                        input_ids, attention_mask, position_ids, cu_seqlens, multi_modal_inputs
                    )

                # for compute the log_prob
                input_ids_rmpad_rolled = torch.roll(input_ids_rmpad, shifts=-1, dims=1)  # (1, total_nnz)

                # pad and slice the inputs if sp > 1
                if self.use_ulysses_sp:
                    is_vlm_model = hasattr(
                        getattr(self.actor_module, "module", self.actor_module).config, "vision_config"
                    )
                    if is_vlm_model:
                        # vlm model's inputs will be sliced after embedding
                        input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad(
                            input_ids_rmpad,
                            position_ids_rmpad=position_ids_rmpad,
                            sp_size=self.ulysses_sequence_parallel_size,
                        )
                    else:
                        input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad_and_slice_inputs(
                            input_ids_rmpad,
                            position_ids_rmpad=position_ids_rmpad,
                            sp_size=self.ulysses_sequence_parallel_size,
                        )
                    input_ids_rmpad_rolled, _, _ = ulysses_pad_and_slice_inputs(
                        input_ids_rmpad_rolled,
                        position_ids_rmpad=None,
                        sp_size=self.ulysses_sequence_parallel_size,
                    )

                input_ids_rmpad_rolled = input_ids_rmpad_rolled.squeeze(0)  # ((total_nnz / sp) + pad)

                # only pass input_ids and position_ids to enable flash_attn_varlen
                extra_args = {}
                if self.use_fused_kernels:
                    extra_args["temperature"] = temperature
                    extra_args["return_dict"] = True

                output = self.actor_module(
                    input_ids=input_ids_rmpad,
                    attention_mask=None,
                    position_ids=position_ids_rmpad,
                    **multi_modal_inputs,
                    use_cache=False,
                    **extra_args,
                )  # prevent model thinks we are generating

                if self.use_fused_kernels:
                    log_probs = output.log_probs.squeeze(0)  # (total_nnz,)
                    entropy_rmpad = output.entropy.squeeze(0)  # (total_nnz,)

                else:
                    logits_rmpad = output.logits.squeeze(0)  # (total_nnz, vocab_size)
                    logits_rmpad.div_(temperature)

                    # if use_sp: ((total_nnz / sp) + pad) ; if not use_sp: (batch, seqlen)
                    inplace_backward = True
                    if calculate_entropy:
                        inplace_backward = False
                    log_probs = logprobs_from_logits(
                        logits=logits_rmpad,
                        labels=input_ids_rmpad_rolled,
                        inplace_backward=inplace_backward,
                    )

                    # compute entropy
                    if calculate_entropy:
                        if not self.config.entropy_checkpointing:
                            entropy_rmpad = self.compute_entropy_from_logits(logits_rmpad)  # ((total_nnz / sp) + pad)
                        else:
                            entropy_rmpad = torch.utils.checkpoint.checkpoint(
                                self.compute_entropy_from_logits, logits_rmpad
                            )

                # gather log_prob if sp > 1
                if self.use_ulysses_sp:
                    # gather and unpad for the ulysses sp
                    log_probs = gather_outputs_and_unpad(
                        log_probs,
                        gather_dim=0,
                        unpad_dim=0,
                        padding_size=pad_size,
                    )
                    if calculate_entropy:
                        entropy_rmpad = gather_outputs_and_unpad(
                            entropy_rmpad,
                            gather_dim=0,
                            unpad_dim=0,
                            padding_size=pad_size,
                        )
                # pad back to (bsz, seqlen)
                if calculate_entropy:
                    full_entropy = pad_input(
                        hidden_states=entropy_rmpad.unsqueeze(-1),
                        indices=indices,
                        batch=batch_size,
                        seqlen=seqlen,
                    )
                full_log_probs = pad_input(
                    hidden_states=log_probs.unsqueeze(-1),
                    indices=indices,
                    batch=batch_size,
                    seqlen=seqlen,
                )

                # only return response part:
                if calculate_entropy:
                    entropy = full_entropy.squeeze(-1)[:, -response_length - 1 : -1]  # (bsz, response_length)
                log_probs = full_log_probs.squeeze(-1)[:, -response_length - 1 : -1]  # (bsz, response_length)

            else:  # not using rmpad and no ulysses sp
                extra_args = {}
                if self.use_fused_kernels:
                    extra_args["temperature"] = temperature
                    extra_args["return_dict"] = True

                output = self.actor_module(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    **multi_modal_inputs,
                    use_cache=False,
                    **extra_args,
                )  # prevent model thinks we are generating

                if self.use_fused_kernels:
                    log_probs = output.log_probs[:, -response_length - 1 : -1]
                    entropy = output.entropy[:, -response_length - 1 : -1]  # (bsz, response_length)

                else:
                    logits = output.logits

                    logits.div_(temperature)
                    logits = logits[:, -response_length - 1 : -1, :]  # (bsz, response_length, vocab_size)
                    log_probs = logprobs_from_logits(logits, micro_batch["responses"])
                    if calculate_entropy:
                        if not self.config.entropy_checkpointing:
                            entropy = verl_F.entropy_from_logits(logits)  # (bsz, response_length)
                        else:
                            entropy = torch.utils.checkpoint.checkpoint(verl_F.entropy_from_logits, logits)

            return entropy, log_probs

    def _forward_micro_batch_with_input_embeds(
        self, micro_batch, temperature, calculate_entropy=False
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward when caller provides precomputed `inputs_embeds`.

        Behavior mirrors the token-id path, including remove-padding and Ulysses SP.
        Labels are still computed from the token sequence (input_ids rolled by 1).
        """
        response_length = micro_batch["responses"].size(-1)
        with torch.autocast(device_type=self.device_name, dtype=torch.bfloat16):
            inputs_embeds = micro_batch["inputs_embeds"]  # (bsz, seqlen, hidden)
            input_ids = micro_batch["input_ids"]  # used to derive labels
            attention_mask = micro_batch["attention_mask"]
            position_ids = micro_batch["position_ids"]
            batch_size, seqlen, hidden_size = inputs_embeds.shape
            entropy = None

            if position_ids.dim() == 3:  # qwen2vl mrope
                position_ids = position_ids.transpose(0, 1)  # (bsz, 4, seqlen) -> (4, bsz, seqlen)

            if self.use_remove_padding:
                # Derive indices from input_ids rmpad to use consistently for embeds/pos ids
                input_ids_rmpad, indices, cu_seqlens, *_ = unpad_input(
                    input_ids.unsqueeze(-1), attention_mask
                )
                input_ids_rmpad = input_ids_rmpad.transpose(0, 1)  # (1, total_nnz)

                # Unpad embeds using the same indices
                embeds_flat = rearrange(inputs_embeds, "b s h -> (b s) h")
                embeds_rmpad = index_first_axis(embeds_flat, indices).unsqueeze(0)  # (1, total_nnz, hidden)

                # Unpad position ids
                if position_ids.dim() == 3:
                    position_ids_rmpad = (
                        index_first_axis(rearrange(position_ids, "c b s ... -> (b s) c ..."), indices)
                        .transpose(0, 1)
                        .unsqueeze(1)
                    )
                else:
                    position_ids_rmpad = index_first_axis(
                        rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."), indices
                    ).transpose(0, 1)

                # Labels from input_ids rmpad (next-token)
                input_ids_rmpad_rolled = torch.roll(input_ids_rmpad, shifts=-1, dims=1)

                # Ulysses SP: pad and slice along seq dim for all rmpad tensors
                if self.use_ulysses_sp:
                    # For ids/pos: reuse helper returning a common pad_size
                    input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad_and_slice_inputs(
                        input_ids_rmpad, position_ids_rmpad=position_ids_rmpad, sp_size=self.ulysses_sequence_parallel_size
                    )
                    input_ids_rmpad_rolled, _, _ = ulysses_pad_and_slice_inputs(
                        input_ids_rmpad_rolled, position_ids_rmpad=None, sp_size=self.ulysses_sequence_parallel_size
                    )

                    # For embeds: manually pad along seq and slice to keep the same pad_size
                    total_len = embeds_rmpad.size(1)
                    if pad_size > 0:
                        embeds_rmpad = torch.nn.functional.pad(embeds_rmpad, (0, 0, 0, pad_size))
                    embeds_rmpad = ulysses_pad_and_slice_inputs.__globals__["slice_input_tensor"](
                        embeds_rmpad, dim=1, padding=False
                    )

                else:
                    pad_size = 0

                # Call model with embeds
                extra_args = {}
                if self.use_fused_kernels:
                    extra_args["temperature"] = temperature
                    extra_args["return_dict"] = True

                output = self.actor_module(
                    inputs_embeds=embeds_rmpad,
                    attention_mask=None,
                    position_ids=position_ids_rmpad,
                    use_cache=False,
                    **extra_args,
                )

                if self.use_fused_kernels:
                    log_probs = output.log_probs.squeeze(0)  # (total_nnz,)
                    entropy_rmpad = output.entropy.squeeze(0)  # (total_nnz,)
                else:
                    logits_rmpad = output.logits.squeeze(0)  # (total_nnz, vocab)
                    logits_rmpad.div_(temperature)

                    inplace_backward = not calculate_entropy
                    log_probs = logprobs_from_logits(
                        logits=logits_rmpad,
                        labels=input_ids_rmpad_rolled,
                        inplace_backward=inplace_backward,
                    )

                    if calculate_entropy:
                        if not self.config.entropy_checkpointing:
                            entropy_rmpad = self.compute_entropy_from_logits(logits_rmpad)
                        else:
                            entropy_rmpad = torch.utils.checkpoint.checkpoint(self.compute_entropy_from_logits, logits_rmpad)

                # Gather and unpad across SP
                if self.use_ulysses_sp:
                    log_probs = gather_outputs_and_unpad(
                        log_probs,
                        gather_dim=0,
                        unpad_dim=0,
                        padding_size=pad_size,
                    )
                    if calculate_entropy:
                        entropy_rmpad = gather_outputs_and_unpad(
                            entropy_rmpad,
                            gather_dim=0,
                            unpad_dim=0,
                            padding_size=pad_size,
                        )

                # Pad back to (bsz, seqlen)
                if calculate_entropy:
                    full_entropy = pad_input(
                        hidden_states=entropy_rmpad.unsqueeze(-1),
                        indices=indices,
                        batch=batch_size,
                        seqlen=seqlen,
                    )
                full_log_probs = pad_input(
                    hidden_states=log_probs.unsqueeze(-1),
                    indices=indices,
                    batch=batch_size,
                    seqlen=seqlen,
                )

                if calculate_entropy:
                    entropy = full_entropy.squeeze(-1)[:, -response_length - 1 : -1]
                log_probs = full_log_probs.squeeze(-1)[:, -response_length - 1 : -1]

            else:
                # No remove-padding: pass embeds with regular masks
                extra_args = {}
                if self.use_fused_kernels:
                    extra_args["temperature"] = temperature
                    extra_args["return_dict"] = True

                output = self.actor_module(
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    use_cache=False,
                    **extra_args,
                )

                if self.use_fused_kernels:
                    log_probs = output.log_probs[:, -response_length - 1 : -1]
                    entropy = output.entropy[:, -response_length - 1 : -1]
                else:
                    logits = output.logits
                    logits.div_(temperature)
                    logits = logits[:, -response_length - 1 : -1, :]
                    log_probs = logprobs_from_logits(logits, micro_batch["responses"])
                    if calculate_entropy:
                        if not self.config.entropy_checkpointing:
                            entropy = verl_F.entropy_from_logits(logits)
                        else:
                            entropy = torch.utils.checkpoint.checkpoint(verl_F.entropy_from_logits, logits)

            return entropy, log_probs

    def _optimizer_step(self):
        assert self.config.grad_clip is not None

        if isinstance(self.actor_module, FSDP):
            grad_norm = self.actor_module.clip_grad_norm_(max_norm=self.config.grad_clip)
        elif isinstance(self.actor_module, FSDPModule):
            grad_norm = fsdp2_clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.grad_clip)
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.grad_clip)

        if isinstance(grad_norm, DTensor):
            grad_norm = grad_norm.full_tensor()

        # if grad_norm is not finite, skip the update
        if not torch.isfinite(grad_norm):
            print(f"WARN: rank {torch.distributed.get_rank()} grad_norm is not finite: {grad_norm}")
            self.actor_optimizer.zero_grad()
        else:
            self.actor_optimizer.step()
        return grad_norm

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def compute_log_prob(self, data: DataProto, calculate_entropy=False) -> torch.Tensor:
        """Compute the log probability of the responses given input_ids, attention_mask and position_ids

        Args:
            data (DataProto): a DataProto containing keys

                ``input_ids``: tensor of shape [batch_size, sequence_length]. torch.int64. Note that input_ids is the
                concatenation of prompt and response. Note that ``sequence_length = prompt_length + response_length``.

                ``attention_mask``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``position_ids``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``responses``:  tensor of shape [batch_size, response_length]. torch.int64.

        Returns:
            torch.Tensor: the log_prob tensor
        """
        # set to eval
        self.actor_module.eval()

        micro_batch_size = data.meta_info["micro_batch_size"]
        temperature = data.meta_info["temperature"]  # temperature must be in the data.meta_info to avoid silent error
        use_dynamic_bsz = data.meta_info["use_dynamic_bsz"]
        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()
        select_keys = ["responses", "input_ids", "attention_mask", "position_ids"]
        non_tensor_select_keys = ["multi_modal_inputs"] if has_multi_modal_inputs else []

        data = data.select(batch_keys=select_keys, non_tensor_batch_keys=non_tensor_select_keys)

        if use_dynamic_bsz:
            max_token_len = data.meta_info["max_token_len"] * self.ulysses_sequence_parallel_size
            micro_batches, batch_idx_list = prepare_dynamic_batch(data, max_token_len=max_token_len)
        else:
            micro_batches = data.split(micro_batch_size)

        log_probs_lst = []
        entropy_lst = []
        for micro_batch in micro_batches:
            micro_batch = micro_batch.to(get_device_id())
            model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}
            with torch.no_grad():
                entropy, log_probs = self._forward_micro_batch(
                    model_inputs, temperature=temperature, calculate_entropy=calculate_entropy
                )
            log_probs_lst.append(log_probs)
            if calculate_entropy:
                entropy_lst.append(entropy)

        log_probs = torch.concat(log_probs_lst, dim=0)
        entropys = None
        if calculate_entropy:
            entropys = torch.concat(entropy_lst, dim=0)

        if use_dynamic_bsz:
            log_probs = restore_dynamic_batch(log_probs, batch_idx_list)
            if calculate_entropy:
                entropys = restore_dynamic_batch(entropys, batch_idx_list)

        return log_probs, entropys

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def update_policy(self, data: DataProto):
        # make sure we are in training mode
        self.actor_module.train()

        temperature = data.meta_info["temperature"]  # temperature must be in the data.meta_info to avoid silent error

        select_keys = [
            "responses",
            "response_mask",
            "input_ids",
            "attention_mask",
            "position_ids",
            "old_log_probs",
            "advantages",
        ]
        if self.config.use_kl_loss:
            select_keys.append("ref_log_prob")
        if self.config.tis_imp_ratio_cap > 0:
            assert "rollout_log_probs" in data.batch.keys(), (
                "Truncated Importance Sampling (TIS) requires to configure "
                "`actor_rollout_ref.rollout.calculate_log_probs=True` "
                "and is not currently supported in Server mode (agent loop)."
            )
            select_keys.append("rollout_log_probs")

        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()
        non_tensor_select_keys = ["multi_modal_inputs"] if has_multi_modal_inputs else []

        data = data.select(batch_keys=select_keys, non_tensor_batch_keys=non_tensor_select_keys)

        # Split to make minibatch iterator for updating the actor
        # See PPO paper for details. https://arxiv.org/abs/1707.06347
        mini_batches = data.split(self.config.ppo_mini_batch_size)

        on_policy = len(mini_batches) == 1 and self.config.ppo_epochs == 1

        metrics = {}
        for _ in range(self.config.ppo_epochs):
            for batch_idx, mini_batch in enumerate(mini_batches):
                if self.config.use_dynamic_bsz:
                    max_token_len = self.config.ppo_max_token_len_per_gpu * self.ulysses_sequence_parallel_size
                    micro_batches, _ = prepare_dynamic_batch(mini_batch, max_token_len=max_token_len)
                else:
                    self.gradient_accumulation = (
                        self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu
                    )
                    micro_batches = mini_batch.split(self.config.ppo_micro_batch_size_per_gpu)

                self.actor_optimizer.zero_grad()

                for micro_batch in micro_batches:
                    micro_batch = micro_batch.to(get_device_id())
                    micro_batch_metrics = {}
                    model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}
                    response_mask = model_inputs["response_mask"]
                    old_log_prob = model_inputs["old_log_probs"]
                    rollout_log_probs = model_inputs["rollout_log_probs"] if self.config.tis_imp_ratio_cap > 0 else None
                    advantages = model_inputs["advantages"]

                    entropy_coeff = self.config.entropy_coeff
                    loss_agg_mode = self.config.loss_agg_mode

                    if self.config.use_dynamic_bsz:
                        loss_scale_factor = response_mask.shape[0] / self.config.ppo_mini_batch_size
                    else:
                        loss_scale_factor = 1 / self.gradient_accumulation

                    # all return: (bsz, response_length)
                    calculate_entropy = False
                    if entropy_coeff != 0:
                        calculate_entropy = True
                    entropy, log_prob = self._forward_micro_batch(
                        model_inputs, temperature=temperature, calculate_entropy=calculate_entropy
                    )

                    if on_policy:
                        old_log_prob = log_prob.detach()
                    else:
                        old_log_prob = model_inputs["old_log_probs"]

                    loss_mode = self.config.policy_loss.get("loss_mode", "vanilla")
                    # vanilla -> verl.trainer.ppo.core_algos.compute_policy_loss_vanilla
                    # gpg -> verl.trainer.ppo.core_algos.compute_policy_loss_gpg
                    # clip_cov -> verl.trainer.ppo.core_algos.compute_policy_loss_clip_cov
                    policy_loss_fn = get_policy_loss_fn(loss_mode)
                    pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower = policy_loss_fn(
                        old_log_prob=old_log_prob,
                        log_prob=log_prob,
                        advantages=advantages,
                        response_mask=response_mask,
                        loss_agg_mode=loss_agg_mode,
                        config=self.config,
                        rollout_log_probs=rollout_log_probs,
                    )

                    if entropy_coeff != 0:
                        entropy_loss = agg_loss(loss_mat=entropy, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)

                        # compute policy loss
                        policy_loss = pg_loss - entropy_loss * entropy_coeff
                    else:
                        policy_loss = pg_loss

                    if self.config.use_kl_loss:
                        ref_log_prob = model_inputs["ref_log_prob"]
                        # compute kl loss
                        kld = kl_penalty(
                            logprob=log_prob, ref_logprob=ref_log_prob, kl_penalty=self.config.kl_loss_type
                        )
                        kl_loss = agg_loss(loss_mat=kld, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)

                        policy_loss = policy_loss + kl_loss * self.config.kl_loss_coef
                        micro_batch_metrics["actor/kl_loss"] = kl_loss.detach().item() * loss_scale_factor
                        micro_batch_metrics["actor/kl_coef"] = self.config.kl_loss_coef

                    if self.config.use_dynamic_bsz:
                        # relative to the dynamic bsz
                        loss = policy_loss * loss_scale_factor
                    else:
                        loss = policy_loss * loss_scale_factor
                    loss.backward()

                    micro_batch_metrics.update(
                        {
                            "actor/pg_loss": pg_loss.detach().item() * loss_scale_factor,
                            "actor/pg_clipfrac": pg_clipfrac.detach().item(),
                            "actor/ppo_kl": ppo_kl.detach().item(),
                            "actor/pg_clipfrac_lower": pg_clipfrac_lower.detach().item(),
                        }
                    )
                    append_to_dict(metrics, micro_batch_metrics)

                grad_norm = self._optimizer_step()
                mini_batch_metrics = {"actor/grad_norm": grad_norm.detach().item()}
                append_to_dict(metrics, mini_batch_metrics)
        self.actor_optimizer.zero_grad()
        return metrics


# Usage Example:
# 
# To integrate pixel_values to embeddings processing in your training pipeline:
# 
# 1. In your actor worker initialization, ensure the actor_module has visual capabilities
# 2. Before calling _forward_micro_batch, process multi_modal_inputs:
# 
#    # Process single sample
#    processed_inputs = actor.process_pixel_values_to_embeddings(multi_modal_inputs)
#    
#    # Or process a batch of samples (more efficient)
#    batch_processed_inputs = actor.process_batch_pixel_values_to_embeddings(batch_multi_modal_inputs)
# 
# 3. The processed inputs will have 'image_embeddings' instead of 'pixel_values'
# 4. This integrates seamlessly with existing multi-modal processing in _forward_micro_batch
# 
# Benefits:
# - Leverages existing GPU-loaded actor_module instead of loading separate visual model
# - Avoids GPU memory issues in dataset processing
# - Supports both individual and batch processing for efficiency
# - Maintains compatibility with existing multi-modal training pipeline
