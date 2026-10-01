# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import torch
from loguru import logger

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.common.utility_functions import comp_pcc
from models.demos.blackhole.qwen36.tt.vision.functional import qwen3_5_vision_transformer_preprocess
from models.tt_transformers.tt.ccl import TT_CCL
from models.tt_transformers.tt.common import get_rot_transformation_mat
from models.tt_transformers.tt.load_checkpoints import (
    convert_hf_to_meta,
    convert_rope_style_hf_to_meta,
    standardize_hf_keys_multimodal,
)

from .patch_merger import PatchMerger
from .vision_block import VisionBlock
from .vision_model_config import VisionModelArgs


class VisionTransformer(LightweightModule):
    """
    Vision Transformer model for Qwen 3 VL.
    This implements only the transformer blocks part of the vision transformer.
    Patch embedding and merging should be done outside this class.
    """

    def __init__(
        self,
        args,
        dtype,
        state_dict,
        weight_cache_path,
        tt_ccl=None,
    ):
        """
        Initialize the Vision Transformer model.

        Args:
            args (VisionModelArgs): Model arguments
            dtype (ttnn.dtype): Data type for computations
            state_dict (dict): State dictionary containing model weights
            weight_cache_path (str): Path to weight cache
            tt_ccl (TT_CCL, optional): CCL helper. The vision tower is always
                tensor-parallel, so one is constructed internally if not given.
        """
        super().__init__()
        self.args = args
        self.dtype = dtype
        self.weight_cache_path = weight_cache_path

        # The vision tower is always tensor-parallel and needs a TT_CCL on hand.
        self.tt_ccl = tt_ccl if tt_ccl is not None else TT_CCL(args.mesh_device)

        # Create transformation matrix for RoPE QK prefill
        transformation_mat_torch = get_rot_transformation_mat(
            args.head_dim
        )  # todo)) args.head_dim is ignored inside the function
        self.transformation_mats = {
            "prefill": ttnn.as_tensor(
                transformation_mat_torch,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=args.mesh_device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ReplicateTensorToMesh(args.mesh_device),
            )
        }

        # Create vision blocks
        self.blocks = []
        for i in range(args.hf_config.vision_config.depth):
            block = VisionBlock(
                mesh_device=args.mesh_device,
                state_dict=state_dict,
                weight_cache_path=weight_cache_path,
                layer_num=i,
                dtype=dtype,
                transformation_mats=self.transformation_mats,
                args=args,
                tt_ccl=self.tt_ccl,
            )
            self.blocks.append(block)

        # The Megatron-style PatchMerger consumes a fractured-along-dim=3 tensor
        # directly (no pre-merger all-gather) and produces a fractured-along-dim=3
        # output, mirroring the LLM's DistributedNorm + LMHead final stretch in
        # `tt_transformers.tt.model`.
        self.patch_merger = PatchMerger(
            mesh_device=args.mesh_device,
            args=args,
            state_dict=state_dict,
            weight_cache_path=weight_cache_path,
            state_dict_prefix=args.get_state_dict_prefix("PatchMerger"),
            dtype=dtype,
            tt_ccl=self.tt_ccl,
        )

    def prepare_input(self, patch_input, seq_len=None):
        """Convert a patchified torch input to a ttnn tensor
        Args:
            patch_input (torch.Tensor): Patchified input tensor
            seq_len (int): Sequence length

        Returns:
            ttnn.Tensor: Prepared input tensor
        """
        patch_seq_len, _ = patch_input.shape
        x = patch_input
        seq_len = ((patch_seq_len // 128) + 1) * 128 if seq_len is None else seq_len
        x = torch.nn.functional.pad(x, (0, 0, 0, seq_len - patch_seq_len)).unsqueeze(0)
        x = self.args.prepare_residual_tensor_prefill(x)
        return x

    def forward(
        self,
        x,
        unpadded_seq_len,
        rot_mats,
    ):
        """
        Forward pass through the Vision Transformer blocks.

        Args:
            x (ttnn.Tensor): Input tensor [batch_size, 1, seq_len, hidden_dim].
                This arrives fractured along dim=3 (the hidden dim) because
                `prepare_residual_tensor_prefill` shards at load time.
            unpadded_seq_len (int): Real sequence length before padding.
            rot_mats (list): Rotation matrices for positional embeddings.

        Returns:
            ttnn.Tensor: Output tensor, fractured along dim=3 (each device owns
                out_hidden_size/TP), matching the LLM lm_head contract.
        """
        # Padded patches must not take part in the real patches' attention: the sequence is zero-padded to a 2048
        # multiple (prepare_input) and the blocks' SDPA is non-causal, so without a window the real queries attend to
        # the padding keys (bias-only K/V) - 62 % of the keys for a 448 px image, PCC 0.88 vs HF at any weight dtype.
        # The SDPA op's on-device block-diagonal windows keep real and padding tokens apart.
        cu_window = None
        seq_len = int(x.shape[2])
        if unpadded_seq_len is not None and int(unpadded_seq_len) < seq_len:
            cu_window = ttnn.from_torch(
                torch.tensor([0, int(unpadded_seq_len), seq_len], dtype=torch.int32),
                dtype=ttnn.int32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=self.args.mesh_device,
                mesh_mapper=ttnn.ReplicateTensorToMesh(self.args.mesh_device),
            )
        for i, block in enumerate(self.blocks):
            x = block(
                x,
                rot_mats=rot_mats,
                cu_window_seqlens=cu_window,
            )
        if cu_window is not None:
            ttnn.deallocate(cu_window)

        # The PatchMerger consumes the block output fractured along dim=3
        # directly (its first op is a DistributedLayerNorm that all-gathers
        # internally) and produces a fractured-along-dim=3 output. No
        # pre-merger all-gather is needed.
        # The merger runs on the PADDED sequence (zero rows merge to discarded rows): slicing to the real length here
        # keyed every merger program on the image's exact patch count, so any image size not seen at warm-up compiled
        # after the traces were parked (served image failure, Qwen3.5-122B 2026-10-01). The caller drops the padding
        # rows on host (DropInVisionTransformer.forward).
        x = self.patch_merger(x)
        return x


class DropInVisionTransformer(torch.nn.Module):
    """Wraps VisionTransformer to be a drop-in replacement for
    Qwen2_5_VisionTransformerPretrainedModel. It uses the reference model
    for certain preprocessing steps like patch embedding and index calculation.
    """

    def __init__(
        self,
        reference_model,
        model_args: VisionModelArgs,
        dtype=ttnn.bfloat8_b,
        debug=False,
        tt_ccl=None,
    ):
        """
        Initialize the TorchVisionTransformer wrapper.

        Args:
            reference_model (Qwen2_5_VisionTransformerPretrainedModel): Initialized reference HF model instance.
            model_args (VisionModelArgs): Model configuration arguments.
            dtype (ttnn.dtype): Compute dtype for weights.
            debug (bool): If True, run the reference path alongside and report PCC.
            tt_ccl (TT_CCL, optional): Reuse a CCL helper across multiple models.
        """
        super().__init__()
        self.reference_model = reference_model
        self.model_args = model_args
        self.debug = debug

        state_dict = standardize_hf_keys_multimodal(reference_model.state_dict())
        state_dict = convert_hf_to_meta(state_dict, model_args.head_dim)
        state_dict_prefix = model_args.get_state_dict_prefix("VisionTransformer")
        state_dict = {f"{state_dict_prefix}.{k}": v for k, v in state_dict.items()}

        # Initialize TT model
        self.tt_model = VisionTransformer(
            args=model_args,
            state_dict=state_dict,
            weight_cache_path=model_args.weight_cache_path(dtype),
            dtype=dtype,
            tt_ccl=tt_ccl,
        )

    @property
    def dtype(self):
        return self.reference_model.dtype

    @property
    def spatial_merge_size(self):
        return self.model_args.hf_config.vision_config.spatial_merge_size

    def forward(self, pixel_values: torch.Tensor, grid_thw: torch.Tensor) -> torch.Tensor:
        """
        Forward pass mimicking the Qwen3_5_VisionTransformerPretrainedModel interface.

        Args:
            pixel_values (torch.Tensor): Input pixel values tensor (equivalent to hidden_states for the ref model start).
                                         Shape typically [num_patches, hidden_size] or similar before patch_embed.
            grid_thw (torch.Tensor): Tensor describing the grid dimensions (time, height, width) for each image/video.
                                     Shape [num_images_or_videos, 3].

        Returns:
            torch.Tensor: Output tensor with shape [1, B, seq_len, out_hidden_size].
        """
        # process pixel_values for each image/video separately
        all_pixel_values = pixel_values
        all_grid_thw = grid_thw
        final_outputs = []
        # todo)) refactor this code to leverage tt-mesh's ttnn.ShardTensorToMesh(mesh_device, dim=batch_size_dim) for data parallelism
        for grid_thw in all_grid_thw:
            # --- pick out the pixel_values for this users' images (grid_thw.prod() pixels) ---
            pixel_values = all_pixel_values[: grid_thw.prod(), :]
            all_pixel_values = all_pixel_values[grid_thw.prod() :, :]
            # --- Preprocessing ---
            # 1. Calculate total unpadded sequence length
            grid_thw = grid_thw.unsqueeze(0)
            unpadded_seq_len = grid_thw.prod(dim=1).sum().item()
            # Calculate padded sequence length (divisible by 2048) required by models/tt_transformers/tt/attention.py::forward_prefill
            seq_len = ((unpadded_seq_len // 2048) + 1) * 2048

            # 2. Use preprocessing function from reference/functional to get indices and embeddings
            cu_seqlens, position_embeddings = qwen3_5_vision_transformer_preprocess(
                seq_len=unpadded_seq_len,
                grid_thw=grid_thw,
                head_dim=self.model_args.head_dim,
                spatial_merge_size=self.model_args.hf_config.vision_config.spatial_merge_size,
            )

            # 3. Use reference model's patch embedding
            patch_input = self.reference_model.patch_embed(pixel_values)
            pos_embeds = self.reference_model.fast_pos_embed_interpolate(grid_thw)
            patch_input = patch_input + pos_embeds

            # 4. Prepare rotational embeddings (cos, sin) -> pad -> convert to TT tensors
            cos_orig, sin_orig = position_embeddings
            cos_orig, sin_orig = convert_rope_style_hf_to_meta(cos_orig, sin_orig)
            # pad sequence length with cos = 1, sin = 0 (identity rotation)
            cos_padded = (
                torch.nn.functional.pad(cos_orig, (0, 0, 0, seq_len - unpadded_seq_len), value=1)
                .unsqueeze(0)
                .unsqueeze(0)
            )
            sin_padded = (
                torch.nn.functional.pad(sin_orig, (0, 0, 0, seq_len - unpadded_seq_len), value=0)
                .unsqueeze(0)
                .unsqueeze(0)
            )
            # Convert to TT tensors on the mesh device
            cos = ttnn.from_torch(
                cos_padded,
                dtype=ttnn.bfloat16,  # Use bfloat16 for RoPE
                layout=ttnn.TILE_LAYOUT,
                device=self.model_args.mesh_device,
                mesh_mapper=ttnn.ReplicateTensorToMesh(self.model_args.mesh_device),
            )
            sin = ttnn.from_torch(
                sin_padded,
                dtype=ttnn.bfloat16,  # Use bfloat16 for RoPE
                layout=ttnn.TILE_LAYOUT,
                device=self.model_args.mesh_device,
                mesh_mapper=ttnn.ReplicateTensorToMesh(self.model_args.mesh_device),
            )
            rot_mats = [cos, sin]

            # 5. Prepare input tensor for the TT model using window_index
            tt_input = self.tt_model.prepare_input(patch_input, seq_len)

            # --- TT Model Execution ---
            tt_out = self.tt_model(
                tt_input,
                unpadded_seq_len=unpadded_seq_len,
                rot_mats=rot_mats,  # Use rot_mats generated in this forward pass
            )

            # deallocate device tensors that are not needed by decode
            ttnn.deallocate(tt_input)
            ttnn.deallocate(cos)
            ttnn.deallocate(sin)
            ttnn.deallocate(rot_mats[0])
            ttnn.deallocate(rot_mats[1])

            # --- Postprocessing (host) ---
            # The merger output is [1, 1, S_padded / merge^2, H_out(/TP)]: fractured along the hidden dim on a mesh
            # (each device owns out_hidden_size / TP channels, possibly padded). The device graph above depends only on
            # the padded length, so every image size maps to one of the warmed buckets; the real rows are taken HERE on
            # host (the consumer, the prefill splice, reads the rows to host anyway) and the images are concatenated on
            # host too (a device concat / slice would compile per image-size combination).
            merge2 = self.model_args.hf_config.vision_config.spatial_merge_size**2
            n_tokens = unpadded_seq_len // merge2
            tp = self.model_args.cluster_shape[1]
            out_hidden_size = self.model_args.hf_config.vision_config.out_hidden_size // tp
            per_dev = [
                ttnn.to_torch(t)[0, 0, :n_tokens, :out_hidden_size] for t in ttnn.get_device_tensors(tt_out)
            ]  # TP x [n_tokens, out_hidden / TP]
            ttnn.deallocate(tt_out)
            rows = torch.cat(per_dev, dim=-1)  # [n_tokens, out_hidden]

            if self.debug:
                logger.info(f"DropInVisionTransformer: Debug enabled, running reference model...")
                reference_output = self.reference_model.forward(pixel_values, grid_thw)
                _, pcc = comp_pcc(reference_output, rows)
                logger.info(f"DropInVisionTransformer: PCC to reference model: {pcc}")

            final_outputs.append(rows)

        # all images' real rows back on the mesh, hidden-fractured like the text embeddings: [1, 1, N, out_hidden]
        rows = torch.cat(final_outputs, dim=0).to(torch.bfloat16)
        tt_out = ttnn.from_torch(
            rows.reshape(1, 1, rows.shape[0], rows.shape[1]),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self.model_args.mesh_device,
            mesh_mapper=ttnn.ShardTensorToMesh(self.model_args.mesh_device, dim=-1),
        )
        return tt_out
