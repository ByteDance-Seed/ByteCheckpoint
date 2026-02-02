################################################################################
#
# Copyright 2025 ByteDance Ltd. and/or its affiliates. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
################################################################################

import os
import time
from typing import Optional

import torch
import torch.distributed as dist

# Defaultly import torch fsdp related method.
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import (
    LocalStateDictConfig,
    ShardedStateDictConfig,
    StateDictType,
)

from bytecheckpoint.checkpointer.base_checkpointer import BaseCheckpointer
from bytecheckpoint.io import bfile
from bytecheckpoint.planner.common import _init_optim_state
from bytecheckpoint.planner.fsdp.fsdp_hack import find_tied_weights, init_fsdp_hack
from bytecheckpoint.planner.fsdp.fsdp_planner import FSDPLoadPlanner, FSDPSavePlanner
from bytecheckpoint.storage import CKPTCounter
from bytecheckpoint.utilities.logger import get_bytecheckpoint_logger

from .meta_type import (
    _EXTRA_STATE_FORMAT,
    DEFAULT_STR,
    EXTRA_STATE_STR,
    FSDP_FLAT_PARAM_META,
    FSDP_STR,
    MODEL_STR,
    OPTIMIZER_STR,
    SUPPORTED_ROLE_SUFFIX_TYPES,
    CheckpointState,
)

logger = get_bytecheckpoint_logger()

FSDP_SUPPORTED_TYPES = {MODEL_STR, OPTIMIZER_STR, EXTRA_STATE_STR}
FSDP_ORDERED_LOADING_KEYS = [OPTIMIZER_STR, MODEL_STR, EXTRA_STATE_STR]


class FSDPCheckpointer(BaseCheckpointer):
    """
    The Checkpointer class for FSDP
    """

    _framework_name = FSDP_STR

    @classmethod
    def save(
        cls,
        ckpt_path: str,
        checkpoint_state: CheckpointState,
        fast_saving=False,
        global_steps: Optional[int] = None,
        callback: Optional[callable] = None,
        role: str = None,
        ignore_append_global_steps_to_folder: Optional[bool] = False,
        save_decomposed_model_optimizer: bool = False,
    ):
        """
        The save method saves the state of an FSDP (Fully Sharded Data Parallel) model, its optimizer, and additional metadata into a specified checkpoint directory.
        It supports advanced configurations for distributed training, sharded states, and asynchronous I/O.

        Args:
            ckpt_path (str): The directory path where the checkpoint will be saved.
            checkpoint_state (CheckpointState): The state object containing components to be saved (e.g., model, optimizer, extra state).
            fast_saving (bool, optional): Enables asynchronous checkpoint saving for faster I/O operations. Defaults to False.
            global_steps (Optional[int], optional): The current global step for naming the checkpoint directory. If provided, the checkpoint is saved in a subdirectory named after the step. Defaults to None.
            callback (Optional[callable], optional): A callback function to execute after the save operation. Defaults to None.
            role (str, optional): The role of the model and optimizer to be saved, e.g. 'actor' or 'critic' in RL. Defaults to None.
            ignore_append_global_steps_to_folder (Optional[bool], optional): Whether to ignore appending the global step number to the folder name. Defaults to False.
            kwargs: Additional arguments for custom configurations.
            save_decomposed_model_optimizer (bool, optional): Whether to save the model and optimizer state in a flattened format for efficient zero communication checkpointing. Defaults to False.

        Returns:
            None
        """
        save_start_time = time.time()
        logger.info("Start save function call. ")
        # Check flatten mode and tied weights.
        _ = init_decompose_mode_fsdp(checkpoint_state)
        # Check role.
        if role and role not in SUPPORTED_ROLE_SUFFIX_TYPES:
            logger.warning("Unsupported role %s, will use default role instead", role)
            role = SUPPORTED_ROLE_SUFFIX_TYPES[DEFAULT_STR]
        # Check version.
        assert "2.1.0" <= torch.__version__.strip(), (
            "ByteCheckpoint now only support torch version from 2.1 and above"
        )
        # Check supported components.
        cls.check_supported_components(checkpoint_state, FSDP_SUPPORTED_TYPES)
        # Check model str.
        assert MODEL_STR in checkpoint_state, "Model is a must"
        # Get checkpoint path for the current step.
        curr_ckpt_path = cls.get_curr_ckpt_path(ckpt_path, global_steps, ignore_append_global_steps_to_folder)
        cls.make_ckpt_dirs(curr_ckpt_path, list(checkpoint_state.keys()), FSDP_SUPPORTED_TYPES)
        # Set the ckpt counter.
        ckpt_counter = CKPTCounter(len(checkpoint_state), fast_saving)
        # Determine state dict type.
        if save_decomposed_model_optimizer:
            state_dict_type = StateDictType.LOCAL_STATE_DICT
            model_state_cfg = LocalStateDictConfig(offload_to_cpu=True)
        else:
            state_dict_type = StateDictType.SHARDED_STATE_DICT
            model_state_cfg = ShardedStateDictConfig(offload_to_cpu=True)
        # Start saving checkpoint.
        save_ckpt_start_time = time.time()
        for key, value in (checkpoint_state).items():
            if key == MODEL_STR:
                # Get model path and state_dict.
                model_path = os.path.join(curr_ckpt_path, MODEL_STR)
                get_model_shard_state_dict_time = time.time()
                with FSDP.state_dict_type(value, state_dict_type, model_state_cfg):
                    model_state_dict = value.state_dict()
                logger.info(
                    "FSDP got model shard state dict, cost time: %s s", time.time() - get_model_shard_state_dict_time
                )
                # Save model to storage.
                cls._save_model(
                    model_path=model_path,
                    model_state=model_state_dict,
                    framework_name=FSDP_STR,
                    role=role,
                    save_planner=FSDPSavePlanner(),
                    ckpt_path=ckpt_path,
                    save_ckpt_start_time=save_ckpt_start_time,
                    ckpt_counter=ckpt_counter,
                    global_steps=global_steps,
                    fast_saving=fast_saving,
                    callback=callback,
                )
            elif key == OPTIMIZER_STR:
                model = checkpoint_state[MODEL_STR]
                # Get optimizer path and state_dict.
                optimizer_path = os.path.join(curr_ckpt_path, OPTIMIZER_STR)
                get_optim_shard_state_dict_time = time.time()
                with FSDP.state_dict_type(model, state_dict_type, model_state_cfg):
                    _init_optim_state(value)
                    osd = FSDP.optim_state_dict(model, value)
                    optimizer_state_dict = osd
                logger.info(
                    "FSDP got optimizer shard state dict, cost time: %s s",
                    time.time() - get_optim_shard_state_dict_time,
                )
                # Save optimizer to storage.
                cls._save_optimizer(
                    optimizer_path=optimizer_path,
                    optimizer_state=optimizer_state_dict,
                    framework_name=FSDP_STR,
                    role=role,
                    save_planner=FSDPSavePlanner(),
                    ckpt_path=ckpt_path,
                    save_ckpt_start_time=save_ckpt_start_time,
                    ckpt_counter=ckpt_counter,
                    global_steps=global_steps,
                    fast_saving=fast_saving,
                    callback=callback,
                )
            elif key == EXTRA_STATE_STR:
                # Get extra path, state and file name.
                extra_state_path = os.path.join(curr_ckpt_path, EXTRA_STATE_STR)
                extra_state_dict = cls.ensure_tensor_on_cpu(value)
                extra_state_file_name = _EXTRA_STATE_FORMAT.format(dist.get_rank())
                # Save extra state.
                cls._save_extra_state(
                    extra_state_path=extra_state_path,
                    extra_state=extra_state_dict,
                    extra_state_file_name=extra_state_file_name,
                    framework_name=FSDP_STR,
                    role=role,
                    ckpt_path=ckpt_path,
                    save_ckpt_start_time=save_ckpt_start_time,
                    ckpt_counter=ckpt_counter,
                    global_steps=global_steps,
                    fast_saving=fast_saving,
                    callback=callback,
                )
        save_cost_time = time.time() - save_start_time
        logger.info("End save function call. Total time cost: %s s", save_cost_time)

    @classmethod
    def load(
        cls,
        ckpt_path: str,
        checkpoint_state: CheckpointState,
        fast_loading: bool = False,
        strict: bool = True,
        role: str = None,
        load_decomposed_model_optimizer: bool = False,
    ):
        """
        The load method restores the state of an FSDP model, its optimizer, and additional metadata from a checkpoint directory.
        It supports loading sharded model states, handling distributed environments, and optimizing file access via shared memory or broadcasting.

        Args:
            ckpt_path (str): The directory path where the checkpoint is stored.
            checkpoint_state (CheckpointState): The state object to populate with the loaded components (e.g., model, optimizer, extra state).
            fast_loading (bool, optional): Whether to enable parallel checkpoint loading. Defaults to False.
            strict (bool, optional): Whether to strictly enforce that all keys in the checkpoint match the model/optimizer state dictionary. Defaults to True.
            role (str, optional): The role of the model and optimizer to be loaded, e.g., 'actor' or 'critic' in RL. Defaults to None.
            load_decomposed_model_optimizer (bool, optional): Whether to load the model and optimizer state in a flattened format. Defaults to False.

        Returns:
            None
        """
        load_start_time = time.time()
        logger.info("Start load function call. ")

        # Init fsdp decompose mode.
        has_tied_weights = False
        if load_decomposed_model_optimizer:
            has_tied_weights = init_decompose_mode_fsdp(checkpoint_state)
        # Check version.
        assert "2.1.0" <= torch.__version__.strip() and torch.__version__.strip() < "2.6.0", (
            "ByteCheckpoint now only support torch version from 2.1 to 2.5"
        )
        # Check role.
        if role and role not in SUPPORTED_ROLE_SUFFIX_TYPES:
            logger.warning("Unsupported role %s, will use default role instead", role)
            role = SUPPORTED_ROLE_SUFFIX_TYPES[DEFAULT_STR]
        # Check supported components.
        cls.check_supported_components(checkpoint_state, FSDP_SUPPORTED_TYPES)
        # Check model str.
        assert MODEL_STR in checkpoint_state, "Model is a must"

        # Determine state dict type.
        if load_decomposed_model_optimizer:
            state_dict_type = StateDictType.LOCAL_STATE_DICT
        else:
            state_dict_type = StateDictType.SHARDED_STATE_DICT

        # Start loading checkpoint.
        # Optimizer must be loaded before model to avoid optimizer.step() to modify model parameters(): param.mul_(1 - lr * weight_decay)
        # https://pytorch.org/docs/stable/_modules/torch/optim/adamw.html#AdamW.
        for key in FSDP_ORDERED_LOADING_KEYS:
            value = checkpoint_state.get(key, None)
            if value is None:
                logger.warning("Key %s not found in checkpoint state, will skip loading.", key)
                continue
            if key == MODEL_STR:
                # Get model path and state_dict.
                model_path = os.path.join(ckpt_path, MODEL_STR)
                get_model_shard_state_dict_time = time.time()
                with FSDP.state_dict_type(value, state_dict_type):
                    model_state_dict = value.state_dict()
                    if load_decomposed_model_optimizer:
                        model_state_dict.pop(FSDP_FLAT_PARAM_META, {})
                logger.info(
                    "FSDP got model shard state dict, cost time: %s s", time.time() - get_model_shard_state_dict_time
                )
                # Load model from storage.
                model_state_dict = cls._load_model(
                    model_path=model_path,
                    model_state=model_state_dict,
                    framework_name=FSDP_STR,
                    role=role,
                    load_planner=FSDPLoadPlanner(strict=strict),
                    fast_loading=fast_loading,
                )
                with FSDP.state_dict_type(value, state_dict_type):
                    value.load_state_dict(model_state_dict, strict=False if has_tied_weights else True)
            elif key == OPTIMIZER_STR:
                model = checkpoint_state[MODEL_STR]
                # Get optimizer path and state_dict.
                optimizer_path = os.path.join(ckpt_path, OPTIMIZER_STR)
                get_optim_shard_state_dict_time = time.time()
                with FSDP.state_dict_type(model, state_dict_type):
                    _init_optim_state(value)
                    osd = FSDP.optim_state_dict(model, value)
                    optimizer_state_dict = osd
                logger.info(
                    "FSDP got optimizer shard state dict, cost time: %s s",
                    time.time() - get_optim_shard_state_dict_time,
                )
                # Load optimizer from storage.
                optimizer_state_dict = cls._load_optimizer(
                    optimizer_path=optimizer_path,
                    optimizer_state=optimizer_state_dict,
                    framework_name=FSDP_STR,
                    role=role,
                    load_planner=FSDPLoadPlanner(strict=strict),
                    fast_loading=fast_loading,
                )
                with FSDP.state_dict_type(model, state_dict_type):
                    flattened_osd = FSDP.optim_state_dict_to_load(model, value, optimizer_state_dict)
                    if state_dict_type != StateDictType.LOCAL_STATE_DICT:
                        value.load_state_dict(flattened_osd)
            elif key == EXTRA_STATE_STR:
                # NOTE: Currently, ByteCheckpoint does not guarantee extra_state reshard correctness.
                # Get path and file name.
                extra_state_path = os.path.join(ckpt_path, EXTRA_STATE_STR)
                extra_state_file_name = _EXTRA_STATE_FORMAT.format(dist.get_rank())
                # Set the file name to be the one produced by rank 0 if it does not exist.
                if not bfile.exists(os.path.join(extra_state_path, extra_state_file_name)):
                    extra_state_file_name = _EXTRA_STATE_FORMAT.format(0)
                # Load extra state from storage.
                checkpoint_state[key] = cls._load_extra_state(
                    extra_state_path=extra_state_path,
                    extra_state_file_name=extra_state_file_name,
                    framework_name=FSDP_STR,
                    role=role,
                )
        dist.barrier()
        load_cost_time = time.time() - load_start_time
        logger.info("End load function call. Total time cost: %s s", load_cost_time)


def init_decompose_mode_fsdp(checkpoint_state: CheckpointState) -> bool:
    tied_weights = find_tied_weights(checkpoint_state[MODEL_STR])
    logger.info("all tied weights %s", tied_weights)
    init_fsdp_hack(tied_weights)
    if tied_weights:
        return True
    return False
