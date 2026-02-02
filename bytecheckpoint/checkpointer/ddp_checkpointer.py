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

from bytecheckpoint.io import bfile
from bytecheckpoint.planner.common import _init_optim_state
from bytecheckpoint.planner.ddp.ddp_planner import DDPLoadPlanner, DDPSavePlanner
from bytecheckpoint.storage import CKPTCounter
from bytecheckpoint.utilities.logger import get_bytecheckpoint_logger

from .base_checkpointer import BaseCheckpointer
from .meta_type import (
    _EXTRA_STATE_FORMAT,
    DDP_STR,
    DEFAULT_STR,
    EXTRA_STATE_STR,
    MODEL_STR,
    OPTIMIZER_STR,
    SUPPORTED_ROLE_SUFFIX_TYPES,
    CheckpointState,
)

logger = get_bytecheckpoint_logger()

DDP_SUPPORTED_TYPES = {MODEL_STR, OPTIMIZER_STR, EXTRA_STATE_STR}
DDP_ORDERED_LOADING_KEYS = [OPTIMIZER_STR, MODEL_STR, EXTRA_STATE_STR]


class DDPCheckpointer(BaseCheckpointer):
    """
    The Checkpointer class for DDP
    """

    _framework_name = DDP_STR

    @classmethod
    def save(
        cls,
        ckpt_path: str,
        checkpoint_state: CheckpointState,
        fast_saving: bool = False,
        global_steps: Optional[int] = None,
        callback: Optional[callable] = None,
        role: str = None,
        ignore_append_global_steps_to_folder: Optional[bool] = False,
    ):
        """
        Saves the model, optimizer, and extra state checkpoint data.
        This method performs distributed data-parallel (DDP) checkpoint saving. It supports saving model states,
        optimizer states, and any additional metadata needed for checkpoint restoration. It also validates
        the checkpoint paths, handles shared memory configurations, and manages asynchronous IO operations.

        Args:
            ckpt_path (str): Root directory where the checkpoint will be saved.
            checkpoint_state (CheckpointState): The state dictionary containing model, optimizer, and extra state data.
            fast_saving (bool, optional): Whether to enable asynchronous checkpoint saving. Defaults to False.
            global_steps (Optional[int], optional): The current global step count, used to name subdirectories for checkpointing. Defaults to None.
            callback (Optional[callable], optional): A callback function to execute after the save operation. Defaults to None.
            role (str, optional): The role of the model and optimizer to be saved, e.g. 'actor' or 'critic' in RL. Defaults to None.
            ignore_append_global_steps_to_folder (Optional[bool], optional): Whether to ignore appending the global step number to the folder name. Defaults to False.

        Returns:
            None
        """

        save_start_time = time.time()
        logger.info("Start save function call. ")

        # Check version.
        assert "2.1.0" <= torch.__version__.strip(), (
            "ByteCheckpoint now only support torch version from 2.1 and above"
        )
        # Check role.
        if role and role not in SUPPORTED_ROLE_SUFFIX_TYPES:
            logger.warning("Unsupported role %s, will use default role instead", role)
            role = SUPPORTED_ROLE_SUFFIX_TYPES[DEFAULT_STR]
        # Check supported components.
        cls.check_supported_components(checkpoint_state, DDP_SUPPORTED_TYPES)
        # Check model str.
        assert MODEL_STR in checkpoint_state, "Model is a must"

        # Get checkpoint path for the current step.
        curr_ckpt_path = cls.get_curr_ckpt_path(ckpt_path, global_steps, ignore_append_global_steps_to_folder)
        cls.make_ckpt_dirs(curr_ckpt_path, list(checkpoint_state.keys()), DDP_SUPPORTED_TYPES)
        # Set the ckpt counter.
        ckpt_counter = CKPTCounter(len(checkpoint_state), fast_saving)

        # Start saving checkpoint.
        save_ckpt_start_time = time.time()
        for key, value in (checkpoint_state).items():
            if key == MODEL_STR:
                # Get model path and state_dict.
                model_path = os.path.join(curr_ckpt_path, MODEL_STR)
                model_state_dict = value.state_dict()
                # Save model to storage.
                cls._save_model(
                    model_path=model_path,
                    model_state=model_state_dict,
                    framework_name=DDP_STR,
                    role=role,
                    save_planner=DDPSavePlanner(),
                    ckpt_path=ckpt_path,
                    save_ckpt_start_time=save_ckpt_start_time,
                    ckpt_counter=ckpt_counter,
                    global_steps=global_steps,
                    fast_saving=fast_saving,
                    callback=callback,
                )
            elif key == OPTIMIZER_STR:
                # Get optimizer path and state_dict.
                optimizer_path = os.path.join(curr_ckpt_path, OPTIMIZER_STR)
                _init_optim_state(value)
                optimizer_state_dict = value.state_dict()
                # Save optimizer to storage.
                cls._save_optimizer(
                    optimizer_path=optimizer_path,
                    optimizer_state=optimizer_state_dict,
                    framework_name=DDP_STR,
                    role=role,
                    save_planner=DDPSavePlanner(),
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
                    framework_name=DDP_STR,
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
    ):
        """
        Loads the model, optimizer, and extra state checkpoint data.

        This method restores checkpoint data in a distributed data-parallel (DDP) environment. It reads model states,
        optimizer states, and any additional metadata from the specified path. It supports shared memory configurations,
        strict loading, and IO performance monitoring.

        Args:
            ckpt_path (str): Root directory where the checkpoint is stored.
            checkpoint_state (CheckpointState): The state dictionary to populate with loaded model, optimizer, and extra state data.
            fast_loading (bool, optional): Whether to enable parallel checkpoint loading. Defaults to False.
            strict (bool, optional): Whether to strictly enforce that all keys in the checkpoint match the model/optimizer state dictionary. Defaults to True.
            role (str, optional): The role of the model and optimizer to be loaded, e.g. 'actor' or 'critic' in RL. Defaults to None.

        Returns:
            None
        """
        load_start_time = time.time()
        logger.info("Start load function call. ")

        # Check version.
        assert torch.__version__.strip() < "2.6.0" and torch.__version__.strip() >= "2.1.0", (
            "ByteCheckpoint now only support torch version from 2.1 to 2.5"
        )
        # Check role.
        if role and role not in SUPPORTED_ROLE_SUFFIX_TYPES:
            logger.warning("Unsupported role %s, will use default role instead", role)
            role = SUPPORTED_ROLE_SUFFIX_TYPES[DEFAULT_STR]
        # Check supported components.
        cls.check_supported_components(checkpoint_state, DDP_SUPPORTED_TYPES)
        # Check model str.
        assert MODEL_STR in checkpoint_state, "Model is a must"

        # Set strict mode for loading.

        # Start loading checkpoint.
        # Optimizer must be loaded before model to avoid optimizer.step() to modify model parameters(): param.mul_(1 - lr * weight_decay)
        # https://pytorch.org/docs/stable/_modules/torch/optim/adamw.html#AdamW.
        for key in DDP_ORDERED_LOADING_KEYS:
            value = checkpoint_state.get(key, None)
            if value is None:
                logger.warning("Key %s not found in checkpoint state, will skip loading.", key)
                continue
            if key == MODEL_STR:
                # Get model path and state_dict.
                model_path = os.path.join(ckpt_path, MODEL_STR)
                model_state_dict = value.state_dict()
                # Load model from storage.
                model_state_dict = cls._load_model(
                    model_path=model_path,
                    model_state=model_state_dict,
                    framework_name=DDP_STR,
                    role=role,
                    load_planner=DDPLoadPlanner(strict=strict),
                    fast_loading=fast_loading,
                )
                value.load_state_dict(model_state_dict)
            elif key == OPTIMIZER_STR:
                # Get optimizer path and state_dict.
                optimizer_path = os.path.join(ckpt_path, OPTIMIZER_STR)
                _init_optim_state(value)
                optimizer_state_dict = value.state_dict()
                # Load optimizer from storage.
                optimizer_state_dict = cls._load_optimizer(
                    optimizer_path=optimizer_path,
                    optimizer_state=optimizer_state_dict,
                    framework_name=DDP_STR,
                    role=role,
                    load_planner=DDPLoadPlanner(strict=strict),
                    fast_loading=fast_loading,
                )
                value.load_state_dict(optimizer_state_dict)
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
                    framework_name=DDP_STR,
                    role=role,
                )
        dist.barrier()
        load_cost_time = time.time() - load_start_time
        logger.info("End load function call. Total time cost: %s s", load_cost_time)
