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
import shutil

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
)
from torch.distributed.fsdp import (
    MixedPrecision,
    ShardingStrategy,
)
from torch.distributed.fsdp.wrap import ModuleWrapPolicy
from torch.testing._internal.common_utils import run_tests

from bytecheckpoint import FSDPCheckpointer
from bytecheckpoint.engine import _store_engine
from bytecheckpoint.utilities.ckpt_format.merge_tool import bytecheckpoint_ckpt_to_pytorch_ckpt
from unittests.common import DEVICE_TYPE, Layer, Model, TestFSDPBase, diff, with_comms

TMP_DIR = "tmp_dir"
NUM_DEVICES = 8
STEPS = 2
HIDDEN_SIZE = 511
LAYER_NUM = 3


class TestFSDPSaveLoadFlatten(TestFSDPBase):
    @property
    def world_size(self) -> int:
        return NUM_DEVICES

    @with_comms
    def test_save_load(self):
        os.makedirs(TMP_DIR, exist_ok=True)
        rank = dist.get_rank()
        torch.manual_seed(0)
        model = Model(HIDDEN_SIZE, LAYER_NUM).to(rank)
        # model = FSDP(model).to(torch.bfloat16)
        device_mesh = init_device_mesh(DEVICE_TYPE, mesh_shape=(2, 4), mesh_dim_names=["dp", "fsdp"])
        model = FSDP(
            model,
            mixed_precision=MixedPrecision(param_dtype=torch.float16, cast_forward_inputs=True),
            use_orig_params=True,
            auto_wrap_policy=ModuleWrapPolicy([Layer]),
            sharding_strategy=ShardingStrategy.HYBRID_SHARD,
            device_mesh=device_mesh,
        )
        parameters = list(model.parameters())
        optimizer = torch.optim.Adam(parameters, lr=0.1, weight_decay=1e-2)
        optimizer.zero_grad()
        # do train steps
        for i in range(STEPS):
            loss = model(torch.rand(HIDDEN_SIZE, HIDDEN_SIZE, device=DEVICE_TYPE, dtype=torch.float16)).sum()
            loss.backward()
            optimizer.step()

        dist.barrier()
        # save with fsdpCheckpointer
        ckpt_state = {"model": model, "optimizer": optimizer}
        os.makedirs(f"{TMP_DIR}/hsdp", exist_ok=True)
        FSDPCheckpointer.save(f"{TMP_DIR}/hsdp", ckpt_state, fast_saving=False, save_decomposed_model_optimizer=True)
        _store_engine.cleanup_resources()

        device_mesh_load = init_device_mesh(DEVICE_TYPE, mesh_shape=(8,), mesh_dim_names=["fsdp"])
        model_load = Model(HIDDEN_SIZE, LAYER_NUM).to(rank)
        model_load = FSDP(
            model_load,
            mixed_precision=MixedPrecision(param_dtype=torch.float16, cast_forward_inputs=True),
            use_orig_params=True,
            auto_wrap_policy=ModuleWrapPolicy([Layer]),
            sharding_strategy=ShardingStrategy.FULL_SHARD,
            device_mesh=device_mesh_load,
        )

        parameters_load = list(model_load.parameters())
        optimizer_load = torch.optim.Adam(parameters_load, lr=0.1, weight_decay=1e-2)
        optimizer_load.zero_grad()
        # load from ckpt
        ckpt_state = {"model": model_load, "optimizer": optimizer_load}
        FSDPCheckpointer.load(f"{TMP_DIR}/hsdp", ckpt_state, load_decomposed_model_optimizer=True)
        os.makedirs(f"{TMP_DIR}/fsdp", exist_ok=True)
        FSDPCheckpointer.save(f"{TMP_DIR}/fsdp", ckpt_state, fast_saving=False, save_decomposed_model_optimizer=True)
        _store_engine.cleanup_resources()

        # verify
        if dist.get_rank() == 0:
            merged_hsdp = bytecheckpoint_ckpt_to_pytorch_ckpt(
                f"{TMP_DIR}/hsdp",
                f"{TMP_DIR}/hsdp_merged",
                "fsdp",
                model_only=True,
                return_dict=True,
                fsdp_save_decomposed_model=True,
            )
            merged_fsdp = bytecheckpoint_ckpt_to_pytorch_ckpt(
                f"{TMP_DIR}/fsdp",
                f"{TMP_DIR}/fsdp_merged",
                "fsdp",
                model_only=True,
                return_dict=True,
                fsdp_save_decomposed_model=True,
            )
            model_diffs = diff(merged_hsdp["model"], merged_fsdp["model"])
            print(f"model diffs: {model_diffs}")
            print(f"fsdp model: {merged_fsdp['model']}, hsdp model: {merged_hsdp['model']}")
            shutil.rmtree(TMP_DIR, ignore_errors=True)
            assert not any(map(bool, model_diffs)), model_diffs


class TestFSDPSaveLoadNaive(TestFSDPBase):
    @property
    def world_size(self) -> int:
        return NUM_DEVICES

    @with_comms
    def test_save_load(self):
        os.makedirs(TMP_DIR, exist_ok=True)
        rank = dist.get_rank()
        torch.manual_seed(0)
        model = Model(HIDDEN_SIZE, LAYER_NUM).to(rank)
        # model = FSDP(model).to(torch.bfloat16)
        device_mesh = init_device_mesh(DEVICE_TYPE, mesh_shape=(2, 4), mesh_dim_names=["dp", "fsdp"])
        model = FSDP(
            model,
            mixed_precision=MixedPrecision(param_dtype=torch.float16, cast_forward_inputs=True),
            use_orig_params=False,
            auto_wrap_policy=ModuleWrapPolicy([Layer]),
            sharding_strategy=ShardingStrategy.HYBRID_SHARD,
            device_mesh=device_mesh,
        )
        parameters = list(model.parameters())
        optimizer = torch.optim.Adam(parameters, lr=0.1, weight_decay=1e-2)
        optimizer.zero_grad()
        # do train steps
        for i in range(STEPS):
            loss = model(torch.rand(HIDDEN_SIZE, HIDDEN_SIZE, device=DEVICE_TYPE, dtype=torch.float16)).sum()
            loss.backward()
            optimizer.step()

        dist.barrier()
        # save with fsdpCheckpointer
        ckpt_state = {"model": model, "optimizer": optimizer}
        os.makedirs(f"{TMP_DIR}/hsdp", exist_ok=True)
        FSDPCheckpointer.save(f"{TMP_DIR}/hsdp", ckpt_state, fast_saving=False, save_decomposed_model_optimizer=False)
        _store_engine.cleanup_resources()

        device_mesh_load = init_device_mesh(DEVICE_TYPE, mesh_shape=(8,), mesh_dim_names=["fsdp"])
        model_load = Model(HIDDEN_SIZE, LAYER_NUM).to(rank)
        model_load = FSDP(
            model_load,
            mixed_precision=MixedPrecision(param_dtype=torch.float16, cast_forward_inputs=True),
            use_orig_params=False,
            auto_wrap_policy=ModuleWrapPolicy([Layer]),
            sharding_strategy=ShardingStrategy.FULL_SHARD,
            device_mesh=device_mesh_load,
        )

        parameters_load = list(model_load.parameters())
        optimizer_load = torch.optim.Adam(parameters_load, lr=0.1, weight_decay=1e-2)
        optimizer_load.zero_grad()
        # load from ckpt
        ckpt_state = {"model": model_load, "optimizer": optimizer_load}
        FSDPCheckpointer.load(f"{TMP_DIR}/hsdp", ckpt_state, load_decomposed_model_optimizer=False)
        os.makedirs(f"{TMP_DIR}/fsdp", exist_ok=True)
        FSDPCheckpointer.save(f"{TMP_DIR}/fsdp", ckpt_state, fast_saving=False, save_decomposed_model_optimizer=False)
        _store_engine.cleanup_resources()

        # verify
        if dist.get_rank() == 0:
            merged_hsdp = bytecheckpoint_ckpt_to_pytorch_ckpt(
                f"{TMP_DIR}/hsdp",
                f"{TMP_DIR}/hsdp_merged",
                "fsdp",
                model_only=True,
                return_dict=True,
                fsdp_save_decomposed_model=False,
            )
            merged_fsdp = bytecheckpoint_ckpt_to_pytorch_ckpt(
                f"{TMP_DIR}/fsdp",
                f"{TMP_DIR}/fsdp_merged",
                "fsdp",
                model_only=True,
                return_dict=True,
                fsdp_save_decomposed_model=False,
            )
            model_diffs = diff(merged_hsdp["model"], merged_fsdp["model"])
            print(f"model diffs: {model_diffs}")
            print(f"fsdp model: {merged_fsdp['model']}, hsdp model: {merged_hsdp['model']}")
            shutil.rmtree(TMP_DIR, ignore_errors=True)
            assert not any(map(bool, model_diffs)), model_diffs


if __name__ == "__main__":
    run_tests()
