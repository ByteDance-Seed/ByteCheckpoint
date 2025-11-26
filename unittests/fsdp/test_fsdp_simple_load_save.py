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
import torch
import torch.distributed as dist
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
)
from torch.distributed.fsdp import (
    MixedPrecision,
    StateDictType,
)
from torch.distributed.fsdp.wrap import ModuleWrapPolicy
from torch.testing._internal.common_utils import run_tests

from bytecheckpoint import FSDPCheckpointer
from bytecheckpoint.engine import _store_engine
from unittests.common import DEVICE_TYPE, Layer, Model, TestFSDPBase, diff, with_comms

TMP_DIR = "tmp_dir"
NUM_DEVICES = 8
STEPS = 2
HIDDEN_SIZE = 64
LAYER_NUM = 16


class TestFSDPLoadSave(TestFSDPBase):
    @property
    def world_size(self) -> int:
        return NUM_DEVICES

    @with_comms
    def test_simple_load_save(self):
        rank = dist.get_rank()
        # print("rank:",rank)
        torch.manual_seed(0)
        model = Model(HIDDEN_SIZE, LAYER_NUM).to(rank)
        # model = FSDP(model).to(torch.bfloat16)
        model = FSDP(
            model,
            mixed_precision=MixedPrecision(param_dtype=torch.float16, cast_forward_inputs=True),
            use_orig_params=True,
            auto_wrap_policy=ModuleWrapPolicy([Layer]),
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=0.1, weight_decay=1e-2)

        optimizer.zero_grad()
        # do one step
        for i in range(STEPS):
            loss = model(torch.rand(HIDDEN_SIZE, HIDDEN_SIZE, device=DEVICE_TYPE)).sum()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        # if rank == 0:
        #     for k ,v in optimizer.state_dict()['state'].items():
        #         print("opt1 k1:",k,"v1:",v)
        with FSDP.state_dict_type(model, StateDictType.SHARDED_STATE_DICT):
            model_state_dict_before_save = model.state_dict()
            optim_state_dict_before_save = FSDP.optim_state_dict(model, optimizer)
        ckpt_state = {"model": model, "optimizer": optimizer}
        FSDPCheckpointer.save(TMP_DIR, ckpt_state, fast_saving=True)
        _store_engine.cleanup_resources()
        # if rank == 0:
        #     for k ,v in model_state_dict_before_save.items():
        #         print("model k1:",k,"v1:",v)
        #     for k ,v in optim_state_dict_before_save.items():
        #         print("opt k1:",k,"v1:",v)

        # print("save finish-----")
        model_new = Model(HIDDEN_SIZE, LAYER_NUM).to(rank)
        model_new = FSDP(
            model_new,
            mixed_precision=MixedPrecision(param_dtype=torch.float16, cast_forward_inputs=True),
            use_orig_params=True,
            auto_wrap_policy=ModuleWrapPolicy([Layer]),
        )
        optimizer_new = torch.optim.Adam(model_new.parameters(), lr=0.1, weight_decay=1e-2)

        ckpt_state = {"model": model_new, "optimizer": optimizer_new}
        FSDPCheckpointer.load(TMP_DIR, ckpt_state)
        with FSDP.state_dict_type(model_new, StateDictType.SHARDED_STATE_DICT):
            model_state_dict_after_load = model_new.state_dict()
            optim_state_dict_after_load = FSDP.optim_state_dict(model_new, optimizer_new)

        # if rank == 0:
        #     for k ,v in optimizer.state_dict()['state'].items():
        #         print("opt2 k1:",k,"v1:",v)
        # if rank == 0:
        #     for k ,v in model_state_dict_after_load.items():
        #         print("model k2:",k,"v2:",v)
        #     for k ,v in optim_state_dict_after_load.items():
        #         print("opt k2:",k,"v2:",v)

        # Verify load and save
        print(f"optim before: {optim_state_dict_before_save}")
        print(f"optim after: {optim_state_dict_after_load}")
        model_diffs = diff(model_state_dict_before_save, model_state_dict_after_load)
        optim_diffs = diff(optim_state_dict_before_save, optim_state_dict_after_load)
        assert not any(map(bool, model_diffs)), model_diffs
        assert not any(map(bool, optim_diffs)), optim_diffs


class TestFSDPLoadSaveOnlyModel(TestFSDPBase):
    @property
    def world_size(self) -> int:
        return NUM_DEVICES

    @with_comms
    def test_simple_load_save(self):
        rank = dist.get_rank()
        # print("rank:",rank)
        torch.manual_seed(0)
        model = Model(HIDDEN_SIZE, LAYER_NUM).to(rank)
        # model = FSDP(model).to(torch.bfloat16)
        model = FSDP(
            model,
            mixed_precision=MixedPrecision(param_dtype=torch.float16, cast_forward_inputs=True),
            use_orig_params=True,
            auto_wrap_policy=ModuleWrapPolicy([Layer]),
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=0.1, weight_decay=1e-2)
        # torch.__version__ = "2.5.0"
        optimizer.zero_grad()
        # do one step
        for i in range(STEPS):
            loss = model(torch.rand(HIDDEN_SIZE, HIDDEN_SIZE, device=DEVICE_TYPE)).sum()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        # if rank == 0:
        #     for k ,v in optimizer.state_dict()['state'].items():
        #         print("opt1 k1:",k,"v1:",v)
        with FSDP.state_dict_type(model, StateDictType.SHARDED_STATE_DICT):
            model_state_dict_before_save = model.state_dict()
        ckpt_state = {"model": model}
        FSDPCheckpointer.save(TMP_DIR, ckpt_state, fast_saving=True)
        _store_engine.cleanup_resources()

        model_new = Model(HIDDEN_SIZE, LAYER_NUM).to(rank)
        model_new = FSDP(
            model_new,
            mixed_precision=MixedPrecision(param_dtype=torch.float16, cast_forward_inputs=True),
            use_orig_params=True,
            auto_wrap_policy=ModuleWrapPolicy([Layer]),
        )
        ckpt_state = {"model": model_new}
        FSDPCheckpointer.load(TMP_DIR, ckpt_state)
        with FSDP.state_dict_type(model_new, StateDictType.SHARDED_STATE_DICT):
            model_state_dict_after_load = model_new.state_dict()
        # Verify load and save
        model_diffs = diff(model_state_dict_before_save, model_state_dict_after_load)
        assert not any(map(bool, model_diffs)), model_diffs


if __name__ == "__main__":
    run_tests()
