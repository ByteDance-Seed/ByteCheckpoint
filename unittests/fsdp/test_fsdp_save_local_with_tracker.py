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
from torch.testing._internal.common_utils import run_tests

from bytecheckpoint import FSDPCheckpointer
from bytecheckpoint.engine import _store_engine
from bytecheckpoint.utilities.ckpt_format.common_utils import find_latest_ckpt_path
from unittests.common import DEVICE_TYPE, Model, TestFSDPBase, diff, with_comms

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
            use_orig_params=False,
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=0.1, weight_decay=1e-2)

        optimizer.zero_grad()
        # do one step
        loss = model(torch.rand(HIDDEN_SIZE, HIDDEN_SIZE, device=DEVICE_TYPE)).sum()
        loss.backward()
        optimizer.step()

        with FSDP.state_dict_type(model, StateDictType.SHARDED_STATE_DICT):
            model_state_dict_before_save = model.state_dict()
            optim_state_dict_before_save = FSDP.optim_state_dict(model, optimizer)
        ckpt_state = {"model": model, "optimizer": optimizer}
        # Add tracker files
        FSDPCheckpointer.save(TMP_DIR, ckpt_state, global_steps=100, fast_saving=True)
        FSDPCheckpointer.save(TMP_DIR, ckpt_state, global_steps=200, fast_saving=True)
        _store_engine.cleanup_resources()

        model_new = Model(HIDDEN_SIZE, LAYER_NUM).to(rank)
        model_new = FSDP(
            model_new,
            mixed_precision=MixedPrecision(param_dtype=torch.float16, cast_forward_inputs=True),
            use_orig_params=False,
        )
        optimizer_new = torch.optim.Adam(model_new.parameters(), lr=0.1, weight_decay=1e-2)

        ckpt_state = {"model": model_new, "optimizer": optimizer_new}
        # When resume, use find_latest_ckpt_path to get latest complete checkpoint path
        latest_ckpt_path = find_latest_ckpt_path(TMP_DIR)

        FSDPCheckpointer.load(latest_ckpt_path, ckpt_state)
        with FSDP.state_dict_type(model_new, StateDictType.SHARDED_STATE_DICT):
            model_state_dict_after_load = model_new.state_dict()
            optim_state_dict_after_load = FSDP.optim_state_dict(model_new, optimizer_new)

        model_diffs = diff(model_state_dict_before_save, model_state_dict_after_load)
        optim_diffs = diff(optim_state_dict_before_save, optim_state_dict_after_load)
        assert not any(map(bool, model_diffs)), model_diffs
        assert not any(map(bool, optim_diffs)), optim_diffs


if __name__ == "__main__":
    run_tests()
