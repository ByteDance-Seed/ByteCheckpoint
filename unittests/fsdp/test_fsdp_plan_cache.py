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
    FullOptimStateDictConfig,
    FullStateDictConfig,
    MixedPrecision,
    StateDictType,
)
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
)
from torch.testing._internal.common_utils import run_tests

from bytecheckpoint import FSDPCheckpointer
from bytecheckpoint.engine import _store_engine
from bytecheckpoint.planner.common import GLOBAL_PLAN_CACHE
from unittests.common import DEVICE_TYPE, Model, TestFSDPBase, diff, with_comms

TMP_DIR = "tmp_dir"
NUM_DEVICES_SAVE = 8
STEPS = 2
HIDDEN_SIZE = 512
LAYER_NUM = 8


class TestFSDPPlanCache(TestFSDPBase):
    @property
    def world_size(self) -> int:
        return NUM_DEVICES_SAVE

    @with_comms
    def test_save(self):
        rank = dist.get_rank()
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
        # do train steps
        for i in range(STEPS):
            loss = model(torch.rand(HIDDEN_SIZE, HIDDEN_SIZE, device=DEVICE_TYPE)).sum()
            loss.backward()
            optimizer.step()
        full_model_state_dcit_config = FullStateDictConfig(rank0_only=True, offload_to_cpu=True)
        full_opt_state_dcit_config = FullOptimStateDictConfig(rank0_only=True, offload_to_cpu=True)
        with FSDP.state_dict_type(
            model, StateDictType.FULL_STATE_DICT, full_model_state_dcit_config, full_opt_state_dcit_config
        ):
            model_state_dict_before_save = model.state_dict()
            optim_state_dict_before_save = FSDP.optim_state_dict(model, optimizer)
        # save for verifying
        if dist.get_rank() == 0:
            torch.save(model_state_dict_before_save, "./model_state_before_save.pt")
            torch.save(optim_state_dict_before_save, "./optim_state_dict_before_save.pt")
        dist.barrier()
        # save with fsdpCheckpointer
        ckpt_state = {"model": model, "optimizer": optimizer}
        FSDPCheckpointer.save(TMP_DIR, ckpt_state, fast_saving=True)
        assert len(GLOBAL_PLAN_CACHE._cache) == 2
        FSDPCheckpointer.save(TMP_DIR, ckpt_state, fast_saving=True)
        FSDPCheckpointer.save(TMP_DIR, ckpt_state, fast_saving=True)
        _store_engine.cleanup_resources()
        FSDPCheckpointer.load(TMP_DIR, ckpt_state)
        with FSDP.state_dict_type(
            model, StateDictType.FULL_STATE_DICT, full_model_state_dcit_config, full_opt_state_dcit_config
        ):
            model_state_dict_after_load = model.state_dict()
            optim_state_dict_after_load = FSDP.optim_state_dict(model, optimizer)
        # Verify load and save
        if rank == 0:
            model_state_dict_before_save = torch.load("./model_state_before_save.pt")
            optim_state_dict_before_save = torch.load("./optim_state_dict_before_save.pt")
            model_diffs = diff(model_state_dict_before_save, model_state_dict_after_load)
            optim_diffs = diff(optim_state_dict_before_save, optim_state_dict_after_load)
            assert not any(map(bool, model_diffs)), model_diffs
            assert not any(map(bool, optim_diffs)), optim_diffs


if __name__ == "__main__":
    run_tests()
