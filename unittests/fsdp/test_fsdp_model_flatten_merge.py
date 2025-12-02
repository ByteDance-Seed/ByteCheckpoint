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
from torch.distributed.fsdp.wrap import ModuleWrapPolicy
from torch.testing._internal.common_utils import run_tests

from bytecheckpoint import FSDPCheckpointer
from bytecheckpoint.engine import _store_engine
from bytecheckpoint.utilities.ckpt_format.merge_tool import bytecheckpoint_ckpt_to_pytorch_ckpt
from unittests.common import DEVICE_TYPE, Layer, Model, TestFSDPBase, diff, with_comms

CKPT_DIR_DISTCP = "./fsdp_flatten_distcp"
CKPT_DIR_MERGE = "./fsdp_flatten_merge"
NUM_DEVICES_SAVE = 8
STEPS = 2
HIDDEN_SIZE = 63
LAYER_NUM = 2


class TestFSDPSaveLoad(TestFSDPBase):
    @property
    def world_size(self) -> int:
        return NUM_DEVICES_SAVE

    @with_comms
    def test_save(self):
        rank = dist.get_rank()
        torch.manual_seed(0)
        model_1 = Model(HIDDEN_SIZE, LAYER_NUM).to(rank)
        # model = FSDP(model).to(torch.bfloat16)
        model_1 = FSDP(
            model_1,
            mixed_precision=MixedPrecision(param_dtype=torch.float16, cast_forward_inputs=True),
            use_orig_params=True,
            auto_wrap_policy=ModuleWrapPolicy([Layer]),
        )
        optimizer_1 = torch.optim.Adam(model_1.parameters(), lr=0.1, weight_decay=1e-2)
        optimizer_1.zero_grad()
        # do train steps
        for i in range(STEPS):
            loss = model_1(torch.rand(HIDDEN_SIZE, HIDDEN_SIZE, device=DEVICE_TYPE)).sum()
            loss.backward()
            optimizer_1.step()
            optimizer_1.zero_grad()
        full_model_state_dcit_config = FullStateDictConfig(rank0_only=True, offload_to_cpu=True)
        full_opt_state_dcit_config = FullOptimStateDictConfig(rank0_only=True, offload_to_cpu=True)
        with FSDP.state_dict_type(
            model_1, StateDictType.FULL_STATE_DICT, full_model_state_dcit_config, full_opt_state_dcit_config
        ):
            model_1_state_dict_before_save = model_1.state_dict()

        # save with fsdpCheckpointer
        ckpt_state = {
            "model": model_1,
            "optimizer": optimizer_1,
        }
        FSDPCheckpointer.save(
            CKPT_DIR_DISTCP,
            ckpt_state,
            fast_saving=True,
            save_decomposed_model_optimizer=True,
        )
        _store_engine.cleanup_resources()
        # verify merged
        if rank == 0:
            bytecheckpoint_ckpt_to_pytorch_ckpt(
                CKPT_DIR_DISTCP,
                CKPT_DIR_MERGE,
                framework="fsdp",
                fsdp_save_decomposed_model=True,
                model_only=True,
            )
            merged_model_path = f"{CKPT_DIR_MERGE}/model.pt"
            model_merged = torch.load(merged_model_path)

            return_dict = bytecheckpoint_ckpt_to_pytorch_ckpt(
                CKPT_DIR_DISTCP,
                CKPT_DIR_MERGE,
                framework="fsdp",
                return_dict=True,
                fsdp_save_decomposed_model=True,
                model_only=True,
            )
            model_merged_return = return_dict["model"]
            model_1_state_dict_after_merge_return = model_merged_return
            model_1_state_dict_after_merge = model_merged
            model_1_diffs = diff(model_1_state_dict_after_merge_return, model_1_state_dict_after_merge)
            assert not any(map(bool, model_1_diffs)), model_1_diffs
            model_1_diffs = diff(model_1_state_dict_before_save, model_1_state_dict_after_merge)
            assert not any(map(bool, model_1_diffs)), model_1_diffs


if __name__ == "__main__":
    run_tests()
