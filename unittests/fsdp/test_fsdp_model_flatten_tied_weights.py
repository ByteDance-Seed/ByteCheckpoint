################################################################################
#
# Copyright 2023 ByteDance Ltd. and/or its affiliates. All rights reserved.
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
)
from torch.distributed.fsdp.wrap import ModuleWrapPolicy
from torch.testing._internal.common_utils import run_tests

from bytecheckpoint import FSDPCheckpointer
from bytecheckpoint.engine import _store_engine
from unittests.common import DEVICE_TYPE, Layer, Model, TestFSDPBase, diff, with_comms

TMP_DIR = "tmp_dir"
NUM_DEVICES = 8
STEPS = 2
HIDDEN_SIZE = 511
LAYER_NUM = 3


class TestFSDPSaveLoad1(TestFSDPBase):
    @property
    def world_size(self) -> int:
        return NUM_DEVICES

    @with_comms
    def test_save_load(self):
        rank = dist.get_rank()
        torch.manual_seed(0)
        model = Model(HIDDEN_SIZE, LAYER_NUM, with_tied_weights=True).to(rank)
        # model = FSDP(model).to(torch.bfloat16)
        model = FSDP(
            model,
            mixed_precision=MixedPrecision(param_dtype=torch.float16, cast_forward_inputs=True),
            use_orig_params=True,
            auto_wrap_policy=ModuleWrapPolicy([Layer]),
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
        FSDPCheckpointer.save(TMP_DIR, ckpt_state, fast_saving=True, save_decomposed_model_optimizer=True)
        _store_engine.cleanup_resources()

        model_load = Model(HIDDEN_SIZE, LAYER_NUM, with_tied_weights=True).to(rank)
        model_load = FSDP(
            model_load,
            mixed_precision=MixedPrecision(param_dtype=torch.float16, cast_forward_inputs=True),
            use_orig_params=True,
            auto_wrap_policy=ModuleWrapPolicy([Layer]),
        )

        parameters_load = list(model_load.parameters())
        optimizer_load = torch.optim.Adam(parameters_load, lr=0.1, weight_decay=1e-2)
        optimizer_load.zero_grad()
        # load from ckpt
        ckpt_state = {"model": model_load, "optimizer": optimizer_load}
        FSDPCheckpointer.load(TMP_DIR, ckpt_state, load_decomposed_model_optimizer=True)
        # verify
        model_diffs = diff(model.state_dict(), model_load.state_dict())
        optim_diffs = diff(optimizer.state_dict(), optimizer_load.state_dict())
        assert not any(map(bool, model_diffs)), model_diffs
        assert not any(map(bool, optim_diffs)), optim_diffs
        self.check_tied_weights(model)
        self.check_tied_weights(model_load)

    @staticmethod
    def check_tied_weights(model):
        from bytecheckpoint.planner.fsdp.fsdp_hack import find_tied_weights

        tied_weights = find_tied_weights(model)
        assert len(tied_weights) > 0


if __name__ == "__main__":
    run_tests()
