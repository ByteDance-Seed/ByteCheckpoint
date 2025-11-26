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
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.testing._internal.common_utils import run_tests

from bytecheckpoint import DDPCheckpointer
from bytecheckpoint.engine import _store_engine
from unittests.common import DEVICE_TYPE, Model, TestFSDPBase, diff, with_comms

TMP_DIR = "tmp_dir"
NUM_DEVICES = 8
STEPS = 2
HIDDEN_SIZE = 64
LAYER_NUM = 16


class TestDDPLoadSave(TestFSDPBase):
    @property
    def world_size(self) -> int:
        return NUM_DEVICES

    @with_comms
    def test_simple_load_save(self):
        rank = dist.get_rank()
        # print("rank:",rank)
        torch.manual_seed(0)
        model = Model(HIDDEN_SIZE, LAYER_NUM).to(rank)
        dist.barrier()
        model = DDP(
            model,
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
        optimizer.zero_grad()
        # do one step
        for i in range(STEPS):
            loss = model(torch.rand(HIDDEN_SIZE, HIDDEN_SIZE, device=DEVICE_TYPE)).sum()
            loss.backward()
            optimizer.step()
        model_state_dict_before_save = model.state_dict()
        optim_state_dict_before_save = optimizer.state_dict()
        ckpt_state = {"model": model, "optimizer": optimizer}
        # ckpt_state = {"model": model}
        DDPCheckpointer.save(TMP_DIR, ckpt_state)
        _store_engine.cleanup_resources()

        # print("save finish-----")
        model_new = Model(HIDDEN_SIZE, LAYER_NUM).to(rank)

        dist.barrier()
        model_new = DDP(model_new)
        optimizer_new = torch.optim.Adam(model_new.parameters(), lr=0.1)
        ckpt_state = {"model": model_new, "optimizer": optimizer_new}
        # ckpt_state = {"model": model_new}
        DDPCheckpointer.load(TMP_DIR, ckpt_state)
        model_state_dict_after_load = model_new.state_dict()
        optim_state_dict_after_load = optimizer_new.state_dict()
        # Verify load and save
        model_diffs = diff(model_state_dict_before_save, model_state_dict_after_load)
        optim_diffs = diff(optim_state_dict_before_save, optim_state_dict_after_load)
        assert not any(map(bool, model_diffs)), model_diffs
        assert not any(map(bool, optim_diffs)), optim_diffs


if __name__ == "__main__":
    run_tests()
