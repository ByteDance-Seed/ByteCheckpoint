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
from unittests.common import DEVICE_TYPE, Model, TestFSDPBase, dict_list_map_inplace, diff, with_comms

TMP_DIR = "tmp_dir"
NUM_DEVICES_SAVE = 8
NUM_DEVICES_LOAD = 4
STEPS = 2
HIDDEN_SIZE = 64
LAYER_NUM = 8


class TestDDPSaveLoad1(TestFSDPBase):
    @property
    def world_size(self) -> int:
        return NUM_DEVICES_SAVE

    @with_comms
    def test_save(self):
        rank = dist.get_rank()
        torch.manual_seed(0)
        model = Model(HIDDEN_SIZE, LAYER_NUM).to(rank)
        model = DDP(
            model,
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
        optimizer.zero_grad()
        # do train steps
        for i in range(STEPS):
            loss = model(torch.rand(HIDDEN_SIZE, HIDDEN_SIZE, device=DEVICE_TYPE)).sum()
            loss.backward()
            optimizer.step()
        # save for verifying
        if dist.get_rank() == 0:
            torch.save(model.state_dict(), "./model_state_before_save.pt")
            torch.save(optimizer.state_dict(), "./optim_state_dict_before_save.pt")
        dist.barrier()
        # save with fsdpCheckpointer
        ckpt_state = {"model": model, "optimizer": optimizer}
        DDPCheckpointer.save(TMP_DIR, ckpt_state)
        _store_engine.cleanup_resources()


class TestDDPSaveLoad2(TestFSDPBase):
    @property
    def world_size(self) -> int:
        return NUM_DEVICES_LOAD

    @with_comms
    def test_load(self):
        rank = dist.get_rank()
        torch.manual_seed(1)
        model = Model(HIDDEN_SIZE, LAYER_NUM).to(rank)
        model = DDP(
            model,
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
        # load from ckpt
        ckpt_state = {"model": model, "optimizer": optimizer}
        DDPCheckpointer.load(TMP_DIR, ckpt_state)

        def apply(x):
            if isinstance(x, torch.Tensor):
                x = x.cpu()
            return x

        model_state_dict_before_save = torch.load("./model_state_before_save.pt")
        optim_state_dict_before_save = torch.load("./optim_state_dict_before_save.pt")
        model_state_dict_after_load = model.state_dict()
        optim_state_dict_after_load = optimizer.state_dict()
        dict_list_map_inplace(apply, model_state_dict_after_load)
        dict_list_map_inplace(apply, optim_state_dict_after_load)
        dict_list_map_inplace(apply, model_state_dict_before_save)
        dict_list_map_inplace(apply, optim_state_dict_before_save)
        model_diffs = diff(model_state_dict_before_save, model_state_dict_after_load)
        optim_diffs = diff(optim_state_dict_before_save, optim_state_dict_after_load)
        assert not any(map(bool, model_diffs)), model_diffs
        assert not any(map(bool, optim_diffs)), optim_diffs


if __name__ == "__main__":
    run_tests()
