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
from torch.distributed._composable.fsdp import fully_shard
from torch.testing._internal.common_utils import run_tests

from bytecheckpoint import FSDP2Checkpointer
from bytecheckpoint.engine import _store_engine
from unittests.common import DEVICE_TYPE, PLATFORM, Model, TestFSDPBase, diff, with_comms

TMP_DIR = "tmp_dir"
NUM_DEVICES = 8
STEPS = 2
HIDDEN_SIZE = 64
LAYER_NUM = 16


class TestFSDP2SaveLoad(TestFSDPBase):
    @property
    def world_size(self) -> int:
        return NUM_DEVICES

    @with_comms
    def test_simple_load_save(self):
        rank = dist.get_rank()
        torch.manual_seed(0)
        model = Model(HIDDEN_SIZE, LAYER_NUM).to(rank)
        model = fully_shard(model)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.1, weight_decay=1e-2)

        optimizer.zero_grad()
        # do one step
        for _ in range(STEPS):
            loss = model(torch.rand(HIDDEN_SIZE, HIDDEN_SIZE, device=DEVICE_TYPE)).sum()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        # Ground truth
        model_state_dict_before_save = model.state_dict()
        optim_state_dict_before_save = optimizer.state_dict()
        # Save
        if PLATFORM == "cuda":
            current_device = torch.cuda.current_device()
        elif PLATFORM == "musa":
            current_device = torch.musa.current_device()
        else:
            current_device = 0
        ckpt_state = {
            "model": model,
            "optimizer": optimizer,
            "extra_state": {"a": torch.tensor(1, device=f"{PLATFORM}:{current_device}")},
        }
        FSDP2Checkpointer.save(TMP_DIR, ckpt_state, fast_saving=True)
        _store_engine.cleanup_resources()

        # Load
        model_new = Model(HIDDEN_SIZE, LAYER_NUM).to(rank)
        model_new = fully_shard(model_new)
        optimizer_new = torch.optim.Adam(model_new.parameters(), lr=0.1, weight_decay=1e-2)
        ckpt_state = {"model": model_new, "optimizer": optimizer_new, "extra_state": {}}
        FSDP2Checkpointer.load(TMP_DIR, ckpt_state)

        # Get state dict
        model_state_dict_after_load = model_new.state_dict()
        optim_state_dict_after_load = optimizer_new.state_dict()
        # Verify load and save
        model_diffs = diff(model_state_dict_before_save, model_state_dict_after_load)
        optim_diffs = diff(optim_state_dict_before_save, optim_state_dict_after_load)
        assert not any(map(bool, model_diffs)), model_diffs
        assert not any(map(bool, optim_diffs)), optim_diffs
        assert torch.equal(ckpt_state["extra_state"]["a"], torch.tensor(1, device="cpu"))


if __name__ == "__main__":
    run_tests()
