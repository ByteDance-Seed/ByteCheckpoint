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

import argparse
import json
import os

import torch.distributed as dist
from safetensors.torch import load_file

from bytecheckpoint import FSDP2Checkpointer


def setup(rank, world_size):
    dist.init_process_group("gloo", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


class MockedModel:
    def __init__(self, state_dict) -> None:
        self.state_dict_ = state_dict

    def state_dict(self):
        return self.state_dict_


if __name__ == "__main__":
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    setup(rank=rank, world_size=world_size)
    parser = argparse.ArgumentParser()
    parser.add_argument("--framework", type=str, default="fsdp")
    parser.add_argument("--load_dir", required=True, type=str)
    # Folder contains model.safetensors.index.json and other data files
    parser.add_argument("--save_dir", required=True, type=str)

    args = parser.parse_args()

    index_file_name = os.path.join(args.load_dir, "model.safetensors.index.json")

    with open(index_file_name) as f:
        index_data = json.load(f)

    config_file_name = os.path.join(args.load_dir, "config.json")
    with open(config_file_name) as f:
        config_data = json.load(f)

    # Get all safetensor files
    safetensor_files = set(index_data["weight_map"].values())

    # Load all weights
    model_weights = {}
    for file in safetensor_files:
        full_path = os.path.join(args.load_dir, file)
        print(f"Loading {full_path}...")
        model_weights.update(load_file(full_path))

    # (Optional): Do all necessary state dict modifications (changing key names or tensor shape) if you want
    final_state_dict = model_weights

    ckpt_state = {"model": MockedModel(final_state_dict)}
    # Save it
    FSDP2Checkpointer.save(path=args.save_dir, checkpoint_state=ckpt_state, fast_saving=True)

    cleanup()
