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
import os

import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP

import bytecheckpoint as bcp

PLATFORM = os.environ.get("PLATFORM", "cuda").lower()
if PLATFORM not in {"cuda", "musa"}:
    PLATFORM = "cuda"

CKPT_PATH = "./tmp_checkpoint_dir_ddp"
HIDDEN_SIZE = 512
LAYER_NUM = 8


class Layer(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.net1 = nn.Linear(hidden_size, hidden_size * 4)
        self.relu = nn.ReLU()
        self.net2 = nn.Linear(hidden_size * 4, hidden_size)

    def forward(self, x):
        return self.net2(self.relu(self.net1(x)))


class Model(nn.Module):
    def __init__(self, hidden_size, layer_num):
        super().__init__()
        self.layers = nn.ModuleList([Layer(hidden_size) for _ in range(layer_num)])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser(description="ByteCheckpoint demo.")
    parser.add_argument(
        "--mode", "-m", type=str, default="normal", choices=["normal", "resume"], help="training mode to run"
    )
    parser.add_argument("--ckpt_path", type=str, default=CKPT_PATH, help="path to load/save the checkpoints")
    parser.add_argument("--iterations", "-i", type=int, default=3, help="the number of training iterations to run")
    args = parser.parse_args()

    # Start Distributed PyTorch
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    if PLATFORM == "cuda":
        torch.cuda.set_device(rank)
    elif PLATFORM == "musa":
        torch.musa.set_device(rank)

    # Define and initalize FSDP model and optimizer
    model = Model(HIDDEN_SIZE, LAYER_NUM).to(rank)
    model = DDP(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    optimizer.zero_grad()
    # Normal mode: training with iteration-level async checkpointing.
    if args.mode == "normal":
        checkpoint_state = {
            "model": model,
            "optimizer": optimizer,
            "extra_state": {"torch_rng_state": torch.get_rng_state()},
        }

        for iter in range(args.iterations):
            loss = model(torch.ones(HIDDEN_SIZE, HIDDEN_SIZE, device=PLATFORM)).sum()
            loss.backward()
            optimizer.step()
            # Save ckpt every step
            bcp.save(
                args.ckpt_path,
                checkpoint_state,
                framework="ddp",
                global_steps=iter,
            )
    else:
        # Resume mode: resume from checkpoint and continue training.
        checkpoint_state = {"model": model, "optimizer": optimizer, "extra_state": {}}
        bcp.load(f"{args.ckpt_path}/global_step_0", checkpoint_state, framework="ddp")
        torch.set_rng_state(checkpoint_state["extra_state"]["torch_rng_state"])
        for iter in range(args.iterations):
            loss = model(torch.ones(HIDDEN_SIZE, HIDDEN_SIZE, device=PLATFORM)).sum()
            loss.backward()
            optimizer.step()
            # Save ckpt every step
            bcp.save(
                args.ckpt_path,
                checkpoint_state,
                framework="ddp",
                global_steps=iter,
            )

    dist.barrier()
    dist.destroy_process_group()

#  torchrun --master_addr=localhost --master_port=6000 --nproc_per_node=8 --nnodes=1 demo/ddp_save_reshard.py --mode normal
#  torchrun --master_addr=localhost --master_port=6000 --nproc_per_node=4 --nnodes=1 demo/ddp_save_reshard.py --mode resume
