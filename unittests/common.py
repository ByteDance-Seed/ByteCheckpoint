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
import collections
import datetime
import sys
from functools import wraps
from typing import Any, Callable, Dict, Tuple, Union

import os

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed._tensor import DTensor
from torch.testing._internal.common_distributed import (
    TEST_SKIPS,
    MultiProcessTestCase,
)

PLATFORM = os.environ.get("PLATFORM", "cuda").lower()
if PLATFORM not in {"cuda", "musa"}:
    PLATFORM = "cuda"

if PLATFORM == "cuda":
    DEVICE_TYPE = "cuda" if torch.cuda.is_available() and torch.cuda.device_count() > 1 else "cpu"
    PG_BACKEND = "nccl" if DEVICE_TYPE == "cuda" else "gloo"
elif PLATFORM == "musa":
    DEVICE_TYPE = "musa" if hasattr(torch, "musa") and torch.musa.is_available() and torch.musa.device_count() > 1 else "cpu"
    PG_BACKEND = "mccl" if DEVICE_TYPE == "musa" else "gloo"

NUM_DEVICES = 4


def diff(x1: Any, x2: Any, prefix: Tuple = ()) -> Tuple[list, list, list]:
    """Recursive diff of dicts.

    Args:
        x1 (object): left dict
        x2 (object): right dict
        prefix (tuple): tracks recursive calls. Used for reporting differing keys.

    Returns:
        Tuple[list, list, list]: tuple of:
            - only_left: Prefixes present only in left dict
            - only_right: Prefixes present only in right dict
            - mismatch: values present in both dicts but not equal across dicts.
                For tensors equality of all elems is checked.
                Each element is a tuple (prefix, type of left value, type of right value).
    """
    mismatch = []
    if isinstance(x1, dict) and isinstance(x2, dict):
        only_left = [prefix + (k,) for k in x1.keys() - x2.keys()]
        only_right = [prefix + (k,) for k in x2.keys() - x1.keys()]
        for k in x2.keys() & x1.keys():
            _left, _right, _mismatch = diff(x1[k], x2[k], prefix + (k,))
            only_left.extend(_left)
            only_right.extend(_right)
            mismatch.extend(_mismatch)
    elif isinstance(x1, list) and isinstance(x2, list):
        only_left = list(range(len(x1) - 1, len(x2) - 1, -1))
        only_right = list(range(len(x1) - 1, len(x2) - 1, -1))
        for i, (v1, v2) in enumerate(zip(x1, x2)):
            _left, _right, _mismatch = diff(v1, v2, prefix + (i,))
            only_left.extend(_left)
            only_right.extend(_right)
            mismatch.extend(_mismatch)
    else:
        only_left = []
        only_right = []
        if isinstance(x1, DTensor) and isinstance(x2, DTensor):
            _is_mismatch = not torch.allclose(
                x1._local_tensor, x2._local_tensor, equal_nan=True, rtol=1e-16, atol=1e-16
            )
        elif isinstance(x1, torch.Tensor) and isinstance(x2, torch.Tensor):
            _is_mismatch = not torch.allclose(x1, x2, equal_nan=True, rtol=1e-16, atol=1e-16)
        else:
            try:
                _is_mismatch = bool(x1 != x2)
            except RuntimeError:
                _is_mismatch = True

        if _is_mismatch:
            print("rank: ", dist.get_rank(), "prefix: ", prefix, " x1:", x1, " x2:", x2)
            mismatch.append((prefix, type(x1), type(x2)))

    return only_left, only_right, mismatch


def dict_list_map_inplace(f: Callable, x: Union[dict, list]):
    """Maps dicts and lists *in-place* with a given function."""
    if isinstance(x, dict):
        for k, v in x.items():
            x[k] = dict_list_map_inplace(f, v)
    elif isinstance(x, list):
        x[:] = (dict_list_map_inplace(f, v) for v in x)
    else:
        return f(x)
    return x


def dict_list_map_outplace(f: Callable, x: Union[dict, list]):
    """Maps dicts and lists *out-of-place* with a given function."""
    if isinstance(x, dict):
        return {k: dict_list_map_outplace(f, v) for k, v in x.items()}
    elif isinstance(x, list):
        return [dict_list_map_outplace(f, v) for v in x]
    elif isinstance(x, tuple):
        return (dict_list_map_outplace(f, v) for v in x)
    elif isinstance(x, set):
        return {dict_list_map_outplace(f, v) for v in x}
    elif isinstance(x, collections.deque):
        return type(x)([dict_list_map_outplace(f, v) for v in x])
    # TODO: deal with more special data containers
    else:
        return f(x)


class Layer(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.net1 = nn.Linear(hidden_size, hidden_size * 4)
        self.relu = nn.ReLU()
        self.net2 = nn.Linear(hidden_size * 4, hidden_size)

    def forward(self, x):
        return self.net2(self.relu(self.net1(x)))


class _Model(nn.Module):
    def __init__(self, hidden_size, layer_num, with_tied_weights=False):
        super().__init__()
        self.embedding = nn.Linear(hidden_size, hidden_size)
        pe = torch.zeros(1, hidden_size, hidden_size)
        freqs = torch.zeros(1, hidden_size, hidden_size)
        self.register_buffer("pe", pe)
        self.register_buffer("freqs", freqs)
        self.layers = nn.ModuleList([Layer(hidden_size) for _ in range(layer_num)])

        if with_tied_weights:
            self.head = nn.Linear(hidden_size, hidden_size, bias=False)
            self.head.weight = self.embedding.weight

    def forward(self, x):
        x = self.embedding(x)
        for layer in self.layers:
            x = layer(x)
        return x


class Model(nn.Module):
    def __init__(self, hidden_size, layer_num, with_tied_weights=False):
        super().__init__()
        self.model = _Model(hidden_size, layer_num, with_tied_weights=with_tied_weights)

    def forward(self, x):
        return self.model(x)


TestFunc = Callable[[object], object]


def with_comms(func: TestFunc) -> TestFunc:
    assert func is not None

    @wraps(func)  # pyre-ignore[6]
    def wrapper(self, *args: Tuple[object], **kwargs: Dict[str, Any]) -> None:  # type: ignore[misc]
        # if backend not specified, and cuda/musa available, then use nccl, else gloo
        if PLATFORM == "cuda":
            if torch.cuda.is_available() and torch.cuda.device_count() >= self.world_size:
                self.device_type = "cuda"
            else:
                self.device_type = "cpu"
        elif PLATFORM == "musa":
            if hasattr(torch, "musa") and torch.musa.is_available() and torch.musa.device_count() >= self.world_size:
                self.device_type = "musa"
            else:
                self.device_type = "cpu"
        else:
            self.device_type = "cpu"

        self.init_pg()
        func(self, *args, **kwargs)  # type: ignore[misc]
        self.destroy_pg()

    return wrapper


class TestFSDPBase(MultiProcessTestCase):
    @property
    def world_size(self) -> int:
        return NUM_DEVICES

    @property
    def backend(self) -> str:
        return PG_BACKEND

    def init_pg(self) -> None:
        if "nccl" in self.backend:
            if PLATFORM == "cuda":
                if torch.cuda.device_count() < self.world_size:
                    sys.exit(TEST_SKIPS[f"multi-gpu-{self.world_size}"].exit_code)
            elif PLATFORM == "musa":
                if not hasattr(torch, "musa") or not torch.musa.is_available() or torch.musa.device_count() < self.world_size:
                    sys.exit(TEST_SKIPS[f"multi-gpu-{self.world_size}"].exit_code)

        if self.backend not in ["nccl", "gloo", "mpi", "cpu:gloo,cuda:nccl"]:
            raise RuntimeError(f"Backend {self.backend} not supported!")

        dist.init_process_group(
            backend=self.backend,
            world_size=self.world_size,
            rank=self.rank,  # pyre-ignore[16]
            init_method=f"file://{self.file_name}",  # pyre-ignore[16]
            timeout=datetime.timedelta(seconds=1200),
        )

        # set device for nccl pg for collectives
        if "nccl" in self.backend:
            if PLATFORM == "cuda":
                torch.cuda.set_device(self.rank)
            elif PLATFORM == "musa":
                torch.musa.set_device(self.rank)

    def destroy_pg(self) -> None:
        dist.barrier()
        dist.destroy_process_group()

    def setUp(self) -> None:
        super().setUp()
        self._spawn_processes()


def torchrun_setup():
    dist.init_process_group("nccl")
    if PLATFORM == "cuda":
        torch.cuda.set_device(dist.get_rank())
    elif PLATFORM == "musa":
        torch.musa.set_device(dist.get_rank())


def torchrun_cleanup():
    dist.destroy_process_group()
