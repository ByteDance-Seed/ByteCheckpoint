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

import io
import os
import pickle
import threading
from typing import DefaultDict

import torch

from bytecheckpoint.utilities.logger import get_bytecheckpoint_logger

logger = get_bytecheckpoint_logger()

PLATFORM = os.environ.get("PLATFORM", "cuda").lower()
if PLATFORM not in {"cuda", "musa"}:
    logger.warning("Unsupported PLATFORM=%s, fallback to cuda", PLATFORM)
    PLATFORM = "cuda"

if hasattr(torch.storage, "TypedStorage"):
    TypedStorage = torch.storage.TypedStorage
elif hasattr(torch.storage, "_TypedStorage"):
    TypedStorage = torch.storage._TypedStorage

# TypedStorage changes in pytorch 2.
if torch.__version__ >= "2":

    def untyped_storage(o):
        return o.untyped_storage()

    def location_caster(o):
        return o
elif torch.__version__ >= "1.11":

    def untyped_storage(o):
        return o.storage()._storage

    def location_caster(o):
        return o._storage if isinstance(o, TypedStorage) else o


if PLATFORM == "cuda":

    def _load_platform_runtime():
        """
        Load CUDA runtime library if available.
        """
        try:
            getter = getattr(torch.cuda, "cudart", None)
            if getter is None:
                raise AttributeError("torch.cuda.cudart missing")
            return getter()
        except Exception as e:
            logger.warning("Failed loading CUDA runtime: %s", e)
            return None

    _host_register_fn = "cudaHostRegister"
    _host_unregister_fn = "cudaHostUnregister"

elif PLATFORM == "musa":

    def _load_platform_runtime():
        """
        Load MUSA runtime library if available.
        """
        try:
            musa_module = getattr(torch, "musa", None)
            getter = getattr(musa_module, "musart", None) if musa_module else None
            if getter is None:
                raise AttributeError("torch.musa.musart missing")
            return getter()
        except Exception as e:
            logger.warning("Failed loading MUSA runtime: %s", e)
            return None

    _host_register_fn = "musaHostRegister"
    _host_unregister_fn = "musaHostUnregister"


lib = _load_platform_runtime()


class PinnedMemoryPool:
    def __init__(self):
        self._l = threading.Lock()
        self._m = DefaultDict(set)

    def _allocate(self, nbytes: int):
        """
        Allocate a tensor storage with the specified number of bytes.

        If there is a suitable storage in the memory pool, it will be popped and returned.
        Otherwise, a new CPU tensor will be created, its memory will be pinned, and its storage will be added to the pool.

        Args:
            nbytes (int): The number of bytes required for the storage.

        Returns:
            torch.storage.TypedStorage: A tensor storage with the specified number of bytes.
        """
        with self._l:
            # We don't really need storage to have the exact size. So in theory we can find a
            # bigger storage that may suit here. But so far we keep everything simple here.
            s = self._m[nbytes]
            if not s:
                cpu_tensor = torch.empty([nbytes], dtype=torch.uint8)
                cpu_tensor = cpu_tensor.share_memory_()
                # Make memory of t pinned
                if lib is not None and nbytes != 0:
                    register = getattr(lib, _host_register_fn, None)
                    assert register is not None, f"{PLATFORM} host register function missing"
                    err = register(cpu_tensor.data_ptr(), cpu_tensor.numel() * cpu_tensor.element_size(), 0)
                    assert err == 0, f"{PLATFORM} error in allocating pinned memory: {err}"
                tensor_storage = untyped_storage(cpu_tensor)
                s.add(tensor_storage)
            return s.pop()

    def _deallocate(self, tensor_storage):
        """
        Deallocate the given tensor storage and add it back to the memory pool.

        WARNING: Call _deallocate when the reference to CPU tensor goes to zero
        so the memory pool will reuse the memory if possible.
        Otherwise, the memory pool will _allocate memory on the used memory range,
        leading to cuda error 712 cudaErrorHostMemoryAlreadyRegistered.

        Args:
            tensor_storage (torch.storage.TypedStorage): The tensor storage to be deallocated.
        """
        with self._l:
            self._m[tensor_storage.nbytes()].add(tensor_storage)

    """
    Interfaces.
    """

    def copy_gpu_tensor_to_cpu_pinned_mem_pool(self, tensor: torch.Tensor, non_blocking=False) -> torch.Tensor:
        """
        Copy a tensor on GPU to pinned memory pool (host CPU memory).
        The input tensor will not be modified

        Args:
            tensor (torch.Tensor): A tensor on a CUDA device.
            non_blocking (bool, optional): Whether the copy operation should be non-blocking. Defaults to False.

        Returns:
            torch.Tensor: A tensor on CPU, whose data is the same as the input tensor.
        """
        m = {}
        _old_warning = getattr(torch.storage, "_warn_typed_storage_removal", None)
        # Suppress the warning about typed storage removal
        torch.storage._warn_typed_storage_removal = lambda *args, **kwags: None

        def persistent_id(o):
            """
            Generate a persistent identifier for a storage object.

            If the storage object is already in the mapping `m`, return its identifier.
            Otherwise, allocate a new storage on the CPU, copy the data from the GPU storage,
            and add it to the mapping.

            Args:
                o: A storage object.

            Returns:
                The identifier of the storage object.
            """
            if torch.is_storage(o) or isinstance(o, TypedStorage):
                storage = o
                if storage._cdata in m:
                    return storage._cdata
                if storage.device.type != "cpu":
                    copied = self._allocate(storage.nbytes())
                    copied.copy_(storage, non_blocking=non_blocking)
                    if isinstance(storage, TypedStorage):
                        copied = storage._new_wrapped_storage(copied)
                    elif isinstance(storage, torch.UntypedStorage):
                        copied._untyped_storage = copied
                else:
                    copied = storage.clone()
                    # fix fp8 dtype untyped_storage error
                    if isinstance(storage, torch.UntypedStorage):
                        copied._untyped_storage = copied
                m[storage._cdata] = copied
                return storage._cdata
            else:
                raise AssertionError(
                    "Passing a non torch.UntypedStorage/TypedStorage type into Pinned memory D2H function."
                )

        b = io.BytesIO()
        p = pickle.Pickler(b)
        p.persistent_id = persistent_id
        p.dump(tensor.untyped_storage())
        b.seek(0)
        up = pickle.Unpickler(b)
        up.persistent_load = lambda i: m[i]
        cpu_memory_storage = up.load()
        # Restore the original warning function
        torch.storage._warn_typed_storage_removal = _old_warning

        return torch.tensor([], dtype=tensor.dtype, device="cpu").set_(cpu_memory_storage, 0, tensor.shape, None)

    def deallocate_cpu_tensor_in_pinned_mem_pool(self, tensor: torch.Tensor):
        """
        Deallocate a CPU tensor from the pinned memory pool.

        This method checks if the given tensor is pinned and, if so, deallocates its storage by
        adding it back to the memory pool. This allows the memory pool to reuse the memory.

        Args:
            tensor (torch.Tensor): A CPU tensor that may be pinned.

        Note:
            The tensor must be pinned for deallocation to occur. If the tensor is not pinned,
            this method will not perform any deallocation.
        """
        if tensor.is_pinned():
            self._deallocate(tensor.untyped_storage())

    def release_memory_pool(self):
        """
        Release all the memory in the pinned memory pool.

        This method iterates through all the stored tensor storages in the memory pool.
        For each storage, it unregisters the pinned CPU memory if CUDA is available and the size is non-zero.
        After unregistering, it clears the storage sets and the memory pool dictionary.

        Note:
            This method should be called when you want to free up all the pinned memory managed by the pool.
        """
        with self._l:
            for nbytes, storages in list(self._m.items()):
                for tensor_storage in storages:
                    # Unregister the CPU memory.
                    if lib is not None and nbytes != 0:
                        unregister = getattr(lib, _host_unregister_fn, None)
                        assert unregister is not None, f"{PLATFORM} host unregister function missing"
                        err = unregister(tensor_storage.data_ptr())
                        assert err == 0, f"{PLATFORM} error in releasing pinned memory: {err}"
                storages.clear()
            self._m.clear()
