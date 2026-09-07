################################################################################
# Copyright (c) Meta Platforms, Inc. and affiliates
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
################################################################################
# Modification Copyright 2025 ByteDance Ltd. and/or its affiliates.
################################################################################

import dataclasses
import operator
from functools import reduce
from typing import Any, Callable, Collection, Dict, List, Mapping, Optional, Sequence, Tuple

import torch
import torch.distributed as dist
from torch.distributed._tensor import DTensor
from torch.distributed.checkpoint._nested_dict import (
    FLATTEN_MAPPING,
)
from torch.distributed.checkpoint._sharded_tensor_utils import (
    _flatten_sharded_tensors,
)
from torch.distributed.checkpoint._traverse import (
    OBJ_PATH,
    STATE_DICT_ITEM,
    _keep_visiting_tensors,
)
from torch.distributed.checkpoint.metadata import (
    BytesStorageMetadata,
)
from torch.distributed.checkpoint.planner import (
    LoadPlan,
    ReadItem,
    SavePlan,
    TensorProperties,
    WriteItemType,
)
from torch.distributed.checkpoint.planner_helpers import (
    _create_read_items,
    _create_write_items,
)
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp._unshard_param_utils import FLAT_PARAM

from bytecheckpoint.checkpointer.meta_type import (
    FSDP_FLAT_PARAM_META,
    FSDP_OPTIMIZER_REQUIRES_GRAD_NUMEL,
    STATE_DICT_STR,
    STATE_DICT_TYPE,
    Metadata,
)
from bytecheckpoint.planner.common import GLOBAL_PLAN_CACHE, custom_dedup_tensors
from bytecheckpoint.planner.default_planner import (
    DefaultLoadPlanner,
    DefaultSavePlanner,
    _check_box_bounds,
    _check_box_overlap,
    create_plan_cache_key,
)
from bytecheckpoint.utilities.logger import get_bytecheckpoint_logger

logger = get_bytecheckpoint_logger()
__all__ = [
    "FSDPSavePlanner",
    "FSDPLoadPlanner",
    "create_default_local_load_plan",
    "create_default_local_save_plan",
]


class FSDPLoadPlanner(DefaultLoadPlanner):
    """
    A planner class for loading fsdp checkpoint using PyTorch DCP
    """

    def __init__(self, strict: bool):
        super().__init__(strict=strict)

    def create_local_plan(self) -> LoadPlan:
        return create_default_local_load_plan(self.state_dict, self.metadata, self.strict)

    def transform_tensor(self, read_item: ReadItem, tensor: torch.Tensor):
        """
        This is an extension from the planner interface to make it easy to extend the default planner.
        """
        return narrow_tensor_by_index_no_grad(tensor, read_item.dest_offsets, read_item.lengths)

    def set_up_planner(
        self,
        state_dict: STATE_DICT_TYPE,
        metadata: Metadata,
        is_coordinator: bool,
    ) -> None:
        self.original_state_dict = state_dict

        if self.flatten_sharded_tensors:
            state_dict = _flatten_sharded_tensors(state_dict)

        if self.flatten_state_dict:
            state_dict, self.mappings = flatten_state_dict(state_dict)

        self.state_dict = state_dict
        self.metadata = metadata
        self.is_coordinator = is_coordinator


class FSDPSavePlanner(DefaultSavePlanner):
    """
    A planner class for saving fsdp checkpoint using PyTorch DCP
    """

    process_group: List[int] = []
    inter_node_pg: List[int] = []
    sharding_strategy: str = ""
    fsdp_flat_param_meta: dict = {}

    def __init__(self):
        super().__init__()

    # this function is subject to change.
    def update(self, model: FSDP):
        """
        Update the process group, inter-node process group, and sharding strategy based on the provided FSDP model.

        Args:
            model (FSDP): The FullyShardedDataParallel model from which to extract process group and sharding strategy information.
        """
        self.process_group = dist.get_process_group_ranks(model.process_group)
        self.inter_node_pg = (
            dist.get_process_group_ranks(model._inter_node_pg) if hasattr(model, "_inter_node_pg") else []
        )
        self.sharding_strategy = model.sharding_strategy

    def verify_all_shards_metadata(self, all_plans: List[SavePlan]):
        """
        Verify the metadata of all shards in the given list of save plans.

        When model zero communication is enabled, the shards metadata verification
        during ShardedTensor initialization is skipped. Therefore, this method
        verifies all shards metadata in the bytecheckpoint global plan.

        Args:
            all_plans (List[SavePlan]): A list of save plans containing write items with shard metadata.

        Raises:
            ValueError: If there is a mismatch in tensor properties, global sizes, or shards size sums.
        """
        # When model zero comm is on
        # We skip shards metadata verification in ShardedTensor initialization
        # So we need to verify all shards metadata in bytecheckpoint global plan
        fqn_to_property: Dict[str, TensorProperties] = {}
        fqn_to_global_size: Dict[str, torch.Size] = {}
        fqn_to_sizes: Dict[str, List[torch.Size]] = {}
        if self.fsdp_flat_param_meta:
            for plan in all_plans:
                for item in plan.items:
                    # for example:
                    # item.tensor_data =
                    # TensorWriteData(chunk=ChunkStorageMetadata(offsets=torch.Size([588]), sizes=torch.Size([147])),
                    # properties=TensorProperties(dtype=torch.float32, layout=torch.strided, requires_grad=True,
                    # memory_format=torch.contiguous_format, pin_memory=False), size=torch.Size([1176]))
                    if item.type == WriteItemType.SHARD:
                        # print(item.tensor_data)
                        if item.index.fqn in fqn_to_property:
                            _raise_if_mismatch(
                                fqn_to_property[item.index.fqn], item.tensor_data.properties, "tensor property"
                            )
                        else:
                            fqn_to_property[item.index.fqn] = item.tensor_data.properties
                        if item.index.fqn in fqn_to_global_size:
                            _raise_if_mismatch(fqn_to_global_size[item.index.fqn], item.tensor_data.size, "global size")
                        else:
                            fqn_to_global_size[item.index.fqn] = item.tensor_data.size
                        fqn_to_sizes.setdefault(item.index.fqn, []).append(item.tensor_data.chunk.sizes)
        for fqn, sizes in fqn_to_sizes.items():
            _raise_if_mismatch(
                fqn_to_global_size[fqn].numel(), sum([size.numel() for size in sizes]), "shards size sum"
            )

    def custom_validate_global_plan(self, global_plan: List[SavePlan], metadata: Metadata) -> bool:
        """
        Validate the global plan and metadata to ensure that all chunks in the state dictionary metadata are valid.

        This function checks the following conditions for each tensor in the state dictionary metadata:
        1. Chunks are within the bounds of the tensor.
        2. Chunks do not overlap with each other.
        3. The combined volume of all chunks equals the volume of the tensor.

        Args:
            global_plan (List[SavePlan]): A list of save plans for the global checkpoint.
            metadata (Metadata): Metadata containing information about the state dictionary.

        Returns:
            bool: True if all checks pass, False otherwise.
        """
        all_good = True
        for key, value in metadata.state_dict_metadata.items():
            if isinstance(value, BytesStorageMetadata):
                continue
            if len(value.size) == 0:
                continue
            chunks_volume = 0
            for chunk_idx, chunk0 in enumerate(value.chunks):
                # Compute the volume
                if not _check_box_bounds(value.size, chunk0):
                    logger.warning(
                        """
                            key:%s has out of bounds chunk:
                            tensor-size:%s chunk: %s
                        """,
                        key,
                        value.size,
                        chunk0,
                    )
                    all_good = False
                chunks_volume += reduce(operator.mul, chunk0.sizes, 1)

                # Check for overlap
                for chunk1 in value.chunks[chunk_idx + 1 :]:
                    if _check_box_overlap(chunk0, chunk1):
                        logger.warning("key:%s has overlapping chunks: %s %s", key, chunk0, chunk1)
                        all_good = False

            # Check whether combined chunk cover the whole tensor
            # optimizer zero comm is not to save padding now. So chunks_volume is not equal to global numel
            tensor_volume = reduce(operator.mul, value.size, 1)
            if FLAT_PARAM in key:
                optimizer_for_validation_key = (
                    f"{key[: key.find(FLAT_PARAM) + len(FLAT_PARAM)]}.{FSDP_OPTIMIZER_REQUIRES_GRAD_NUMEL}"
                )
                if optimizer_for_validation_key in self.state_dict:
                    tensor_volume = reduce(operator.mul, [self.state_dict[optimizer_for_validation_key].item()])

            if chunks_volume != tensor_volume:
                logger.warning(
                    """
                        key:%s invalid fill tensor-volume:
                        %s chunks-volume: %s
                    """,
                    key,
                    tensor_volume,
                    chunks_volume,
                )
                all_good = False

        return all_good

    def set_up_planner(self, state_dict: STATE_DICT_TYPE, is_coordinator: bool) -> None:
        self.fsdp_flat_param_meta = state_dict.pop(FSDP_FLAT_PARAM_META, {})
        if self.flatten_state_dict:
            state_dict, self.mappings = flatten_state_dict(state_dict)
        if self.flatten_sharded_tensors:
            state_dict = _flatten_sharded_tensors(state_dict)
        self.state_dict = state_dict
        self.is_coordinator = is_coordinator

    def lookup_plan_meta(self, extra_info: Optional[str] = "") -> Optional[Tuple[SavePlan, Metadata]]:
        if not hasattr(self, STATE_DICT_STR):
            return None
        else:
            plan_key = create_plan_cache_key(self.state_dict, extra_info)
            return GLOBAL_PLAN_CACHE.get(plan_key)

    def cache_plan_meta(self, new_plan: SavePlan, new_metadata: Metadata, extra_info: Optional[str] = "") -> None:
        # Need to take sharding_strategy into account?
        plan_key = create_plan_cache_key(self.state_dict, extra_info)
        GLOBAL_PLAN_CACHE.put(plan_key, new_plan, new_metadata)

    def clear_cache(self) -> None:
        GLOBAL_PLAN_CACHE.clear()

    def create_local_plan(self) -> SavePlan:
        plan = create_default_local_save_plan(self.state_dict, self.is_coordinator)
        if self.flatten_state_dict:
            plan = dataclasses.replace(plan, planner_data=self.mappings)
        self.plan = plan
        return self.plan

    def create_global_plan(self, all_plans: List[SavePlan]) -> Tuple[List[SavePlan], Metadata]:
        """
        Create a global save plan from a list of local save plans.

        This method first applies a custom deduplication function to the local save plans if the `dedup_replicated_tensors` flag is set.
        It then verifies the metadata of all shards in the local save plans if the `fsdp_flat_param_meta` dictionary is not empty.
        After that, it disables the default DCP's deduplication of replicated tensors and calls the superclass's `create_global_plan` method.
        Finally, it validates the global plan using a custom validation function and updates the global metadata with the `fsdp_flat_param_meta` dictionary.

        Args:
            all_plans (List[SavePlan]): A list of local save plans to be combined into a global plan.

        Returns:
            Tuple[List[SavePlan], Metadata]: A tuple containing the list of local save plans and the global metadata.

        Raises:
            ValueError: If the custom validation of the global plan fails.
        """
        # Use customized deduplicate function for load balance.
        if self.dedup_replicated_tensors:
            all_plans = custom_dedup_tensors(all_plans)
        # Verify shards metadata in all plans.
        if self.fsdp_flat_param_meta:
            self.verify_all_shards_metadata(all_plans)
        # Disable DCP's dedup replicated tensors function.
        self.dedup_replicated_tensors = False
        all_local_plans, global_metatadata = super().create_global_plan(all_plans, do_validate_global_plan=False)
        # Run FSDP custom validate global plan.
        if not self.custom_validate_global_plan(all_local_plans, global_metatadata):
            raise ValueError("Failed to validate global plan")
        global_metatadata.user_defined_dict.update(self.fsdp_flat_param_meta)
        return all_local_plans, global_metatadata


# The api is referenced from https://github.com/pytorch/pytorch/blob/release/2.1/torch/distributed/checkpoint/_traverse.py#L37-L77
# This is because this PR(https://github.com/pytorch/pytorch/pull/125335) modifies the logic for terminating recursion: mapping types
# are always recursive when encountered, causing the `keep_traversing` function to fail. This makes us can't avoid flatten param_groups
# to keep compatibility with existing checkpoints. so we copy the torch2.1 api here.
def traverse_state_dict(
    state_dict: STATE_DICT_TYPE,
    visitor: Callable[[OBJ_PATH, STATE_DICT_ITEM], None],
    keep_traversing: Callable[[STATE_DICT_ITEM], bool],
) -> None:
    """
    Invoke ``visitor`` for each value recursively in ``state_dict``.
    Traversal is short-circuited when if finds a collection for which ``keep_visiting_tensors`` evaluates
    to false for all elements.
    By default, all collections with at least one ``torch.Tensor`` element are traversed.
    Visitor takes a path argument that is a tuple of the keys used to reach it.

    Args:
        state_dict (STATE_DICT_TYPE): The state dictionary to traverse.
        visitor (Callable[[OBJ_PATH, STATE_DICT_ITEM], None]): A function to be called for each value in the state dictionary.
            It takes two arguments: the path to the value and the value itself.
        keep_traversing (Callable[[STATE_DICT_ITEM], bool]): A function that determines whether to continue traversing a collection.
            If it returns False for all elements in a collection, the traversal of that collection is short-circuited.
    """

    # a value is terminal if it has no other containers values inside it
    def _is_terminal(value: STATE_DICT_ITEM, path: OBJ_PATH = None) -> bool:
        values: Collection[STATE_DICT_ITEM]
        if path is not None and isinstance(path[0], str) and "param_groups" in path[0]:
            return True
        if isinstance(value, Mapping):
            values = value.values()
        elif isinstance(value, list):
            values = value
        else:
            return True

        for entry in values:
            if isinstance(entry, (Mapping, list)) and not _is_terminal(entry):
                return False
            if keep_traversing is not None and keep_traversing(entry):
                return False
        return True

    def _traverse_obj(path: OBJ_PATH, value: STATE_DICT_ITEM) -> None:
        if _is_terminal(value, path):
            visitor(path, value)
        elif isinstance(value, Mapping):
            for k, v in value.items():
                _traverse_obj(path + (str(k),), v)
        elif isinstance(value, list):
            for i, v in enumerate(value):
                _traverse_obj(path + (i,), v)

    for key, value in state_dict.items():
        _traverse_obj((str(key),), value)


def flatten_state_dict(
    state_dict: STATE_DICT_TYPE,
) -> Tuple[STATE_DICT_TYPE, FLATTEN_MAPPING]:
    """
    Flatten a nested state dictionary consisting of dictionaries and lists into a top-level dictionary.
    This function is useful for simplifying the structure of a complex state dictionary.

    Args:
        state_dict (STATE_DICT_TYPE): The nested state dictionary to be flattened.

    Returns:
        Tuple[STATE_DICT_TYPE, FLATTEN_MAPPING]: A tuple containing the flattened state dictionary
        and a mapping from the original keys to the new keys in the flattened dictionary.

    Note:
        - The new keys in the flattened dictionary are derived from the object paths, joined by dots.
          For example, a nested dictionary {'a': {'b': ...}} will result in the key 'a.b' in the flattened dictionary.
        - Use `unflatten_state_dict` to revert the flattening process.
    """
    flattened: STATE_DICT_TYPE = {}
    mappings: FLATTEN_MAPPING = {}

    def flat_copy(path: OBJ_PATH, value: STATE_DICT_ITEM) -> None:
        new_fqn = ".".join(map(str, path))
        if new_fqn in flattened:
            raise ValueError(f"duplicated flatten key {new_fqn}")
        flattened[new_fqn] = value
        mappings[new_fqn] = path

    traverse_state_dict(state_dict, flat_copy, keep_traversing=_keep_visiting_tensors)
    return flattened, mappings


def _raise_if_mismatch(expected, actual, prop_name):
    if expected != actual:
        raise ValueError(
            f"Shard {prop_name} property does not match from different ranks! "
            f"Found {prop_name}={expected}, and {prop_name}={actual}."
        )


def narrow_tensor_by_index_no_grad(tensor: torch.Tensor, offsets: Sequence[int], sizes: Sequence[int]) -> torch.Tensor:
    """
    Narrow the tensor according to ``offsets`` and ``sizes``.

    Args:
        tensor (torch.Tensor): The input tensor to be narrowed.
        offsets (Sequence[int]): A sequence of integers representing the starting indices for each dimension.
        sizes (Sequence[int]): A sequence of integers representing the sizes for each dimension after narrowing.

    Returns:
        torch.Tensor: The narrowed tensor.
    """
    with torch.no_grad():
        narrowed_tensor = tensor
        for idx, (offset, size) in enumerate(zip(offsets, sizes)):
            if size < tensor.size(idx):
                # Reshape to get shard for this rank and we don't want autograd
                # recording here for the narrow op and 'local_shard' should be a
                # leaf variable in the autograd graph.
                narrowed_tensor = narrowed_tensor.narrow(idx, offset, size)
        return narrowed_tensor


def create_default_local_load_plan(state_dict: Dict[str, Any], metadata: Metadata, strict: bool = True) -> LoadPlan:
    """
    Create a default local load plan for loading a checkpoint.

    Args:
        state_dict (Dict[str, Any]): The state dictionary containing the objects to be loaded.
        metadata (Metadata): The metadata associated with the checkpoint.
        strict (bool, optional): If True, raise an exception if a key is missing in the metadata.
                                 If False, log a warning and continue. Defaults to True.

    Returns:
        LoadPlan: A load plan containing the read requests for the objects in the state dictionary.
    """
    requests = []
    for fqn, obj in state_dict.items():
        try:
            md = metadata.state_dict_metadata[fqn]
        except Exception as e:
            if strict:
                raise e
            else:
                logger.warning("Ignore missing key %s when loading checkpoint as strict=%s", fqn, strict)
                continue
        logger.debug("Creating load plan for %s successfully", fqn)
        # Since DTensor supports submesh, adding extra check to ensure _create_read_items()
        # gets called only when the current rank is part of the mesh for the corresponding DTensor.
        if isinstance(obj, DTensor):
            if obj.device_mesh.get_coordinate() is not None:
                requests += _create_read_items(fqn, md, obj)
            logger.debug("Load plan for %s ", obj)
        else:
            requests += _create_read_items(fqn, md, obj)
            logger.debug("Load plan for other %s ", obj)

    logger.debug("Finish load plan")
    return LoadPlan(requests)


def create_default_local_save_plan(state_dict: Dict[str, Any], is_coordinator: bool) -> SavePlan:
    """
    Create a default local save plan for saving a checkpoint.

    This function iterates through the state dictionary and creates write requests for each object.
    For Distributed Tensors (DTensors), it checks if the current rank is part of the mesh before creating write requests.
    For regular tensors or if the current process is the coordinator, it creates write requests directly.

    Args:
        state_dict (Dict[str, Any]): The state dictionary containing the objects to be saved.
        is_coordinator (bool): A boolean indicating whether the current process is the coordinator.

    Returns:
        SavePlan: A save plan containing the write requests for the objects in the state dictionary.
    """
    requests = []
    # Key: fqn
    # Value: dictionary (Key is the process rank, value is tensor to receive)

    for fqn, obj in state_dict.items():
        # Since DTensor supports submesh, adding extra check to ensure _create_write_items()
        # gets called only when the current rank is part of the mesh for the corresponding DTensor.
        if isinstance(obj, DTensor):
            if obj.device_mesh.get_coordinate() is not None:
                requests += _create_write_items(fqn, obj)
        elif isinstance(obj, (torch.Tensor)) or is_coordinator:
            requests += _create_write_items(fqn, obj)

    return SavePlan(requests)
