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

import asyncio
import dataclasses
import ipaddress
import multiprocessing
import pickle
import socket
import threading
import time
import zlib
from typing import DefaultDict, Dict

import grpc

from ..logger import get_bytecheckpoint_logger
from . import report_service_interceptor, report_service_pb2, report_service_pb2_grpc

logger = get_bytecheckpoint_logger()


def current_timestamp():
    """
    Returns the current Unix timestamp

    Returns:
        int: Current Unix timestamp in seconds
    """
    return int(time.time())


@dataclasses.dataclass
class Item:
    """
    Data class for storing gather/broadcast state and synchronization primitives

    Args:
        cv (asyncio.Condition): Condition variable for synchronization
        contents (dict): Dictionary to store rank-wise contents
        ranks (set): Set of ranks that have reported
        created_time (int): Timestamp when item was created
    """

    cv: asyncio.Condition = dataclasses.field(default_factory=asyncio.Condition)
    contents: dict = dataclasses.field(default_factory=dict)
    ranks: set = dataclasses.field(default_factory=set)

    # Timestamp field with a default factory function
    created_time: int = dataclasses.field(default_factory=current_timestamp)

    def statedict(self):
        """
        Returns a dictionary representation of the item's state

        Returns:
            dict: Dictionary containing contents, ranks and created_time
        """
        return {
            "contents": dict(self.contents),
            "ranks": set(self.ranks),
            "created_time": self.created_time,
        }


_GRPC_OPTIONS = [
    ("grpc.max_send_message_length", 1024 * 1024 * 1024),
    ("grpc.max_receive_message_length", 1024 * 1024 * 1024),
    ("grpc.enable_http_proxy", 0),
]


class ReportServicer(report_service_pb2_grpc.ByteCheckpointReportServiceServicer):
    """
    gRPC servicer that implements gather/broadcast operations for distributed training

    Args:
        world_size (int): Total number of ranks in the distributed setup
    """

    def __init__(self, world_size: int):
        self._l = asyncio.Lock()
        self._world_size = world_size

        self._gather_dict = DefaultDict(Item)
        self._bc_dict = DefaultDict(Item)

    async def Gather(self, req: report_service_pb2.ByteCheckpointGatherRequest, ctx: grpc.aio.ServicerContext):
        """
        Handler for gather operation RPC

        Args:
            req (ByteCheckpointGatherRequest): Request containing tag, rank and content
            ctx (grpc.aio.ServicerContext): gRPC context

        Returns:
            ByteCheckpointGatherResponse: Response containing gathered contents if requested
        """
        i = await self._record(self._gather_dict, req, ctx)
        resp = report_service_pb2.ByteCheckpointGatherResponse()
        if req.with_result:
            resp.contents.extend([v for k, v in sorted(i.contents.items(), key=lambda x: x[0])])

        return resp

    async def Broadcast(self, req: report_service_pb2.ByteCheckpointBroadcastRequest, ctx: grpc.aio.ServicerContext):
        """
        Handler for broadcast operation RPC

        Args:
            req (ByteCheckpointBroadcastRequest): Request containing tag, rank and content
            ctx (grpc.aio.ServicerContext): gRPC context

        Returns:
            ByteCheckpointBroadcastResponse: Response containing broadcasted content
        """

        i = await self._record(self._bc_dict, req, ctx)
        return report_service_pb2.ByteCheckpointBroadcastResponse(content=i.contents[req.src_rank])

    async def _record(self, d: Dict[str, Item], req, ctx: grpc.aio.ServicerContext):
        async with self._l:
            i = d[req.tag]
        async with i.cv:
            if req.rank in i.ranks:
                # ctx.abort(
                #     grpc.StatusCode.INTERNAL,
                #     f"Using the same tag in multiple threads/processes. tag: {req.tag}",
                # )
                logger.warning("GRPC server got same req tag %s from rank %s", req.tag, req.rank)
            i.ranks.add(req.rank)
            if req.content:
                i.contents[req.rank] = req.content
            if len(i.ranks) == self._world_size:
                logger.info(
                    "GRPC server got req tag %s from all %s ranks after %s seconds",
                    req.tag,
                    self._world_size,
                    int(time.time() - i.created_time),
                )
                async with self._l:
                    del d[req.tag]
                i.cv.notify_all()
            await i.cv.wait_for(lambda: len(i.ranks) == self._world_size)
        return i

    async def GetStatus(self, req: report_service_pb2.ByteCheckpointGetStatusRequest, ctx: grpc.aio.ServicerContext):
        """
        Handler for getting server status

        Args:
            req (ByteCheckpointGetStatusRequest): Empty request
            ctx (grpc.aio.ServicerContext): gRPC context

        Returns:
            ByteCheckpointGetStatusResponse: Response containing serialized server state
        """

        async with self._l:
            _gather_dict = {k: v.statedict() for k, v in self._gather_dict.items()}
            _bc_dict = {k: v.statedict() for k, v in self._bc_dict.items()}
            b = pickle.dumps(
                {
                    "world_size": self._world_size,
                    "gather_dict": _gather_dict,
                    "bc_dict": _bc_dict,
                }
            )
        return report_service_pb2.ByteCheckpointGetStatusResponse(status=b)


def _is_ipv6_address(ip: str):
    """
    Check if given IP address is IPv6

    Args:
        ip (str): IP address string

    Returns:
        bool: True if IPv6, False otherwise
    """

    try:
        ip_obj = ipaddress.ip_address(ip)
    except ValueError:
        return False
    return ip_obj.version == 6


def _concat_ip_and_port(ip: str, port: int):
    """
    Concatenate IP and port with appropriate formatting

    Args:
        ip (str): IP address
        port (int): Port number

    Returns:
        str: Formatted address string
    """
    if not _is_ipv6_address(ip):
        return f"{ip}:{port}"
    else:
        return f"[{ip}]:{port}"


def get_host_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(('8.8.8.8', 80))
        ip = s.getsockname()[0]
    finally:
        s.close()
    return ip

def _get_local_ip():
    """
    Get local machine's IP address

    Returns:
        str: Local IP address
    """
    return get_host_ip()

@dataclasses.dataclass
class _AsyncObj:
    e: threading.Event = dataclasses.field(default_factory=threading.Event)
    obj: object = None


async def async_serve(servicer, async_addr: _AsyncObj):
    """
    Start gRPC server asynchronously

    Args:
        servicer (ReportServicer): Service implementation
        async_addr (_AsyncObj): Object to store server address
    """
    server: grpc.Server = grpc.aio.server(options=_GRPC_OPTIONS)
    report_service_pb2_grpc.add_ByteCheckpointReportServiceServicer_to_server(servicer, server)
    port = server.add_insecure_port("[::]:0")
    await server.start()
    async_addr.obj = _concat_ip_and_port(_get_local_ip(), port)
    async_addr.e.set()
    await server.wait_for_termination()


def serve(servicer) -> str:
    """
    Start gRPC server in a separate thread

    Args:
        servicer (ReportServicer): Service implementation

    Returns:
        str: Server address
    """
    async_addr = _AsyncObj()
    th = threading.Thread(
        target=lambda servicer=servicer, async_addr=async_addr: asyncio.run(async_serve(servicer, async_addr)),
        daemon=True,
    )
    th.start()
    async_addr.e.wait()
    return async_addr.obj


def _serve_in_loop(world_size, conn):
    servicer = ReportServicer(world_size)
    addr = serve(servicer)
    conn.send(addr)
    conn.close()
    while True:
        time.sleep(1)


def start_server_in_new_process(world_size: int):
    """
    Start gRPC server in a new process

    Args:
        world_size (int): Total number of ranks

    Returns:
        str: Server address
    """
    parent_conn, child_conn = multiprocessing.Pipe()
    p = multiprocessing.get_context("spawn").Process(target=_serve_in_loop, args=(world_size, child_conn), daemon=True)
    p.start()
    return parent_conn.recv()


def get_stub(addr: str):
    """
    Create gRPC stub with retry interceptor

    Args:
        addr (str): Server address

    Returns:
        ByteCheckpointReportServiceStub: gRPC stub
    """
    channel = grpc.insecure_channel(addr, options=_GRPC_OPTIONS)
    interceptors = (
        report_service_interceptor.RetryOnRpcErrorClientInterceptor(
            max_attempts=4,
            sleeping_policy=report_service_interceptor.ExponentialBackoff(
                init_backoff_ms=1000, max_backoff_ms=32000, multiplier=2
            ),
            status_for_retry=(grpc.StatusCode.UNAVAILABLE,),
        ),
    )
    interceptor_channel = grpc.intercept_channel(channel, *interceptors)
    return report_service_pb2_grpc.ByteCheckpointReportServiceStub(interceptor_channel)


def _get_tag():
    return "_default_tag"


def gather(
    stub: report_service_pb2_grpc.ByteCheckpointReportServiceStub,
    gather_rank: int,
    rank: int,
    obj,
    tag: str = None,
    timeout=None,
):
    """
    Perform gather operation across ranks

    Args:
        stub (ByteCheckpointReportServiceStub): gRPC stub
        gather_rank (int): Rank that collects results
        rank (int): Current rank
        obj: Object to gather
        tag (str, optional): Operation tag
        timeout: RPC timeout

    Returns:
        list: Gathered objects if current rank is gather_rank
    """
    tag = tag or _get_tag()
    req = report_service_pb2.ByteCheckpointGatherRequest(
        tag=tag, rank=rank, content=pickle.dumps(obj), with_result=(gather_rank == rank)
    )
    resp = stub.Gather(req, timeout=timeout)
    if gather_rank != rank:
        return
    return [pickle.loads(content) for content in resp.contents]


def broadcast(
    stub: report_service_pb2_grpc.ByteCheckpointReportServiceStub,
    src_rank: int,
    rank: int,
    obj=None,
    tag: str = None,
    timeout=None,
):
    """
    Broadcast object from source rank to all ranks

    Args:
        stub (ByteCheckpointReportServiceStub): gRPC stub
        src_rank (int): Source rank for broadcast
        rank (int): Current rank
        obj: Object to broadcast
        tag (str, optional): Operation tag
        timeout: RPC timeout

    Returns:
        object: Broadcasted object
    """
    tag = tag or _get_tag()
    content = b"" if rank != src_rank else pickle.dumps(obj)
    # Since we will transfer this to all machines, compression here is important.
    c_content = zlib.compress(content)
    resp = stub.Broadcast(
        report_service_pb2.ByteCheckpointBroadcastRequest(tag=tag, rank=rank, content=c_content, src_rank=src_rank),
        timeout=timeout,
    )
    content = zlib.decompress(resp.content)
    return pickle.loads(content)


def barrier(
    stub: report_service_pb2_grpc.ByteCheckpointReportServiceStub,
    rank: int,
    tag: str = None,
    timeout=None,
):
    """
    Synchronization barrier across ranks

    Args:
        stub (ByteCheckpointReportServiceStub): gRPC stub
        rank (int): Current rank
        tag (str, optional): Operation tag
        timeout: RPC timeout
    """
    gather(stub, 0, rank, tag=tag, obj=None, timeout=timeout)


def get_server_status(stub: report_service_pb2_grpc.ByteCheckpointReportServiceStub, timeout=None):
    """
    Get current server status

    Args:
        stub (ByteCheckpointReportServiceStub): gRPC stub
        timeout: RPC timeout

    Returns:
        dict: Server status dictionary
    """
    resp = stub.GetStatus(report_service_pb2.ByteCheckpointGetStatusRequest(), timeout=timeout)
    return pickle.loads(resp.status)
