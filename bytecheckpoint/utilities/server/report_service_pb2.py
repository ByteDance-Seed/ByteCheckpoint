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
# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: bytecheckpoint/utilities/server/report_service.proto
"""Generated protocol buffer code."""

from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import symbol_database as _symbol_database
from google.protobuf.internal import builder as _builder

# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()


DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(
    b'\n4bytecheckpoint/utilities/server/report_service.proto"^\n\x1b\x42yteCheckpointGatherRequest\x12\x0b\n\x03tag\x18\x01 \x01(\t\x12\x0c\n\x04rank\x18\x02 \x01(\x05\x12\x0f\n\x07\x63ontent\x18\x03 \x01(\x0c\x12\x13\n\x0bwith_result\x18\x04 \x01(\x08"0\n\x1c\x42yteCheckpointGatherResponse\x12\x10\n\x08\x63ontents\x18\x01 \x03(\x0c"^\n\x1e\x42yteCheckpointBroadcastRequest\x12\x0b\n\x03tag\x18\x01 \x01(\t\x12\x0c\n\x04rank\x18\x02 \x01(\x05\x12\x0f\n\x07\x63ontent\x18\x03 \x01(\x0c\x12\x10\n\x08src_rank\x18\x04 \x01(\x05"2\n\x1f\x42yteCheckpointBroadcastResponse\x12\x0f\n\x07\x63ontent\x18\x01 \x01(\x0c" \n\x1e\x42yteCheckpointGetStatusRequest"1\n\x1f\x42yteCheckpointGetStatusResponse\x12\x0e\n\x06status\x18\x01 \x01(\x0c\x32\x8a\x02\n\x1b\x42yteCheckpointReportService\x12G\n\x06Gather\x12\x1c.ByteCheckpointGatherRequest\x1a\x1d.ByteCheckpointGatherResponse"\x00\x12P\n\tBroadcast\x12\x1f.ByteCheckpointBroadcastRequest\x1a .ByteCheckpointBroadcastResponse"\x00\x12P\n\tGetStatus\x12\x1f.ByteCheckpointGetStatusRequest\x1a .ByteCheckpointGetStatusResponse"\x00\x62\x06proto3'
)

_globals = globals()
_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, _globals)
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, "bytecheckpoint.utilities.server.report_service_pb2", _globals)
if _descriptor._USE_C_DESCRIPTORS is False:
    DESCRIPTOR._options = None
    _globals["_BYTECHECKPOINTGATHERREQUEST"]._serialized_start = 56
    _globals["_BYTECHECKPOINTGATHERREQUEST"]._serialized_end = 150
    _globals["_BYTECHECKPOINTGATHERRESPONSE"]._serialized_start = 152
    _globals["_BYTECHECKPOINTGATHERRESPONSE"]._serialized_end = 200
    _globals["_BYTECHECKPOINTBROADCASTREQUEST"]._serialized_start = 202
    _globals["_BYTECHECKPOINTBROADCASTREQUEST"]._serialized_end = 296
    _globals["_BYTECHECKPOINTBROADCASTRESPONSE"]._serialized_start = 298
    _globals["_BYTECHECKPOINTBROADCASTRESPONSE"]._serialized_end = 348
    _globals["_BYTECHECKPOINTGETSTATUSREQUEST"]._serialized_start = 350
    _globals["_BYTECHECKPOINTGETSTATUSREQUEST"]._serialized_end = 382
    _globals["_BYTECHECKPOINTGETSTATUSRESPONSE"]._serialized_start = 384
    _globals["_BYTECHECKPOINTGETSTATUSRESPONSE"]._serialized_end = 433
    _globals["_BYTECHECKPOINTREPORTSERVICE"]._serialized_start = 436
    _globals["_BYTECHECKPOINTREPORTSERVICE"]._serialized_end = 702
# @@protoc_insertion_point(module_scope)
