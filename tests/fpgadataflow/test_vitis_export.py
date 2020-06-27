# Copyright (c) 2020, Xilinx
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of FINN nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

# as of Feb'20 there is a bug that segfaults ONNX shape inference if we
# import pytorch before onnx, so we make sure to import onnx first
import onnx  # NOQA

import pytest
from onnx import TensorProto, helper
from finn.core.datatype import DataType
from finn.core.modelwrapper import ModelWrapper
from finn.custom_op.registry import getCustomOp

from finn.transformation.fpgadataflow.vitis_build import VitisBuild
from finn.util.basic import alveo_part_map, alveo_default_platform


def make_model(idt, ch, k):
    inp = helper.make_tensor_value_info("inp", TensorProto.FLOAT, [1, ch])
    outp = helper.make_tensor_value_info("outp", TensorProto.FLOAT, [1, k])

    labelselect_node = helper.make_node(
        "LabelSelect_Batch",
        [inp.name],
        [outp.name],
        domain="finn",
        backend="fpgadataflow",
        Labels=ch,
        PE=1,
        K=k,
        inputDataType=idt.name,
    )

    graph = helper.make_graph(
        nodes=[labelselect_node], name="graph", inputs=[inp], outputs=[outp],
    )

    model = helper.make_model(graph, producer_name="model")
    model = ModelWrapper(model)

    model.set_tensor_datatype(inp.name, idt)
    model.set_tensor_datatype(outp.name, DataType.UINT32)

    model.graph.value_info.append(inp)
    model.graph.value_info.append(outp)

    return model


# input datatype -- checked by assertion in HLSCustomOp
@pytest.mark.parametrize("idt", [DataType.UINT16])
# input channels
@pytest.mark.parametrize("ch", [1024])
# number of top labels to select
@pytest.mark.parametrize("k", [5])
# board
@pytest.mark.parametrize("board", ["U250"])
# clock period
@pytest.mark.parametrize("period_ns", [5])
@pytest.mark.slow
@pytest.mark.vivado
def test_vitis_export(idt, ch, k, board, period_ns):
    platform = alveo_default_platform[board]
    fpga_part = alveo_part_map[board]
    model = make_model(idt, ch, k)
    getCustomOp(model.graph.node[0]).set_nodeattr("inFIFODepth", 64)
    model = model.transform(VitisBuild(fpga_part, period_ns, platform))
    model.save("vitis_dataflow.onnx")
