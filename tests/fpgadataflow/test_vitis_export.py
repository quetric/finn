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

from finn.transformation.fpgadataflow.vitis_build import VitisBuild
from finn.util.basic import alveo_part_map, alveo_default_platform


def make_model(dt, width=32):
    inp = helper.make_tensor_value_info("inp", TensorProto.FLOAT, [1, 32, 32, 16])
    outp = helper.make_tensor_value_info("outp", TensorProto.FLOAT, [1, 32, 32, 16])
    inter = helper.make_tensor_value_info("inter", TensorProto.FLOAT, [1, 32, 32, 16])

    idma_node = helper.make_node(
        "IODMA",
        [inp.name],
        [inter.name],
        ImgDim=32,
        NumChannels=16,
        dataType=dt.name,
        intfWidth=width,
        direction="in",
        domain="finn",
        backend="fpgadataflow",
    )

    odma_node = helper.make_node(
        "IODMA",
        [inter.name],
        [outp.name],
        ImgDim=32,
        NumChannels=16,
        dataType=dt.name,
        intfWidth=width,
        direction="out",
        domain="finn",
        backend="fpgadataflow",
    )

    graph = helper.make_graph(
        nodes=[idma_node, odma_node], name="graph", inputs=[inp], outputs=[outp],
    )

    model = helper.make_model(graph, producer_name="model")
    model = ModelWrapper(model)

    model.set_tensor_datatype("inp", dt)
    model.set_tensor_datatype("outp", dt)
    model.set_tensor_datatype("inter", dt)

    model.graph.value_info.append(inp)
    model.graph.value_info.append(outp)
    model.graph.value_info.append(inter)

    return model


@pytest.mark.slow
@pytest.mark.vivado
def test_vitis_export(board="U250", period_ns=10):
    platform = alveo_default_platform[board]
    fpga_part = alveo_part_map[board]
    model = make_model(DataType.UINT32)
    model = model.transform(VitisBuild(fpga_part, period_ns, platform))
    model.save("vitis_dataflow.onnx")
