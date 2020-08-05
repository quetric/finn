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

import pytest
import numpy as np
import random

from onnx import TensorProto, helper

from finn.core.datatype import DataType
from finn.core.modelwrapper import ModelWrapper
from finn.transformation.fpgadataflow.prepare_ip import PrepareIP
from finn.transformation.fpgadataflow.hlssynth_ip import HLSSynthIP
from finn.transformation.general import GiveUniqueNodeNames


def make_memstreamer_modelwrapper(
    nstreams, pe, simd, wmem, odt, values, strategy, height
):

    outp = []
    for i in range(nstreams):
        outp.append(
            helper.make_tensor_value_info(
                "outp" + str(i), TensorProto.FLOAT, list(values[i].dims)
            )
        )

    memstreamer_node = helper.make_node(
        "MemStreamer",
        [],
        [x.name for x in outp],
        domain="finn",
        backend="fpgadataflow",
        max_height=height,
        nstreams=nstreams,
        PE=pe,
        SIMD=simd,
        WMEM=wmem,
        weights=values,
        dataType=[x.value for x in odt],
        strategy=strategy,
    )
    graph = helper.make_graph(
        nodes=[memstreamer_node], name="graph", inputs=[], outputs=outp,
    )

    model = helper.make_model(graph, producer_name="model")
    model = ModelWrapper(model)

    for i in range(nstreams):
        model.set_tensor_datatype(outp[i].name, odt[i])

    return model


# number of streams
@pytest.mark.parametrize("nstreams", [5])
# number of streams
@pytest.mark.parametrize("height", [3, 4])
# intra or inter-layer packing
@pytest.mark.parametrize("strategy", ["intra", "inter"])
# number of iterations (tests with random starting seeds)
@pytest.mark.parametrize("iteration", range(1))
@pytest.mark.vivado
def test_fpgadataflow_memstreamer(nstreams, strategy, iteration, height):
    # generate attributes and data
    odt = []
    simd = []
    pe = []
    wmem = []
    values = []
    for i in range(nstreams):
        odt.append(random.choice([DataType.BIPOLAR, DataType.INT2]))
        simd.append(random.choice([8, 16]))
        pe.append(random.choice([4, 8]))
        wmem.append(random.choice([256, 288, 576]))
        tensor_shape = (pe[i], wmem[i], simd[i])
        if odt[i] == DataType.BIPOLAR:
            tensor = (
                np.random.randint(
                    DataType.BINARY.min(), DataType.BINARY.max() + 1, size=tensor_shape
                )
                .flatten()
                .astype(float)
            )
            tensor = 2 * tensor - 1
        else:
            tensor = (
                np.random.randint(odt[i].min(), odt[i].max() + 1, size=tensor_shape)
                .flatten()
                .astype(float)
            )
        tensor = helper.make_tensor(
            name="weights" + str(i),
            data_type=TensorProto.FLOAT,
            dims=tensor_shape,
            vals=tensor,
        )
        values.append(tensor)

    model = make_memstreamer_modelwrapper(
        nstreams, pe, simd, wmem, odt, values, strategy, height
    )

    # do IP preparation and check result
    model = model.transform(GiveUniqueNodeNames())
    model = model.transform(PrepareIP("xc7z020clg400-1", 5))
    model = model.transform(HLSSynthIP())
