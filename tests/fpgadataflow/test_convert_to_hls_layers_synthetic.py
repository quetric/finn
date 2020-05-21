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

import os
import numpy as np

from onnx import TensorProto, helper

import finn.core.onnx_exec as oxe
from finn.core.datatype import DataType
from finn.core.modelwrapper import ModelWrapper
from finn.transformation.fold_constants import FoldConstants
from finn.transformation.general import GiveReadableTensorNames, GiveUniqueNodeNames
from finn.transformation.streamline.reorder import (
    MoveLinearPastEltwiseAdd,
    MoveScalarLinearPastInvariants,
)
from finn.transformation.infer_shapes import InferShapes
from finn.transformation.infer_datatypes import InferDataTypes
from finn.transformation.infer_data_layouts import InferDataLayouts
from finn.util.basic import gen_finn_dt_tensor
from finn.util.test import soft_verify_topk
from finn.transformation.double_to_single_float import DoubleToSingleFloat
from finn.transformation.insert_topk import InsertTopK
import finn.transformation.fpgadataflow.convert_to_hls_layers as to_hls
from finn.transformation.fpgadataflow.prepare_cppsim import PrepareCppSim
from finn.transformation.fpgadataflow.compile_cppsim import CompileCppSim
from finn.transformation.fpgadataflow.set_exec_mode import SetExecMode

import pytest

export_onnx_path = "test_output_synthetic.onnx"

# construct a synthetic graph to test:
# topk insertion, topk conversion to hls, add conversion to hls
# graph should just be a sum


def make_model(ch, ifmdim):
    shape = [1, ch, ifmdim, ifmdim]
    inp = helper.make_tensor_value_info("inp", TensorProto.FLOAT, shape)
    inp1_add = helper.make_tensor_value_info("inp1_add", TensorProto.FLOAT, shape)
    inp1_add_ct = helper.make_tensor_value_info("inp1_add_ct", TensorProto.FLOAT, [1])
    inp2_add = helper.make_tensor_value_info("inp2_add", TensorProto.FLOAT, shape)
    inp2_add_ct = helper.make_tensor_value_info("inp2_add_ct", TensorProto.FLOAT, [1])
    inp1_mul = helper.make_tensor_value_info("inp1_mul", TensorProto.FLOAT, shape)
    inp1_mul_ct = helper.make_tensor_value_info("inp1_mul_ct", TensorProto.FLOAT, [1])
    inp2_mul = helper.make_tensor_value_info("inp2_mul", TensorProto.FLOAT, shape)
    inp2_mul_ct = helper.make_tensor_value_info("inp2_mul_ct", TensorProto.FLOAT, [1])
    eltwise_add = helper.make_tensor_value_info("eltwise_add", TensorProto.FLOAT, shape)
    pool = helper.make_tensor_value_info("pool", TensorProto.FLOAT, [1, ch, 1, 1])
    reshape_ct = helper.make_tensor_value_info("reshape_ct", TensorProto.INT64, [2])
    outp = helper.make_tensor_value_info("outp", TensorProto.FLOAT, [1, ch])

    add1_node = helper.make_node("Add", [inp.name, inp1_add_ct.name], [inp1_add.name])
    add2_node = helper.make_node("Add", [inp.name, inp2_add_ct.name], [inp2_add.name])
    mul1_node = helper.make_node(
        "Mul", [inp1_add.name, inp1_mul_ct.name], [inp1_mul.name]
    )
    mul2_node = helper.make_node(
        "Mul", [inp2_add.name, inp2_mul_ct.name], [inp2_mul.name]
    )
    eltwise_add_node = helper.make_node(
        "Add", [inp1_mul.name, inp2_mul.name], [eltwise_add.name]
    )
    globalavgpool_node = helper.make_node(
        "GlobalAveragePool", [eltwise_add.name], [pool.name]
    )
    reshape_node = helper.make_node(
        "Reshape", [pool.name, reshape_ct.name], [outp.name]
    )
    graph = helper.make_graph(
        nodes=[
            add1_node,
            add2_node,
            mul1_node,
            mul2_node,
            eltwise_add_node,
            globalavgpool_node,
            reshape_node,
        ],
        name="graph",
        inputs=[inp],
        outputs=[outp],
    )

    model = helper.make_model(graph, producer_name="add-model")
    model = ModelWrapper(model)

    # set initializers for scalar add/mul nodes
    model.set_initializer(add1_node.input[1], np.array([7.0]))
    model.set_initializer(add2_node.input[1], np.array([8.0]))
    model.set_initializer(mul1_node.input[1], np.array([3.0]))
    model.set_initializer(mul2_node.input[1], np.array([3.0]))
    model.set_initializer(reshape_node.input[1], np.array([1, -1]))

    return model


@pytest.mark.vivado
# data types
@pytest.mark.parametrize("idt", [DataType.UINT4])
# channels
@pytest.mark.parametrize("ch", [64])
# ifmdim
@pytest.mark.parametrize("ifmdim", [7])
def test_convert_to_hls_layers_synthetic(ch, ifmdim, idt):
    model = make_model(ch, ifmdim)
    model.save(export_onnx_path)
    model = ModelWrapper(export_onnx_path)
    model = model.transform(DoubleToSingleFloat())
    model = model.transform(InferShapes())
    model = model.transform(FoldConstants())
    model = model.transform(GiveUniqueNodeNames())
    model = model.transform(GiveReadableTensorNames())
    model = model.transform(InferDataLayouts())
    # model.save("golden.onnx")
    # generate test vectors of correct shape
    if ifmdim == -1:
        input_tensor_shape = (1, ch)
    else:
        input_tensor_shape = (1, ch, ifmdim, ifmdim)

    x = gen_finn_dt_tensor(idt, input_tensor_shape)

    # generate expected value from streamlined net
    input_dict = {model.graph.input[0].name: x}

    output_dict = oxe.execute_onnx(model, input_dict, True)
    produced_sum = output_dict[model.graph.output[0].name]
    expected_sum = np.sum(3.0 * ((x + x) + 15.0), axis=(2, 3)) / (ifmdim * ifmdim)
    assert (produced_sum.flatten() == expected_sum.flatten()).all()

    model = model.transform(MoveLinearPastEltwiseAdd())
    model = model.transform(InferDataLayouts())
    model = model.transform(MoveScalarLinearPastInvariants())

    # verify again, to check we didnt break anything
    output_dict = oxe.execute_onnx(model, input_dict, True)
    produced_sum = output_dict[model.graph.output[0].name]
    assert np.isclose(expected_sum.flatten(), produced_sum.flatten(), atol=1e-3).all()

    # convert to hls
    model.set_tensor_datatype(model.graph.input[0].name, idt)
    model = model.transform(to_hls.InferAddStreamsLayer())
    model = model.transform(to_hls.InferGlobalAccPoolLayer())
    model = model.transform(MoveScalarLinearPastInvariants())
    # insert top-k node, which should absorb linear ops before it
    model = model.transform(InsertTopK())
    model = model.transform(InferShapes())
    model = model.transform(InferDataLayouts())
    model = model.transform(GiveUniqueNodeNames())
    model = model.transform(GiveReadableTensorNames())
    model = model.transform(InferDataTypes())
    model = model.transform(to_hls.InferLabelSelectLayer())

    # model.save("golden_hls.onnx")
    # check topology status
    finn_nodes = model.get_finn_nodes()
    assert len(finn_nodes) == 3
    add_nodes = model.get_nodes_by_op_type("AddStreams_Batch")
    assert len(add_nodes) == 1
    pool_nodes = model.get_nodes_by_op_type("GlobalAccPool_Batch")
    assert len(pool_nodes) == 1
    label_nodes = model.get_nodes_by_op_type("LabelSelect_Batch")
    assert len(label_nodes) == 1

    model = model.transform(PrepareCppSim())
    model = model.transform(CompileCppSim())
    model = model.transform(SetExecMode("cppsim"))

    output_dict = oxe.execute_onnx(model, input_dict, True)
    produced_topk_hls = output_dict[model.graph.output[0].name]
    topk_input = output_dict[model.graph.node[-1].input[0]]
    assert soft_verify_topk(topk_input, produced_topk_hls, 5)

    os.remove(export_onnx_path)
