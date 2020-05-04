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
from onnx import TensorProto, helper

import finn.core.onnx_exec as oxe
from finn.analysis.fpgadataflow.hls_synth_res_estimation import hls_synth_res_estimation
from finn.core.datatype import DataType
from finn.core.modelwrapper import ModelWrapper
from finn.custom_op.multithreshold import multithreshold
from finn.transformation.fpgadataflow.codegen_ipgen import CodeGen_ipgen
from finn.transformation.fpgadataflow.codegen_npysim import CodeGen_npysim
from finn.transformation.fpgadataflow.compile import Compile
from finn.transformation.fpgadataflow.hlssynth_ipgen import HLSSynth_IPGen
from finn.transformation.fpgadataflow.set_exec_mode import SetExecMode
from finn.transformation.general import GiveUniqueNodeNames
from finn.transformation.fpgadataflow.prepare_rtlsim import PrepareRTLSim
from finn.util.basic import gen_finn_dt_tensor
from finn.transformation.fpgadataflow.replace_verilog_relpaths import (
    ReplaceVerilogRelPaths,
)


def make_single_thresholding_modelwrapper(T, pe, idt, odt):
    NumChannels = T.shape[0]

    inp = helper.make_tensor_value_info("inp", TensorProto.FLOAT, [1, NumChannels])
    outp = helper.make_tensor_value_info("outp", TensorProto.FLOAT, [1, NumChannels])

    node_inp_list = ["inp", "thresh"]

    FCLayer_node = helper.make_node(
        "Thresholding_Batch",
        node_inp_list,
        ["outp"],
        domain="finn",
        backend="fpgadataflow",
        NumChannels=NumChannels,
        PE=pe,
        inputDataType=idt.name,
        outputDataType=odt.name,
    )
    graph = helper.make_graph(
        nodes=[FCLayer_node], name="fclayer_graph", inputs=[inp], outputs=[outp]
    )

    model = helper.make_model(graph, producer_name="thresholding-model")
    model = ModelWrapper(model)

    model.set_tensor_datatype("inp", idt)
    model.set_tensor_datatype("outp", odt)

    model.set_tensor_datatype("thresh", idt)
    model.set_initializer("thresh", T)
    return model


def prepare_inputs(input_tensor, idt):
    if idt == DataType.BIPOLAR:
        # convert bipolar to binary
        return {"inp": (input_tensor + 1) / 2}
    else:
        return {"inp": input_tensor}


# TODO binary/bipolar inputs/outputs
# activation: None or DataType
@pytest.mark.parametrize("act", [DataType.INT4])
# input datatype
@pytest.mark.parametrize("idt", [DataType.INT2, DataType.INT4])
# folding, -1 is maximum possible
@pytest.mark.parametrize("nf", [-1, 2, 1])
# number of input features
@pytest.mark.parametrize("ich", [16])
# execution mode
@pytest.mark.parametrize("exec_mode", ["rtlsim", "npysim"])
def test_fpgadataflow_thresholding(idt, act, nf, ich, exec_mode):
    if nf == -1:
        nf = ich
    pe = ich // nf
    assert ich % pe == 0

    # generate input data
    x = gen_finn_dt_tensor(idt, (1, ich))

    odt = act
    n_steps = act.get_num_possible_values() - 1
    T = np.random.randint(idt.min(), idt.max() + 1, (ich, n_steps)).astype(np.float32)
    # provide non-decreasing thresholds
    T = np.sort(T, axis=1)
    # generate thresholds for activation
    if idt == DataType.BIPOLAR:
        # bias thresholds to be positive
        T = np.ceil((T + ich) / 2)
        assert (T >= 0).all()

    model = make_single_thresholding_modelwrapper(T, pe, idt, odt)

    if exec_mode == "npysim":
        model = model.transform(SetExecMode("npysim"))
        model = model.transform(CodeGen_npysim())
        model = model.transform(Compile())
    elif exec_mode == "rtlsim":
        model = model.transform(SetExecMode("rtlsim"))
        model = model.transform(GiveUniqueNodeNames())
        model = model.transform(CodeGen_ipgen("xc7z020clg400-1", 5))
        model = model.transform(HLSSynth_IPGen())
        model = model.transform(ReplaceVerilogRelPaths())
        model = model.transform(PrepareRTLSim())
    else:
        raise Exception("Unknown exec_mode in test_fpgadataflow_thresholding")

    # prepare input data
    input_dict = prepare_inputs(x, idt)

    y = multithreshold(x, T)
    if act == DataType.BIPOLAR:
        # binary to bipolar
        y = 2 * y - 1
    else:
        # signed offset
        y += act.min()

    oshape = model.get_tensor_shape("outp")
    y_expected = y.reshape(oshape)
    # execute model
    y_produced = oxe.execute_onnx(model, input_dict)["outp"]

    y_produced = y_produced.reshape(y_expected.shape)

    assert (y_produced == y_expected).all(), "npysim failed"

    if exec_mode == "rtlsim":
        hls_synt_res_est = model.analysis(hls_synth_res_estimation)
        assert "Thresholding_Batch_0" in hls_synt_res_est
