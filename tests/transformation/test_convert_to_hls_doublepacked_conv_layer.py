from onnx import TensorProto, helper
import numpy as np
import pytest

from finn.core.datatype import DataType
from finn.transformation.infer_shapes import InferShapes
from finn.transformation.infer_datatypes import InferDataTypes
from finn.transformation.general import GiveUniqueNodeNames
from finn.transformation.lower_convs_to_matmul import LowerConvsToMatMul

from finn.transformation.fpgadataflow.prepare_ip import PrepareIP
from finn.transformation.fpgadataflow.prepare_rtlsim import PrepareRTLSim
from finn.transformation.fpgadataflow.hlssynth_ip import HLSSynthIP
from finn.transformation.fpgadataflow.replace_verilog_relpaths import (
    ReplaceVerilogRelPaths,
)
import finn.core.onnx_exec as oxe
from finn.core.modelwrapper import ModelWrapper
from finn.util.basic import gen_finn_dt_tensor
import finn.transformation.fpgadataflow.convert_to_hls_layers as to_hls
from finn.transformation.fpgadataflow.infer_doublepacked_dsp import InferDoublePackedConv

from finn.transformation.fpgadataflow.prepare_cppsim import PrepareCppSim
from finn.transformation.fpgadataflow.compile_cppsim import CompileCppSim
from finn.transformation.fpgadataflow.set_exec_mode import SetExecMode
from finn.custom_op.im2col import compute_conv_output_dim
from finn.custom_op.registry import getCustomOp
import os

# kernel size
@pytest.mark.parametrize("kernel_size", [7])
# stride
@pytest.mark.parametrize("stride", [2])
# padding
@pytest.mark.parametrize("pad", [3])
# execution mode
@pytest.mark.parametrize("exec_mode", ["cppsim", "rtlsim"])
def test_convert_to_hls_doublepacked_conv_layer(kernel_size, stride, pad, exec_mode):
    idt = DataType.UINT8 # DataType.INT8

    in_feature_dim = 7
    in_chn = 3
    out_chn = 5

    out_feature_dim = compute_conv_output_dim(in_feature_dim, kernel_size, stride, pad)

    input_shape = [1, in_chn, in_feature_dim, in_feature_dim]
    output_shape = [1, out_chn, out_feature_dim, out_feature_dim]

    conv_param_shape = [out_chn, in_chn, kernel_size, kernel_size]
    conv_weight_dt = DataType.INT8

    conv_attr = {}
    conv_attr["dilations"] = [1, 1]
    conv_attr["group"] = 1
    conv_attr["kernel_shape"] = [kernel_size, kernel_size]
    conv_attr["pads"] = [pad, pad, pad, pad]
    conv_attr["strides"] = [stride, stride]

    top_in = helper.make_tensor_value_info("top_in", TensorProto.FLOAT, input_shape)
    top_out = helper.make_tensor_value_info("top_out", TensorProto.FLOAT, output_shape)
    value_info = [
        helper.make_tensor_value_info("p1", TensorProto.FLOAT, conv_param_shape)
    ]

    modelproto = helper.make_model(
        helper.make_graph(
            name="conv_test",
            inputs=[top_in],
            outputs=[top_out],
            value_info=value_info,
            nodes=[
                helper.make_node("Conv", ["top_in", "p1"], ["top_out"], **conv_attr)
            ],
        )
    )

    model = ModelWrapper(modelproto)
    model.set_tensor_datatype("top_in", idt)
    model.set_tensor_datatype("top_out", idt)
    model.set_tensor_datatype("p1", conv_weight_dt)
    model.set_initializer("p1", gen_finn_dt_tensor(conv_weight_dt, conv_param_shape))

    model = model.transform(InferShapes())
    model = model.transform(InferDataTypes())


    include_file = "./double_packed_conv.hpp"
    include_file = os.path.abspath(include_file)
    new_model = model.transform(InferDoublePackedConv([1]))
    new_model = new_model.transform(GiveUniqueNodeNames())
    new_model = new_model.transform(InferDataTypes())


    inst = getCustomOp(new_model.graph.node[1])
    simd = in_chn
    inst.set_nodeattr("SIMD", simd)
    pe = out_chn
    inst.set_nodeattr("PE", pe)
    pe = out_chn

    mmv = 16
    while out_feature_dim% mmv != 0:
        mmv //= 2
    assert mmv >=2 and mmv%2 ==0
    print("Using mmv = ",mmv)
    inst.set_nodeattr("MMV", mmv)


    if exec_mode == "cppsim":
        new_model = new_model.transform(PrepareCppSim())
        new_model = new_model.transform(CompileCppSim())
        new_model = new_model.transform(SetExecMode("cppsim"))
    elif exec_mode == "rtlsim":
        new_model = new_model.transform(SetExecMode("rtlsim"))
        new_model = new_model.transform(GiveUniqueNodeNames())
        new_model = new_model.transform(PrepareIP("xc7z020clg400-1", 5))
        new_model = new_model.transform(HLSSynthIP())
        new_model = new_model.transform(ReplaceVerilogRelPaths())
        new_model = new_model.transform(PrepareRTLSim())
    else:
        raise Exception("Unknown exec_mode")


    x = gen_finn_dt_tensor(idt, input_shape)
    inp_dict = {model.graph.input[0].name: x}
    expected_dict = oxe.execute_onnx(
        model, inp_dict, return_full_exec_context=True, start_node=None, end_node=None
    )
    produced_dict = oxe.execute_onnx(
        new_model, inp_dict, return_full_exec_context=True, start_node=None, end_node=None
    )

    exp_out = expected_dict["top_out"]
    prod_out = produced_dict["top_out"]
    assert (exp_out == prod_out).all(), "Not equal"

# kernel size
@pytest.mark.parametrize("kernel_size", [7])
# stride
@pytest.mark.parametrize("stride", [2])
# padding
@pytest.mark.parametrize("pad", [3])
# execution mode
@pytest.mark.parametrize("exec_mode", ["cppsim", "rtlsim"])
def test_convert_to_hls_doublepacked_conv_layer_with_fused_thres(kernel_size, stride, pad, exec_mode):
    idt = DataType.UINT8 # DataType.INT8
    odt = DataType.UINT2

    in_feature_dim = 7
    in_chn = 3
    out_chn = 5

    out_feature_dim = compute_conv_output_dim(in_feature_dim, kernel_size, stride, pad)

    input_shape = [1, in_chn, in_feature_dim, in_feature_dim]
    output_shape = [1, out_chn, out_feature_dim, out_feature_dim]

    conv_param_shape = [out_chn, in_chn, kernel_size, kernel_size]
    conv_weight_dt = DataType.UINT4

    conv_attr = {}
    conv_attr["dilations"] = [1, 1]
    conv_attr["group"] = 1
    conv_attr["kernel_shape"] = [kernel_size, kernel_size]
    conv_attr["pads"] = [pad, pad, pad, pad]
    conv_attr["strides"] = [stride, stride]

    top_in = helper.make_tensor_value_info("top_in", TensorProto.FLOAT, input_shape)
    top_out = helper.make_tensor_value_info("top_out", TensorProto.FLOAT, output_shape)
    value_info = [
        helper.make_tensor_value_info("p1", TensorProto.FLOAT, conv_param_shape),
        helper.make_tensor_value_info("conv_out", TensorProto.FLOAT, output_shape),
        helper.make_tensor_value_info("thres", TensorProto.FLOAT, [1,odt.max()]),
        
    ]

    # Thresholds

    NumChannels = out_chn

    modelproto = helper.make_model(
        helper.make_graph(
            name="conv_test",
            inputs=[top_in],
            outputs=[top_out],
            value_info=value_info,
            nodes=[
                helper.make_node("Conv", ["top_in", "p1"], ["conv_out"], **conv_attr),
                helper.make_node( "MultiThreshold", ["conv_out","thres"],["top_out"],
                    domain="finn",out_dtype=odt.name
                    )
            ],
        )
    )

    model = ModelWrapper(modelproto)
    model.set_tensor_datatype("top_in", idt)
    model.set_tensor_datatype("top_out", odt)
    model.set_tensor_datatype("p1", conv_weight_dt)
    model.set_tensor_datatype("thres", idt)
    model.set_initializer("p1", gen_finn_dt_tensor(conv_weight_dt, conv_param_shape))
    model.set_initializer("thres", np.round(np.arange(odt.max())*idt.max()/odt.max()).reshape((1,odt.max())))

    model = model.transform(InferShapes())
    model = model.transform(InferDataTypes())


    new_model = model.transform(InferDoublePackedConv([1]))
    new_model = new_model.transform(GiveUniqueNodeNames())
    new_model = new_model.transform(InferDataTypes())


    inst = getCustomOp(new_model.graph.node[1])
    simd = in_chn
    inst.set_nodeattr("SIMD", simd)
    pe = out_chn
    inst.set_nodeattr("PE", pe)
    pe = out_chn

    mmv = 16
    while out_feature_dim% mmv != 0:
        mmv //= 2
    assert mmv >=2 and mmv%2 ==0
    print("Using mmv = ",mmv)
    inst.set_nodeattr("MMV", mmv)


    if exec_mode == "cppsim":
        new_model = new_model.transform(PrepareCppSim())
        new_model = new_model.transform(CompileCppSim())
        new_model = new_model.transform(SetExecMode("cppsim"))
    elif exec_mode == "rtlsim":
        new_model = new_model.transform(SetExecMode("rtlsim"))
        new_model = new_model.transform(GiveUniqueNodeNames())
        new_model = new_model.transform(PrepareIP("xc7z020clg400-1", 5))
        new_model = new_model.transform(HLSSynthIP())
        new_model = new_model.transform(ReplaceVerilogRelPaths())
        new_model = new_model.transform(PrepareRTLSim())
    else:
        raise Exception("Unknown exec_mode")


    x = gen_finn_dt_tensor(idt, input_shape)
    inp_dict = {model.graph.input[0].name: x}
    expected_dict = oxe.execute_onnx(
        model, inp_dict, return_full_exec_context=True, start_node=None, end_node=None
    )
    produced_dict = oxe.execute_onnx(
        new_model, inp_dict, return_full_exec_context=True, start_node=None, end_node=None
    )

    exp_out = expected_dict["top_out"]
    prod_out = produced_dict["top_out"]
    assert (exp_out == prod_out).all(), "Not equal"

