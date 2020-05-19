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

from onnx import TensorProto, helper
import numpy as np

from finn.core.datatype import DataType
from finn.transformation import Transformation
from finn.custom_op.registry import getCustomOp
from finn.transformation.infer_shapes import InferShapes
from finn.transformation.infer_datatypes import InferDataTypes


class InferConvInpGen(Transformation):
    """Convert Im2Col layers to ConvolutionInputGenerator layers."""

    def apply(self, model):
        graph = model.graph
        node_ind = 0
        graph_modified = False
        for n in graph.node:
            node_ind += 1
            if n.op_type == "Im2Col":
                i2c_input = n.input[0]
                i2c_output = n.output[0]
                i2c_in_shape = model.get_tensor_shape(i2c_input)
                i2c_out_shape = model.get_tensor_shape(i2c_output)
                dt = model.get_tensor_datatype(i2c_input)
                i2c_inst = getCustomOp(n)
                stride = i2c_inst.get_nodeattr("stride")
                k = i2c_inst.get_nodeattr("kernel_size")
                pad = i2c_inst.get_nodeattr("pad_amount")
                pad_val = i2c_inst.get_nodeattr("pad_value")
                ifm_ch = i2c_in_shape[-1]
                ifm_dim = i2c_in_shape[1]
                ofm_dim = i2c_out_shape[1]
                # if padding enabled, ensure pad_val supported by DataType
                if pad > 0:
                    assert dt.allowed(pad_val), "Im2Col DataType must support pad_val"
                # create equivalent ConvolutionInputGenerator node
                # TODO support padding
                new_node = helper.make_node(
                    "ConvolutionInputGenerator",
                    [i2c_input],
                    [i2c_output],
                    domain="finn",
                    backend="fpgadataflow",
                    ConvKernelDim=k,
                    IFMChannels=ifm_ch,
                    IFMDim=ifm_dim,
                    OFMDim=ofm_dim,
                    SIMD=ifm_ch,
                    Stride=stride,
                    inputDataType=dt.name,
                    outputDataType=dt.name,
                )
                graph.node.insert(node_ind, new_node)
                # remove old nodes
                graph.node.remove(n)
                graph_modified = True
        if graph_modified:
            model = model.transform(InferShapes())
            model = model.transform(InferDataTypes())
        return (model, graph_modified)


class InferStreamingMaxPool(Transformation):
    """Convert MaxPoolNHWC layers to StreamingMaxPool layers."""

    def apply(self, model):
        graph = model.graph
        node_ind = 0
        graph_modified = False
        for n in graph.node:
            node_ind += 1
            if n.op_type == "MaxPoolNHWC":
                mp_input = n.input[0]
                mp_output = n.output[0]
                mp_in_shape = model.get_tensor_shape(mp_input)
                # mp_out_shape = model.get_tensor_shape(mp_output)
                dt = model.get_tensor_datatype(mp_input)
                mp_inst = getCustomOp(n)
                # stride = mp_inst.get_nodeattr("strides")[0]
                k = mp_inst.get_nodeattr("kernel_shape")[0]
                # pad = mp_inst.get_nodeattr("pads")[0]
                ifm_ch = mp_in_shape[-1]
                ifm_dim = mp_in_shape[1]
                # ofm_dim = mp_out_shape[1]
                if ifm_dim % k == 0:
                    # create equivalent StreamingMaxPool_Batch node
                    # TODO support non-k strides
                    new_node = helper.make_node(
                        "StreamingMaxPool_Batch",
                        [mp_input],
                        [mp_output],
                        domain="finn",
                        backend="fpgadataflow",
                        PoolDim=k,
                        NumChannels=ifm_ch,
                        ImgDim=ifm_dim,
                        dataType=dt.name,
                    )
                    graph.node.insert(node_ind, new_node)
                    # remove old nodes
                    graph.node.remove(n)
                    graph_modified = True
        if graph_modified:
            model = model.transform(InferShapes())
            model = model.transform(InferDataTypes())
        return (model, graph_modified)


class InferBinaryStreamingFCLayer(Transformation):
    """Convert XnorPopcountMatMul layers to
    StreamingFCLayer_Batch layers. Any immediately following MultiThreshold
    layers will also be absorbed into the MVTU."""

    def __init__(self, mem_mode="const"):
        super().__init__()
        self.mem_mode = mem_mode

    def apply(self, model):
        graph = model.graph
        node_ind = 0
        graph_modified = False
        for n in graph.node:
            node_ind += 1
            if n.op_type == "XnorPopcountMatMul":
                mm_input = n.input[0]
                mm_weight = n.input[1]
                mm_output = n.output[0]
                mm_in_shape = model.get_tensor_shape(mm_input)
                mm_out_shape = model.get_tensor_shape(mm_output)
                assert (
                    model.get_tensor_datatype(mm_input) == DataType.BINARY
                ), """First
                input for xnorpopcount is not set to FINN DataType BINARY."""
                assert (
                    model.get_tensor_datatype(mm_weight) == DataType.BINARY
                ), """Second
                input (weights) for xnorpopcount is not set to FINN DataType BINARY."""
                idt = DataType.BINARY
                wdt = DataType.BINARY
                mm_output = n.output[0]
                W = model.get_initializer(mm_weight)
                # extract weight shape, note that ONNX and finn-hlslib
                # make different assumptions about dim order here
                # ONNX assumes W has (in, out) shape
                # finn-hlslib assumes W has (out, in) shape
                mh = int(W.shape[1])
                mw = int(W.shape[0])
                # create node with no parallelization first
                pe = 1
                simd = 1
                assert mh % pe == 0, "Requirement MH divisable by PE is violated."
                assert mw % simd == 0, "Requirement MW divisable by SIMD is violated."
                wmem = mw * mh // (pe * simd)
                assert (
                    mw * mh == wmem * pe * simd
                ), """Requirement (MW * MH) divisiable by
                (WMEM * PE * SIMD) is violated."""
                # see if we have any following thresholds
                consumer = model.find_consumer(mm_output)
                if consumer is not None and consumer.op_type == "MultiThreshold":
                    # TODO ensure integer thresholds?
                    # create MVTU (i.e. including activation)
                    mt_output = consumer.output[0]
                    mt_out_shape = model.get_tensor_shape(mt_output)
                    mt_thres = consumer.input[1]
                    T = model.get_initializer(mt_thres)
                    assert (
                        T.shape[0] == 1 or T.shape[0] == mh
                    ), """First dimension of
                    thresholds neither 1 nor MH."""
                    odt = model.get_tensor_datatype(mt_output)
                    if odt.bitwidth() == 1:
                        # covers both bipolar and binary
                        actval = 0
                    else:
                        actval = odt.min()
                    model.set_tensor_shape(mm_input, mm_in_shape)
                    model.set_tensor_shape(mt_output, mt_out_shape)
                    # create and insert new StreamingFCLayer node
                    new_node = helper.make_node(
                        "StreamingFCLayer_Batch",
                        [mm_input, mm_weight, mt_thres],
                        [mt_output],
                        domain="finn",
                        backend="fpgadataflow",
                        resType="ap_resource_lut()",
                        MW=mw,
                        MH=mh,
                        SIMD=simd,
                        PE=pe,
                        inputDataType=idt.name,
                        weightDataType=wdt.name,
                        outputDataType=odt.name,
                        ActVal=actval,
                        binaryXnorMode=1,
                        noActivation=0,
                        numInputVectors=list(mm_in_shape[:-1]),
                        mem_mode=self.mem_mode,
                    )
                    graph.node.insert(node_ind, new_node)
                    # remove old nodes
                    graph.node.remove(n)
                    graph.node.remove(consumer)
                    graph_modified = True
                else:
                    # no activation, matmul only
                    odt = model.get_tensor_datatype(mm_output)
                    model.set_tensor_shape(mm_input, mm_in_shape)
                    model.set_tensor_shape(mm_output, mm_out_shape)
                    # create and insert new StreamingFCLayer node
                    new_node = helper.make_node(
                        "StreamingFCLayer_Batch",
                        [mm_input, mm_weight],
                        [mm_output],
                        domain="finn",
                        backend="fpgadataflow",
                        resType="ap_resource_lut()",
                        MW=mw,
                        MH=mh,
                        SIMD=simd,
                        PE=pe,
                        inputDataType=idt.name,
                        weightDataType=wdt.name,
                        outputDataType=odt.name,
                        ActVal=0,
                        binaryXnorMode=1,
                        noActivation=1,
                        numInputVectors=list(mm_in_shape[:-1]),
                        mem_mode=self.mem_mode,
                    )
                    graph.node.insert(node_ind, new_node)
                    # remove old node
                    graph.node.remove(n)
                    graph_modified = True
        if graph_modified:
            model = model.transform(InferShapes())
            model = model.transform(InferDataTypes())
        return (model, graph_modified)


class InferQuantizedStreamingFCLayer(Transformation):
    """Convert MatMul layers with quantized inputs and weights to
    StreamingFCLayer_Batch layers. Any immediately following MultiThreshold
    layers will also be absorbed into the MVTU."""

    def __init__(self, mem_mode="const"):
        super().__init__()
        self.mem_mode = mem_mode

    def apply(self, model):
        graph = model.graph
        node_ind = 0
        graph_modified = False
        for n in graph.node:
            node_ind += 1
            if n.op_type == "MatMul":
                mm_input = n.input[0]
                mm_weight = n.input[1]
                mm_output = n.output[0]
                mm_in_shape = model.get_tensor_shape(mm_input)
                mm_out_shape = model.get_tensor_shape(mm_output)
                idt = model.get_tensor_datatype(mm_input)
                wdt = model.get_tensor_datatype(mm_weight)
                if idt.is_integer() and wdt.is_integer():
                    mm_output = n.output[0]
                    W = model.get_initializer(mm_weight)
                    # extract weight shape, note that ONNX and finn-hlslib
                    # make different assumptions about dim order here
                    # ONNX assumes W has (in, out) shape
                    # finn-hlslib assumes W has (out, in) shape
                    mh = int(W.shape[1])
                    mw = int(W.shape[0])
                    # create node with no parallelization first
                    pe = 1
                    simd = 1
                    assert mh % pe == 0, "Requirement MH divisable by PE is violated."
                    assert (
                        mw % simd == 0
                    ), "Requirement MW divisable by SIMD is violated."
                    wmem = mw * mh // (pe * simd)
                    assert (
                        mw * mh == wmem * pe * simd
                    ), """Requirement (MW * MH) divisiable by
                    (WMEM * PE * SIMD) is violated."""
                    # see if we have any following thresholds
                    consumer = model.find_consumer(mm_output)
                    if consumer is not None and consumer.op_type == "MultiThreshold":
                        # TODO ensure integer thresholds?
                        # create MVTU (i.e. including activation)
                        mt_output = consumer.output[0]
                        mt_out_shape = model.get_tensor_shape(mt_output)
                        mt_thres = consumer.input[1]
                        T = model.get_initializer(mt_thres)
                        assert (
                            T.shape[0] == 1 or T.shape[0] == mh
                        ), """First dimension of
                        thresholds neither 1 nor MH."""
                        odt = model.get_tensor_datatype(mt_output)
                        scale = getCustomOp(consumer).get_nodeattr("out_scale")
                        assert (
                            scale == 1.0
                        ), "out_scale must be equal to 1.0 for HLS conversion."
                        actval = getCustomOp(consumer).get_nodeattr("out_bias")
                        assert (
                            int(actval) == actval
                        ), "out_bias must be integer for HLS conversion."
                        actval = int(actval)
                        assert (not odt.signed()) or (
                            actval < 0
                        ), "Signed output requres actval < 0"
                        model.set_tensor_shape(mm_input, mm_in_shape)
                        model.set_tensor_shape(mt_output, mt_out_shape)
                        # create and insert new StreamingFCLayer node
                        new_node = helper.make_node(
                            "StreamingFCLayer_Batch",
                            [mm_input, mm_weight, mt_thres],
                            [mt_output],
                            domain="finn",
                            backend="fpgadataflow",
                            resType="ap_resource_lut()",
                            MW=mw,
                            MH=mh,
                            SIMD=simd,
                            PE=pe,
                            inputDataType=idt.name,
                            weightDataType=wdt.name,
                            outputDataType=odt.name,
                            ActVal=actval,
                            binaryXnorMode=0,
                            noActivation=0,
                            numInputVectors=list(mm_in_shape[:-1]),
                            mem_mode=self.mem_mode,
                        )
                        graph.node.insert(node_ind, new_node)
                        # remove old nodes
                        graph.node.remove(n)
                        graph.node.remove(consumer)
                        graph_modified = True
                    else:
                        # no activation, matmul only
                        odt = model.get_tensor_datatype(mm_output)
                        model.set_tensor_shape(mm_input, mm_in_shape)
                        model.set_tensor_shape(mm_output, mm_out_shape)
                        # create and insert new StreamingFCLayer node
                        new_node = helper.make_node(
                            "StreamingFCLayer_Batch",
                            [mm_input, mm_weight],
                            [mm_output],
                            domain="finn",
                            backend="fpgadataflow",
                            resType="ap_resource_lut()",
                            MW=mw,
                            MH=mh,
                            SIMD=simd,
                            PE=pe,
                            inputDataType=idt.name,
                            weightDataType=wdt.name,
                            outputDataType=odt.name,
                            ActVal=0,
                            binaryXnorMode=0,
                            noActivation=1,
                            numInputVectors=list(mm_in_shape[:-1]),
                            mem_mode=self.mem_mode,
                        )
                        graph.node.insert(node_ind, new_node)
                        # remove old node
                        graph.node.remove(n)
                        graph_modified = True
        if graph_modified:
            model = model.transform(InferShapes())
            model = model.transform(InferDataTypes())
        return (model, graph_modified)


class InferLabelSelectLayer(Transformation):
    """Convert any TopK into a LabelSelect HLS layer."""

    def apply(self, model):
        graph = model.graph
        node_ind = 0
        graph_modified = False
        for node in graph.node:
            node_ind += 1
            if node.op_type == "TopK":
                fc_input = node.input[0]
                k_input = node.input[1]
                val_output = node.output[0]
                idx_output = node.output[1]
                fc_in_shape = model.get_tensor_shape(fc_input)

                idt = model.get_tensor_datatype(fc_input)

                # skip conversion for layers with float input
                if not idt.is_integer():
                    continue

                # skip conversion for if value output is connected (not supported)
                if model.find_consumer(val_output) is not None:
                    continue

                num_labels = int(fc_in_shape[-1])
                # create node with no parallelization first
                pe = 1
                assert (
                    num_labels % pe == 0
                ), "Requirement Labels divisable by PE is violated."

                k = model.get_initializer(k_input)[0]

                # create and insert new StreamingFCLayer node
                new_node = helper.make_node(
                    "LabelSelect_Batch",
                    [fc_input],
                    [idx_output],
                    domain="finn",
                    backend="fpgadataflow",
                    Labels=num_labels,
                    PE=pe,
                    K=k,
                    inputDataType=idt.name,
                )
                graph.node.insert(node_ind, new_node)
                # remove old node
                graph.node.remove(node)
                graph_modified = True

        if graph_modified:
            model = model.transform(InferShapes())
            model = model.transform(InferDataTypes())
        return (model, graph_modified)


class InferAddStreamsLayer(Transformation):
    """Convert any Add into a AddStreams HLS layer."""

    def apply(self, model):
        graph = model.graph
        node_ind = 0
        graph_modified = False
        for node in graph.node:
            node_ind += 1
            if node.op_type == "Add":
                in0 = node.input[0]
                in1 = node.input[1]
                result = node.output[0]
                in0_shape = model.get_tensor_shape(in0)
                in1_shape = model.get_tensor_shape(in1)

                # skip if different shapes on inputs
                if in0_shape != in1_shape:
                    continue

                idt0 = model.get_tensor_datatype(in0)
                idt1 = model.get_tensor_datatype(in1)

                # skip if different data types on inputs
                if idt0 != idt1:
                    continue

                idt = idt0
                in_shape = in0_shape

                # skip conversion for layers with float input
                if not idt.is_integer():
                    continue

                num_channels = int(in_shape[-1])
                # create node with no parallelization first
                pe = 1
                assert (
                    num_channels % pe == 0
                ), "Requirement Channels divisable by PE is violated."

                # create and insert new StreamingFCLayer node
                new_node = helper.make_node(
                    "AddStreams_Batch",
                    [in0, in1],
                    [result],
                    domain="finn",
                    backend="fpgadataflow",
                    NumChannels=num_channels,
                    PE=pe,
                    inputDataType=idt.name,
                )
                graph.node.insert(node_ind, new_node)
                # remove old node
                graph.node.remove(node)
                graph_modified = True

        if graph_modified:
            model = model.transform(InferShapes())
            model = model.transform(InferDataTypes())
        return (model, graph_modified)


class InferGlobalAccPoolLayer(Transformation):
    """Convert any GlobalAveragePool into a GlobalAccPool HLS layer and a scalar Mul."""

    def apply(self, model):
        graph = model.graph
        node_ind = 0
        graph_modified = False
        for node in graph.node:
            node_ind += 1
            if node.op_type == "GlobalAveragePool":
                in0 = node.input[0]
                in0_shape = model.get_tensor_shape(in0)

                idt = model.get_tensor_datatype(in0)

                # skip conversion for layers with float input
                if not idt.is_integer():
                    continue

                num_ch = int(in0_shape[-1])
                vecs = in0_shape[:-1]
                # create node with no parallelization first
                pe = 1
                assert (
                    num_ch % pe == 0
                ), "Requirement Labels divisable by PE is violated."

                # create and insert HLS pool node
                out_shape = model.get_tensor_shape(node.output[0])
                pool_out = helper.make_tensor_value_info(
                    model.make_new_valueinfo_name(), TensorProto.FLOAT, out_shape
                )
                model.graph.value_info.append(pool_out)
                new_pool = helper.make_node(
                    "GlobalAccPool_Batch",
                    [in0],
                    [pool_out.name],
                    domain="finn",
                    backend="fpgadataflow",
                    NumChannels=num_ch,
                    PE=pe,
                    inputDataType=idt.name,
                    numInputVectors=vecs,
                )

                mul_value = helper.make_tensor_value_info(
                    model.make_new_valueinfo_name(), TensorProto.FLOAT, [1]
                )
                model.graph.value_info.append(mul_value)
                model.set_initializer(
                    mul_value.name, np.array(1 / (vecs[-1] * vecs[-2]))
                )
                new_mul = helper.make_node(
                    "Mul", [in0, mul_value.name], [node.output[0]],
                )
                graph.node.insert(node_ind, new_pool)
                node_ind += 1
                graph.node.insert(node_ind, new_mul)
                # remove old node
                graph.node.remove(node)
                graph_modified = True

        if graph_modified:
            model = model.transform(InferShapes())
            model = model.transform(InferDataTypes())
        return (model, graph_modified)


class InferDuplicateStreamsLayer(Transformation):
    """Convert any tensor with fanout > 1 into a DuplicateStreams HLS layer."""

    def apply(self, model):
        graph = model.graph
        node_ind = 0
        graph_modified = False
        for node in graph.node:
            node_ind += 1
            successors = model.find_direct_successors(node.output[0])
            if len(successors) == 2:
                output_tensor = node.output[0]

                dt = model.get_tensor_datatype(output_tensor)

                # skip conversion for layers with float input
                if not dt.is_integer():
                    continue

                # create two identical tensors
                out_shape = model.get_tensor_shape(output_tensor)
                dup0 = helper.make_tensor_value_info(
                    model.make_new_valueinfo_name(), TensorProto.FLOAT, out_shape
                )
                dup1 = helper.make_tensor_value_info(
                    model.make_new_valueinfo_name(), TensorProto.FLOAT, out_shape
                )

                num_ch = int(out_shape[-1])
                vecs = out_shape[:-1]
                # create node with no parallelization first
                pe = 1
                assert (
                    num_ch % pe == 0
                ), "Requirement Labels divisable by PE is violated."

                model.graph.value_info.append(dup0)
                model.graph.value_info.append(dup1)
                dup_node = helper.make_node(
                    "DuplicateStreams_Batch",
                    [output_tensor],
                    [dup0, dup1],
                    domain="finn",
                    backend="fpgadataflow",
                    NumChannels=num_ch,
                    PE=pe,
                    inputDataType=dt.name,
                    numInputVectors=vecs,
                )

                graph.node.insert(node_ind, dup_node)

                # connect successors to dup0/dup1
                assert (
                    len(successors[0].input) == 1 and len(successors[1].input) == 1
                ), "Duplicating to multi-input node not supported"

                successors[0].input[0] = dup0
                successors[1].input[0] = dup0

                # remove old node
                graph.node.remove(node)
                graph_modified = True

        if graph_modified:
            model = model.transform(InferShapes())
            model = model.transform(InferDataTypes())
        return (model, graph_modified)
