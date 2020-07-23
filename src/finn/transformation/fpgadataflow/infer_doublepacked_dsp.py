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

from onnx import TensorProto
from onnx import helper

from finn.transformation import Transformation
from finn.transformation.infer_shapes import InferShapes
from finn.util.basic import get_by_name
from finn.core.datatype import DataType
import numpy as np


class InferDoublePackedConv(Transformation):
    """InferDoublePackedConv """

    def __init__(self, conv_position_to_replace, include_file=""):
        super(InferDoublePackedConv, self).__init__()
        self.conv_position_to_replace = tuple(conv_position_to_replace)
        self.include_file = include_file

    def get_smallest_possible(self, vals):
        """Returns smallest (fewest bits) possible DataType that can represent
        value. Prefers unsigned integers where possible."""
        vals = np.array(vals)
        for v in vals:
            assert int(v) == v, "Error float value"

        for k in DataType.__members__:
            dt = DataType[k]

            if dt in [DataType.FLOAT32]:
                # not currently supported
                continue

            if (dt.min() <= vals).all() and (vals <= dt.max()).all():
                return dt

    def apply(self, model):
        graph = model.graph
        node_ind = 0
        conv_position = 0
        graph_modified = False
        for n in graph.node:
            node_ind += 1
            if n.op_type == "Conv":
                conv_position += 1
                if conv_position not in self.conv_position_to_replace:
                    continue

                cnv_input = n.input[0]
                cnv_output = n.output[0]
                idt = model.get_tensor_datatype(cnv_input)
                odt = model.get_tensor_datatype(cnv_output)
                # extract conv parameters
                k = get_by_name(n.attribute, "kernel_shape").ints[-1]
                pad = get_by_name(n.attribute, "pads").ints[-1]
                stride = get_by_name(n.attribute, "strides").ints[-1]
                weight_name = n.input[1]
                W_conv = model.get_initializer(weight_name)
                ifm_ch = W_conv.shape[1]
                ofm_ch = W_conv.shape[0]
                ifm_dim = model.get_tensor_shape(n.input[0])[-1]  # assume NCHW
                ofm_dim = model.get_tensor_shape(n.output[0])[-1]  # assume NCHW
                # reuse conv weights for new matmul weights
                # conv weights are [OFM][IFM][k][k]
                # first convert to [OFM][k][k][IFM] (to remain compatible with
                # finn-hlslib and how it does im2col/sliding window)
                W_matmul = W_conv.transpose(0, 2, 3, 1)
                # reshape into [OFM][k*k*IFM] matrix

                mh = ofm_ch
                mw = ifm_ch * k * k
                W_matmul = W_matmul.reshape(mh, mw)
                # transpose to get ONNX-compatible [k*k*IFM][OFM] matrix
                W_matmul = W_matmul.T

                model.set_initializer(weight_name, W_matmul)
                wdt = self.get_smallest_possible(
                    [min(W_matmul.flatten()), max(W_matmul.flatten())]
                )

                if wdt.bitwidth() > 8:
                    print(
                        "Can't infer double packed conv as weight bits =",
                        wdt.bitwidth(),
                    )
                    continue
                if wdt.signed():
                    wdt = DataType.INT8
                else:
                    wdt = DataType.UINT8

                model.set_tensor_datatype(weight_name, wdt)
                idtypes = [idt, wdt]
                has_signed_inp = len(list(filter(lambda x: x.signed(), idtypes))) != 0
                if has_signed_inp:
                    conv_odt = DataType.INT32
                else:
                    conv_odt = DataType.UINT32

                # create new intermediate values
                inp_trans_out = helper.make_tensor_value_info(
                    model.make_new_valueinfo_name(),
                    TensorProto.FLOAT,
                    (1, ifm_dim, ifm_dim, ifm_ch),  # NHWC
                )
                graph.value_info.append(inp_trans_out)
                inp_trans_out = inp_trans_out.name
                model.set_tensor_datatype(inp_trans_out, idt)

                matmul_out = helper.make_tensor_value_info(
                    model.make_new_valueinfo_name(),
                    TensorProto.FLOAT,
                    (1, ofm_dim, ofm_dim, ofm_ch),
                )
                graph.value_info.append(matmul_out)
                matmul_out = matmul_out.name
                model.set_tensor_datatype(matmul_out, conv_odt)

                # TODO:absorb the transpose
                has_activation = False

                # consumer = model.find_consumer(mm_output)
                # if consumer is not None and consumer.op_type == "MultiThreshold":
                if has_activation:
                    pass
                    # odt = thres odt
                    # get thres_params
                else:
                    odt = conv_odt

                # create new nodes
                # NCHW -> NHWC
                inp_trans_node = helper.make_node(
                    "Transpose", [cnv_input], [inp_trans_out], perm=[0, 2, 3, 1]
                )

                # dp conv
                simd = 1
                pe = 1
                assert mh % pe == 0, "Requirement MH divisible by PE is violated."
                assert mw % simd == 0, "Requirement MW divisible by SIMD is violated."
                wmem = mw * mh // (pe * simd)
                assert (
                    mw * mh == wmem * pe * simd
                ), "Requirement (MW * MH) divisiable by(WMEM * PE * SIMD) is violated."

                conv_node_inputs = [inp_trans_out, weight_name]
                # if has_activation:
                #     conv_node_inputs += [thres_params]

                dp_conv_node = helper.make_node(
                    "ConvDoublePacked_Batch",
                    conv_node_inputs,
                    [matmul_out],
                    domain="finn",
                    backend="fpgadataflow",
                    ConvKernelDim=k,  # ("i",True,0),
                    IFMChannels=ifm_ch,  # ("i",True,0),
                    IFMDim=ifm_dim,  # ("i",True,0),
                    OFMChannels=ofm_ch,  # ("i",True,0),
                    OFMDim=ofm_dim,  # ("i", True, 0),
                    Stride=stride,  # ("i",True,0),
                    Padding=pad,  # ("i",True,0),
                    SIMD=simd,  # ("i",True,0),
                    PE=pe,  # ("i",True,0),           #num
                    MW=mw,  # ("i", True, 0),
                    MH=mh,  # ("i", True, 0),
                    inputDataType=idt.name,  # ("s", True, ""),
                    weightDataType=wdt.name,  # ("s", True, ""),
                    outputDataType=odt.name,  # ("s", True, ""),
                    noActivation=0
                    if has_activation
                    else 1,  # ("i", False, 0), #"ActivationType ("i",True,0)
                    numInputVectors=[1, ofm_dim, ofm_dim],  # ("ints", False, [1]),
                    include_file=self.include_file,  # ("s", False, ""),
                )

                # NHWC -> NCHW
                out_trans_node = helper.make_node(
                    "Transpose", [matmul_out], [cnv_output], perm=[0, 3, 1, 2]
                )
                # insert nodes where the conv is to preserve topological ordering
                graph.node.insert(node_ind, inp_trans_node)
                graph.node.insert(node_ind + 1, dp_conv_node)
                graph.node.insert(node_ind + 2, out_trans_node)
                node_ind += 2
                # remove old nodes
                graph.node.remove(n)
                graph_modified = True

        if graph_modified:
            model = model.transform(InferShapes())

        # This transform only requires one pass
        # Also, a second pass would generate unwanted behavior
        return (model, False)
