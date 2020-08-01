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

from onnx import helper, TensorProto
import numpy as np

from finn.transformation import Transformation
from finn.custom_op.registry import getCustomOp
from finn.util.basic import interleave_matrix_outer_dim_from_partitions
from finn.transformation.general import SortGraph


class CreateMemSubsystem(Transformation):
    """Analyze a graph and generate a memory subsystem IP for it, whereby
    the various weight buffers are efficiently packed into block RAMs and
    the memory subsystem serves external weight interface streams."""

    def __init__(self, strategy="inter", ignore_slr=False):
        super().__init__()
        self.strategy = strategy
        self.ignore_slr = ignore_slr

    def apply(self, model):
        # parse the graph, looking for fc layers with BRAM efficiency < 0.8
        # set the fc node to external mem_mode
        # put the parameters of those layers into lists: PE, SIMD, WMEM, weights,
        # create a MemStreamer for those layers
        pe = []
        simd = []
        wmem = []
        values = []
        dataType = []
        outp = []
        # gather info about the floorplan - how many SLRs?
        # we will add one mem subsystem per SLR
        # TODO: analysis pass to gather floorplan info
        if not self.ignore_slr:
            nslr = -1
            for node in model.graph.node:
                node_inst = getCustomOp(node)
                node_slr = node_inst.get_nodeattr("slr")
                if node_slr is not None:
                    nslr = max(node_slr, nslr)
        else:
            nslr = 0
        # scan for partition IDs
        highest_partition_id = 0
        for node in model.graph.node:
            node_inst = getCustomOp(node)
            pid = node_inst.get_nodeattr("partition_id")
            highest_partition_id = max(pid, highest_partition_id)
        # insert a mem subsystem for each SLR
        for slr in range(nslr + 1):
            for node in model.graph.node:
                node_inst = getCustomOp(node)
                if node.op_type != "StreamingFCLayer_Batch":
                    continue
                mem_mode = node_inst.get_nodeattr("mem_mode")
                if mem_mode == "external":
                    continue
                if not self.ignore_slr:
                    node_slr = node_inst.get_nodeattr("slr")
                    if node_slr != slr:
                        continue
                if (
                    node_inst.bram_efficiency_estimation() < 0.8
                ):  # and node_inst.calc_wmem() > 128:
                    pe.append(node_inst.get_nodeattr("PE"))
                    simd.append(node_inst.get_nodeattr("SIMD"))
                    wmem.append(node_inst.calc_wmem())
                    dataType.append(node_inst.get_weight_datatype())
                    tensor = model.get_initializer(node.input[1])
                    # reshape to (PE, WMEM, SIMD) - same as in streamingfclayer
                    tensor = interleave_matrix_outer_dim_from_partitions(
                        tensor.T, pe[-1]
                    )
                    tensor = tensor.reshape(pe[-1], wmem[-1], simd[-1])
                    tensor = np.flip(tensor, axis=-1)
                    # TODO check this^
                    tensor = helper.make_tensor(
                        name="weights_" + node.name,
                        data_type=TensorProto.FLOAT,
                        dims=tensor.shape,
                        vals=tensor.flatten(),
                    )
                    values.append(tensor)
                    outp.append(node.input[1])
                    node_inst.set_nodeattr("mem_mode", "external")

            if len(outp) == 0:
                continue
            # instantiate streamer
            memstreamer_node = helper.make_node(
                "MemStreamer",
                [],
                [x for x in outp],
                domain="finn",
                backend="fpgadataflow",
                nstreams=len(outp),
                PE=pe,
                SIMD=simd,
                WMEM=wmem,
                weights=values,
                dataType=[x.value for x in dataType],
                strategy=self.strategy,
            )
            if not self.ignore_slr:
                getCustomOp(memstreamer_node).set_nodeattr("slr", slr)
            # add a partition ID beyond the existing IDs
            highest_partition_id += 1
            getCustomOp(memstreamer_node).set_nodeattr(
                "partition_id", highest_partition_id
            )
            model.graph.node.insert(0, memstreamer_node)
        model = model.transform(SortGraph())
        return (model, False)
