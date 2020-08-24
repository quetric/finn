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

from finn.custom_op.registry import getCustomOp
from finn.transformation import Transformation
from finn.util.basic import get_by_name
import warnings


class Floorplan(Transformation):
    """Perform Floorplanning of the dataflow design. Separate DMAs into their own
    partitions IDs, and TODO: split the design into sections of defined size"""

    def __init__(self, floorplan=None):
        super().__init__()
        self.user_floorplan = floorplan

    def apply(self, model):
        # we currently assume that all dataflow nodes belonging to the same partition
        # are connected to each other and there is a single input/output to/from each.
        all_nodes = list(model.graph.node)
        df_nodes = list(
            filter(lambda x: get_by_name(x.attribute, "backend") is not None, all_nodes)
        )

        # if floorplan provided,
        if self.user_floorplan is not None:
            unassigned_nodes = 0
            for node in df_nodes:
                node_inst = getCustomOp(node)
                # assign SLR
                if node.name in self.user_floorplan:
                    node_slr = self.user_floorplan[node.name]["slr"]
                elif node.op_type == "StreamingDataWidthConverter_Batch":
                    # optimize for possible SLR crossing
                    in_width = node_inst.get_nodeattr("inWidth")
                    out_width = node_inst.get_nodeattr("outWidth")
                    if in_width > out_width:
                        # use consumer config (node not yet configured)
                        consumer = model.find_consumer(node.output[0])
                        if consumer.name in self.user_floorplan:
                            node_slr = self.user_floorplan[consumer.name]["slr"]
                        elif "default" in self.user_floorplan:
                            node_slr = self.user_floorplan["default"]["slr"]
                        else:
                            unassigned_nodes += 1
                            node_slr = -1  # no pblock assignment in linking
                    else:
                        # use producer config (node already configured)
                        producer = model.find_producer(node.input[0])
                        prod_inst = getCustomOp(producer)
                        node_slr = prod_inst.get_nodeattr("slr")
                elif "default" in self.user_floorplan:
                    node_slr = self.user_floorplan["default"]["slr"]
                else:
                    unassigned_nodes += 1
                    node_slr = -1  # no pblock assignment in linking
                node_inst.set_nodeattr("slr", node_slr)
                # assign memory port
                if node.op_type == "IODMA":
                    if node.name in self.user_floorplan:
                        mem_port = self.user_floorplan[node.name]["mem_port"]
                    elif "default" in self.user_floorplan:
                        mem_port = self.user_floorplan["default"]["mem_port"]
                    node_inst.set_nodeattr("mem_port", mem_port)

            if unassigned_nodes > 0:
                warnings.warn(
                    str(unassigned_nodes)
                    + " nodes have no entry in the provided floorplan "
                    + "and no default value was set"
                )

        else:
            # partition id generation
            partition_cnt = 0

            # Assign IODMAs to their own partitions
            dma_nodes = list(filter(lambda x: x.op_type == "IODMA", df_nodes))
            for node in dma_nodes:
                node_inst = getCustomOp(node)
                node_inst.set_nodeattr("partition_id", partition_cnt)
                partition_cnt += 1

            non_dma_nodes = list(filter(lambda x: x not in dma_nodes, df_nodes))
            dyn_tlastmarker_nodes = list(
                filter(
                    lambda x: x.op_type == "TLastMarker"
                    and getCustomOp(x).get_nodeattr("DynIters") == "true",
                    non_dma_nodes,
                )
            )
            non_dma_nodes = list(
                filter(lambda x: x not in dyn_tlastmarker_nodes, non_dma_nodes)
            )

            for node in dyn_tlastmarker_nodes:
                node_inst = getCustomOp(node)
                node_inst.set_nodeattr("partition_id", partition_cnt)
                partition_cnt += 1

            for node in non_dma_nodes:
                pre_node = model.find_producer(node.input[0])
                node_inst = getCustomOp(node)
                if pre_node not in non_dma_nodes:
                    # input node
                    node_inst.set_nodeattr("partition_id", partition_cnt)
                    partition_cnt += 1
                    continue
                elif not (
                    node.op_type == "StreamingFCLayer_Batch"
                    and node_inst.get_nodeattr("mem_mode") is not None
                    and node_inst.get_nodeattr("mem_mode") == "external"
                ):
                    pre_nodes = model.find_direct_predecessors(node)
                else:
                    pre_nodes = [pre_node]

                node_slr = node_inst.get_nodeattr("slr")
                for pre_node in pre_nodes:
                    pre_inst = getCustomOp(pre_node)
                    pre_slr = pre_inst.get_nodeattr("slr")
                    if node_slr == pre_slr:
                        partition_id = pre_inst.get_nodeattr("partition_id")
                        node_inst.set_nodeattr("partition_id", partition_id)
                        break
                else:
                    # no matching, new partition
                    node_inst.set_nodeattr("partition_id", partition_cnt)
                    partition_cnt += 1

        return (model, False)
