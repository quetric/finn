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
import json


class Floorplan(Transformation):
    """Perform Floorplanning of the dataflow design. Separate DMAs into their own
    partitions IDs, and TODO: split the design into sections of defined size"""

    def __init__(self, limits=None, floorplan_file=None):
        super().__init__()
        self.resource_limits = limits

        # dict key = node.name, value = dict ( key="slr" ,..
        #                                            (future) key="device_id" )
        if floorplan_file is None:
            self.user_floorplan = None
        else:
            with open(floorplan_file, "r") as f:
                self.user_floorplan = json.load(f)

    def apply(self, model):
        target_partition_id = 0
        # we currently assume that all dataflow nodes belonging to the same partition
        # are connected to each other and there is a single input/output to/from each.
        all_nodes = list(model.graph.node)
        df_nodes = list(
            filter(lambda x: get_by_name(x.attribute, "backend") is not None, all_nodes)
        )
        if self.user_floorplan is not None:
            no_config = 0
            for node in df_nodes:
                node_inst = getCustomOp(node)
                if node.name in self.user_floorplan:
                    node_slr = self.user_floorplan[node.name]["slr"]
                elif "default" in self.user_floorplan:
                    node_slr = self.user_floorplan["default"]["slr"]
                else:
                    no_config += 1
                    node_slr = -1  # no pblock assignment in linking

                node_inst.set_nodeattr("slr", node_slr)

            if no_config > 0:
                warnings.warn(
                    str(no_config)
                    + " nodes have no entry in the provided floorplan "
                    + "and no default value was set"
                )

            # partition id gen: first pass
            partition_cnt = 0
            for node in df_nodes:
                pre_node = model.find_producer(node.input[0])
                if pre_node is None:
                    # input node
                    node_inst = getCustomOp(node)
                    node_inst.set_nodeattr("partition_id", partition_cnt)
                    partition_cnt += 1
                    continue
                elif not (
                    node.op_type == "StreamingFCLayer_Batch"
                    and get_by_name(node.attribute, "mem_mode") is not None
                    and get_by_name(node.attribute, "mem_mode").s.decode("UTF-8")
                    == "external"
                ):
                    pre_nodes = model.find_direct_predecessors(node)
                else:
                    # StreamingFCLayer_Batch with external doesn't take the same
                    # partition weight generator
                    pre_nodes = [pre_node]

                node_inst = getCustomOp(node)
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

            # TODO: merge contiguous partitions in the same slr
            # (ex. required when a residual block has two SLR crossings to the same SRL)
            # not currently supported by VitisLink transform

        else:
            dma_nodes = list(filter(lambda x: x.op_type == "IODMA", df_nodes))

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

            for node in dma_nodes:
                node_inst = getCustomOp(node)
                node_inst.set_nodeattr("partition_id", target_partition_id)
                target_partition_id += 1

            for node in dyn_tlastmarker_nodes:
                node_inst = getCustomOp(node)
                node_inst.set_nodeattr("partition_id", target_partition_id)
                target_partition_id += 1

            for node in non_dma_nodes:
                # TODO: implement proper floorplanning; for now just a single partition
                node_inst = getCustomOp(node)
                node_inst.set_nodeattr("partition_id", target_partition_id)

            # create model.link_configuration ["slr"] mapping
        return (model, False)
