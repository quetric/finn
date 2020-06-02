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

import finn.util.basic as util
from finn.transformation import Transformation
from toposort import toposort_flatten


class GiveUniqueNodeNames(Transformation):
    """Give unique names to each node in the graph using enumeration."""

    def apply(self, model):
        optype_count = {}
        for n in model.graph.node:
            if n.op_type not in optype_count.keys():
                optype_count[n.op_type] = 0
            n.name = "%s_%d" % (n.op_type, optype_count[n.op_type])
            optype_count[n.op_type] += 1
        # return model_was_changed = False as single iteration is always enough
        return (model, False)


class GiveRandomTensorNames(Transformation):
    """Give random tensor names to all tensors."""

    def apply(self, model):
        names = model.get_all_tensor_names()
        for name in names:
            model.rename_tensor(name, util.random_string())
        # return model_was_changed = False as single iteration is always enough
        return (model, False)


class GiveReadableTensorNames(Transformation):
    """Give more human-readable names to all internal tensors. It's recommended
    to apply give_unique_node_names prior to this transform."""

    def apply(self, model):
        # to ensure we can use rename_tensor safely (without renaming existing
        # tensors) we start by giving random names to all tensors
        model = model.transform(GiveRandomTensorNames())
        graph = model.graph
        for n in graph.node:
            out_num = 0
            for o in n.output:
                model.rename_tensor(o, "%s_out%d" % (n.name, out_num))
                out_num += 1
            init_in_num = 0
            for i in n.input:
                if model.get_initializer(i) is not None:
                    model.rename_tensor(i, "%s_param%d" % (n.name, init_in_num))
                    init_in_num += 1
        # give special names to the main model input and output
        model.rename_tensor(model.graph.input[0].name, "global_in")
        model.rename_tensor(model.graph.output[0].name, "global_out")
        # return model_was_changed = False as single iteration is always enough
        return (model, False)


class GiveUniqueParameterTensors(Transformation):
    """Make every parameter tensor unique. The aim is to avoid affecting
    other nodes apart from the one the system is currently operating on."""

    def apply(self, model):
        graph = model.graph
        graph_modified = False
        seen_parameters = []
        for n in graph.node:
            # copy inputs since they may be modified
            node_inputs_list = [x for x in n.input]
            for input_idx, node_input in enumerate(node_inputs_list):
                # check if it's a parameter
                input_init = model.get_initializer(node_input)
                if input_init is None:
                    # dynamic input
                    continue

                # check if repeated
                if node_input not in seen_parameters:
                    # first occurance
                    seen_parameters += [node_input]
                    continue

                new_param_name = model.make_new_valueinfo_name()

                model.set_initializer(new_param_name, input_init)
                model.set_tensor_datatype(
                    new_param_name, model.get_tensor_datatype(node_input)
                )

                # point node input to new tensor
                n.input[input_idx] = new_param_name

        return (model, graph_modified)


class SortGraph(Transformation):
    """ Returns the model with its node list sorted topologically.
    Any ONNX graph to be executed must have a topologically sorted node list, as dictated
    by the ONNX standard.
    """
    
    # Notes on SortGraph performance:
    #         benchmark in  tests/transformation/test_sort_graph.py
    # 
    #         The algorithm doesn't move initializers so its performance should only depend on
    #         the number of nodes
    # 
    #         Relative order of magnitudes for time per step:
    #             - Gather graph structure:       base
    #             - Sort nodes:                   0.1 of base
    #             - Remove and insert in order :  0.001 of base
    # 
    #     Notes:
    #         Remove nodes and insert them in order:
    #           Probably this is faster than copying initializers and more robust in general

    def apply(self, model):
        # Gather graph structure
        graph_dependencies = {}
        node_list = [
            n for n in model.graph.node
        ]  # I also need the list to remove the nodes
        for node_idx, n in enumerate(node_list):
            node_pred = model.find_direct_predecessors(n)
            if node_pred is None:
                # Will also eliminate nodes that are floating around for some reason
                continue

            node_dependencies = [node_list.index(pred) for pred in node_pred]
            graph_dependencies[node_idx] = set(node_dependencies)

        # Sort nodes
        sorted_node_indexes = toposort_flatten(graph_dependencies)

        # Remove nodes and insert them in order
        # Can't remove nodes before if I want to use model.find_direct_predecessors()
        for n in node_list:
            model.graph.node.remove(n)

        for new_idx, sorted_idx in enumerate(sorted_node_indexes):
            model.graph.node.insert(new_idx, node_list[sorted_idx])

        return model, False


class ConvertSubToAdd(Transformation):
    """Convert subtract-a-constant nodes to add-a-constant nodes."""

    def apply(self, model):
        graph = model.graph
        for n in graph.node:
            if n.op_type == "Sub":
                A = model.get_initializer(n.input[1])
                if A is not None:
                    n.op_type = "Add"
                    model.set_initializer(n.input[1], -A)
        # return model_was_changed = False as single iteration is always enough
        return (model, False)


class ConvertDivToMul(Transformation):
    """Convert divide by constant nodes to multiply by constant nodes."""

    def apply(self, model):
        graph = model.graph
        for n in graph.node:
            if n.op_type == "Div":
                A = model.get_initializer(n.input[1])
                if A is not None:
                    n.op_type = "Mul"
                    model.set_initializer(n.input[1], 1.0 / A)
        # return model_was_changed = False as single iteration is always enough
        return (model, False)
