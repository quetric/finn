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
import subprocess

from finn.transformation import Transformation
from finn.custom_op.registry import getCustomOp


class CreateVitisXO(Transformation):
    """Create a Vitis object file from a stitched FINN ip.

    Outcome if successful: sets the vitis_xo attribute in the ONNX
    ModelProto's metadata_props field with the name of the object file as value.
    The object file can be found under the ip subdirectory.
    """

    def __init__(self, ip_name="finn_design"):
        super().__init__()
        self.ip_name = ip_name

    def apply(self, model):
        vivado_proj_dir = model.get_metadata_prop("vivado_stitch_proj")
        stitched_ip_dir = vivado_proj_dir + "/ip"
        args_string = []
        # NOTE: this assumes the graph is Vitis-compatible: max one axi lite interface
        # developed from instructions in UG1393 (v2019.2) and package_xo documentation
        # package_xo is responsible for generating the kernel xml
        for node in model.graph.node:
            node_inst = getCustomOp(node)
            arg_id = 0
            if node.op_type == "TLastMarker":
                stream_width = node_inst.get_nodeattr("StreamWidth")
                # add a stream input or output port, based on direction
                if node_inst.get_nodeattr("Direction") == "in":
                    args_string.append(
                        "{in:4:%s:s_axis:0x0:0x0:ap_uint&lt;%s>:0}"
                        % (str(arg_id), str(stream_width))
                    )
                else:
                    args_string.append(
                        "{out:4:%s:m_axis:0x0:0x0:ap_uint&lt;%s>:0}"
                        % (str(arg_id), str(stream_width))
                    )
                arg_id += 1
                # add a axilite port if dynamic
                # add a count parameter if dynamic
                if node_inst.get_nodeattr("DynIters") == "true":
                    args_string.append(
                        "{numReps:0:%s:s_axi_control:0x4:0x10:uint:0}" % str(arg_id)
                    )
                    arg_id += 1
            elif node.op_type == "IODMA":
                port_width = node_inst.get_nodeattr("intfWidth")
                # add an address parameter
                # add a count parameter
                args_string.append(
                    "{addr:1:%s:m_axi_gmem0:0x8:0x10:ap_uint&lt;%s>*:0}"
                    % (str(arg_id), str(port_width))
                )
                arg_id += 1
                args_string.append(
                    "{numReps:0:%s:s_axi_control:0x4:0x1C:uint:0}" % str(arg_id)
                )
                arg_id += 1

        # save kernel xml then run package_xo
        xo_name = self.ip_name + ".xo"
        xo_path = vivado_proj_dir + "/" + xo_name
        model.set_metadata_prop("vitis_xo", xo_path)

        # generate the package_xo command in a tcl script
        package_xo_string = (
            "package_xo -force -xo_path %s -kernel_name %s -ip_directory %s"
            % (xo_path, self.ip_name, stitched_ip_dir)
        )
        for arg in args_string:
            package_xo_string += " -kernel_xml_args " + arg
        with open(vivado_proj_dir + "/gen_xo.tcl", "w") as f:
            f.write(package_xo_string)

        # create a shell script and call Vivado
        package_xo_sh = vivado_proj_dir + "/gen_xo.sh"
        working_dir = os.environ["PWD"]
        with open(package_xo_sh, "w") as f:
            f.write("#!/bin/bash \n")
            f.write("cd {}\n".format(vivado_proj_dir))
            f.write("vivado -mode batch -source gen_xo.tcl\n")
            f.write("cd {}\n".format(working_dir))
        bash_command = ["bash", package_xo_sh]
        process_compile = subprocess.Popen(bash_command, stdout=subprocess.PIPE)
        process_compile.communicate()
        return (model, False)
