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
import copy
import numpy as np
import math

from finn.core.datatype import DataType
from finn.custom_op.fpgadataflow import HLSCustomOp
from finn.util.data_packing import pack_innermost_dim_as_hex_string
from finn.util.basic import roundup_to_integer_multiple

# This block implements a packed memory subsystem connected to the weight inputs
# of 2 or more computational blocks. it gets parameters resulting from a
# mempack analysis pass, and weight_shapes analysis pass
# the point is to generate using memstreamers the IP that serves all target
# layers with the specified packing solution

memclkrst_template = """
create_bd_cell -type ip -vlnv xilinx.com:ip:clk_wiz:6.0 memclk
connect_bd_net [get_bd_ports ap_clk] [get_bd_pins memclk/clk_in1]
set_property -dict [list CONFIG.PRIM_IN_FREQ.VALUE_SRC USER] [get_bd_cells memclk]
set_property -dict [list CONFIG.PRIM_IN_FREQ $computefreqmhz] [get_bd_cells memclk]
set_property -dict [list CONFIG.USE_LOCKED {false} CONFIG.USE_RESET {false}] [get_bd_cells memclk]
set_property -dict [list CONFIG.OVERRIDE_MMCM {true} CONFIG.MMCM_CLKFBOUT_MULT_F {$CLKMULT$}] [get_bd_cells memclk]
set_property -dict [list CONFIG.MMCM_CLKOUT0_DIVIDE_F {$CLKDIV$}] [get_bd_cells memclk]
create_bd_cell -type ip -vlnv xilinx.com:ip:proc_sys_reset:5.0 psrst_memclk
connect_bd_net [get_bd_ports ap_rst_n] [get_bd_pins psrst_memclk/ext_reset_in]
connect_bd_net [get_bd_pins memclk/clk_out1] [get_bd_pins psrst_memclk/slowest_sync_clk]
save_bd_design
"""


class MemStreamer(HLSCustomOp):
    """Class that implements a packed memory streamer. Takes
    multiple tensors as attributes, and delivers the values in the tensors on
    its outputs."""

    def __init__(self, onnx_node):
        super().__init__(onnx_node)
        self.shapes = self.merge_pe_simd(self.get_shapes_dict())
        self.outputs = {}
        for key in self.shapes.keys():
            self.outputs[key] = "out_" + key

    def get_nodeattr_types(self):
        my_attrs = {
            # Number of output streams
            "nstreams": ("i", True, 0),
            "strategy": ("s", False, "inter"),
            "max_height": ("i", False, 4),
            # parallelism parameters
            "PE": ("ints", True, []),
            "SIMD": ("ints", True, []),
            "WMEM": ("ints", True, []),
            # Data types
            "dataType": ("ints", True, []),
            # Data for each output stream, in shape (pe, wmem, simd)
            "weights": ("tensors", True, []),
        }
        my_attrs.update(super().get_nodeattr_types())
        return my_attrs

    def make_shape_compatible_op(self, model):
        pass

    def infer_node_datatype(self, model):
        pass

    def verify_node(self):
        info_messages = []
        # verify that "domain" is set to "finn"
        domain_value = self.onnx_node.domain
        if domain_value == "finn":
            info_messages.append("Attribute domain is set correctly")
        else:
            info_messages.append('Attribute domain should be set to "finn"')

        # verify that "backend" is set to "fpgadataflow"
        backend_value = self.get_nodeattr("backend")
        if backend_value == "fpgadataflow":
            info_messages.append("Attribute backend is set correctly")
        else:
            info_messages.append('Attribute backend should be set to "fpgadataflow"')

        strategy = self.get_nodeattr("strategy")
        if strategy != "intra" and strategy != "inter":
            info_messages.append(
                'Attribute strategy should be set to "inter" or "intra"'
            )

        # TODO:
        # verify that lengths of PE, SIMD, dataType, WMEM and weights are the same
        # verify that the size of weights corresponds to PE, SIMD, WMEM
        # verify that the weight values fit in dataType

        return info_messages

    def get_shapes_dict(self):
        """Assembles a dictionary as expected by the mempack analysis pass."""
        nstreams = self.get_nodeattr("nstreams")
        ret = {}
        for i in range(nstreams):
            pe = self.get_nodeattr("PE")
            simd = self.get_nodeattr("SIMD")
            wmem = self.get_nodeattr("WMEM")
            dt = self.get_nodeattr("dataType")
            values = self.get_nodeattr("weights")
            ret["layer" + str(i)] = {
                "PE": pe[i],
                "SIMD": simd[i],
                "WMEM": wmem[i],
                "DataType": DataType(dt[i]),
                "Values": np.array(values[i].float_data).reshape(
                    pe[i], wmem[i], simd[i]
                ),
            }
        return ret

    def merge_pe_simd(self, shapes):
        """Trades off PE and SIMD dimensions to take advantage of wider memories."""
        ret = shapes
        for layer in shapes:
            pe = shapes[layer]["PE"]
            simd = shapes[layer]["SIMD"]
            wmem = shapes[layer]["WMEM"]
            values = shapes[layer]["Values"]
            dataw = shapes[layer]["DataType"].bitwidth()
            max_width = pe * simd * dataw
            # pack narrow buffers into a single PE, for efficiency
            # TODO: take depth into consideration
            if max_width < 18:
                scaled_pe = 1
                factor = pe
            else:
                # find minimum value of PE where we get some
                # benefit from using all 18 bits of a BRAM18
                scaled_pe = 1
                while scaled_pe < pe:
                    if pe % scaled_pe == 0:
                        factor = int(pe / scaled_pe)
                        width = scaled_pe * simd * dataw
                        if math.ceil(width / 18) < math.ceil(width / 16):
                            break
                    scaled_pe += 1
            if scaled_pe < pe:
                assert values.shape == (pe, wmem, simd), "Unexpected weight array shape"
                # TODO check this results in correct data layout
                values = (
                    values.transpose(1, 0, 2)
                    .reshape(wmem, scaled_pe, simd * factor)
                    .transpose(1, 0, 2)
                )
                ret[layer]["Values"] = values
                ret[layer]["PE"] = scaled_pe
                ret[layer]["SIMD"] = factor * simd
        return ret

    def ipgen_singlenode_code(self):
        """Builds the bash script for ip generation."""
        node = self.onnx_node
        code_gen_dir = self.get_nodeattr("code_gen_dir_ipgen")

        make_ip_sh = code_gen_dir + "/make_ip.sh"
        working_dir = os.environ["PWD"]
        with open(make_ip_sh, "w") as f:
            f.write("#!/bin/bash \n")
            f.write("cd {}\n".format(code_gen_dir))
            f.write("vivado -mode batch -source bd_assembly.tcl\n")
            f.write("cd {}\n".format(working_dir))
        bash_command = ["bash", make_ip_sh]
        process_compile = subprocess.Popen(bash_command, stdout=subprocess.PIPE)
        process_compile.communicate()

        # finalize
        self.set_nodeattr("ipgen_path", code_gen_dir + "/mem_subsystem")
        self.set_nodeattr("ip_path", code_gen_dir + "/mem_subsystem/" + node.name)
        vlnv = "xilinx.com:user:%s:1.0" % node.name
        self.set_nodeattr("ip_vlnv", vlnv)

    def code_generation_ipgen(self, model, fpgapart, clk):
        """Generates tcl script for ip generation."""
        # generate top cpp file for ip generation
        path = self.get_nodeattr("code_gen_dir_ipgen")
        # generate packing solution for assigned buffers (by calling mempack)
        # import packing function here to avoid circular dependency
        from finn.analysis.mempacking import pack_memory_shapes, packDefaultConfig

        config = packDefaultConfig()
        strategy = self.get_nodeattr("strategy")
        if strategy == "intra":
            config.enableInter = False
        elif strategy == "inter":
            config.enableInter = True
        else:
            raise Exception("Strategy parameter set incorrectly")
        config.gold = 0
        config.thresh_min = 0
        stack_height = self.get_nodeattr("max_height")
        config.max_stack_height = stack_height
        self.packed_shapes = pack_memory_shapes(self.shapes, args=config)
        self.build_connectivity_spec()
        self.build_bin_spec()
        self.gen_artefacts(path, fpgapart, clk)
        self.generate_params(path)

    def generate_params(self, tcl_folder):
        """Generate readmemh include files for RTL streamers."""
        for bin in self.bin_spec:
            # identify bin width (max of all widths), depth (sum of all depths)
            bin_width = self.bin_spec[bin]["width"]
            bin_data = []
            nstreams = len(self.bin_spec[bin]["streams"])
            # get data
            for stream in self.bin_spec[bin]["streams"]:
                # get a PE memory and pack it into hex list
                pe_weight_tensor = stream["data"]
                idimbits = roundup_to_integer_multiple(bin_width, 4)
                bin_data += list(
                    pack_innermost_dim_as_hex_string(
                        pe_weight_tensor, stream["dtype"], idimbits, prefix=""
                    )
                )

            # create dedicated folder for bin artefacts
            directory = tcl_folder + "/" + bin
            if not os.path.exists(directory):
                os.makedirs(directory)

            # write these to files, one per bin
            if nstreams <= 2:
                # write single readmemh file for 1 or 2 streams
                with open(directory + "/memblock_0.dat", "w") as f:
                    for item in bin_data:
                        f.write("%s\n" % item)
            else:
                # write data to files, in 1k blocks
                block_idx = 0
                for line_idx in range(len(bin_data)):
                    if (line_idx % 1024) == 0:
                        f = open(
                            directory + "/memblock_" + str(block_idx) + ".dat", "w"
                        )
                    f.write("%s\n" % bin_data[line_idx])
                    if (line_idx % 1024) == 1023:
                        f.close()
                        block_idx += 1

    def blackboxfunction(self):
        pass

    def dataoutstrm(self):
        pass

    def defines(self):
        pass

    def docompute(self):
        pass

    def get_number_output_values(self):
        pass

    def global_includes(self):
        pass

    def pragmas(self):
        pass

    def read_npy_data(self):
        pass

    def save_as_npy(self):
        pass

    def strm_decl(self):
        pass

    def build_connectivity_spec(self):
        """ construct a connectivity spec of streams """
        # each stream has a source, waypoints, a destination, and attributes
        # the source is defined by the bin number and the stream index in the bin
        # waypoints is a list of dicts describing IP instances
        # that the data goes through, e.g. FIFOs
        #    -for each waypoint, specify the name of the IP
        #     and the name of the input/output ports
        #    -e.g. [{'instance':'fifo0', 'in':'s_axis', 'out':'m_axis'},
        #           {'instance':'comb0', 'in':'s_axis2', 'out':'m_axis'}]
        #    -waypoints is initially empty but is filled by each
        #     assembly pass until src is connected to dst
        # the destination is defined by the layer name and the PE index in the layer
        # the stream attributes are:
        #    -width at source and depth at source (number of words before repeating)
        #    -width at destination and depth at destination
        #     (number of words before repeating)
        #    -substream index - determines order in case this needs to be combined
        #     with another stream to serve a PE
        #    -width and depth at source don't need to match width and depth at
        #     destination but must be convertible to the width and depth at destination
        #        -e.g two 256x64b substreams can be concatenated and reshaped into
        #         a 512x64b stream using substream indices
        #        -e.g two 256x64b substreams can be concatenated 256x128b stream
        #         using substream indices
        #        -if stream depth requires conversion, it is achieved through
        #         interleaving the source substreams
        solution = self.packed_shapes
        net = self.shapes
        self.connectivity = {}
        pe_index = dict.fromkeys(net.keys(), 0)
        stream_idx = 0
        for bin in solution:
            bin_strm_idx = 0
            for layer in solution[bin]:
                sname = "stream" + str(stream_idx)
                self.connectivity[sname] = {}
                self.connectivity[sname]["dst"] = {
                    "name": layer,
                    "pe_idx": pe_index[layer],
                }
                self.connectivity[sname]["src"] = {"name": bin, "s_idx": bin_strm_idx}
                self.connectivity[sname]["waypoints"] = []
                self.connectivity[sname]["attributes"] = {}
                self.connectivity[sname]["attributes"]["dtype"] = net[layer]["DataType"]
                self.connectivity[sname]["attributes"]["dst_w"] = (
                    net[layer]["SIMD"] * net[layer]["DataType"].bitwidth()
                )
                self.connectivity[sname]["attributes"]["dst_h"] = net[layer]["WMEM"]
                self.connectivity[sname]["attributes"]["src_w"] = self.connectivity[
                    sname
                ]["attributes"]["dst_w"]
                self.connectivity[sname]["attributes"]["src_h"] = self.connectivity[
                    sname
                ]["attributes"]["dst_h"]
                self.connectivity[sname]["attributes"]["substrm_idx"] = None
                self.connectivity[sname]["data"] = copy.deepcopy(
                    self.shapes[layer]["Values"][pe_index[layer]]
                )
                pe_index[layer] += 1
                bin_strm_idx += 1
                stream_idx += 1
            # create an even number of streams per bin by
            # splitting last stream into half-height substreams
            if bin_strm_idx % 2 != 0 and bin_strm_idx > 2:
                new_sname = "stream" + str(stream_idx)
                self.connectivity[new_sname] = copy.deepcopy(self.connectivity[sname])
                self.connectivity[sname]["attributes"]["substrm_idx"] = 0
                self.connectivity[new_sname]["attributes"]["substrm_idx"] = 1
                self.connectivity[sname]["attributes"]["src_h"] //= 2
                self.connectivity[new_sname]["attributes"]["src_h"] //= 2
                self.connectivity[new_sname]["src"]["s_idx"] = bin_strm_idx
                self.connectivity[sname]["data"] = copy.deepcopy(
                    self.shapes[layer]["Values"][pe_index[layer] - 1][::2]
                )
                self.connectivity[new_sname]["data"] = copy.deepcopy(
                    self.shapes[layer]["Values"][pe_index[layer] - 1][1::2]
                )
                stream_idx += 1

    def build_bin_spec(self):
        """Generate spec of bin - width and height,
        parameters of each stream - from connectivity spec"""
        # we need for each bin
        #    -the number of streams
        #    -overall width and height of bin
        #    -width, height and offset of each stream
        self.bin_spec = {}
        # first parse streams and set width and height of each stream
        for stream in self.connectivity:
            bin = self.connectivity[stream]["src"]["name"]
            if bin not in self.bin_spec.keys():
                self.bin_spec[bin] = {"streams": [], "width": 0, "height": 0}
            s_width = self.connectivity[stream]["attributes"]["src_w"]
            s_height = self.connectivity[stream]["attributes"]["src_h"]
            s_data = self.connectivity[stream]["data"]
            s_dtype = self.connectivity[stream]["attributes"]["dtype"]
            self.bin_spec[bin]["streams"].append(
                {
                    "name": stream,
                    "width": s_width,
                    "height": s_height,
                    "offset": 0,
                    "data": s_data,
                    "dtype": s_dtype,
                }
            )
        # go through streams and set offset plus bin overall width and height
        for bin in self.bin_spec:
            offset = 0
            width = 0
            for i in range(len(self.bin_spec[bin]["streams"])):
                stream = self.bin_spec[bin]["streams"][i]
                stream["offset"] = offset
                offset += stream["height"]
                width = max(width, stream["width"])
            self.bin_spec[bin]["width"] = width
            self.bin_spec[bin]["height"] = offset

    def gen_artefacts(self, tcl_folder, fpgapart, clk_ns):
        """Generate TCL file which assembles the memory subsystem from
        RTL streamer components and AXI infrastructure IP"""
        tcl = []
        # define some essential stuff in the BD assembly script
        tcl.append("set computefreqmhz " + str(math.floor(1000 / clk_ns)) + "\n")
        tcl.append(
            "create_project mem_subsystem ./mem_subsystem -part " + fpgapart + "\n"
        )
        tcl.append("create_bd_design " + self.onnx_node.name + "\n")
        tcl.append(
            "create_bd_port -dir I -type clk "
            + "-freq_hz [expr $computefreqmhz*1000000] ap_clk\n"
        )
        tcl.append("create_bd_port -dir I -type rst ap_rst_n\n")
        tcl.append(
            "set_property ip_repo_paths "
            + "/workspace/finn/finn-rtllib/ [current_project]\n"
        )
        tcl.append("update_ip_catalog\n")

        # Step 0: create memory clocks and resets
        max_streams_per_bin = 0
        for bin in self.packed_shapes:
            max_streams_per_bin = max(max_streams_per_bin, len(self.packed_shapes[bin]))
        if max_streams_per_bin > 2:
            # if any bin produces more than 2 streams, we need a high freq clock
            mmcm_f_mult = (
                math.ceil(float(clk_ns) / 0.125) * 0.125
            )  # we want 1000 MHz for VCO
            mmcm_f_div = (
                math.floor(((2 * mmcm_f_mult) / max_streams_per_bin) / 0.125) * 0.125
            )
            tclcode = memclkrst_template
            tclcode = tclcode.replace("$CLKMULT$", ("%.2f" % mmcm_f_mult))
            tclcode = tclcode.replace("$CLKDIV$", ("%.2f" % mmcm_f_div))
            tcl.append(tclcode)

        # Step 1: instantiate RTL streamers for bins
        for bin in self.bin_spec:
            nstreams = len(self.bin_spec[bin]["streams"])
            bin_depth = self.bin_spec[bin]["height"]
            bin_width = self.bin_spec[bin]["width"]
            tcl.append(
                "create_bd_cell -type ip -vlnv xilinx.com:user:memstream:1.0 "
                + bin
                + "\n"
            )
            tcl.append(
                "set_property -dict [list CONFIG.NSTREAMS "
                + str(nstreams)
                + "] [get_bd_cells "
                + bin
                + "]\n"
            )
            tcl.append(
                "set_property -dict [list CONFIG.MEM_DEPTH "
                + str(bin_depth)
                + "] [get_bd_cells "
                + bin
                + "]\n"
            )
            tcl.append(
                "set_property -dict [list CONFIG.MEM_WIDTH "
                + str(bin_width)
                + "] [get_bd_cells "
                + bin
                + "]\n"
            )
            tcl.append(
                "set_property -dict [list CONFIG.MEM_INIT "
                + tcl_folder
                + "/"
                + bin
                + "/] [get_bd_cells "
                + bin
                + "]\n"
            )
            if nstreams > 2:
                # bin depth higher than number of ports, use higher freq. clock
                tcl.append(
                    "connect_bd_net [get_bd_pins memclk/clk_out1] [get_bd_pins "
                    + bin
                    + "/aclk]\n"
                )
                tcl.append(
                    "connect_bd_net "
                    + "[get_bd_pins psrst_memclk/peripheral_aresetn] [get_bd_pins "
                    + bin
                    + "/aresetn]\n"
                )
            else:
                tcl.append(
                    "connect_bd_net [get_bd_ports ap_clk] [get_bd_pins "
                    + bin
                    + "/aclk]\n"
                )
                tcl.append(
                    "connect_bd_net [get_bd_ports ap_rst_n] [get_bd_pins "
                    + bin
                    + "/aresetn]\n"
                )
            for i in range(nstreams):
                stream = self.bin_spec[bin]["streams"][i]
                s_width = stream["width"]
                assert (
                    s_width % 8 == 0 and s_width / 8 >= 1
                ), "Stream widths must be multiples of 1 byte"
                s_height = stream["height"]
                s_offset = stream["offset"]
                tcl.append(
                    "set_property -dict [list CONFIG.STRM"
                    + str(i)
                    + "_WIDTH "
                    + str(s_width)
                    + "] [get_bd_cells "
                    + bin
                    + "]\n"
                )
                tcl.append(
                    "set_property -dict [list CONFIG.STRM"
                    + str(i)
                    + "_DEPTH "
                    + str(s_height)
                    + "] [get_bd_cells "
                    + bin
                    + "]\n"
                )
                tcl.append(
                    "set_property -dict [list CONFIG.STRM"
                    + str(i)
                    + "_OFFSET "
                    + str(s_offset)
                    + "] [get_bd_cells "
                    + bin
                    + "]\n"
                )
        tcl.append("save_bd_design\n")

        # Step 2: instantiate FIFOs (if needed) for each stream
        # (make sure to connect afull back to the bin)
        fifo_idx = 0
        for stream in self.connectivity:
            bin = self.connectivity[stream]["src"]["name"]
            bin_out_idx = self.connectivity[stream]["src"]["s_idx"]
            if len(self.bin_spec[bin]["streams"]) <= 2:
                # no FIFO if 1 or 2 streams, set waypoint to bin output
                self.connectivity[stream]["waypoints"].append(
                    {"instance": bin, "port": "m_axis_" + str(bin_out_idx)}
                )
                continue
            fifo = "fifo" + str(fifo_idx)
            tcl.append(
                "create_bd_cell -type ip -vlnv xilinx.com:ip:axis_data_fifo:2.0 "
                + fifo
                + "\n"
            )
            tcl.append(
                "set_property -dict [list "
                + "CONFIG.FIFO_DEPTH {32} CONFIG.HAS_PROG_FULL {1} "
                + "CONFIG.PROG_FULL_THRESH {16} "
                + "CONFIG.FIFO_MEMORY_TYPE {distributed}] "
                + "[get_bd_cells "
                + fifo
                + "]\n"
            )
            tcl.append(
                "set_property -dict [list CONFIG.IS_ACLK_ASYNC {1}] [get_bd_cells "
                + fifo
                + "]\n"
            )
            tcl.append(
                "connect_bd_net [get_bd_pins memclk/clk_out1] [get_bd_pins "
                + fifo
                + "/s_axis_aclk]\n"
            )
            tcl.append(
                "connect_bd_net [get_bd_ports ap_clk] [get_bd_pins "
                + fifo
                + "/m_axis_aclk]\n"
            )
            tcl.append(
                "connect_bd_net "
                + "[get_bd_pins psrst_memclk/peripheral_aresetn] [get_bd_pins "
                + fifo
                + "/s_axis_aresetn]\n"
            )
            tcl.append(
                "connect_bd_intf_net [get_bd_intf_pins "
                + bin
                + "/m_axis_"
                + str(bin_out_idx)
                + "] [get_bd_intf_pins "
                + fifo
                + "/S_AXIS]\n"
            )
            tcl.append(
                "connect_bd_net [get_bd_pins "
                + bin
                + "/m_axis_"
                + str(bin_out_idx)
                + "_afull] [get_bd_pins "
                + fifo
                + "/prog_full]\n"
            )
            self.connectivity[stream]["waypoints"].append(
                {"instance": fifo, "port": "M_AXIS"}
            )
            fifo_idx += 1
        tcl.append("save_bd_design\n")

        # Step 3: merge substreams
        merged_substreams = []
        for ref_stream in self.connectivity:
            # for each stream, detect substreams
            # skip of no substreams
            if self.connectivity[ref_stream]["attributes"]["substrm_idx"] is None:
                continue
            # skip if this layer/PE combination has already been solved
            if ref_stream in merged_substreams:
                continue
            substreams = {
                k: v
                for k, v in self.connectivity.items()
                if v["dst"] == self.connectivity[ref_stream]["dst"]
            }
            if len(substreams) > 1:
                # if multiple substreams exist, instantiate combiner and DWC for them
                # combiner for up to 16 substreams (we only really need 2)
                comb = (
                    "sc_"
                    + self.connectivity[ref_stream]["dst"]["name"]
                    + "_"
                    + str(self.connectivity[ref_stream]["dst"]["pe_idx"])
                )
                tcl.append(
                    "create_bd_cell -type ip -vlnv xilinx.com:ip:axis_combiner:1.1 "
                    + comb
                    + "\n"
                )
                tcl.append(
                    "set_property -dict [list CONFIG.NUM_SI "
                    + str(len(substreams))
                    + "] [get_bd_cells "
                    + comb
                    + "]\n"
                )
                tcl.append(
                    "connect_bd_net [get_bd_ports ap_clk] [get_bd_pins "
                    + comb
                    + "/aclk]\n"
                )
                tcl.append(
                    "connect_bd_net [get_bd_ports ap_rst_n] [get_bd_pins "
                    + comb
                    + "/aresetn]\n"
                )
                for substream in substreams:
                    source_port = (
                        self.connectivity[substream]["waypoints"][-1]["instance"]
                        + "/"
                        + self.connectivity[substream]["waypoints"][-1]["port"]
                    )
                    tcl.append(
                        "connect_bd_intf_net [get_bd_intf_pins "
                        + source_port
                        + "] [get_bd_intf_pins "
                        + comb
                        + "/S"
                        + (
                            "%02d"
                            % self.connectivity[substream]["attributes"]["substrm_idx"]
                        )
                        + "_AXIS]\n"
                    )
                    self.connectivity[substream]["waypoints"].append(
                        {"instance": comb, "port": "M_AXIS"}
                    )
                # output of combiner to DWC
                dwc = (
                    "dwc_"
                    + self.connectivity[ref_stream]["dst"]["name"]
                    + "_"
                    + str(self.connectivity[ref_stream]["dst"]["pe_idx"])
                )
                tcl.append(
                    "create_bd_cell -type ip -vlnv xilinx.com:ip:axis_dwidth_converter:1.1 "
                    + dwc
                    + "\n"
                )
                tcl.append(
                    "set_property -dict [list CONFIG.M_TDATA_NUM_BYTES "
                    + str(
                        int(
                            math.ceil(
                                self.connectivity[ref_stream]["attributes"]["dst_w"] / 8
                            )
                        )
                    )
                    + "] [get_bd_cells "
                    + dwc
                    + "]\n"
                )
                tcl.append(
                    "connect_bd_net [get_bd_ports ap_clk] [get_bd_pins "
                    + dwc
                    + "/aclk]\n"
                )
                tcl.append(
                    "connect_bd_net [get_bd_ports ap_rst_n] [get_bd_pins "
                    + dwc
                    + "/aresetn]\n"
                )
                tcl.append(
                    "connect_bd_intf_net [get_bd_intf_pins "
                    + comb
                    + "/M_AXIS] [get_bd_intf_pins "
                    + dwc
                    + "/S_AXIS]\n"
                )
                for substream in substreams:
                    self.connectivity[substream]["waypoints"].append(
                        {"instance": dwc, "port": "M_AXIS"}
                    )
                    merged_substreams.append(substream)
        tcl.append("save_bd_design\n")

        # Step 4: instantiate combiner(s) (if needed)
        # inspect network to determine the number of PEs for each layer
        # which gives us the number of inputs to the combiner tree
        for layer in self.shapes:
            num_pe = self.shapes[layer]["PE"]
            assert num_pe <= 256, "Maximum supported value of PE is 256"
            if num_pe == 1:
                continue
            ncombiners_l1 = math.ceil(num_pe / 16)
            ninputs = num_pe
            for i in range(ncombiners_l1):
                tcl.append(
                    "create_bd_cell -type ip -vlnv xilinx.com:ip:axis_combiner:1.1 "
                    + layer
                    + "_"
                    + str(i)
                    + "\n"
                )
                tcl.append(
                    "set_property -dict [list CONFIG.NUM_SI "
                    + str(min(ninputs, 16))
                    + "] [get_bd_cells "
                    + layer
                    + "_"
                    + str(i)
                    + "]\n"
                )
                tcl.append(
                    "connect_bd_net [get_bd_ports ap_clk] [get_bd_pins "
                    + layer
                    + "_"
                    + str(i)
                    + "/aclk]\n"
                )
                tcl.append(
                    "connect_bd_net [get_bd_ports ap_rst_n] [get_bd_pins "
                    + layer
                    + "_"
                    + str(i)
                    + "/aresetn]\n"
                )
                ninputs -= 16
            # if multiple combiners, instantiate a combiner combiner
            # (up to 256 streams - total maximum output size of 512 bytes)
            if ncombiners_l1 > 1:
                tcl.append(
                    "create_bd_cell -type ip -vlnv xilinx.com:ip:axis_combiner:1.1 "
                    + layer
                    + "_out\n"
                )
                tcl.append(
                    "set_property -dict [list CONFIG.NUM_SI "
                    + str(ncombiners_l1)
                    + "] [get_bd_cells "
                    + layer
                    + "_out]\n"
                )
                tcl.append(
                    "connect_bd_net [get_bd_ports ap_clk] [get_bd_pins "
                    + layer
                    + "_out/aclk]\n"
                )
                tcl.append(
                    "connect_bd_net [get_bd_ports ap_rst_n] [get_bd_pins "
                    + layer
                    + "_out/aresetn]\n"
                )
                for i in range(ncombiners_l1):
                    tcl.append(
                        "connect_bd_intf_net [get_bd_intf_pins "
                        + layer
                        + "_"
                        + str(i)
                        + "/M_AXIS] [get_bd_intf_pins "
                        + layer
                        + "_out/S"
                        + ("%02d" % i)
                        + "_AXIS]\n"
                    )
        tcl.append("save_bd_design\n")

        # Step 5: connect combiner(s) to FIFOs or substream merge logic
        for stream in self.connectivity:
            layer = self.connectivity[stream]["dst"]["name"]
            # skip substreams with index > 0
            if (
                self.connectivity[stream]["attributes"]["substrm_idx"] is not None
                and self.connectivity[stream]["attributes"]["substrm_idx"] > 0
            ):
                continue
            # skip if this layer only has one PE
            if self.shapes[layer]["PE"] == 1:
                continue
            target_pe = self.connectivity[stream]["dst"]["pe_idx"]
            # connect stream to combiner(s)
            combiner_index = math.floor(target_pe / 16)
            combiner_input = target_pe % 16
            source_port = (
                self.connectivity[stream]["waypoints"][-1]["instance"]
                + "/"
                + self.connectivity[stream]["waypoints"][-1]["port"]
            )
            tcl.append(
                "connect_bd_intf_net [get_bd_intf_pins "
                + source_port
                + "] [get_bd_intf_pins "
                + layer
                + "_"
                + str(combiner_index)
                + "/S"
                + ("%02d" % combiner_input)
                + "_AXIS]\n"
            )
            self.connectivity[stream]["waypoints"].append(
                {"instance": layer + "_" + str(combiner_index), "port": "M_AXIS"}
            )
            if self.shapes[layer]["PE"] > 16:
                self.connectivity[stream]["waypoints"].append(
                    {"instance": layer + "_out", "port": "M_AXIS"}
                )
        tcl.append("save_bd_design\n")

        # Step 6: create/connect stream outputs
        connected_outputs = []
        for stream in self.connectivity:
            source_port = (
                self.connectivity[stream]["waypoints"][-1]["instance"]
                + "/"
                + self.connectivity[stream]["waypoints"][-1]["port"]
            )
            target_layer = self.connectivity[stream]["dst"]["name"]
            if target_layer in connected_outputs:
                continue
            oname = self.outputs[target_layer]
            connected_outputs.append(target_layer)
            tcl.append(
                "create_bd_intf_port -mode master -vlnv "
                + "xilinx.com:interface:axis_rtl:1.0 "
                + oname
                + "\n"
            )
            tcl.append(
                "connect_bd_intf_net [get_bd_intf_ports "
                + oname
                + "] [get_bd_intf_pins "
                + source_port
                + "]\n"
            )
        tcl.append("save_bd_design\n")

        # export the BD as an IP
        tcl.append(
            "ipx::package_project "
            + "-root_dir "
            + "[get_property DIRECTORY [current_project]]/[current_bd_design] "
            + "-vendor xilinx.com "
            + "-library user "
            + "-taxonomy /UserIP "
            + "-module [current_bd_design] "
            + "-import_files\n"
        )

        # write to file
        tcl_file = tcl_folder + "/bd_assembly.tcl"
        with open(tcl_file, "w") as f:
            f.write("".join(tcl))

    def get_verilog_top_module_intf_names(self):
        intf_names = super().get_verilog_top_module_intf_names()
        intf_names["clk"] = ["ap_clk"]
        intf_names["rst"] = ["ap_rst_n"]
        intf_names["s_axis"] = []
        intf_names["m_axis"] = []
        for outp in self.outputs:
            intf_names["m_axis"].append(self.outputs[outp])
        return intf_names

    def get_stream_widths(self):
        ret = []
        for key in self.shapes:
            ret.append(
                self.shapes[key]["PE"]
                * self.shapes[key]["SIMD"]
                * self.shapes[key]["DataType"].bitwidth()
            )
        return ret
