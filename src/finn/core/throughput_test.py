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


def throughput_test(model, batchsize=1000):
    """Runs the throughput test for the given model remotely on the pynq board.
    The metadata properties related to the pynq board have to be set.
    Returns a dictionary with results of the throughput test"""

    pynq_ip = model.get_metadata_prop("pynq_ip")
    pynq_port = int(model.get_metadata_prop("pynq_port"))
    pynq_username = model.get_metadata_prop("pynq_username")
    pynq_password = model.get_metadata_prop("pynq_password")
    pynq_target_dir = model.get_metadata_prop("pynq_target_dir")
    deployment_dir = model.get_metadata_prop("pynq_deploy_dir")
    # extracting last folder of absolute path (deployment_dir)
    deployment_folder = os.path.basename(os.path.normpath(deployment_dir))

    cmd = (
        "sshpass -p {} ssh {}@{} -p {} "
        '"cd {}/{}; echo "{}" | '
        'sudo -S python3.6 driver.py --exec_mode="throughput_test" --batchsize=%d"'
        % batchsize
    ).format(
        pynq_password,
        pynq_username,
        pynq_ip,
        pynq_port,
        pynq_target_dir,
        deployment_folder,
        pynq_password,
    )
    bash_command = ["/bin/bash", "-c", cmd]
    process_compile = subprocess.Popen(bash_command, stdout=subprocess.PIPE)
    process_compile.communicate()

    cmd = "sshpass -p {} scp -P{} {}@{}:{}/{}/nw_metrics.txt {}".format(
        pynq_password,
        pynq_port,
        pynq_username,
        pynq_ip,
        pynq_target_dir,
        deployment_folder,
        deployment_dir,
    )
    bash_command = ["/bin/bash", "-c", cmd]
    process_compile = subprocess.Popen(bash_command, stdout=subprocess.PIPE)
    process_compile.communicate()

    with open("{}/nw_metrics.txt".format(deployment_dir), "r") as file:
        res = eval(file.read())

    return res
