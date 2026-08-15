"""Test LDMOS Degradation template using cINN generated params.
Extended from Weiman's cmd_creator.py

This script generates TCAD nodes using cINN predictions.
"""

import os
import argparse
import glob
import re
import shutil

import torch


def create_cmd_file(template_file, output_file, params):
    """Create a new cmd file with randomly generated parameters and save to par.csv."""
    # Params from NN.
    lsti, tox, nwellD1, nwellD2, nwellD3,dose_energy1,dose_energy2,dose_energy3,aneal_temp,xn= params[:10]
    # The rest of the params.
    nn = 3
    Lextra = 2.5
    nwellscale = 1
    psubD_scale = 1
    polyThickness = 1

    # Fixed parameters
    params = {
        "Lsti": lsti,
        "tox": tox,
        "nwellD1": nwellD1,
        "nwellD2": nwellD2,
        "nwellD3": nwellD3,
        "dose_energy1": dose_energy1,
	"dose_energy2": dose_energy2,
        "dose_energy3": dose_energy3,
        "aneal_temp": aneal_temp,
	"nn":nn,
	"xn":xn,
        "node": 14,
        "Lch": 1.50,
        "Xmax": 5,
        "Lacc": 0.2,
        "psubD": 5e16,
        "pbD": 1.0e15,
        "nsD": 1.0e15,
        "tsti": 0.4,
        "pmesh_0.01": 0.01,
        "pmesh_0.02": 0.02,
        "pmesh_0.03": 0.03,
        "pmesh_0.3": 0.3,
        "pmesh_1.0": 1.0,
        "Lextra" : Lextra,
        "nwellscale" : nwellscale,
        "psubD_scale" : psubD_scale,
        "polyThickness" : polyThickness
    }

    try:
        with open(template_file, 'r') as f:
            content = f.read()

        # Replace all placeholders
        for key, value in params.items():
            placeholder = f"@{key}@" if not str(key).startswith("pmesh_") else f"@<{key.replace('_', ' * ')}>@"
            content = content.replace(placeholder, str(value))

        content = content.replace('@node@', str(params["node"]))

        # Write the modified CMD file
        with open(output_file, 'w') as f:
            f.write(content)

        # Write parameters to par.csv
        folder = os.path.dirname(output_file)
        with open(os.path.join(folder, "par.csv"), "w") as f:
            f.write("Parameter,Value\n")
  #          for k, v in params.items():
 #               f.write(f"{k},{v}\n")
            for k, v in params.items():
                if not k.startswith("pmesh_"):  # exclude mesh params
                    f.write(f"{k},{v}\n")
        print(f"✅ Created {output_file} with parameters: Lsti={lsti:.4f}, tox={tox:.4f}, nwellD1={nwellD1:.2e},nwellD2={nwellD2:.2e}, nwellD3={nwellD3:.2e}, dose_energy1={dose_energy1:.2e}, dose_energy2={dose_energy2:.2e}, dose_energy3={dose_energy3:.2e}, Lextra={Lextra:.4f}, nwellscale={nwellscale:.4f}, psubD_scale={psubD_scale:.4f}, polyThickness={polyThickness:.4f}, nn={nn}, xn={xn:.4f}")
        return True

    except FileNotFoundError:
        print(f"❌ Error: Template file '{template_file}' not found.")
        return False
    except Exception as e:
        print(f"❌ Error occurred: {e}")
        return False


def copy_cmd_to_each_node(src_file, dest_path):
    try:
        shutil.copy(src_file, dest_path)
        print(f"Copied {src_file} → {dest_path}")
        return True
    except Exception as e:
        print(f"Failed to copy: {e}")
        return False


def generate_multiple_cmd_files(template_file, num_files, params):
    """Generate multiple cmd files with different random parameters."""
    print(f"Generating {num_files} cmd files using template '{template_file}'...")
    
    success_count = 0
    for i in range(1, num_files + 1):
        os.mkdir(f"node{i}")

	#step 1 Sprocess
        cmd_file = f"./node{i}/pp14_fps.cmd"
        prf_flag = copy_cmd_to_each_node("LDMOSprocess_fps.prf",f"./node{i}/LDMOSprocess_fps.prf")

	#step 2 Dev Mesh
        prf_flag = prf_flag and copy_cmd_to_each_node("pp16_fps.cmd",f"./node{i}/pp16_fps.cmd")
        prf_flag = prf_flag and copy_cmd_to_each_node("DevMesh_fps.prf",f"./node{i}/DevMesh_fps.prf")

	#step 3 Sdevice IDVG
        prf_flag = prf_flag and copy_cmd_to_each_node("pp20_des.cmd",f"./node{i}/pp20_des.cmd")
        prf_flag = prf_flag and copy_cmd_to_each_node("pp20_des.par",f"./node{i}/pp20_des.par")

	#step 4 Svisual IDVG
        prf_flag = prf_flag and copy_cmd_to_each_node("pp21_vis.cmd",f"./node{i}/pp21_vis.cmd")
        prf_flag = prf_flag and copy_cmd_to_each_node("n21_vis.tcl",f"./node{i}/n21_vis.tcl")

	#step 5 Sdevice IDVG
        prf_flag = prf_flag and copy_cmd_to_each_node("pp23_des.cmd",f"./node{i}/pp23_des.cmd")
        prf_flag = prf_flag and copy_cmd_to_each_node("pp23_des.par",f"./node{i}/pp23_des.par")

	#step 6 Svisual IDVG
        prf_flag = prf_flag and copy_cmd_to_each_node("pp24_vis.cmd",f"./node{i}/pp24_vis.cmd")
        prf_flag = prf_flag and copy_cmd_to_each_node("n24_vis.tcl",f"./node{i}/n24_vis.tcl")

	#step 7 Sdevice IDVD
        prf_flag = prf_flag and copy_cmd_to_each_node("pp25_des.cmd",f"./node{i}/pp25_des.cmd")
        prf_flag = prf_flag and copy_cmd_to_each_node("pp25_des.par",f"./node{i}/pp25_des.par")

	#step 8 Svisual IDVD
        prf_flag = prf_flag and copy_cmd_to_each_node("pp26_vis.cmd",f"./node{i}/pp26_vis.cmd")
        prf_flag = prf_flag and copy_cmd_to_each_node("n26_vis.tcl",f"./node{i}/n26_vis.tcl")

	#step 7 Run.sh
       # prf_flag = prf_flag and copy_cmd_to_each_node("Run.sh",f"./node{i}/Run.sh")

        if create_cmd_file(template_file, cmd_file, params[i - 1]) and prf_flag:
            success_count += 1

    
    print(f"\nSuccessfully generated {success_count} out of {num_files} requested cmd files.")


def generate_run_sh_with_error_handling():
    """Generate Run.sh in each node*/ folder with cd, ordered steps, and error handling.
    
    Note: sdevice commands (pp23_des and pp25_des) have individual timeout limits (7200 seconds).
    If they timeout or fail, the script continues to the next step (their dependent svisual commands
    will still run).
    """
    run_script = [
        "#!/bin/bash",
        'cd "$(dirname "$0")"',  # Make sure the script runs from its own folder
        'echo "▶️ Starting simulation in $(pwd)"',
        "",
        "sprocess -n pp14_fps.cmd > /dev/null 2>&1",
        'if [ $? -ne 0 ]; then echo "❌ pp14_fps.cmd failed"; exit 1; fi',
        "",
        "sprocess -n pp16_fps.cmd > /dev/null 2>&1",
        'if [ $? -ne 0 ]; then echo "❌ pp16_fps.cmd failed"; exit 1; fi',
        "",
        "sdevice -q pp20_des.cmd > /dev/null 2>&1",
        'if [ $? -ne 0 ]; then echo "❌ pp20_des.cmd failed"; exit 1; fi',
        "",
        "svisual -b n21_vis.tcl > /dev/null 2>&1",
        'if [ $? -ne 0 ]; then echo "❌ n21_vis.tcl failed"; exit 1; fi',
        "",
        "# pp23_des.cmd with timeout (7200s = 2 hours) - continue even if killed or fails",
        "timeout 7200 sdevice -q pp23_des.cmd > /dev/null 2>&1",
        "RESULT=$?",
        'if [ $RESULT -eq 124 ]; then echo "⚠️  pp23_des.cmd timed out (killed after 7200 seconds)"; elif [ $RESULT -ne 0 ]; then echo "⚠️  pp23_des.cmd failed with code $RESULT but continuing"; fi',
        "",
        "svisual -b n24_vis.tcl > /dev/null 2>&1",
        'if [ $? -ne 0 ]; then echo "❌ n24_vis.tcl failed"; exit 1; fi',
        "",
        "# pp25_des.cmd with timeout (7200s = 2 hours) - continue even if killed or fails",
        "timeout 7200 sdevice -q pp25_des.cmd > pp25_run.log 2>&1",
        "RESULT=$?",
        'if [ $RESULT -eq 124 ]; then echo "⚠️  pp25_des.cmd timed out (killed after 7200 seconds)"; elif [ $RESULT -ne 0 ]; then echo "⚠️  pp25_des.cmd failed with code $RESULT but continuing"; fi',
        "",
        "svisual -b n26_vis.tcl > /dev/null 2>&1",
        'if [ $? -ne 0 ]; then echo "❌ n26_vis.tcl failed"; exit 1; fi',
        "",
        'echo "✅ Done with simulation in $(pwd)"'
    ]

    folders = sorted(glob.glob("node*/"), key=lambda x: int(re.search(r'node(\d+)', x).group(1)))

    for folder in folders:
        runsh_path = os.path.join(folder, "Run.sh")
        try:
            with open(runsh_path, "w") as f:
                f.write("\n".join(run_script) + "\n")
            os.chmod(runsh_path, 0o755)
            print(f"✅ Created Run.sh in {folder}")
        except Exception as e:
            print(f"❌ Failed to create Run.sh in {folder}: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate CMD files with random parameters.')
    parser.add_argument('--params', help='NN test results')
    parser.add_argument('--count', '-c', type=int, default=10, help='Number of CMD files to generate (default: 1)')
    args = parser.parse_args()
    params = torch.load(args.params)
    generate_multiple_cmd_files("./LDMOSprocess_fps.cmd", args.count, params)
    generate_run_sh_with_error_handling()
