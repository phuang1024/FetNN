"""Call Weiman's script generator with hardcoded recipe params.
Generate a LDMOS Degradation sim.
"""

import argparse
import os


# Copied from Weiman's script.
def create_cmd_file(template_file, output_file, params):
    """Create a new cmd file with randomly generated parameters and save to par.csv."""
    # Generate random parameters
    lsti, tox, nwellD1, nwellD2, nwellD3,dose_energy1,dose_energy2,dose_energy3,aneal_temp,nn,xn,Lextra, nwellscale, psubD_scale, polyThickness= params

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("file")
    args = parser.parse_args()
    
    # Aug 14 50 epochs training.
    # LogitX:  tensor([-0.4148,  1.3484,  0.1107,  0.8823, -0.4397,  0.2967,  1.5418,  1.3965, -0.3753, -0.9692])
    # LogitY:  tensor([0.0916, 0.2357, 1.1038, 1.1691])
    # UnnormX: tensor([2.5035e+00, 9.3038e+00, 2.5265e+13, 1.3488e+13, 2.0717e+10, 3.1634e+02, 3.8872e+02, 3.8933e+02, 1.0532e+03, 4.5574e-02])
    # UnnormY: tensor([4.7994e-03, 1.9763e+01, 5.3478e+04, 1.7854e+00])
    params = [
        # NN: Lsti ... anneal_temp
        2.5035e+00, 9.3038e+00, 2.5265e+13, 1.3488e+13, 2.0717e+10, 3.1634e+02, 3.8872e+02, 3.8933e+02, 1.0532e+03,
        # Manual: nn
        3,
        # NN: xn
        4.5574e-02,
        # Manual: Lextra, nwellscale, psubD_scale, polyThickness
        2.5, 1, 1, 1,
    ]
    create_cmd_file(args.file, args.file, params)
