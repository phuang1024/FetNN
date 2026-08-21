"""Preprocess TCAD simulation template.
"""

import os
from pathlib import Path
from shutil import copytree

from params import Params

RUN_SH = """
#!/bin/bash

sprocess sprocess_fps.cmd
sdevice IdVg_des.cmd
"""


def replace_in_file(dir: Path, file, replacements: dict):
    """Replace each key with value.
    Inplace on file.
    """
    with open(dir / file, "r") as f:
        data = f.read()
    for k, v in replacements.items():
        if k not in data:
            print(f"replace_in_file: Key not found: file={file}, key={k}")
        data = data.replace(k, v)
    with open(dir / file, "w") as f:
        f.write(data)


def fill_params(dir: Path, params: Params, node):
    """Fill parameter values in TCAD scripts.
    """
    # sprocess
    replace_in_file(dir, "sprocess_fps.cmd", {
        "@node@": node,
        "@adapt@": "1",
        "@nsub@": params.substr_n,
        "tepi": params.epi_thick,
        "@pepi_hi@": params.epi_p_high,
        "@pepi_low@": params.epi_p_low,
    })
    sproc_tdr = f"n{node}_fps.tdr"

    # IdVg
    replace_in_file(dir, "IdVg_des.cmd", {
        "@tdr@": sproc_tdr,
        "@parameter@": params.sdev_params,
        "@log@": "IdVg_des.log",
        "@plot@": "IdVg_current.plt",
        "@tdrdat@": "IdVg_plot.tdr",
        "@Vdslin@": params.idvg_vd,
        "@Vgs@": params.idvg_vg,
    })

    # Write run.sh
    with open(dir / "run.sh", "w") as f:
        f.write(RUN_SH)


if __name__ == "__main__":
    copytree("../../SentaurusFiles/LDMOS_Processing", "test_node")
    fill_params(Path("test_node"), Params(), "178")
