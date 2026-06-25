#!/usr/bin/env python3
"""
Generates a fred treatment plan with single spot using get_plan_monospot code and runs fred for all energies of interest

Author: Generated for proton therapy workflow
"""

from pathlib import Path
import sys
import subprocess
import shutil

def clear_directories(*dirs):
    """Clear all files in directories without removing the directories themselves."""
    for d in dirs:
        path = Path(d)
        if path.exists():
            for file in path.iterdir():
                if file.is_file():
                    file.unlink()
        else:
            path.mkdir(parents=True)

def remove_out_folders(base_dir="."):
    base = Path(base_dir)

    for d in base.iterdir():
        if d.is_dir() and d.name.startswith("out"):
            shutil.rmtree(d)

def write_beam_model(energy, alpha_x=None, beta_x=None, alpha_y=None, beta_y=None):
    cmd = [
        sys.executable, Path(__file__).parent / "get_plan_monospot.py",
        BM_FILE, CT_FILE,
        "--energy", str(energy),
        "--gantry-angle", str(GANTRY_ANGLE),
        "--couch-angle", str(COUCH_ANGLE),
        "--snout-pos", str(SNOUT_POS_MM),
        "--isocenter", *[str(x) for x in ISOCENTER_MM],
        "--nprim", str(NPRIM),
    ]
    if alpha_x is not None:
        cmd += ["--alpha-x", str(alpha_x)]
    if beta_x is not None:
        cmd += ["--beta-x", str(beta_x)]
    if alpha_y is not None:
        cmd += ["--alpha-y", str(alpha_y)]
    if beta_y is not None:
        cmd += ["--beta-y", str(beta_y)]
    return subprocess.run(cmd)


def run_fred(energy):
    fred_inp = next(Path("freds").glob(f"fred_*{energy:.1f}MeV.inp"))
    sim = subprocess.run([FRED_EXE, "-f", str(fred_inp)], shell=True)

    # Move FRED output to an energy-specific name
    shutil.move("out", f"out_{energy:.1f}MeV")
    return sim


# =============================================================================
# Configuration of key parameters
# =============================================================================

BM_FILE = "CustomBeamModel.bm"
CT_FILE = "CT.mhd"
FRED_EXE = "fred"
ENERGIES = [80, 100, 120, 140, 160, 180, 200, 220]  # MeV

# Shared beam / geometry settings
GANTRY_ANGLE = 0.0
COUCH_ANGLE = 0.0
SNOUT_POS_MM = 200.0
ISOCENTER_MM = [0.0, 0.0, 0.0]
NPRIM = 1E6

# =============================================================================

if __name__ == "__main__":
    clear_directories("rtplans", "freds", "regions")
    remove_out_folders()

    for energy in ENERGIES:
        #TODO: these are placeholders, they will change once we have the experimental data to update
        ax = None
        bx = None
        ay = None
        by = None

        print(f"\n--- {energy:.1f} MeV ---")
        prep = write_beam_model(energy, alpha_x=ax, beta_x=bx, alpha_y=ay, beta_y=by)
        if prep.returncode != 0:
            print(f"  ERROR: input generation failed for {energy:.1f} MeV")
            continue

        sim = run_fred(energy)
        if sim.returncode != 0:
            print(f"  ERROR: FRED failed for {energy:.1f} MeV")

    print("\n=== Sweep done ===")
