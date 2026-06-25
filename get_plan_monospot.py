#!/usr/bin/env python3
"""
Generate FRED simulation input file (.inp) for a single spot at a determined energy and gantry angle.

This script reads a CT mhd file and a beam model file,
then generates a formatted .inp file for FRED Monte Carlo simulation.

Usage:
    python get_plan_monospot.py <beam_model_file> --energy <proton_energy_MeV> --gantry_angle <gantry_angle_deg>

Example:
    python get_plan_monospot.py CustomBeamModel.bm --energy 150.0 --gantry_angle 90.0

Author: Generated for proton therapy workflow
"""

import sys
import argparse
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path

import fredtools as ft
import re


# =============================================================================
# Configuration Constants
# =============================================================================

# Nozzle geometry (in mm, will be converted to cm for output)
NOZZLE_TO_ISO_DISTANCE_MM = 600.0  # Nozzle exit to isocenter
SMX_TO_ISO_DISTANCE_MM = 2000.0    # SMX magnet to isocenter
SMY_TO_ISO_DISTANCE_MM = 2560.0    # SMY magnet to isocenter

# Convert to cm for .inp file
NOZZLE_TO_ISO_DISTANCE_CM = NOZZLE_TO_ISO_DISTANCE_MM / 10.0  # 60 cm

# Reference plane distance for field origin (SMY magnet to iso)
FIELD_ORIGIN_DISTANCE_CM = SMY_TO_ISO_DISTANCE_MM / 10.0  # 256 cm

# emittanceRefPlaneDistance - distance from isocenter to reference plane
EMITTANCE_REF_PLANE_DISTANCE_CM = (SMY_TO_ISO_DISTANCE_MM / 10 - NOZZLE_TO_ISO_DISTANCE_CM) #  194 cm


# =============================================================================
# Helper Functions
# =============================================================================

def interpolate_beam_params(bm_energy_df, energy):
    """
    Interpolate beam model parameters for a given energy.
    
    Parameters
    ----------
    bm_energy_df : pd.DataFrame
        Beam model energy DataFrame (indexed by nomEnergy)
    energy : float
        Nominal energy to interpolate for (MeV)
    
    Returns
    -------
    dict
        Dictionary with interpolated beam parameters
    """
    # Get the energy values from the index
    energies = bm_energy_df.index.values
    
    # Check bounds
    if energy < energies.min() or energy > energies.max():
        raise ValueError(f"Energy {energy} MeV is outside beam model range [{energies.min()}, {energies.max()}] MeV")
    
    # Interpolate each parameter
    params = {}
    for col in bm_energy_df.columns:
        params[col] = np.interp(energy, energies, bm_energy_df[col].values)
    
    return params


def calculate_field_vectors(gantry_angle_deg):
    """
    Calculate field direction vectors (f, u) from gantry angle.
    
    For a gantry rotating around the Z-axis (IEC convention):
    - Gantry at 0°: beam comes from +Y direction (anterior)
    - Gantry at 90°: beam comes from -X direction (left lateral)
    - Gantry at 180°: beam comes from -Y direction (posterior)
    - Gantry at 270°: beam comes from +X direction (right lateral)
    
    Parameters
    ----------
    gantry_angle_deg : float
        Gantry angle in degrees
    
    Returns
    -------
    tuple
        (f_vector, u_vector) - forward and up direction vectors
    """
    # For now, use fixed vectors as in the example
    # f points in beam direction, u points up
    # TODO: Calculate properly based on gantry angle if needed
    f = [0.0, 1.0, 0.0]
    u = [0.0, 0.0, 1.0]
    
    return f, u


def format_float(value, width=12, precision=4, sign=True):
    """Format a float with consistent width and sign."""
    if sign:
        return f"{value:+{width}.{precision}f}"
    else:
        return f"{value:{width}.{precision}f}"


def format_scientific(value, precision=10):
    """Format a float in scientific notation."""
    return f"{value:+.{precision}E}"


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

def extract_rs_value(rs_id):
    """
    Extract the numeric value from a range shifter ID.

    e.g., "RS=2cm" -> 2
    """
    if rs_id is None or pd.isna(rs_id):
        return None

    match = re.search(r'\d+', str(rs_id))
    return int(match.group()) if match else None

# =============================================================================
# Main INP Generation Functions
# =============================================================================

def build_single_field_info(gantry_angle=0.0, couch_angle=0.0, snout_pos_mm=200.0, rs_id=None, rs_setting=None):
    """Build a one-row fields_info DataFrame for a single synthetic field."""
    data = {
        'FDeliveryNo': [1],
        'FNo': [1],
        'FName': ['SingleSpot'],
        'FGantryAngle': [gantry_angle],
        'FCouchAngle': [couch_angle],
        'FCouchPitchAngle': [0.0],
        'FCouchRollAngle': [0.0],
        'FIsoPos': [None],
        'PBSnoutPos': [snout_pos_mm],
        'PBRSID': [rs_id],
        'PBRSSetting': [rs_setting],
    }
    return pd.DataFrame(data)

def generate_header(beam_model, num_spots, total_primaries):
    """Generate the .inp file header section."""
    lines = []
    
    # Get beam model description
    bm_desc = beam_model.get('BM Description', {})
    bm_name = bm_desc.get('name', 'Unknown')
    bm_time = bm_desc.get('creationTime', datetime.now().strftime('%Y/%m/%d %H:%M:%S'))

    # Format header
    lines.append("#" * 143)
    lines.append(f"# Beam model: {bm_name} ({bm_time})")
    lines.append(f"# Fields no.: 1")
    lines.append(f"# Fields PB no.: {num_spots} (total: {num_spots})")
    
    # Format primaries
    lines.append(f"# Fields prim. no.: [{total_primaries:.2f}] (total: {total_primaries:.3E})")
    lines.append("#" * 143)
    
    return lines

def generate_field_definitions(fields_info):
    """Generate field definition lines."""
    lines = []
    
    for _, field in fields_info.iterrows():
        field_id = int(field['FDeliveryNo'])
        field_name = field['FName']

        # Field origin - at SMY magnet position (256 cm from iso)
        # O is defined in the FRED coordinate system
        O = [0.0, -FIELD_ORIGIN_DISTANCE_CM, 0.0]
        
        # Get direction vectors
        f, u = calculate_field_vectors(field['FGantryAngle'])
        
        # Format the line
        O_str = f"[ {format_float(O[0])}, {format_float(O[1])}, {format_float(O[2])} ]"
        f_str = f"[ {format_float(f[0], precision=3)}, {format_float(f[1], precision=3)}, {format_float(f[2], precision=3)} ]"
        u_str = f"[ {format_float(u[0], precision=3)}, {format_float(u[1], precision=3)}, {format_float(u[2], precision=3)} ]"
        
        line = f"field: {field_id} ; O={O_str} ;   f={f_str} ;   u={u_str} ;   # Definition of field {field_id} (field name: '{field_name}')"
        lines.append(line)
    
    return lines


def generate_pbmaster_definitions(fields_info):
    """Generate pbmaster definition lines."""
    lines = []
    
    for _, field in fields_info.iterrows():
        field_id = int(field['FDeliveryNo'])
        
        line = (f"pbmaster: {field_id} ; particle=proton ; Xsec=emittance ; "
                f"emittanceRefPlaneDistance={EMITTANCE_REF_PLANE_DISTANCE_CM:.4f} ; "
                f"columns=[P.x, P.y, P.z, v.x, v.y, v.z, Emean, Estdev, N, "
                f"twissAlphaX, twissBetaX, emittanceX, twissAlphaY, twissBetaY, emittanceY]")
        lines.append(line)
    
    return lines


def generate_group_region_definitions(fields_info):
    """Generate group and region definitions."""
    lines = []
    
    # Create field group string
    field_names = " ".join([f"field_{int(f['FDeliveryNo'])}" for _, f in fields_info.iterrows()])
    
    lines.append(f"group: fieldGroup {field_names}")
    lines.append("region: gantry ; O = [0,0,0] ; L = [1,1,1] ; lAdaptiveSize=t ; material = vacuum")
    lines.append("set_parent: gantry fieldGroup nozzleGroup")
    lines.append("save_regions: 0")
    
    return lines


def clean_rs_id(rs_id):
    """
    Clean range shifter ID to match region naming convention.
    
    Converts DICOM RS ID (e.g., "RS=2cm") to region name (e.g., "RS_2cm")
    """
    if rs_id is None or pd.isna(rs_id):
        return None
    
    # Convert "RS=2cm" -> "RS_2cm"
    cleaned = str(rs_id).replace("=", "_") # TODO fix
    cleaned = "RS"
    return cleaned



def generate_setup_delivery_sequence(fields_info, isocenter_mm):
    """Generate setup and delivery sequence for all fields."""
    lines = []
    rs_ids = []
    
    lines.append("#" * 50)
    lines.append("###### Start of setup and delivery sequence ######")
    lines.append("#" * 50)
    
    # Convert isocenter from mm to cm AND flip the signs
    # DICOM isocenter [40, -240, -570] mm -> [-4, 24, 57] cm in FRED
    iso_cm = [-x / 10.0 for x in isocenter_mm]
    
    for _, field in fields_info.iterrows():
        field_id = int(field['FDeliveryNo'])
        field_name = field['FName']
        gantry_angle = field['FGantryAngle']
        couch_angle = field['FCouchAngle']
        snout_pos_mm = field['PBSnoutPos']
        rs_id = field.get('PBRSID', None)
        rs_setting = field.get('PBRSSetting', None)
        
        # Convert snout position to cm
        snout_pos_cm = snout_pos_mm / 10.0
        
        # Calculate snout shift (snout position relative to nozzle)
        # The snout moves along the beam direction (Y in FRED coords)
        snout_shift_cm = NOZZLE_TO_ISO_DISTANCE_CM - snout_pos_cm
        
        lines.append(f"###### setup sequence for field {field_id} (field name: '{field_name}')")
        
        # Nozzle translation
        lines.append("# translate nozzle regions by snout position")
        lines.append(f"transform: nozzleGroup shift_by {format_float(0.0)} {format_float(-snout_shift_cm)} {format_float(0.0)}")
        
        # Gantry rotation
        lines.append("# rotate gantry by gantry angle")
        lines.append(f"transform: gantry rotate z {format_float(gantry_angle, sign=False)}")
        
        # Phantom translation and rotation
        lines.append("# translate and rotate phantom by iso position and couch rotation")
        lines.append(f"transform: phantom shift_by {format_float(iso_cm[0])} {format_float(iso_cm[1])} {format_float(iso_cm[2])}")
        lines.append(f"transform: phantom rotate y {format_float(-couch_angle)}")
        
        # Delivery sequence
        lines.append(f"# delivery sequences for slices in field {field_id}")
        lines.append("deactivate: fieldGroup, nozzleGroup")
        
        rs_ids.append(rs_id)
        
        # Activate range shifter if present and IN
        rs_name = clean_rs_id(rs_id)
        if rs_name and rs_setting == 'IN':
            lines.append(f"activate: field_{field_id} {rs_name}")
        else:
            lines.append(f"activate: field_{field_id}")
        
        lines.append(f"deliver: field_{field_id}")
        
        # Reset
        lines.append("# reset fieldGroup, nozzleGroup and phantom to default FoR")
        lines.append("restore: 0")
    
    lines.append("#" * 50)
    lines.append("###### End of setup and delivery sequence ########")
    lines.append("#" * 50)
    
    return lines, rs_ids

def generate_pb_single_spot(energy, bm_energy_df, field_id=1, n_weight=1,
                            alpha_x=None, beta_x=None, alpha_y=None, beta_y=None):
    """Generate pencil beam definition lines for a single centred spot.
    If new alpha and beta parameters are provided, they are used instead of the ones in the custom beam model."""
    lines = []
    lines.append("#" * 36)
    lines.append("###### Start of PB definition ######")
    lines.append("#" * 36)
    lines.append(f"###### PB definitions for field {field_id}")
    lines.append(
        "# spotID    fieldID           P.x           P.y           P.z"
        "             v.x           v.y           v.z        Emean     Estdev"
        "           N               twissAlphaX       twissBetaX        emittanceX"
        "          twissAlphaY       twissBetaY        emittanceY"
    )

    # Interpolate beam model parameters
    bm_params = interpolate_beam_params(bm_energy_df, energy)
    alpha_x = alpha_x if alpha_x is not None else bm_params['alphaX']
    beta_x  = beta_x  if beta_x  is not None else bm_params['betaX']
    alpha_y = alpha_y if alpha_y is not None else bm_params['alphaY']
    beta_y  = beta_y  if beta_y  is not None else bm_params['betaY']
    n_calc = n_weight * bm_params['scalingFactor']

    # Single spot at field centre, reference plane
    px, py, pz = 0.0, 0.0, 0.0  # cm, at the reference plane (origin)
    vx, vy, vz = 0.0, 0.0, 1.0  # direction along beam axis

    energy_delivered = bm_params['Energy']
    estdev = bm_params['dEnergy']
    epsilon_x = bm_params['epsilonX']
    epsilon_y = bm_params['epsilonY']

    line = (
        f"pb: {1:<10d} {int(field_id):<4d} \t"
        f"{format_scientific(px)} {format_scientific(py)} {format_scientific(pz)}   "
        f"{format_scientific(vx)} {format_scientific(vy)} {format_scientific(vz)}   "
        f"{energy_delivered:7.3f} {estdev:.5E}   {n_calc:.10E}   "
        f"{format_scientific(alpha_x)} {format_scientific(beta_x)} {format_scientific(epsilon_x)}   "
        f"{format_scientific(alpha_y)} {format_scientific(beta_y)} {format_scientific(epsilon_y)}"
    )
    lines.append(line)

    return lines, n_calc


def generate_inp_file_single_spot(bm_file, energy, gantry_angle=0.0, couch_angle=0.0, snout_pos_mm=200.0,
                                  isocenter=None, n_weight=1, rs_id=None, rs_setting=None,
                                  alpha_x=None, beta_x=None, alpha_y=None, beta_y=None,
                                  output_dir="rtplans"):
    """
    Generate FRED .inp file(s) from a single centred spot at a given energy and using beam model files.
    Returns output filename and rs_ids.
    """
    if isocenter is None:
        isocenter = [0.0, 0.0, 0.0]

    print(f"Reading beam model: {bm_file}")
    beam_model = ft.readBeamModel(bm_file)
    bm_energy_df = beam_model['BM Energy']

    fields_info = build_single_field_info(gantry_angle, couch_angle, snout_pos_mm, rs_id, rs_setting)
    print(fields_info)

    # Ensure output directory exists
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    field_id = int(fields_info.iloc[0]['FDeliveryNo'])
    pb_lines, total_primaries = generate_pb_single_spot(energy, bm_energy_df, field_id, n_weight,
                                                        alpha_x, beta_x, alpha_y, beta_y)

    print(f"Generating .inp file for single spot at {energy:.3f} MeV...")

    all_lines = []
    all_lines.extend(generate_header(beam_model, num_spots=1, total_primaries=total_primaries))
    all_lines.extend(generate_field_definitions(fields_info))
    all_lines.extend(generate_pbmaster_definitions(fields_info))
    all_lines.extend(generate_group_region_definitions(fields_info))
    lines, rs_ids = generate_setup_delivery_sequence(fields_info, isocenter)
    all_lines.extend(lines)
    all_lines.extend(pb_lines)

    output_file = f"rtplan_SingleSpot_{energy:.1f}MeV.inp"
    full_path = output_path / output_file

    print(f"Writing to: {full_path}")
    with open(full_path, 'w') as f:
        f.write('\n'.join(all_lines))

    print(f"  Done: {output_file}")

    return output_file, rs_ids


def generate_fred_inp(rtplan_inp_filename, regions_filename, output_dir='freds', ct_file='CT.mhd',
                      nprim=5e4, ionization_potential=78.0,
                      materials_file='materials.inp', rtplan_dir = "rtplans", regions_dir = "regions"):
    """
    Generate FRED main simulation input file for a single rtplan .inp.
    rtplan_inp_filename is just the filename (e.g. 'rtplan_Field1.inp'),
    not a full path.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    lines = []

    lines.append("### Phantom ###")
    lines.append("region<")
    lines.append("    ID=phantom")
    lines.append(f"    CTscan= '{ct_file}'")
    lines.append("    score = [dose-to-water, RBE]")
    lines.append("    lWriteLETd_parts = false")
    lines.append("    lWriteCTHU = false")
    lines.append("  ")
    lines.append("region>")
    lines.append("")

    lines.append("### Materials ###")
    lines.append(f"include: {materials_file}")
    lines.append("")

    lines.append("### Regions ###")
    lines.append(f"include: {regions_dir}/{regions_filename}")
    lines.append("")

    lines.append("### Beam ###")
    lines.append(f"include: {rtplan_dir}/{rtplan_inp_filename}")
    lines.append(f"nprim = {nprim:.0E}")
    lines.append("")

    lines.append("### Physics ###")
    lines.append(f"IonizPotential={int(ionization_potential)}")
    lines.append("")

    lines.append("### Control ###")
    lines.append("lAllowHUClamping = true")
    lines.append("lTracking_nuc_el = true")
    lines.append("lTracking_nuc_inel = true")
    lines.append("lTracking_nuc = true")
    lines.append("lTracking_fluc = true")
    # lines.append("lUseInternalHU2Mat = true")
    lines.append("lUseInternalHU2Mat = false")
    lines.append(f"include: hu2materials_fred_.txt")
    lines.append(f"WaterIpot = {int(ionization_potential)}")
    lines.append("RBE_Constant = 1.1")
    lines.append("")

    # Derive fred filename from the rtplan filename
    # e.g. "rtplan_Field1.inp" -> "fred_Field1.inp"
    stem = rtplan_inp_filename
    if stem.startswith("rtplan_"):
        stem = stem[len("rtplan_"):]
    output_file = f"fred_{stem}"

    full_path = output_path / output_file

    with open(full_path, 'w') as f:
        f.write('\n'.join(lines))

    print(f"  FRED input file written: {full_path}")
    return output_file



def generate_region_inp(rtplan_inp_filename, output_dir='regions', rs_length=None):
    """
    Generate FRED main simulation input file for a single rtplan .inp.
    rtplan_inp_filename is just the filename (e.g. 'rtplan_Field1.inp'),
    not a full path.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    if not rs_length:
        rs_length = 1
        lines = []
        lines.append("region<")
        lines.append("  ID=RS")
        lines.append(f" L=[40, 40, {int(rs_length)}]")
        lines.append("  O=[0, 0, 0]")
        lines.append("  f=[0, 1, 0]")
        lines.append("  material=HU -1000")
        lines.append("  pivot=[0.5, 0.5, 1]")
        lines.append("  u=[0, 0, 1]")
        lines.append("  voxels=[1, 1, 1]")
        lines.append("region>")
        lines.append("")
        lines.append("group: rangeShifterGroup RS")
        lines.append("group: nozzleGroup rangeShifterGroup")
        lines.append("")

        
        
        
    else:
        lines = []
        lines.append("region<")
        lines.append("  ID=RS")
        lines.append(f" L=[40, 40, {int(rs_length)}]")
        lines.append("  O=[0, 0, 0]")
        lines.append("  f=[0, 1, 0]")
        lines.append("  material=uclhRS")
        lines.append("  pivot=[0.5, 0.5, 1]")
        lines.append("  u=[0, 0, 1]")
        lines.append("  voxels=[1, 1, 1]")
        lines.append("region>")
        lines.append("")
        lines.append("group: rangeShifterGroup RS")
        lines.append("group: nozzleGroup rangeShifterGroup")
        lines.append("")

    # Derive fred filename from the rtplan filename
    # e.g. "rtplan_Field1.inp" -> "fred_Field1.inp"
    stem = rtplan_inp_filename
    if stem.startswith("rtplan_"):
        stem = stem[len("rtplan_"):]
    output_file = f"region_{stem}"

    full_path = output_path / output_file

    with open(full_path, 'w') as f:
        f.write('\n'.join(lines))

    print(f"  REGION input file written: {full_path}")
    return output_file


def main():
    parser = argparse.ArgumentParser(
        description='Generate FRED .inp file for a single pencil beam spot',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
    Examples:
    python get_plan_monospot.py CustomBeamModel.bm --energy 150.0 --gantry-angle 90
        """
    )

    parser.add_argument('bm_file', help='Path to beam model (.bm) file')
    parser.add_argument('ct_file', help='Path to CT (.mhd) file')
    parser.add_argument('--energy',       type=float, required=True,
                        help='Nominal beam energy in MeV')
    parser.add_argument('--gantry-angle', type=float, default=0.0,
                        help='Gantry angle in degrees (default: 0)')
    parser.add_argument('--couch-angle',  type=float, default=0.0,
                        help='Couch angle in degrees (default: 0)')
    parser.add_argument('--snout-pos',    type=float, default=200.0,
                        help='Snout position in mm (default: 200)')
    parser.add_argument('--isocenter',    type=float, nargs=3,
                        default=[0.0, 0.0, 0.0], metavar=('X', 'Y', 'Z'),
                        help='Isocenter position in mm (default: 0 0 0)')
    parser.add_argument('--nprim',        type=float, default=5e4,
                        help='Number of primary particles for FRED (default: 5e4)')
    parser.add_argument('--n-weight',     type=float, default=1.0,
                        help='Spot weight / N value written to the pb line (default: 1.0)')
    parser.add_argument('--alpha-x',      type=float, default=None,
                        help="Twiss alphaX parameter (default: read from beam model)")
    parser.add_argument('--beta-x',      type=float, default=None,
                        help="Twiss betaX parameter (default: read from beam model)")
    parser.add_argument('--alpha-y',      type=float, default=None,
                        help="Twiss alphaY parameter (default: read from beam model)")
    parser.add_argument('--beta-y',      type=float, default=None,
                        help="Twiss betaY parameter (default: read from beam model)")

    args = parser.parse_args()

    if not Path(args.bm_file).exists():
        print(f"Error: Beam model file not found: {args.bm_file}")
        sys.exit(1)

    if not Path(args.ct_file).exists():
        print(f"Error: CT file not found: {args.ct_file}")
        sys.exit(1)

    try:
        # clear_directories("rtplans", "freds", "regions")

        rtplan_file, rs_ids = generate_inp_file_single_spot(
            bm_file=args.bm_file,
            energy=args.energy,
            gantry_angle=args.gantry_angle,
            couch_angle=args.couch_angle,
            snout_pos_mm=args.snout_pos,
            isocenter=args.isocenter,
            n_weight=args.n_weight,
            alpha_x=args.alpha_x,
            beta_x=args.beta_x,
            alpha_y=args.alpha_y,
            beta_y=args.beta_y,
            output_dir="rtplans",
        )

        rs_length = extract_rs_value(rs_ids[0])
        region_file = generate_region_inp(rtplan_file, output_dir="regions",
                                          rs_length=rs_length)
        fred_file = generate_fred_inp(rtplan_file, region_file,
                                      output_dir="freds",
                                      nprim=args.nprim,
                                      ct_file=args.ct_file)

        print(f"\nSuccessfully generated rtplan file in: rtplans/{rtplan_file}")
        print(f"Successfully generated fred file in:   freds/{fred_file}")
        print(f"Successfully generated region file in: regions/{region_file}")

        print(f"\nSuccessfully generated {len(rtplan_file)} rtplan file in: rtplans/")
        print(f"\nSuccessfully generated {len(fred_file)} fred file in: freds/")
        print(f"\nSuccessfully generated {len(region_file)} region file in: regions/")

    except Exception as e:
        print(f"Error generating .inp file: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()