import pandas as pd
import SimpleITK as sitk
import numpy as np
from scipy.optimize import curve_fit
from matplotlib import pyplot as plt
import fredtools as ft
from scipy.interpolate import interp1d

distances_mm = np.array([-200, -100, 0, 100, 200])

def gaussian_2d(xy, amplitude, x0, z0, sigma_x, sigma_z, offset):
    """2D Gaussian function for curve fitting."""
    x, z = xy
    return (
        amplitude * np.exp(
            -((x - x0)**2 / (2 * sigma_x**2) + (z - z0)**2 / (2 * sigma_z**2))
        ) + offset
    ).ravel()

def _build_circular_mask(nz, nx, dz, dx, radius_mm):
    """Build a 2D boolean mask of voxels whose centres lie within `radius_mm`
    of the central voxel of an (nz, nx) slice. Computed once, reused for all depths."""
    # Voxel-centre coordinates relative to array centre
    z = (np.arange(nz) - (nz - 1) / 2.0) * dz
    x = (np.arange(nx) - (nx - 1) / 2.0) * dx
    zz, xx = np.meshgrid(z, x, indexing='ij')
    r2 = zz ** 2 + xx ** 2
    return r2 <= radius_mm ** 2

def get_experimental_sigmas_air(energy):
    df = pd.read_excel("UCLH_Spot_Profiles_All_Sigmas.xlsx", sheet_name="Sheet3")
    row = df[df["Energy"] == energy]
    row = row.iloc[0]
    sigma_x = [row[f"sigmaX_{int(d/10)}"] for d in distances_mm]
    sigma_y = [row[f"sigmaY_{int(d/10)}"] for d in distances_mm]
    return sigma_x, sigma_y

def get_experimental_IDD_water(energy):
    """Retrieve experimental data from IDD measurements for a given energy."""
    df = pd.read_excel("UCLH_reference_IDD.xlsx", sheet_name=str(energy))
    x = df['Depth (mm)']
    y = df['Dose (normalised to 1.0 at 20mm deep)']
    return x, y


def get_experimental_IDD_water_abs(energy):
    """Retrieve experimental data from IDD measurements for a given energy, no normalisation."""
    df = pd.read_excel("UCLH_reference_IDD.xlsx", sheet_name=str(energy))
    x = df['Depth (mm)']
    y = df['Dose (Gy/MU/mm^2)']
    return x, y

def get_fred_IDD(energy, norm_depth=20, aperture_radius_mm=40.8):
    """Retrieve MC IDD, option to normalise to a given depth."""
    dose_img = ft.readMHD(f"out_{energy:.1f}MeV/Dose.mhd")
    dose_array = sitk.GetArrayFromImage(dose_img)
    dx, dy, dz = dose_img.GetSpacing()
    nz, ny, nx = dose_array.shape
    depths = np.arange(ny) * dy
    if aperture_radius_mm is None:
        idd_raw = np.sum(dose_array, axis=(0, 2))
    else:
        mask = _build_circular_mask(nz, nx, dz, dx, aperture_radius_mm)
        idd_raw = (dose_array * mask[:, None, :]).sum(axis=(0, 2)) # Gy/MU
    dose_at_norm = float(np.interp(norm_depth, depths, idd_raw))
    idd = idd_raw / dose_at_norm

    return depths, idd


def get_fred_IDD_abs(energy, aperture_radius_mm=40.8):
    """
    Compute the integral depth dose from a FRED dose-to-water MHD file in
    units of Gy / MU / mm² (broad-field equivalent dose representation).

    Parameters
    ----------
    energy : int or float
        Beam energy, used to locate the output folder and the protons/MU.
    aperture_radius_mm : float or None
        Defaults to 40.8 mm to match the standard PTW Bragg Peak Chamber.
        If None, falls back to full lateral integration.
    n_sim : float
        Number of primaries simulated in FRED.
    """
    # Loading biological dose directly (inherently contains the 1.1 RBE multiplier)
    dose_img = ft.readMHD(f"out_{energy:.1f}MeV/RBE/Phantom.DoseBio_Constant.mhd")
    dose_array = sitk.GetArrayFromImage(dose_img)
    dx, dy, dz = dose_img.GetSpacing()  # mm
    nz, ny, nx = dose_array.shape
    depths = np.arange(ny) * dy

    # 1. ALWAYS calculate the full, unmasked lateral sum for absolute energy scaling
    total_slice_sum = dose_array.sum(axis=(0, 2))

    # 2. Calculate the masked sum to match the physical chamber shape
    if aperture_radius_mm is None:
        slice_sum_shape = total_slice_sum
    else:
        mask = _build_circular_mask(nz, nx, dz, dx, aperture_radius_mm)
        slice_sum_shape = (dose_array * mask[:, None, :]).sum(axis=(0, 2))

    # 3. CRITICAL CRITERIA: Re-introduce the voxel area factor (dx * dz)
    # This converts raw voxel values to an extensive quantity (Gy·mm²)
    idd_gy_mm2_shape = slice_sum_shape * dx * dz
    idd_gy_mm2_total = total_slice_sum * dx * dz

    # 4. Calculate the Halo Correction Factor at a reference depth of 2.0 cm (20 mm)
    # This finds the closest voxel slice index corresponding to 20 mm depth
    ref_depth_mm = 20.0
    ref_idx = np.argmin(np.abs(depths - ref_depth_mm))

    # Correction Factor = Total Energy / Masked Chamber Energy (at 2cm)
    halo_correction_factor = idd_gy_mm2_total[ref_idx] / idd_gy_mm2_shape[ref_idx]

    # 5. Convert from "per simulated proton" to "per MU"
    # TODO: Not necessary if we account for them in FRED directly
    #protons_per_mu = PROTONS_PER_MU[energy]
    idd_per_mu = idd_gy_mm2_shape #/ n_sim * protons_per_mu

    # 6. Apply the Halo Correction to scale the shape to the absolute broad-field magnitude
    idd_absolute = idd_per_mu * halo_correction_factor

    return depths, idd_absolute

def get_fred_sigmas(energy, dist):
    """Retrieve MC sigmas from a 2D gaussian fit at a given distance from isocentre in mm."""
    dose_img = ft.readMHD(f"out_{energy:.1f}MeV/Dose.mhd")
    dose_array = sitk.GetArrayFromImage(dose_img)
    dx, dy, dz = dose_img.GetSpacing()
    origin = dose_img.GetOrigin()
    nz, ny, nx = dose_array.shape

    iy = int(round((dist - origin[1]) / dy))
    iy = np.clip(iy, 0, ny - 1)
    actual_y_mm = origin[1] + iy * dy

    slice_2d = dose_array[:, iy, :]
    # Build coordinate arrays (in mm, centred on image origin)
    x_coords = origin[0] + np.arange(nx) * dx   # (nx,)
    z_coords = origin[2] + np.arange(nz) * dz   # (nz,)
    X, Z = np.meshgrid(x_coords, z_coords)       # both (nz, nx)

    # --- Initial parameter guesses ---
    amp0 = slice_2d.max()
    iz0, ix0 = np.unravel_index(slice_2d.argmax(), slice_2d.shape)
    x0_0 = x_coords[ix0]
    z0_0 = z_coords[iz0]
    sig0 = 5.0  # mm, reasonable starting sigma
    offset0 = slice_2d.min()

    p0 = [amp0, x0_0, z0_0, sig0, sig0, offset0]

    # --- Fit ---
    try:
        popt, pcov = curve_fit(gaussian_2d,(X.ravel(), Z.ravel()), slice_2d.ravel(),
            p0=p0, maxfev=10_000)
        amplitude, x0, z0, sigma_x, sigma_z, offset = popt

        # perr = np.sqrt(np.diag(pcov))
        # print(f"\n2D Gaussian fit at y = {actual_y_mm:.2f} mm:")
        # print(f"σ_x = {sigma_x:.3f} ± {perr[3]:.3f} mm")
        # print(f"σ_z = {sigma_z:.3f} ± {perr[4]:.3f} mm")
        # print(f"Centre: x0 = {x0:.2f} mm, z0 = {z0:.2f} mm")

    except RuntimeError as e:
        print(f"Fit failed: {e}")
        return None, None, None

    return sigma_x, sigma_z, popt


def compare_IDD(energy, abs=False):
    """Plots the relevant IDDs and calculates the ratio/scaling between the two."""
    if abs: # compare with absolute IDD values
        x_exp, y_exp = get_experimental_IDD_water_abs(energy)
        x_fred, y_fred = get_fred_IDD_abs(energy)
        ylabel = 'IDD (Gy/MU/mm$^2$)'
    else:    # compare normalised IDD curves
        x_exp, y_exp = get_experimental_IDD_water(energy)
        x_fred, y_fred = get_fred_IDD(energy)
        ylabel = 'IDD (normalised to 1.0 at 20mm deep)'
    x_interp = np.linspace(7, 400, 3970)
    interp_exp = interp1d(x_exp, y_exp, bounds_error=False, fill_value=0)
    interp_fred = interp1d(x_fred, y_fred, bounds_error=False, fill_value=0)

    y_exp_interp = interp_exp(x_interp)
    y_fred_interp = interp_fred(x_interp)

    plt.figure()
    plt.plot(x_interp, y_exp_interp, label='Experimental')
    plt.plot(x_interp, y_fred_interp, label='FRED')
    plt.legend()
    plt.xlabel('Depth (mm)')
    plt.ylabel(ylabel)
    plt.title(f'{energy} MeV beam', fontweight='bold')
    plt.show()

    print(f"----Energy = {energy} MeV Stats -----")
    scale = np.dot(y_exp_interp, y_fred_interp) / np.dot(y_fred_interp, y_fred_interp)
    print(f"Least-squares scale factor (exp = scale * fred): {scale:.4f}")
    mask = y_fred_interp > 0.01 * y_fred_interp.max()
    ratios = y_exp_interp[mask] / y_fred_interp[mask]
    print(f"Mean ratio:   {ratios.mean():.4f}")
    print(f"Median ratio: {np.median(ratios):.4f}")
    print(f"Std of ratio: {ratios.std():.4f}")


def compare_sigmas(energy):
    sigmasX_exp, sigmasY_exp = get_experimental_sigmas_air(energy)
    sigmasX_fred = []
    sigmasY_fred = []
    for d in distances_mm:
        sx, sy, _ = get_fred_sigmas(energy, d)
        sigmasX_fred.append(sx)
        sigmasY_fred.append(sy)

    plt.figure()
    plt.scatter(distances_mm, sigmasX_exp, label=r'Experimental $\sigma_x$', c='orange', marker='x')
    plt.scatter(distances_mm, sigmasX_fred, label=r'FRED $\sigma_x$', c='orange', marker='o')
    plt.scatter(distances_mm, sigmasY_exp, label=r'Experimental $\sigma_y$', c='blue', marker='x')
    plt.scatter(distances_mm, sigmasY_fred, label=r'FRED $\sigma_y$', c='blue', marker='o')
    plt.legend()
    plt.xlabel('Distance from isocentre (mm)')
    plt.ylabel(r'$\sigma$ (mm)')
    plt.title(f'{energy} MeV beam', fontweight='bold')
    plt.show()

def plot_IDD_comparison_grid(abs=False):

    if abs:
        ylabel = 'IDD (Gy/MU/mm$^2$)'
    else:
        ylabel = 'IDD (normalised to 1.0 at 20 mm depth)'

    fig, axes = plt.subplots(2, 4, figsize=(18, 10), sharex=True, sharey=True)
    axes = axes.flatten()

    for ax, energy in zip(axes, ENERGIES):

        if abs:
            x_exp, y_exp = get_experimental_IDD_water_abs(energy)
            x_fred, y_fred = get_fred_IDD_abs(energy)
        else:
            x_exp, y_exp = get_experimental_IDD_water(energy)
            x_fred, y_fred = get_fred_IDD(energy)

        x_interp = np.linspace(7, 400, 3970)

        interp_exp = interp1d(
            x_exp, y_exp,
            bounds_error=False,
            fill_value=0
        )

        interp_fred = interp1d(
            x_fred, y_fred,
            bounds_error=False,
            fill_value=0
        )

        y_exp_interp = interp_exp(x_interp)
        y_fred_interp = interp_fred(x_interp)

        ax.plot(x_interp, y_exp_interp, label='Experimental')
        ax.plot(x_interp, y_fred_interp, label='FRED')

        ax.set_title(f'{energy} MeV')
        ax.grid(alpha=0.3)

    # Common labels
    fig.supxlabel('Depth (mm)')
    fig.supylabel(ylabel)

    # Single legend for entire figure
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=2)

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.show()


def plot_sigma_comparison_grid():

    fig, axes = plt.subplots(2, 4, figsize=(18, 10), sharex=True)
    axes = axes.flatten()

    for ax, energy in zip(axes, ENERGIES):

        sigmasX_exp, sigmasY_exp = get_experimental_sigmas_air(energy)

        sigmasX_fred = []
        sigmasY_fred = []

        for d in distances_mm:
            sx, sy, _ = get_fred_sigmas(energy, d)
            sigmasX_fred.append(sx)
            sigmasY_fred.append(sy)

        ax.plot(distances_mm, sigmasX_exp,
                color='k', marker='o', linestyle='-',
                label=r'Exp $\sigma_x$')

        ax.plot(distances_mm, sigmasY_exp,
                color='k', marker='s', linestyle='-',
                label=r'Exp $\sigma_y$')

        ax.plot(distances_mm, sigmasX_fred,
                color='tab:red', marker='o', linestyle='--',
                label=r'FRED $\sigma_x$')

        ax.plot(distances_mm, sigmasY_fred,
                color='tab:red', marker='s', linestyle='--',
                label=r'FRED $\sigma_y$')

        ax.set_title(f'{energy} MeV')
        ax.grid(alpha=0.3)

    fig.supxlabel('Distance from isocentre (mm)')
    fig.supylabel(r'$\sigma$ (mm)')

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=4)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

ENERGIES = [80, 100, 120, 140, 160, 180, 200, 220]  # MeV

# plot_IDD_comparison_grid(abs=True)
plot_sigma_comparison_grid()