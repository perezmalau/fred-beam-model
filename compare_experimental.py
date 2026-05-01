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

def get_experimental_sigmas_air(energy):
    df = pd.read_excel("UCLH_Spot_Profiles_All_Sigmas.xlsx", sheet_name="Sheet3")
    row = df[df["Energy"] == energy]
    row = row.iloc[0]
    sigma_x = [row[f"sigmaX_{int(d/10)}"] for d in distances_mm]
    sigma_y = [row[f"sigmaY_{int(d/10)}"] for d in distances_mm]
    return sigma_x, sigma_y

def get_experimental_IDD_water(energy, norm=False):
    """Retrieve experimental data from IDD measurements for a given energy."""
    df = pd.read_excel("UCLH_reference_IDD.xlsx", sheet_name=str(energy))
    depths = df['Depth (mm)']
    if norm:
        dose = df['Dose (normalised to 1.0 at 20mm deep)']
    else:
        dose = df['Dose (Gy/MU/mm^2)']
    return depths, dose

def get_fred_IDD(energy, norm=False, norm_depth=20):
    """Retrieve MC IDD, option to normalise to a given depth."""
    dose_img = ft.readMHD(f"out_{energy:.1f}MeV/Dose.mhd")
    dose_array = sitk.GetArrayFromImage(dose_img)
    dx, dy, dz = dose_img.GetSpacing()
    nz, ny, nx = dose_array.shape
    depths = np.arange(ny) * dy

    idd = np.sum(dose_array, axis=(0, 2)) #/ (nx * dx * nz * dz)

    if norm:
        dose_at_norm = float(np.interp(norm_depth, depths, idd))
        idd = idd / dose_at_norm

    return depths, idd

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


def compare_IDD(energy, norm=False):
    """Plots the relevant IDDs and calculates the ratio/scaling between the two."""
    x_exp, y_exp = get_experimental_IDD_water(energy, norm=norm)
    x_fred, y_fred = get_fred_IDD(energy, norm=norm)

    x_interp = np.linspace(7, 400, 397)
    interp_exp = interp1d(x_exp, y_exp, bounds_error=False, fill_value=0)
    interp_fred = interp1d(x_fred, y_fred, bounds_error=False, fill_value=0)

    y_exp_interp = interp_exp(x_interp)
    y_fred_interp = interp_fred(x_interp)

    plt.figure()
    plt.plot(x_interp, y_exp_interp, label='Experimental')
    plt.plot(x_interp, y_fred_interp, label='FRED')
    plt.legend()
    plt.xlabel('Depth (mm)')
    plt.ylabel('IDD')
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


energies = [80, 100, 120, 140, 160, 180, 200]
for energy in energies:
    # Uncomment if you just ran a simulation with the water CT
    # compare_IDD(e, norm=False)

    # Uncomment if ran a simulation with the water CT
    compare_sigmas(energy)