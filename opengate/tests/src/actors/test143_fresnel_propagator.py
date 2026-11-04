#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Analytic tests of the Fresnel propagator opengate.fresnel_propagation.propagate.

Sign convention (the one of the field recorded by the WavefrontActor): time
dependence exp(-i w t), plane wave exp(+i k z), phase in matter exp(-i k delta L).
With the carrier exp(i k z) removed, the paraxial wave equation is

    dPsi/dz = (i / 2k) Laplacian_perp Psi.

With Psi(x) = sum Psi^(f) exp(+2 pi i f x) (inverse DFT), the Laplacian is
-4 pi^2 f^2, so Psi^(f, z) = Psi^(f, 0) exp(-i pi lambda z f^2): this is the
transfer function H of the module. In this convention a wave diverging from a
point source at distance R has the phase +pi r^2 / (lambda R) on a plane, and a
converging lens of focal length f the phase -pi r^2 / (lambda f).

Gaussian beam (T2, T4). The solution of the paraxial equation is

    Psi(r, z) = (q0 / q(z)) exp(i k r^2 / (2 q(z))),  q(z) = q0 + z,

with q0 = -i z_R and z_R = pi w0^2 / lambda (Psi(r, 0) = exp(-r^2 / w0^2)).
Writing 1/q = (z + i z_R) / (z^2 + z_R^2):
  - the real part of i k r^2 / (2q) is -r^2 / w(z)^2, with
    w(z) = w0 sqrt(1 + (z / z_R)^2);
  - its imaginary part is +k r^2 / (2 R(z)), R(z) = z (1 + (z_R / z)^2), the
    curvature of a diverging wave (positive, as for the point source);
  - q0 / q = 1 / (1 + i z / z_R) has the phase -arctan(z / z_R) (Gouy phase).
Hence the phase is phi(r, z) = -arctan(z / z_R) + k r^2 / (2 R(z)). In 2D with a
separable (elliptical) beam, each axis contributes the factor
(q0a / qa)^(1/2) exp(i k a^2 / (2 qa)), so the Gouy phase is
-(arctan(z / z_Rx) + arctan(z / z_Ry)) / 2. The intensity is proportional to
exp(-2 a^2 / w_a^2) on each axis, so its second moment is <a^2> = w_a^2 / 4.

A thin lens multiplies the field by exp(-i k r^2 / (2 f)), so 1/q' = 1/q0 - 1/f
and the axial intensity is |q' / (q' + z)|^2, maximum at the waist
z_w = f / (1 + (f / z_R)^2), just before f.

Fresnel scaling theorem (T5). For a point source at R1, the field after the plane
is exp(+i pi r^2 / (lambda R1)) T(r). Propagated by D, it equals

    (1 / M) exp(i pi r^2 / (lambda (R1 + D))) [P_{z_eff} T](r / M),

with M = (R1 + D) / R1 and z_eff = D / M. The intensity at the detector is thus
|P_{z_eff} T|^2 / M^2 evaluated at r / M: the propagation uses the pixel of the
plane, and the detector pixel is the plane pixel times M.
"""

import time

import numpy as np
from scipy.interpolate import RectBivariateSpline

import opengate.tests.utility as utility
from opengate import fresnel_propagation as fp

LAM = fp.wavelength_mm(20.0)  # mm
K = 2 * np.pi / LAM


def centred(n, d):
    """Coordinates of n pixels of size d, with 0 on the pixel n // 2."""
    return (np.arange(n) - n // 2) * d


def gaussian(ny, nx, dy, dx, w0y, w0x):
    y = centred(ny, dy)[:, None]
    x = centred(nx, dx)[None, :]
    return np.exp(-(y**2) / w0y**2 - x**2 / w0x**2).astype(np.complex128), y, x


def second_moment_width(i2d, coord, axis):
    """1/e^2 width 2 sqrt(<a^2>) of the marginal of i2d along axis."""
    marg = i2d.sum(axis=1 - axis)
    c = coord.ravel()
    return 2 * np.sqrt(np.sum(c**2 * marg) / np.sum(marg))


def gauss_case(ny, nx, dy, dx, w0y, w0x, label):
    """T2: widths and phases of a Gaussian beam at z = 0.5 and 2 z_R (of y)."""
    ok = True
    zry = np.pi * w0y**2 / LAM
    zrx = np.pi * w0x**2 / LAM
    errs = []
    for zf in (0.5, 2.0):
        z = zf * zry
        psi, y, x = gaussian(ny, nx, dy, dx, w0y, w0x)
        out = fp.propagate(psi, (dy, dx), LAM, z)
        i2d = np.abs(out) ** 2
        wy = w0y * np.sqrt(1 + (z / zry) ** 2)
        wx = w0x * np.sqrt(1 + (z / zrx) ** 2)
        ry = z * (1 + (zry / z) ** 2)
        rx = z * (1 + (zrx / z) ** 2)
        e_wy = abs(second_moment_width(i2d, y, 0) / wy - 1)
        e_wx = abs(second_moment_width(i2d, x, 1) / wx - 1)
        gouy = -(np.arctan(z / zrx) + np.arctan(z / zry)) / 2
        cy, cx = ny // 2, nx // 2
        e_g = abs(np.angle(out[cy, cx]) - gouy)
        # curvature term at r = w/2 (nearest pixel) on each axis
        jx = cx + int(round(wx / 2 / dx))
        jy = cy + int(round(wy / 2 / dy))
        ph_x = gouy + K * x.ravel()[jx] ** 2 / (2 * rx)
        ph_y = gouy + K * y.ravel()[jy] ** 2 / (2 * ry)
        e_cx = abs(np.angle(out[cy, jx]) - ph_x)
        e_cy = abs(np.angle(out[jy, cx]) - ph_y)
        e_ph = max(e_g, e_cx, e_cy)
        this = max(e_wx, e_wy) < 1e-6 and e_ph < 1e-6
        ok &= this
        errs.append((zf, max(e_wx, e_wy), e_ph))
        utility.print_test(
            this,
            f"T2 {label} z = {zf} z_R: w_y = {wy * 1e3:.4f} um, w_x = "
            f"{wx * 1e3:.4f} um, width rel. err. {max(e_wx, e_wy):.2e}; "
            f"Gouy {gouy:.6f} rad, phase at w/2 x {ph_x:.6f} y {ph_y:.6f} rad, "
            f"phase err. {e_ph:.2e} rad",
        )
    return ok, errs


def axis_intensity(phase_sign, f, n, d, w0, zs):
    psi, y, x = gaussian(n, n, d, d, w0, w0)
    lens = np.exp(phase_sign * 1j * np.pi * (x**2 + y**2) / (LAM * f))
    out = []
    for z in zs:
        res = fp.propagate(psi * lens, (d, d), LAM, z)
        out.append(abs(res[n // 2, n // 2]) ** 2)
    return np.array(out)


def spline_on(img, coord_src, coord_dst_y, coord_dst_x):
    s = RectBivariateSpline(coord_src, coord_src, img, kx=5, ky=5)
    return s(coord_dst_y, coord_dst_x)


if __name__ == "__main__":
    t_start = time.time()
    is_ok = True

    # T1: plane wave
    psi = np.ones((256, 256), np.complex128)
    out = fp.propagate(psi.copy(), (1e-3, 1e-3), LAM, 100.0)
    err = np.max(np.abs(out - 1))
    is_ok &= utility.print_test(err < 1e-12, f"T1 plane wave, D = 100 mm: {err:.2e}")

    # D = 0 returns |psi|^2 (no propagation)
    rng = np.random.default_rng(3)
    psi = rng.normal(size=(64, 48)) + 1j * rng.normal(size=(64, 48))
    ok = True
    for pad in ("none", "zero", "mirror"):
        out = fp.propagate(psi.copy(), (1e-3, 2e-3), LAM, 0.0, padding=pad)
        err = np.max(np.abs(out - psi))
        ok &= err < 1e-15 and np.all(np.isfinite(out))
        ok &= np.allclose(np.abs(out) ** 2, np.abs(psi) ** 2, rtol=0, atol=0)
    is_ok &= utility.print_test(ok, f"D = 0: output = input (max err. {err:.1e})")

    # T2: Gaussian beam, square and rectangular grids
    w0 = 20e-3
    ok, errs_sq = gauss_case(1024, 1024, 1e-3, 1e-3, w0, w0, "1024x1024")
    is_ok &= ok
    # rectangular grid, dx != dy and an elliptical beam: a swap of the axes
    # (shape or spacing) changes both widths
    ok, errs_rc = gauss_case(1024, 512, 1e-3, 2e-3, 20e-3, 30e-3, "1024x512")
    is_ok &= ok
    psi, _, _ = gaussian(1024, 512, 1e-3, 2e-3, 20e-3, 30e-3)
    z = 2 * np.pi * 20e-3**2 / LAM
    wrong = fp.propagate(psi.copy(), (2e-3, 1e-3), LAM, z)
    right = fp.propagate(psi, (1e-3, 2e-3), LAM, z)
    diff = np.max(np.abs(np.abs(wrong) - np.abs(right)))
    is_ok &= utility.print_test(
        diff > 1e-2, f"T2 swapped (dx, dy) gives another field: max diff {diff:.3f}"
    )

    # Slit grid: the grid of the slits (P3), (ny, nx) = (64, 8000), dx = 0.05 um:
    # a Gaussian along x, uniform along y
    dx, w0x = 5e-5, 2e-3
    psi = np.tile(gaussian(1, 8000, 1.0, dx, 1.0, w0x)[0], (64, 1))
    zrx = np.pi * w0x**2 / LAM
    out = fp.propagate(psi, (1e-3, dx), LAM, zrx)
    wx = second_moment_width(np.abs(out) ** 2, centred(8000, dx), 1)
    e_w = abs(wx / (w0x * np.sqrt(2)) - 1)
    e_y = np.max(np.abs(out - out[:1]))
    ok = out.shape == (64, 8000) and e_w < 1e-6 and e_y < 1e-12
    is_ok &= utility.print_test(
        ok,
        f"Slit grid 64 x 8000: w_x(z_R) rel. err. {e_w:.1e}, "
        f"variation along y {e_y:.1e}",
    )

    # T3: Parseval
    psi = rng.normal(size=(300, 200)) + 1j * rng.normal(size=(300, 200))
    e0 = np.sum(np.abs(psi) ** 2)
    out = fp.propagate(psi, (1e-4, 2e-4), LAM, 50.0)
    err = abs(np.sum(np.abs(out) ** 2) / e0 - 1)
    is_ok &= utility.print_test(err < 1e-12, f"T3 Parseval: rel. err. {err:.2e}")
    is_ok &= utility.print_test(out is psi, "T3 padding none works in place")

    # T4: sign, thin lens f = 50 mm on a wide Gaussian
    f, n, d = 50.0, 1024, 4e-5
    w0 = n * d / 10
    zr = np.pi * w0**2 / LAM
    zs = np.array([0.8 * f, f, 1.2 * f])
    i_conv = axis_intensity(-1, f, n, d, w0, zs)
    q0 = -1j * zr
    qp = 1 / (1 / q0 - 1 / f)
    i_an = np.abs(qp / (qp + zs)) ** 2
    e_an = np.max(np.abs(i_conv / i_an - 1))
    ok = i_conv[1] > i_conv[0] and i_conv[1] > i_conv[2] and e_an < 1e-4
    is_ok &= utility.print_test(
        ok,
        f"T4 phase -pi r^2/(lambda f): I(0) = 1, I(0.8f, f, 1.2f) = "
        f"{i_conv[0]:.3f}, {i_conv[1]:.3f}, {i_conv[2]:.3f}; analytic "
        f"{i_an[0]:.3f}, {i_an[1]:.3f}, {i_an[2]:.3f} (rel. err. {e_an:.1e})",
    )
    z_w = f / (1 + (f / zr) ** 2)
    zscan = np.linspace(0.96 * f, 1.02 * f, 61)
    iscan = axis_intensity(-1, f, n, d, w0, zscan)
    z_max = zscan[np.argmax(iscan)]
    ok = abs(z_max - z_w) <= zscan[1] - zscan[0]
    is_ok &= utility.print_test(
        ok, f"T4 axial maximum at z = {z_max:.3f} mm, waist z_w = {z_w:.3f} mm"
    )
    zs_div = np.array([0.1, 0.4, 0.8, 1.0, 1.2]) * f
    i_div = axis_intensity(+1, f, n, d, w0, zs_div)
    ok = np.all(np.diff(i_div) < 0) and i_div[0] < 1
    is_ok &= utility.print_test(
        ok,
        "T4 phase +pi r^2/(lambda f): no focus, I(z) = "
        + ", ".join(f"{v:.3f}" for v in i_div),
    )

    # T5: Fresnel scaling theorem (cone beam, D31)
    r1, dist = 400.0, 100.0
    n, d = 2048, 1e-4
    xs = centred(n, d)
    y2, x2 = xs[:, None], xs[None, :]
    r2 = x2**2 + y2**2
    aperture = np.exp(-((r2 / 0.035**2) ** 4))  # smooth super-Gaussian window
    s, phi0 = 5e-3, -1.0
    obj = aperture * np.exp(1j * phi0 * np.exp(-r2 / s**2))
    samp = fp.check_sampling(d, n * d, LAM, dist)
    edge = 0.05 / (LAM * r1)  # local frequency of the spherical wave at r = 50 um
    ok = samp["ok"] and edge < 0.5 / d
    is_ok &= utility.print_test(
        ok,
        f"T5 direct grid {n}^2, {d * 1e3} um: {samp['message']} Local frequency "
        f"of the spherical wave {edge:.0f}/mm < Nyquist {0.5 / d:.0f}/mm",
    )
    sph = np.exp(1j * np.pi * r2 / (LAM * r1))
    i_dir = np.abs(fp.propagate(obj * sph, (d, d), LAM, dist)) ** 2
    m, z_eff = fp.cone_beam_scaling(r1, dist)
    roi = np.abs(xs) <= 0.02
    xr = xs[roi]
    ref = i_dir[np.ix_(roi, roi)]
    # module recipe: plane pixel in the FFT, z_eff, / M^2, detector coordinates x M
    i_mod, det_pix, det_org, meta = fp.propagate_to_detector(
        obj.copy(), (d, d), (xs[0], xs[0]), LAM, dist, source_distance_mm=r1
    )
    ok = meta["M"] == m and meta["z_eff_mm"] == z_eff
    f10 = meta["fresnel_number_10px"]
    ok &= len(f10) == 2 and f10[0] == f10[1] == fp.fresnel_number(d, LAM, z_eff)
    ok &= det_pix == (d * m, d * m) and det_org == (xs[0] * m, xs[0] * m)
    is_ok &= utility.print_test(ok, "T5 propagate_to_detector meta and pixel x M")
    i_mod = spline_on(i_mod, xs * m, xr, xr)
    e_mod = np.max(np.abs(i_mod - ref)) / np.max(ref)
    contrast = (ref.max() - ref.min()) / ref.max()
    is_ok &= utility.print_test(
        e_mod < 1e-3,
        f"T5 module recipe (M = {m}, z_eff = {z_eff} mm) vs direct spherical "
        f"propagation, |x|,|y| <= 20 um (contrast {contrast:.3f}): rel. err. "
        f"{e_mod:.2e}",
    )
    # PCI-GAMOS: pixel / M in the FFT and at the output (not an assertion)
    i_gam = fp.intensity(fp.propagate(obj.copy(), (d / m, d / m), LAM, z_eff), m)
    i_gam = spline_on(i_gam, xs / m, xr, xr)
    e_gam = np.max(np.abs(i_gam - ref)) / np.max(ref)
    print(f"T5 PCI-GAMOS recipe (pixel / M) vs direct propagation: rel. err. "
          f"{e_gam:.2e} (reported only)")

    # T7: padding, a slit next to the right border
    ny, nx, d = 64, 512, 1e-4
    psi = np.zeros((ny, nx), np.complex128)
    psi[:, nx - 12 : nx - 4] = 1.0
    zp = 50.0
    res = {}
    for pad in ("none", "zero", "mirror"):
        out = fp.propagate(psi.copy(), (d, d), LAM, zp, padding=pad)
        res[pad] = np.abs(out) ** 2
        ok = out.shape == psi.shape and out.dtype == np.complex128
        is_ok &= utility.print_test(ok, f"T7 padding {pad}: shape {out.shape}")
    left = {p: res[p][:, :32].max() / res[p].max() for p in res}
    ok = left["none"] > 0.1 and left["zero"] < 1e-2
    is_ok &= utility.print_test(
        ok,
        f"T7 max of |Psi|^2 on the 32 columns of the opposite (left) border / max: "
        f"none {left['none']:.3f}, zero {left['zero']:.1e}",
    )
    # mirror: a constant field stays constant, and the periodic wrap-around
    # fringe disappears as with zero padding
    flat = np.ones((64, 128), np.complex128)
    out = fp.propagate(flat.copy(), (d, d), LAM, zp, padding="mirror")
    err = np.max(np.abs(out - 1))
    ok = err < 1e-12 and left["mirror"] < 1e-2
    is_ok &= utility.print_test(
        ok,
        f"T7 mirror: constant field err. {err:.1e}, opposite border "
        f"{left['mirror']:.1e}",
    )
    out = fp.propagate(psi, (d, d), LAM, zp, padding="zero")
    ok = out is not psi and np.all(psi[:, nx - 12 : nx - 4] == 1.0)
    is_ok &= utility.print_test(ok, "T7 with padding the input is left unchanged")
    msg = None
    try:
        fp.propagate(psi, (d, d), LAM, zp, padding="foo")
    except ValueError as e:
        msg = str(e)
    is_ok &= utility.print_test(msg is not None, f"T7 unknown padding: {msg!r:.60}")

    dt = time.time() - t_start
    print(f"Total time {dt:.1f} s")
    utility.test_ok(is_ok)
