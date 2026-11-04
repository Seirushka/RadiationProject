import numpy as np


def pow_interp(x, y, x_new, extrapolate=True, threshold=1e-20):
    """
    Кусочно-степенная интерполяция в лог-лог пространстве.
    """
    mask = (x > 0) & (y > 0)
    xp, yp = x[mask], y[mask]

    if len(xp) < 2:
        return np.zeros_like(x_new, dtype=float)

    lx, ly = np.log10(xp), np.log10(yp)
    k = (ly[1:] - ly[:-1]) / (lx[1:] - lx[:-1])

    y_vals = np.zeros_like(x_new, dtype=float)

    if extrapolate:
        inside = np.ones_like(x_new, dtype=bool)
    else:
        inside = (x_new >= xp.min()) & (x_new <= xp.max())

    if np.any(inside):
        xs = x_new[inside]

        idx = np.searchsorted(xp, xs, side='right') - 1
        idx = np.clip(idx, 0, len(k) - 1)

        y_left = yp[idx]
        x_left = xp[idx]
        k_interv = k[idx]

        y_vals[inside] = y_left * (xs / x_left) ** k_interv

        if extrapolate:
            right_ext = xs > xp.max()
            y_vals[right_ext & (y_vals < threshold)] = threshold

    return y_vals
