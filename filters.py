import math

import numpy as np
import scipy.signal as sig


def moving_average(x, n, type):
    if type == "ma":
        weights = np.ones(n)
    elif type == "ewma":
        weights = np.exp(np.linspace(-1., 0., n))
    weights /= weights.sum()

    out = np.zeros(x.shape)
    for i in range(x.shape[1]):
        out[:, i] = np.convolve(x[:, i], weights, mode="full")[:-(n - 1)]
        out[:n, i] = out[n, i]
    return out


def butter_filter(v: np.ndarray, smoothing: dict, t: np.ndarray):
    dt = np.mean(t[1:] - t[:-1])
    if type(smoothing["cutoff"]) == list:
        v_filt = np.zeros(v.shape)
        for i, cutoff in enumerate(smoothing["cutoff"]):
            w = 2 * dt * cutoff
            b, a = sig.butter(smoothing["order"], w, btype="lowpass")
            if smoothing["filtfilt"]:
                v_filt[:, i] = sig.filtfilt(b, a, v[:, i])
            else:
                v_filt[:, i] = sig.lfilter(b, a, v[:, i])
    else:
        w = 2*dt*smoothing["cutoff"]
        b, a = sig.butter(smoothing["order"], w, btype="lowpass")
        if smoothing["filtfilt"]:
            v_filt = sig.filtfilt(b, a, v, axis=0)
        else:
            v_filt = sig.lfilter(b, a, v, axis=0)
    return v_filt


# def butter_filter(x:np.array, w:float, dt:float):
#     ans = np.zeros(x.shape)
#     Q = 0.7071
#     K = np.tan(dt / (2.0 * w))
#     poly = K * K + K / Q + 1.0
#     a0 = 2.0 * (K * K - 1.0) / poly
#     a1 = (K * K - K / Q + 1.0) / poly
#     b0 = K * K / poly
#     b1 = 2.0 * b0
#     i0 = i1 = o0 = o1 = 0
#     for i, value in enumerate(x):
#         out = b0*value + b1*i0 + b0*i1 - a0*o0 - a1*o1
#         i1 = i0
#         i0 = value
#         o1 = o0
#         ans[i] = o0 = out
#     return ans


# 1 Euro Velocity Filter
class OneEuroFilters:
    def __init__(self, t0, vel0, min_cutoff=1.0, beta=0.0, d_cutoff=1.0):
        for d in range(3):
            setattr(self, f"OneEuroFilter{d}", OneEuroFilter(t0, vel0[d], min_cutoff, beta, d_cutoff))

    def filter(self, t: np.ndarray, vel: np.ndarray):
        v_body_filt = np.zeros((len(t), 3))
        for d in range(3):
            for i in range(1, len(t)):
                v_body_filt[i, d] = getattr(self, f"OneEuroFilter{d}")(t[i], vel[i, d])
        return v_body_filt


# 1 Euro Filter Implementation from: https://github.com/jaantollander/OneEuroFilter
class OneEuroFilter:
    def __init__(self, t0, x0, dx0=0.0, min_cutoff=1.0, beta=0.0, d_cutoff=1.0):
        """Initialize the one euro filter."""
        # The parameters.
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        # Previous values.
        self.x_prev = float(x0)
        self.dx_prev = float(dx0)
        self.t_prev = float(t0)

    def __call__(self, t, x):
        """Compute the filtered signal."""
        t_e = t - self.t_prev

        # The filtered derivative of the signal.
        a_d = self.smoothing_factor(t_e, self.d_cutoff)
        dx = (x - self.x_prev) / t_e
        dx_hat = self.exponential_smoothing(a_d, dx, self.dx_prev)

        # The filtered signal.
        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        a = self.smoothing_factor(t_e, cutoff)
        x_hat = self.exponential_smoothing(a, x, self.x_prev)

        # Memorize the previous values.
        self.x_prev = x_hat
        self.dx_prev = dx_hat
        self.t_prev = t

        return x_hat

    @staticmethod
    def smoothing_factor(t_e, cutoff):
        r = 2 * math.pi * cutoff * t_e
        return r / (r + 1)

    @staticmethod
    def exponential_smoothing(a, x, x_prev):
        return a * x + (1 - a) * x_prev