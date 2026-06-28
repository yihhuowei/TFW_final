import numpy as np
import pywt
import matplotlib.pyplot as plt
from pathlib import Path

OUTDIR = Path("./")  # 改成 Path("/mnt/data") 也可
WAVELET = "db4"
CWT_WAVELET = "morl"
CWT_MAX_SCALE = 128
SIGMA = 0.3
N = 2048
RANDOM_SEED = 0

np.random.seed(RANDOM_SEED)
t = np.linspace(0, 1, N)
signal_clean = np.sin(2*np.pi*5*t) + np.sin(2*np.pi*40*t*t)  # sine + chirp
noise = SIGMA * np.random.randn(N)
signal_noisy = signal_clean + noise

def wavelet_denoise_universal(x, sigma, wavelet="db4", level=None, mode="symmetric"):
    coeffs = pywt.wavedec(x, wavelet, level=level, mode=mode)
    cA, details = coeffs[0], coeffs[1:]
    lam = sigma * np.sqrt(2*np.log(len(x)))  # Universal threshold (Donoho)
    details_th = [pywt.threshold(d, lam, mode='soft') for d in details]
    coeffs_th = [cA] + details_th
    x_rec = pywt.waverec(coeffs_th, wavelet, mode=mode)
    return x_rec, coeffs, coeffs_th, lam

signal_uni, coeffs_noisy, coeffs_th, lam = wavelet_denoise_universal(
    signal_noisy, SIGMA, wavelet=WAVELET
)


def dwt_details_to_matrix(coeffs, N):

    details = coeffs[1:]
    J = len(details)
    mat = np.zeros((J, N))
    for j, d in enumerate(details, start=1):
        up = np.repeat(d, 2**j)
        if len(up) < N:
            up = np.pad(up, (0, N - len(up)), mode='edge')
        else:
            up = up[:N]
        mat[J - j, :] = up
    return mat

def savefig(fname):
    OUTDIR.mkdir(parents=True, exist_ok=True)
    p = OUTDIR / fname
    plt.tight_layout()
    plt.savefig(p, dpi=200)
    plt.show()
    print(f"saved: {p.resolve()}")

widths = np.arange(1, CWT_MAX_SCALE)
cwt_coeffs, cwt_freqs = pywt.cwt(signal_noisy, widths, CWT_WAVELET)
cwt_power = np.abs(cwt_coeffs)

# DWT details (noisy vs after soft-threshold)
mat_dwt_noisy = dwt_details_to_matrix(coeffs_noisy, N)
mat_dwt_soft  = dwt_details_to_matrix(coeffs_th,     N)

# 1) Clean signal
plt.figure(figsize=(12,4))
plt.plot(t, signal_clean, label="Clean signal")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.title("Clean signal (ground truth)")
plt.legend()
savefig("fig1_clean_signal.png")

# 2) Known Gaussian noise
plt.figure(figsize=(12,4))
plt.plot(t, noise, label=f"Gaussian noise (σ = {SIGMA:.3f})")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.title("Known Gaussian noise")
plt.legend()
savefig("fig2_known_noise.png")

# 3) Noisy observation
plt.figure(figsize=(12,4))
plt.plot(t, signal_noisy, label="Noisy signal")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.title("Signal with additive Gaussian noise")
plt.legend()
savefig("fig3_noisy_signal.png")


# 4) CWT scalogram (2D)
plt.figure(figsize=(12,6))
plt.imshow(cwt_power, extent=[t[0], t[-1], widths[-1], widths[0]], aspect='auto')
plt.xlabel("Time (s)")
plt.ylabel("Scale (lower = higher freq)")
plt.title("Scalogram (CWT magnitude) of noisy signal")
savefig("fig4_scalogram_noisy.png")

# 5) DWT detail coefficients (noisy) pseudo-2D map
plt.figure(figsize=(12,6))
plt.imshow(np.abs(mat_dwt_noisy), aspect='auto', vmin=0, vmax=8)
plt.colorbar(label="|detail coeff| (upsampled)")
plt.xlabel("Time (s)")
plt.ylabel("DWT level (top = coarse)")
plt.title(f"DWT detail coefficients (noisy) – wavelet = {WAVELET}")
savefig("fig5_dwt_noisy_coeffs.png")

# 6) DWT detail coefficients after soft-threshold
plt.figure(figsize=(12,6))
plt.imshow(np.abs(mat_dwt_soft), aspect='auto', vmin=0, vmax=8)
plt.colorbar(label="|detail coeff| (upsampled)")
plt.xlabel("Time (s)")
plt.ylabel("DWT level (top = coarse)")
plt.title(f"DWT detail coefficients after soft-threshold (λ = {lam:.3f})")
savefig("fig6_dwt_soft_coeffs.png")

# 7) Reconstruction
# comparison
plt.figure(figsize=(12,4))
plt.plot(t, signal_noisy, label="Noisy", alpha=0.6)
plt.plot(t, signal_uni, label="Denoised (Universal soft-threshold)")
plt.plot(t, signal_clean, label="Clean", linestyle='--')
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.title(f"Reconstruction after soft-thresholding (λ = {lam:.3f})")
plt.legend()
savefig("fig7_reconstruction_compare.png")

# 8) Denoised signal
plt.figure(figsize=(12,4))
plt.plot(t, signal_uni, label="Denoised (Universal soft-threshold)")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.title(f"Reconstruction after soft-thresholding (λ = {lam:.3f})")
plt.legend()
savefig("fig8_denoised_signal.png")