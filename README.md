# pyccutof

Python package to work with mass spectrometry data files generated from a JEOL AccuTOF mass spectrometer.

This code was developed for an academic research project and is in no way affiliated with or sponsored by JEOL. The included functions have been suitable for my purposes, but no guarantee is made of their more general applicability.

## Capabilities

- Parse spectra saved in raw data folders from AccuTOF mass spectrometer (including `MsData.FFT` and `MsData.FFC` binary data files generated from the onboard FastFlight-Plus Digital Signal Averager).
- Parse spectra exported from JEOL software to "JEOL-DX" format.
- Work with parsed spectra as a numpy array, either during an interactive Python session or as part of a user-generated script.

## Example usage

Given the FFC and FFT files, along with a previously determined calibration curve, create a DataFrame containing all mass spectra in the FFT file, with m/z value as the index and mass spectrum timestamp as column name:

```
import pyccutof as pt

FFC_FN = "MsData.FFC"
FFT_FN = "MsData.FFT"
cal_curve = np.poly1d([
                8.97324469e-19,
                -5.01087595e-14,
                3.47362513e-07,
                4.90251082e-03,
                1.73793870e+01
                ])

numpts, index_records = pt.readffc(FFC_FN)
fft = pt.read_fft_lazy(FFT_FN, index_records)
mz = pt.apply_mz_cal(fft, calpoly)
df_specs = pt.create_df_specs(fft, mz)
```

The DataFrame `df_specs` can then be further analyzed or saved to disk (e.g., `df_specs.to_csv()`, `df_specs.to_pickle()`, ...).

## Documentation

Documentation is hosted at pyccutof.readthedocs.io and created with `sphinx`. This includes using docstrings written in `numpydocs` style.
