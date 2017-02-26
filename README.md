# pyccutof

Python package to work with mass spectrometry data files generated from a JEOL AccuTOF mass spectrometer.

This code was developed for an academic research project and is in no way affiliated with or sponsored by JEOL. The included functions have been suitable for my purposes, but no guarantee is made of their more general applicability.

## Capabilities

- Parse spectra saved in raw data folders from AccuTOF mass spectrometer (including `MsData.FFT` and `MsData.FFC` binary data files generated from the onboard FastFlight-Plus Digital Signal Averager).
- Parse spectra exported from JEOL software to "JEOL-DX" format.
- Work with parsed spectra as a numpy array, either during an interactive Python session or as part of a user-generated script.
