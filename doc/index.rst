.. pyccutof documentation master file, created by
   sphinx-quickstart on Tue Nov 13 08:12:50 2018.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to pyccutof's documentation!
====================================

.. toctree::
   :hidden:

   Home <self>

pyccutof is a Python package for working with JEOL AccuTOF mass spectrometer data files.

The project repository is hosted `on Github <https://github.com/awbirdsall/pyccutof/>`_.

Installation
------------

:code:`pip install pyccutof`

Example usage
-------------

Given the FFC and FFT files, along with a previously determined calibration curve, create a pandas DataFrame containing all mass spectra in the FFT file, with m/z value as the index and mass spectrum timestamp as column name::

    import numpy as np
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
    mz = pt.apply_mz_cal(fft, cal_curve)
    df_specs = pt.create_df_specs(fft, mz)

The DataFrame :code:`df_specs` can then be further analyzed or saved to disk (e.g., :code:`df_specs.to_csv()`, :code:`df_specs.to_pickle()`, ...).

Main module
-----------

Import pyccutof using :code:`import pyccutof`. This imports the entirety of :code:`pyccutof.main`, which contains the following (descriptions auto-generated from docstrings):

.. automodule:: pyccutof.main
    :members:



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
