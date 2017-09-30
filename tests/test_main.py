import pytest
import pyccutof as ptof
import numpy as np
from pkg_resources import resource_filename

@pytest.fixture(scope='module')
def ffc():
    fn = resource_filename('pyccutof', 'data/sample.FFC')
    return ptof.readffc(fn)

def test_upconvert_uint32_catches_size_mismatch():
    # should complain if raw_bytes don't fit into the number of bytes per word
    raw_bytes_mismatch = np.array([0xff, 0xee, 0xdd, 0xcc, 0xbb])
    with pytest.raises(ValueError):
        ptof.upconvert_uint32(raw_bytes_mismatch, 3)

def test_upconvert_uint32_correct_calc():
    raw_bytes = np.array([0xff, 0xee, 0xdd, 0xcc, 0xbb, 0xaa])
    output_words = np.array([0xddeeff, 0xaabbcc])
    assert np.equal(ptof.upconvert_uint32(raw_bytes, 3), output_words).all()

def test_readffc_returns_value(ffc):
    assert ffc

def test_import_fft_returns_value(ffc):
    fn = resource_filename('pyccutof', 'data/sample.FFT')
    assert ptof.import_fft(fn, ffc[1])
