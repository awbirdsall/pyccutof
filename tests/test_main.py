import pytest
import pyccutof as ptof
import numpy as np

def test_upconvert_uint32_correct_calc():
    raw_bytes = np.array([0xff, 0xee, 0xdd, 0xcc, 0xbb, 0xaa])
    output_words = np.array([0xddeeff, 0xaabbcc])
    assert np.equal(ptof.upconvert_uint32(raw_bytes, 3), output_words).all()
