import pytest
import pyccutof as ptof
import numpy as np
from pkg_resources import resource_filename

@pytest.fixture(scope='module')
def ffc():
    fn = resource_filename('pyccutof', 'data/sample_tenspec.FFC')
    return ptof.readffc(fn)

@pytest.fixture(scope='module')
def ffc_onespec():
    fn = resource_filename('pyccutof', 'data/sample_onespec.FFC')
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
    fn = resource_filename('pyccutof', 'data/sample_tenspec.FFT')
    assert ptof.import_fft(fn, ffc[1])

def test_extract_counts_parses_uncompressed_spectrum_correctly():
    # modify sample compressed spectrum from Fastflight manual, appendix d, so
    # that it is uncompressed and follows assumptions of
    # extract_counts_from_spectrum.
    uncompressed_spectrum = np.array([0xffffff, 0xff400f, 0x000010, 0xff8000,
                                      0x000000, 0xff0000, 0x000050, 0x000054,
                                      0x000045, 0x000060, 0x000050, 0x000050,
                                      0x000070, 0x000500, 0x000800, 0xffffff,
                                      0xff8018, 0xff8000, 0x000009, 0xff0000,
                                      0x000485, 0x000120, 0x000045, 0x000050,
                                      0x000070, 0x000500, 0x000800, 0x000485,
                                      0x000120, 0x000045, 0x000054, 0x000045,
                                      0x000060, 0x000050, 0x000054, 0x000045,
                                      0x000060, 0x003052, 0x000000])
    output = np.array([0x50, 0x54, 0x45, 0x60, 0x50, 0x50, 0x70, 0x500, 0x800,
                       0x485, 0x120, 0x45, 0x50, 0x70, 0x500, 0x800, 0x485,
                       0x120, 0x45, 0x54, 0x45, 0x60, 0x50, 0x54, 0x45, 0x60,
                       0x3052, 0x0])
    extracted = ptof.extract_counts_from_spectrum(uncompressed_spectrum)
    assert np.array_equal(extracted, output)

def test_extract_counts_catches_extended_address_gap():
    # check error is raised if the data is not in contiguous bins because the
    # extended address values don't line up properly
    ea_gap_spectrum = np.array([0xffffff, 0xff4007, 0x000010, 0xff8000,
                                0x000000, 0xff0000,
                                0x000050, # single data value in first frame
                                0xffffff, 0xff8008, 0xff8000,
                                0x000002, # second frame starts in bin *2*
                                0xff0000, 0x000485, 0x003052, 0x000000])
    with pytest.raises(ptof.ParserError):
        ptof.extract_counts_from_spectrum(ea_gap_spectrum)

def test_extract_counts_catches_unexpected_data_tag():
    # check error is raised if the data is not in contiguous bins because there
    # are unexpected data tag values (meaning compression took place)
    data_tag_spectrum = np.array([0xffffff, 0xffc00b, 0x000010, 0xff8000,
                                  0x000000, 0xff0000, 0x000050,
                                  0xff0010, # data tag skips ahead
                                  0x000045, 0x003052, 0x000000])
    with pytest.raises(ptof.ParserError):
        ptof.extract_counts_from_spectrum(data_tag_spectrum)
    pass

def test_extract_count_catches_mismatch_length_frame_header():
    # check error is raised if number of words in frame is not same as what the
    # frame header claims. Only expect to happen if data is malformed or
    # there's some unexpected bug in the code.
    length_mismatch_spectrum = np.array([0xffffff, 0xff4008, # claims 8 words
                                         0x000010, 0xff8000, 0x000000,
                                         0xff0000, 0x000050,
                                         0xffffff, 0xff8008, 0xff8000,
                                         0x000001, 0xff0000, 0x000485,
                                         0x003052, 0x000000])
    with pytest.raises(ptof.ParserError):
        ptof.extract_counts_from_spectrum(length_mismatch_spectrum)

def test_extract_count_catches_incorrect_frame_headers():
    # check error is raised if frame header bits 15 and 14 for a spectrum do
    # not go from "first frame ... middle frames ... last frame" or "first and
    # last frame" (single-frame spectrum). Only expect to happen if data is
    # malformed or there's unexpected bug in code.
    # For now, not an exhaustive test of all cases -- just consider one. Most
    # important that check for multiframe spectra is working.
    wrong_fh_bits_spectrum = np.array([0xffffff,
                                       0xff0007, # claims middle frame
                                       0x000010, 0xff8000, 0x000000, 0xff0000,
                                       0x000050, 0xffffff, 0xff8008, 0xff8000,
                                       0x000001, 0xff0000, 0x000485, 0x003052,
                                       0x000000])
    with pytest.raises(ptof.ParserError):
        ptof.extract_counts_from_spectrum(wrong_fh_bits_spectrum)

def test_import_fft_extract_counts_single_index_recs_row(ffc_onespec):
    # check for unexpected failure of single-spectrum data import
    fn = resource_filename('pyccutof', 'data/sample_onespec.FFT')
    fft_onespec = ptof.import_fft(fn, ffc_onespec[1])
    onespec = ptof.extract_counts_from_spectrum(fft_onespec[0])
    assert onespec.size == 13891
