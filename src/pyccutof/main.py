# don't import readfft -- currently not working as "valid Win32"
# from readfftstream import readfft

import numpy as np
import os
import pandas as pd

def readffc(fn):
    ffc_dtype = np.dtype([('protocol', '<i4'),
                          ('time', '<i4'), # multiply by TimeMultiplier to obtain ms
                          ('reserved', '<i4'), # number bytes in record for TOF spectrum
                          ('counts', '<i8'), # total ion counts above background in spectrum
                          ('offset', '<i8'), # starting location of spectrum in FFT file
                          ('timemultiplier', '<i8'),]) # 2.5 with dsp; 5.0 without (so 2.5)
    with open(fn, 'rb') as f:
        header = f.read(4) # numberofpoints
        index_records = np.fromfile(f, dtype=ffc_dtype)
    return header, index_records

class ParserError(Exception):
    # Raise exception when parser didn't work correctly
    pass

def ibits(i, pos, l):
    # convenience bitwise function: extract l bits of i, starting from pos.
    # like fortran ibits
    return (i>>pos)&~(-1<<l)

def upconvert_uint32(raw_bytes, num_bytes):
    num_words = raw_bytes.size/num_bytes
    if num_words != int(num_words):
        ValueError("raw bytes are not evenly divisible into native num_bytes")
    four_bytes = np.zeros((int(num_words), 4), dtype=np.uint8)
    four_bytes[:, :num_bytes] = raw_bytes.reshape(-1, 3)
    words = four_bytes.view('uint32').reshape(four_bytes.shape[:-1])
    return words

def is_tag(word):
    # tags have ff as most significant byte
    high_ff = ibits(word, 16, 8)==0xff
    return high_ff

def is_data_tag(word):
    # check 0<xxxx<8000 for FFxxxx
    # except drop high bit (sets sign)
    # 0xff8000 (ext_addr_tag) should return false
    small_lower_bytes = ibits(word, 0, 15)<0x8000
    return is_tag(word) and not(is_ext_addr_tag(word)) and small_lower_bytes

def is_ext_addr_tag(word):
    return word == 0xff8000

def extract_data_record(word):
    '''bits 0-14 are value and bit 15 is sign'''
    value = ibits(word, 0, 15)
    if (word>>15)&1 == 1:
        value = -value
    return value

# for lazy way
def extract_counts_from_spectrum(word_list):
    '''Use boolean mask to extract counts from spectrum.
    
    Return array includes first bin (timestamp) and last two bins (total ion
    count) of spectrum, even though they don't correspond to "real" ion counts.
    
    Brittle assumptions about structure of frames in spectrum (hex words):
    ffffff # start of frame
    ffxxxx # frame header: length + position in spectrum
    xxxxxx # spectrum protocol + timestamp (first frame only)
    ff8000 # extended address tag
    xxxxxx # extended address value
    ff0000 # data tag offset of 0 from extended address
    xxxxxx # first data value
    ...... # continuation of contiguous data (no ffxxxx words skipping bins)
    xxxxxx # last data value
    xxxxxx # total ion sum, low word (last frame only)
    xxxxxx # total ion sum, high word (last frame only)

    NB: The "total ion sum" will not equal the sum of the raw data values
    in the spectrum. According to the FastFlight manual (Sect. 3.15 "Data
    Compression"), the total ion count is calculated using the net peak area
    above background, where peaks and background are determined by the digital
    signal processor, regardless of whether data compression is on or off.
    '''
    # FIXME what if there are a different number of frames per spectrum? (expected
    # if mass range differs)
    # TODO check consistency of all descriptions
    
    # check assumptions about structure
    # (use flatnonzero rather than where for ease of math)
    frame_start_idxs = np.flatnonzero(word_list==0xffffff) 
    # all frames should have frame header as second word
    frame_header_idxs = frame_start_idxs + 1
    frame_header_expected = (ibits(word_list[frame_header_idxs], 16, 8) == 0xff).all()
    # all frames should have external address tag as third word
    ea_tag_idxs = frame_start_idxs + 2
    # index is one larger for first frame since it includes timestamp as third word.
    ea_tag_idxs[0] = ea_tag_idxs[0] + 1
    ea_tag_expected = (word_list[ea_tag_idxs] == 0xff8000).all()
    # all frames should have data tag of 0xff0000 as fifth word (sixth for first frame)
    data_tag_idxs = frame_start_idxs + 4
    # index is one larger for first frame since it includes timestamp as third word.
    data_tag_idxs[0] = data_tag_idxs[0] + 1
    data_tag_expected = (word_list[data_tag_idxs] == 0xff0000).all()
    if not(frame_header_expected and ea_tag_expected and data_tag_expected):
        raise ParserError("Tags at starts of frames not as required for lazy parsing.")
    
    # external address values should match with expectation of no skipped bins
    framelengths = ibits(word_list[frame_header_idxs], 0, 14)
    # calculate number of data points by assuming each word in frame, other than
    # those accounted for above, is a data point
    numdata = framelengths - 5
    numdata[0] = numdata[0] - 1 # timestamp/protocol word in first frame
    ea_expected_values = np.roll(numdata.cumsum(), 1)
    ea_expected_values[0] = 0
    # extract actual values
    ea_value_idxs = frame_start_idxs + 3
    ea_value_idxs[0] = ea_value_idxs[0] + 1
    ea_actual_values = word_list[ea_value_idxs]
    if not(np.array_equal(ea_expected_values, ea_actual_values)):
        raise ParserError("Extended address values inconsistent with no skipped bins.")

    # use Boolean mask to strip all non-data values
    mask = np.ones(word_list.shape, dtype=bool)
    mask[frame_start_idxs] = False
    mask[frame_header_idxs] = False
    mask[ea_tag_idxs] = False
    mask[ea_value_idxs] = False
    mask[data_tag_idxs] = False
    timestamp_protocol = 2
    mask[timestamp_protocol] = False
    data_words = word_list[mask]

    # no more tagged words (0xff####) should be present in data_words
    if (ibits(data_words, 16, 8)==0xff).any():
        raise ParserError("Unexpected tagged words in spectrum.")
        
    return data_words

def import_fft(fn, index_recs):
    '''Read FFT file and return list of spectra.
    
    Requires filename and list of offsets (from FFC file).
    
    Assume the spectrum lasts until the start of the
    following spectrum. This allows for gulping up the spectrum at once
    but relies on there not being
    any cruft between spectra. Also requires stripping off 
    partial final spectrum from last full spectrum.
    '''
    if index_recs.shape[0] == 1:
        raise ValueError("import_fft cannot handle single-row index_recs.")
    with open(fn, 'rb') as f:
        # load all spectra into memory
        # timestamp_list = []
        spectra_list = []
        for i, offset in enumerate(index_recs['offset']):
            # go to start of spectrum
            f.seek(offset, os.SEEK_SET)

            if i+1<index_recs['offset'].size:
                spectrum_size = index_recs['offset'][i+1]-offset
#             else:
                # for now, assume we can use penultimate spectrum_size for final spectrum
                # This breaks import_fft when index_recs only has 1 row.
                # TODO: better way?
                # final spectrum grabs the rest (including possible fragment):
#                                 spectrum_size = -1
                # TODO: fix that if only pass portion of index_recs, "final" spectrum
                # isn't final, will grab *entire* rest of file, which could be huge
                # this code leads to only grabbing the first frame
#                 first_two_words = np.fromfile(f, dtype=np.uint8, spectrum_size = 6)
#                 fh = upconvert_uint32(first_two_words[3:], 3)[0] # frame header
#                 framelength = ibits(fh, 0, 14) # number of entries
#                 spectrum_size = framelength*3
#                 # return to start of first frame of final spectrum
#                 f.seek(offset, os.SEEK_SET)
            spectrum_raw = np.fromfile(f, dtype=np.uint8, count=spectrum_size)
            # strip off incomplete byte at end, if any
            if spectrum_raw.size%3 != 0:
                spectrum_raw = spectrum_raw[:-(spectrum_raw.size%3)]
            spectrum = upconvert_uint32(spectrum_raw, 3)
            spectrum_timestamp = ibits(spectrum[2], 0, 21)
            #timestamp_list.append(spectrum_timestamp)
            spectra_list.append(spectrum)
    # strip any trailing spectrum fragment from last spectra
    # (might happen if collection terminated manually)
    if spectra_list[-1].size != spectra_list[0].size:
        spectra_list[-1] = spectra_list[-1][:spectra_list[0].size]
    return spectra_list

def read_fft_lazy(fn, index_recs):
    spectra_list = import_fft(fn, index_recs)
    return np.array([extract_counts_from_spectrum(spectrum) for spectrum in spectra_list])

# disabled until figure out correct packaging of fortran-based readfft
# def read_fft_f2py(fn, numspec, index_recs):
#     # same as above but use `lazy` method to get specbins
#     startbyte = index_recs['offset'][0]+1
#     # kills python kernel if specbins wrong...
#     # pass more than 1 row of index_recs to read_fft_lazy because of bug
#     # with single index_recs
#     specbins = read_fft_lazy(fn, index_recs[:2]).shape[1]
#     specbytes = index_recs['offset'][1]-index_recs['offset'][0] + 1 # plus 1 for luck
#     # f2py-compiled call:
#     spectra = readfft(numspec, fn, startbyte, specbins, specbytes)
#     # just return slice of output for purpose of comparison
#     return spectra[:,:,1]

def read_jeoldx(fn):
    '''read jeol-dx `.jmc` file as pandas dataframe.'''
    df = pd.read_csv(fn, sep='\t', comment='#', names=['time', 'signal'], index_col='time')
    return df
