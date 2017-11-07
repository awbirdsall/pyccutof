# don't import readfft -- currently not working as "valid Win32"
# from readfftstream import readfft

import numpy as np
import os
import pandas as pd

def readffc(fn):
    '''Read an FFC file from the filename.

    The FFC record starts with a 4-byte header, and then continues with index
    records providing information about each spectrum in the corresponding FFT
    file.

    Parameters
    ----------
    fn : str
    Filename of FFC file to read.

    Returns
    -------
    numpts : int
    Header of FFC, "NumberOfPoints", converted to int.
    index_records : numpy.recarray
    recarray of spectra in corresponding FFT file, with field names 'protocol',
    'time', 'reserved', 'counts', 'offset', and 'timemultiplier'.

    '''
    ffc_dtype = np.dtype([('protocol', '<i4'),
                          ('time', '<i4'), # in units of "TimeMultiplier"
                          ('reserved', '<i4'), # number bytes
                          ('counts', '<i8'), # total ion counts above background
                          ('offset', '<i8'), # starting location byte in fft
                          ('timemultiplier', '<i8'),]) # 2.5 with dsp installed
                                                       # 5.0 without dsp
    with open(fn, 'rb') as f:
        header = f.read(4)
        index_records = np.fromfile(f, dtype=ffc_dtype)
    numpts = int(np.array([header]).view("int32"))
    return numpts, index_records

class ParserError(Exception):
    '''Raise exception when parser didn't work correctly.'''
    pass

def ibits(i, pos, l):
    '''Extract l bits of i, starting from pos.

    Convenience function, like fortran `ibits`.

    '''
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
def extract_counts_from_spectrum(word_list, clean_after_end=True):
    '''Use boolean mask to extract counts from spectrum.
    
    Return array includes first bin (timestamp) and last two bins (total ion
    count) of spectrum, even though they don't correspond to "real" ion counts.
    If the word list extends past a frame flagged as the last frame in
    a spectrum (as seen for first entry in a sample FFC), strip off all further
    words if `clean_after_end`.
    
    Requires brittle assumptions about structure of frames in spectrum:
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
    # check assumptions about location of frame header, external address, and
    # single data tag (see docstring)

    # start with indices of 0xffffff frame start words
    frame_start_idxs = np.flatnonzero(word_list==0xffffff)
    # all frames should have frame header as second word, which starts with ff
    frame_header_idxs = frame_start_idxs + 1
    frame_header_high_byte = ibits(word_list[frame_header_idxs], 16, 8)
    frame_header_correct = (frame_header_high_byte == 0xff).all()
    if not(frame_header_correct):
        raise ParserError("Incorrect value for frame header high bytes (0xff)")

    # strip values after first "final" frame, if clean_after_end
    # otherwise, if there are extra values, will throw ParserError below
    if clean_after_end:
        last_frames = ibits(word_list[frame_header_idxs], 15, 1) == 1
        first_last = frame_header_idxs[last_frames][0]
        # extract number of words in this last frame
        last_words = ibits(word_list[first_last], 0, 14)
        # strip off word_list after this frame
        word_list = word_list[:first_last+last_words-1]

        # recalculate values derived above
        frame_start_idxs = np.flatnonzero(word_list==0xffffff) 
        frame_header_idxs = frame_start_idxs + 1

    # check frame header spectrum positions are correct
    fh_pos = ibits(word_list[frame_header_idxs], 14, 2)
    if len(fh_pos)>1:
        first_correct = (fh_pos[0] == 0b01)
        mid_correct = (np.all(fh_pos[1:-1] == 0b00))
        last_correct = (fh_pos[-1] == 0b10)
        fh_correct = first_correct and mid_correct and last_correct
    else: # single frame header means single-frame spectrum
        fh_correct = (fh_pos[0] == 0b11)
    if not fh_correct:
        raise ParserError("Frame header spectrum positions are incorrect. Bad "
                          "input or problem with parsing.")

    # all frames should have external address tag as third word (fourth for
    # first frame, which containes timestamp as third word)
    ea_tag_idxs = frame_start_idxs + 2
    ea_tag_idxs[0] = ea_tag_idxs[0] + 1
    ea_tag_correct = (word_list[ea_tag_idxs] == 0xff8000).all()

    # all frames should have data tag 0xff0000 as fifth word (sixth for first
    # frame)
    data_tag_idxs = frame_start_idxs + 4
    # index is one larger for first frame since it includes timestamp as third
    # word.
    data_tag_idxs[0] = data_tag_idxs[0] + 1
    data_tag_correct = (word_list[data_tag_idxs] == 0xff0000).all()
    if not(ea_tag_correct and data_tag_correct):
        raise ParserError("Unallowed external address or data tag values for "
                          "extract_counts_from_spectrum")
    
    # external address values should match with expectation of no skipped bins
    # calculate number of data points by assuming each word in frame, other
    # than those accounted for above, is a data point. Count length of frames
    # using locations of 0xffffff as references
    framelengths = np.zeros_like(frame_start_idxs)
    framelengths[:-1] = frame_start_idxs[1:]-frame_start_idxs[:-1]
    framelengths[-1] = len(word_list) - frame_start_idxs[-1]
    numdata = framelengths - 5
    numdata[0] = numdata[0] - 1 # timestamp/protocol word in first frame
    ea_expected_values = np.roll(numdata.cumsum(), 1)
    ea_expected_values[0] = 0
    # extract actual values
    ea_value_idxs = frame_start_idxs + 3
    ea_value_idxs[0] = ea_value_idxs[0] + 1
    ea_actual_values = word_list[ea_value_idxs]
    if not(np.array_equal(ea_expected_values, ea_actual_values)):
        raise ParserError("Extended address values are inconsistent with no "
                          "skipped bins.")

    # check agreement between frame header lengths and actual frame lengths
    fh_lengths = ibits(word_list[frame_header_idxs], 0, 14)
    if not np.array_equal(fh_lengths, framelengths):
        raise ParserError("Frame header frame length inconsistent with actual "
                          "number of words in frame")

    # finally ready to prepare output data_words!
    # use Boolean mask to strip all non-data values
    mask = np.ones(word_list.shape, dtype=bool)
    mask[frame_start_idxs] = False
    mask[frame_header_idxs] = False
    mask[ea_tag_idxs] = False
    mask[ea_value_idxs] = False
    mask[data_tag_idxs] = False
    # stripping out timestamp_protocol in header region, but note that first
    # "data" point in spectrum also appears to have the timestamp value, so
    # timestamp is still included in what is returned
    timestamp_protocol = 2
    mask[timestamp_protocol] = False
    data_words = word_list[mask]

    # no more tagged words (0xff####) should be present in data_words
    if (ibits(data_words, 16, 8)==0xff).any():
        raise ParserError("Unexpected tagged words in spectrum.")
        
    return data_words

def import_fft(fn, index_recs):
    '''Read FFT file and return list of spectra, each containing raw words.
    
    Requires filename and list of offsets (from FFC file).
    
    Assume the spectrum lasts for the length of the corresponding 'reserved'
    value in the FFC. Sample data shows that the 'reserved' value is usually
    okay, except it's too big for the first entry. Don't clean up issues like
    this here; instead, use `extract_counts_from_spectrum()`.
    '''
    with open(fn, 'rb') as f:
        # load all spectra into memory
        # timestamp_list = []
        spectra_list = []
        for reserved, offset in zip(index_recs['reserved'],
                                    index_recs['offset']):
            f.seek(offset, os.SEEK_SET)
            spectrum_raw = np.fromfile(f, dtype=np.uint8, count=reserved)
            if spectrum_raw.size == 0:
                raise ValueError("no spectrum in {} at offset {}"
                                 .format(f, offset))
            # silently strip off incomplete byte at end, if any
            if spectrum_raw.size%3 != 0:
                spectrum_raw = spectrum_raw[:-(spectrum_raw.size%3)]
            spectrum = upconvert_uint32(spectrum_raw, 3)
            spectrum_timestamp = ibits(spectrum[2], 0, 21)
            #timestamp_list.append(spectrum_timestamp)
            spectra_list.append(spectrum)
    return spectra_list

def read_fft_lazy(fn, index_recs):
    spectra_list = import_fft(fn, index_recs)
    return np.array([extract_counts_from_spectrum(s) for s in spectra_list])

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
    '''Read jeol-dx `.jmc` file as pandas DataFrame.
    '''
    df = pd.read_csv(fn, sep='\t', comment='#', names=['time', 'signal'],
                     index_col='time')
    return df
