import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pyodbc
import scipy.integrate as spi
import scipy.signal as sps

# don't import readfft -- currently not working as "valid Win32"
# from readfftstream import readfft

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

def readmassrange(acqfn):
    '''Fetch StartMass and EndMass from acquisition data database file.
    '''
    cnxn = pyodbc.connect("Driver={{Microsoft Access Driver "
                          "(*.mdb)}};Dbq={};Uid=;Pwd=;".format(acqfn))
    cursor = cnxn.cursor()
    cursor.execute("SELECT StartMass FROM T_AcquisitionSettingTOFCollection")
    start = cursor.fetchone().StartMass
    cursor.execute("SELECT EndMass FROM T_AcquisitionSettingTOFCollection")
    end = cursor.fetchone().EndMass
    cnxn.close()
    return start, end

def readana(anafn):
    with open(anafn, 'rb') as f:
        f.seek(4, os.SEEK_SET) # seek past header
        data_raw = np.fromfile(f, dtype=np.uint8, count=-1).reshape((-1,14))
    time_bytes = data_raw[:,:6]
    data_bytes = data_raw[:,6:]
    times = upconvert_uint64(time_bytes, 6)
    # hacky conversion factor
    conv_slope = 500./0xffffffffffffffff
    conv_offset = -124
    data = upconvert_uint64(data_bytes, 8)*conv_slope+conv_offset
    return np.vstack([times, data])

def protocol_info(fftfn):
    # TODO -- see fastflight manual
    with open(fftfn, 'rb') as f:
        protocol_raw = np.fromfile(f, dtype=np.uint4, count=96)
    return protocol_raw

# cannot use this as written without working readfft
# def processdatafolder(fldr):
#     # mass spectra in FFT and FFC
#     fftfn = os.path.join(fldr, "MsData.FFT")
#     ffcfn = os.path.join(fldr, "MsData.FFC")
#     ffc = readffc(ffcfn)
#     fft = readfft(fftfn, ffc)
#     # analog signals
#     ana = [None, None]
#     for i in [1,2]:
#         anafn = "Analog"+str(i)+".7an"
#         anapth = os.path.join(fldr, anafn)
#         if os.path.isfile(anapth):
#             ana[i-1] = readana("Analog1.7an")
#     # mass range
#     acqfn = os.path.join(fldr, "AcquisitionData.7dd")
#     massrange = readmassrange(acqfn)
#     return ffc, fft, ana, massrange

def create_cal_poly(cal_times, cal_mz, deg=4):
    '''Create a calibration polynomial from fitting times to known m/z.
    '''
    cal_poly = np.poly1d(np.polyfit(cal_times, cal_mz, deg))
    return cal_poly

def apply_mz_cal(fft, cal_poly):
    '''Given fft array and calibration polynomial, create array of m/z values.

    This means every bin of counts in the fft is given a corresponding m/z.
    '''
    # all fft entries in each row are time bins, except first one and last two
    num_fft_time_bins = fft[:,1:-2].shape[1]
    time_bins = np.arange(1,num_fft_time_bins+1,step=1)
    mz_values = cal_poly(time_bins)
    return mz_values

def extract_tic(fft):
    '''Create dataframe of total ion chromatogram from fft.
    '''
    chrom_time = fft[:,0]*2.5e-3 # fft times are in units of 2.5 ms
    tic = fft[:,-2] # penultimate entry in each spectrum is total ion count
    # (caution: if tic very large, overflows into last entry. TODO: check for
    # this.)
    df_tic = pd.DataFrame({'chrom_time':chrom_time, 'tic':tic}).set_index('chrom_time')
    return df_tic

def create_df_specs(fft, mz):
    '''Create DataFrame of all spectra in fft, with shared m/z index.

    Assumes fft array is in the format passed from read_fft_lazy(), in which
    each row contains the timestamp in the first entry, the total ion count in
    the last two entries, and the mass spectrum in between.

    Array of m/z values can be constructed using apply_mz_cal().

    Parameters
    ----------
    fft : 2D numpy array
    Array of FastFlight data, as output from read_fft_lazy.
    mz : 1D numpy array
    Array of m/z values corresponding to intensities in each fft spectrum.

    Returns
    -------
    df : pd.DataFrame
    DataFrame in which each row is a single m/z (m/z value is index), and each
    column is a different chromatogram time (time is column name).
    '''
    # in fft, timestamps are given in units of 2.5 ms
    timestamps = fft[:,0]*2.5e-3
    specs = fft[:,1:-2]
    data_dict = {timestamp: spec for timestamp,spec in zip(timestamps,specs)}
    data_dict.update({'mz': mz})
    df = pd.DataFrame(data_dict).set_index('mz')
    return df

def extract_eic(spec_df, mz_min, mz_max):
    '''Calculate extracted ion chromatogram over given m/z range.

    Parameters
    ----------
    spec_df : pd.DataFrame
    DataFrame of all spectrum, as output from create_df_specs().
    mz_min, mz_max : float
    Values of m/z between which the EIC is limited to.

    Returns
    -------
    eic : pd.Series
    Extracted ion chromatogram, intensities indexed by chromatogram timestamp.
    '''
    # assume all mass spectra in spec_array share same m/z index
    selected_range_mask = (spec_df.index>mz_min) & (spec_df.index<mz_max)
    eic = spec_df[selected_range_mask].apply(max)
    return eic

def detect_peak_heights(eic, num_peaks=1, make_plot=True, ax=None):
    '''Find a given number of largest peaks in a chromatogram.

    Parameters
    ----------
    eic : pd.Series
    Chromatogram of intensities indexed by timestamps, as output by
    extract_eic().
    num_peaks : float
    Number of peaks to extract, starting with largest. Returned peaks are kept
    in chromatogram order (i.e., largest peak not necessarily returned first).
    If None, all peaks are returned.
    make_plot : Boolean
    Whether to make a diagnostic plot showing the peaks, heights, and bases.
    ax : matplotlib.Axis
    Axis to plot on. If None, make new axis.

    Returns
    -------
    df_heights : pd.DataFrame
    DataFrame with each entry containing the peak height, along with its
    timestamp and the left and right bases to which the prominences are
    measured.
    '''
    peak_idxs, _ = sps.find_peaks(eic)

    prominences, leftbase_idxs, rightbase_idxs = sps.peak_prominences(eic, peak_idxs)

    if num_peaks is not None:
        # only keep num_peaks largest peaks. on use of argpartition, see
        # https://stackoverflow.com/a/23734295/4280216
        largest_idxs = np.argpartition(prominences, -num_peaks)[-num_peaks:]
        prominences = prominences[largest_idxs]
        leftbase_idxs = leftbase_idxs[largest_idxs]
        rightbase_idxs = rightbase_idxs[largest_idxs]
        peak_idxs = peak_idxs[largest_idxs]

    # convert from numpy index to Series index (i.e., timestamps)
    peak_ts = eic.iloc[peak_idxs].index.values
    leftbase_ts = eic.iloc[leftbase_idxs].index.values
    rightbase_ts = eic.iloc[rightbase_idxs].index.values

    if make_plot:
        if ax is None:
            fig, ax = plt.subplots()
        peak_values = eic.loc[peak_ts]
        contour_heights = peak_values - prominences

        ax.plot(eic, label='data')
        ax.plot(peak_ts, peak_values, "x", label='peaks')
        ax.plot(leftbase_ts, eic.loc[leftbase_ts], "o",
                 alpha=0.3, label='leftbases')
        ax.plot(rightbase_ts, eic.loc[rightbase_ts], "o",
                 alpha=0.3, label='rightbases')
        ax.vlines(x=peak_ts, ymin=contour_heights, ymax=peak_values,
                   label='heights')
        for height, x, y in zip(prominences, peak_ts, peak_values):
            ax.annotate("{:.2f}".format(height), xy=(x, y))
        ax.legend()
        ax.set_title("Check peak heights")

    df_heights = pd.DataFrame({'height': prominences,
                               'peak_time': peak_ts,
                               'leftbase_time': leftbase_ts,
                               'rightbase_time': rightbase_ts})
    return df_heights

def integrate_area(int_region, bl_function=None, leftedge=None, rightedge=None,
                  make_plot=True, ax=None):
    '''Integrate Series values, with optional baseline and/or index cutoff.

    Parameters
    ----------
    int_region : pd.Series
    Series of intensities, with index values of chromatogram time.
    bl_function : function
    Function that returns baseline intensity as a function of chromatogram
    time. If provided, baseline is subtracted before integration is performed.
    leftedge, rightedge : float
    Chromatogram times over which to integrate, if provided, defined as
    (leftedge, rightedge].
    make_plot : Boolean
    Make plot for visual check. Only helpful if baseline and/or edges defined.
    ax : matplotlib.Axis
    Axis to plot on. If None, create new axis.

    Returns
    -------
    integral : float
    Integral of region.

    Integral is calculated using Scipy implementation of composite trapezoidal
    rule.
    '''
    int_region_original = int_region.copy()
    if leftedge is not None:
        int_region = int_region[int_region.index>leftedge]
    if rightedge is not None:
        int_region = int_region[int_region.index<=rightedge]
    if bl_function is not None:
        int_region = int_region - bl_function(int_region.index.values)
    integral = spi.trapz(y=int_region, x=int_region.index.values)

    if make_plot:
        if ax is None:
            fig, ax = plt.subplots()
        ax.plot(int_region_original, label='original data')
        max_idx = int_region_original.idxmax()
        max_coords = (max_idx, int_region_original.loc[max_idx])
        ax.annotate("area: {:.2f}".format(integral), xy=max_coords,
                    xycoords='data')
        if bl_function is not None:
            ax.plot(int_region_original.index,
                     bl_function(int_region_original.index.values),
                     label='baseline')
        if leftedge is not None:
            ax.plot(leftedge, int_region_original.loc[leftedge], 'o',
                     label='left edge')
        if rightedge is not None:
            ax.plot(rightedge, int_region_original.loc[rightedge], 'o',
                     label='right edge')
        ax.set_title('Check integration region')
        ax.legend(loc='best')

    return integral

def calc_linear_baseline(eic, leftbase, rightbase):
    '''Given EIC and two indices, extract baseline and start/end times.

    Helpful in going between detect_peak_heights() and integrate_area().

    Parameters
    ----------
    eic : pd.Series
    Extracted ion chromatogram, intensities indexed by timestamp.
    leftbase, rightbase : float
    Timestamps of the start and end of the linear baseline.

    Returns
    -------
    baseline : function
    Linear function giving baseline intensity as a function of time.
    '''
    x1, y1 = leftbase, eic.loc[leftbase]
    x2, y2 = rightbase, eic.loc[rightbase]
    slope = (y2-y1)/(x2-x1)
    baseline = lambda x: slope*(x-x1) + y1
    return baseline
