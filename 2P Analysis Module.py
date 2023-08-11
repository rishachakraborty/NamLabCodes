#!/usr/bin/env python
# coding: utf-8

# In[4]:


def getstimcells(xmlfilepoints,opsdict,envfileforz,envfileforpoints):
    stimcelldict = {}
    for i in range(len(xmlfilepoints)):
        xpix = opsdict['Lx'] * xmlfilepoints['X'][i]
        ypix = opsdict['Ly'] * xmlfilepoints['Y'][i]
        index = xmlfilepoints['Index'][i]
        for j in range(len(envfileforpoints['Indices'])):
            if type(envfileforpoints['Indices'][j]) == str:
                if " " not in envfileforpoints['Indices'][j]:
                    if int(envfileforpoints['Indices'][j]) == int(index):
                        point = envfileforpoints['Points'][j]
                    for k in range(len(envfileforz['Name'])):
                        if envfileforz['Name'][k] == point:
                            z = envfileforz['Z'][k]
                            stimcelldict[i] = [index, ypix, xpix, z]
            if type(envfileforpoints['Indices'][j]) == np.float64:
                if np.isnan(envfileforpoints['Indices'][j]) == False:
                    if int(envfileforpoints['Indices'][j]) == int(index):
                        point = envfileforpoints['Points'][j]
                    for k in range(len(envfileforz['Name'])):
                        if envfileforz['Name'][k] == point:
                            z = envfileforz['Z'][k]
                            stimcelldict[i] = [index, ypix, xpix, z]
    stimcelldf = ((pd.DataFrame.from_dict(stimcelldict, orient='index')).drop_duplicates()).transpose()
    
    return stimcelldf
    
def getdicts(directory, reference_plane):
    files = [os.path.join(root, name)
             for root, dirs, files in os.walk(directory)
             for name in files if os.path.splitext(name)[-1] in '.xml']
    sub = "MarkPoints"
    for i,file in enumerate(files):
        if sub in file:
            stimtimedict, numiter, numstim = gettimedict(file)
            xmlfilepoints = pd.read_xml(file, xpath='.//Point')
        elif '.xml' in file:
            xmlfiletime = pd.read_xml(file,xpath='.//Frame')     
            
    filesenv = [os.path.join(root, name)
                for root, dirs, files in os.walk(directory)
                for name in files if os.path.splitext(name)[-1] in '.env']
    for i, file in enumerate(filesenv):
        if '.env' in file:
            envfileforz = pd.read_xml(file, xpath='.//PVGalvoPoint')
            envfileforpoints = pd.read_xml(file, xpath='.//PVGalvoPointElement')
            
    directory = directory + 'suite2p'
    sub = "plane"
    databindict = {}
    for i, file in enumerate(os.listdir(directory)):
        if sub in file:
            if file == (sub + str(reference_plane)):
                statdictall = np.load(directory + '/' + file + '/stat.npy', allow_pickle = True)
                fluoresencedict = np.load(directory + '/' + file + '/F.npy', allow_pickle = True)
                neuropildict = np.load(directory + '/' + file + '/Fneu.npy', allow_pickle = True)
                spikesdict = np.load(directory + '/' + file + '/spks.npy', allow_pickle = True)
                opsdict = ((np.load(directory + '/' + file + '/ops.npy', allow_pickle = True)).reshape(1,))[0]
                iscelldict = np.sum((np.load(directory + '/' + file + '/iscell.npy', allow_pickle = True)).astype(int), axis =1)
                statdict = statdictall[np.where(iscelldict>=1)]
                fluoresencedict = fluoresencedict[np.where(iscelldict>=1)]
                neuropildict = neuropildict[np.where(iscelldict>=1)]
                spikesdict = spikesdict[np.where(iscelldict>=1)]
                databindict[int(file.replace(sub, ''))] = directory + '/' + file + '/data.bin'
            else:
                databindict[int(file.replace(sub, ''))] = directory + '/' + file + '/data.bin'
        
    stimcelldf = getstimcells(xmlfilepoints,opsdict,envfileforz,envfileforpoints)
    ROIdict = roidict(statdict)
    
    return stimtimedict, numiter, numstim, xmlfiletime, fluoresencedict, neuropildict, spikesdict, opsdict, databindict, statdict, stimcelldf, ROIdict


def gettrace(ROIindex, fluoresencedict, opsdict, neuropildict):
    return fluoresencedict[ROIindex] - opsdict['neucoeff']*neuropildict[ROIindex]

def roidict(statdict):
    ROIdict = {}
    for i in range(len(statdict)):
        ROIdict[i] = statdict[i]['med']
    return ROIdict

def getframetimes(framearray,planeofinterest,numplanes):
    timearray = []
    for i in range(int(len(framearray)/numplanes)):
        timearray = np.append(timearray,framearray[numplanes*i + planeofinterest])
    return timearray


def closestframe(time, timearray):
    timearray = [frame - time for frame in timearray]
    return np.argmin(np.absolute(timearray))


def closestROI(XYPoint,ROIdict):
    distancearray = np.zeros(len(ROIdict))
    for i in range(len(ROIdict)):
        distancearray[i] = math.dist(XYPoint,ROIdict[i])
    return np.argmin(distancearray)
    
def preprocess(F: np.ndarray, baseline: str, win_baseline: float, sig_baseline: float,
               fs: float, prctile_baseline: float = 8) -> np.ndarray:
    """ preprocesses fluorescence traces for spike deconvolution

    baseline-subtraction with window "win_baseline"
    
    Parameters
    ----------------

    F : float, 2D array
        size [neurons x time], in pipeline uses neuropil-subtracted fluorescence

    baseline : str
        setting that describes how to compute the baseline of each trace

    win_baseline : float
        window (in seconds) for max filter

    sig_baseline : float
        width of Gaussian filter in frames

    fs : float
        sampling rate per plane

    prctile_baseline : float
        percentile of trace to use as baseline if using `constant_prctile` for baseline
    
    Returns
    ----------------

    F : float, 2D array
        size [neurons x time], baseline-corrected fluorescence

    """
    win = int(win_baseline * fs)
    if baseline == "maximin":
        Flow = gaussian_filter(F, [sig_baseline])
        Flow = minimum_filter1d(Flow, win)
        Flow = maximum_filter1d(Flow, win)
    elif baseline == "constant":
        Flow = gaussian_filter(F, [sig_baseline])
        Flow = np.amin(Flow)
    elif baseline == "constant_prctile":
        Flow = np.percentile(F, prctile_baseline, axis=1)
        Flow = np.expand_dims(Flow, axis=1)
    else:
        Flow = 0.

    F = F - Flow

    return F
    
def indextonum(string):
    indexarray = []
    indices = string.split(',')
    for i in range(len(indices)):
        indexlist = indices[i].split('-')
        for j in range(len(indexlist)):
            index = int(indexlist[j])
            indexarray = np.append(indexarray,index)
    return indexarray
    
    
def gettimedict(xmlfile):
    xmlfirstlayer = pd.read_xml(xmlfile)
    xmlsecondlayer = pd.read_xml(xmlfile, xpath='.//PVGalvoPointElement ')
    tree = etree.parse(xmlfile)
    xmlroot = str(etree.tostring(tree.getroot()))

    timedict = {}
    dictindex = 0
    #[time on, time off, roi#, interpointdelay, duration,stim power

    numiterations = xmlroot.split(' ')[1]
    numiterations = int(numiterations.split('"')[1])
    iterationdelay = xmlroot.split(' ')[2]
    iterationdelay = float(iterationdelay.split('"')[1])


    for j in range(numiterations):
        for i in range(len(xmlfirstlayer)):
            power = xmlfirstlayer['UncagingLaserPower'][i]
            reps = xmlfirstlayer['Repetitions'][i]
            if i== 0:
                if j==0:
                    starttime = xmlsecondlayer['InitialDelay'][0]  
                else: 
                    starttime = xmlsecondlayer['InitialDelay'][0] + iterationdelay + timedict[dictindex-1][1]
            else:
                starttime = xmlsecondlayer['InitialDelay'][i] + timedict[dictindex-1][1] #the time off of the previous point
            endtime = starttime + reps*(xmlsecondlayer['InterPointDelay'][i] + xmlsecondlayer['Duration'][i])
            if type(xmlsecondlayer['Indices'][i]) == str:
                indices = indextonum(xmlsecondlayer['Indices'][i])
            else:
                indices = [xmlsecondlayer['Indices'][i]]
            timedict[dictindex] = [starttime,endtime,indices, xmlsecondlayer['InterPointDelay'][i],xmlsecondlayer['Duration'][i],power]
            dictindex = dictindex + 1
            
    return timedict, numiterations, len(xmlfirstlayer)


# In[5]:


from typing import Optional, Tuple, Sequence
from contextlib import contextmanager
from tifffile import TiffWriter

import os

import numpy as np


class BinaryFile:

    def __init__(self, Ly: int, Lx: int, filename: str, n_frames: int = None,
                 dtype: str = "int16"):
        """
        Creates/Opens a Suite2p BinaryFile for reading and/or writing image data that acts like numpy array

        Parameters
        ----------
        Ly: int
            The height of each frame
        Lx: int
            The width of each frame
        filename: str
            The filename of the file to read from or write to
        """
        self.Ly = Ly
        self.Lx = Lx
        self.filename = filename
        self.dtype = dtype
        write = (not os.path.exists(self.filename))

        if write and n_frames is None:
            raise ValueError(
                "need to provide number of frames n_frames when writing file")
        elif not write:
            n_frames = self.n_frames
        shape = (n_frames, self.Ly, self.Lx)
        mode = "w+" if write else "r+"
        self.file = np.memmap(self.filename, mode=mode, dtype=self.dtype, shape=shape)
        self._index = 0
        self._can_read = True

    @staticmethod
    def convert_numpy_file_to_suite2p_binary(from_filename: str,
                                             to_filename: str) -> None:
        """
        Works with npz files, pickled npy files, etc.

        Parameters
        ----------
        from_filename: str
            The npy file to convert
        to_filename: str
            The binary file that will be created
        """
        np.load(from_filename).tofile(to_filename)

    @property
    def nbytesread(self):
        """number of bytes per frame (FIXED for given file)"""
        return np.int64(2 * self.Ly * self.Lx)

    @property
    def nbytes(self):
        """total number of bytes in the file."""
        return os.path.getsize(self.filename)

    @property
    def n_frames(self) -> int:
        """total number of frames in the file."""
        return int(self.nbytes // self.nbytesread)

    @property
    def shape(self) -> Tuple[int, int, int]:
        """
        The dimensions of the data in the file

        Returns
        -------
        n_frames: int
            The number of frames
        Ly: int
            The height of each frame
        Lx: int
            The width of each frame
        """
        return self.n_frames, self.Ly, self.Lx

    @property
    def size(self) -> int:
        """
        Returns the total number of pixels

        Returns
        -------
        size: int
        """
        return np.prod(np.array(self.shape).astype(np.int64))

    def close(self) -> None:
        """
        Closes the file.
        """
        self.file._mmap.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def __setitem__(self, *items):
        indices, data = items
        if data.dtype != "int16":
            self.file[indices] = np.minimum(data, 2**15 - 2).astype("int16")
        else:
            self.file[indices] = data

    def __getitem__(self, *items):
        indices, *crop = items
        return self.file[indices]

    def sampled_mean(self) -> float:
        """
        Returns the sampled mean.
        """
        n_frames = self.n_frames
        nsamps = min(n_frames, 1000)
        inds = np.linspace(0, n_frames, 1 + nsamps).astype(np.int64)[:-1]
        frames = self.file[inds].astype(np.float32)
        return frames.mean(axis=0)

    @property
    def data(self) -> np.ndarray:
        """
        Returns all the frames in the file.

        Returns
        -------
        frames: n_frames x Ly x Lx
            The frame data
        """
        return self.file[:]

    def bin_movie(self, bin_size: int, x_range: Optional[Tuple[int, int]] = None,
                  y_range: Optional[Tuple[int, int]] = None,
                  bad_frames: Optional[np.ndarray] = None,
                  reject_threshold: float = 0.5) -> np.ndarray:
        """
        Returns binned movie that rejects bad_frames (bool array) and crops to (y_range, x_range).

        Parameters
        ----------
        bin_size: int
            The size of each bin
        x_range: int, int
            Crops the data to a minimum and maximum x range.
        y_range: int, int
            Crops the data to a minimum and maximum y range.
        bad_frames: int array
            The indices to *not* include.
        reject_threshold: float

        Returns
        -------
        frames: nImg x Ly x Lx
            The frames
        """

        good_frames = ~bad_frames if bad_frames is not None else np.ones(
            self.n_frames, dtype=bool)

        batch_size = min(np.sum(good_frames), 500)
        batches = []
        for k in np.arange(0, self.n_frames, batch_size):
            indices = slice(k, min(k + batch_size, self.n_frames))
            data = self.file[indices]

            if x_range is not None and y_range is not None:
                data = data[:, slice(*y_range), slice(*x_range)]  # crop

            good_indices = good_frames[indices]
            if np.mean(good_indices) > reject_threshold:
                data = data[good_indices]

            if data.shape[0] > bin_size:
                data = binned_mean(mov=data, bin_size=bin_size)
                batches.extend(data)

        mov = np.stack(batches)
        return mov

    def write_tiff(self, fname, range_dict={}):
        "Writes BinaryFile's contents using selected ranges from range_dict into a tiff file."
        n_frames, Ly, Lx = self.shape
        frame_range, y_range, x_range = (0,n_frames), (0, Ly), (0, Lx)
        with TiffWriter(fname, bigtiff=True) as f:
            # Iterate through current data and write each frame to a tiff
            # All ranges should be Tuples(int,int)
            if 'frame_range' in range_dict:
                frame_range = range_dict['frame_range']
            if 'x_range' in range_dict:
                x_range = range_dict['x_range']
            if 'y_range' in range_dict:
                y_range = range_dict['y_range']
            print('Frame Range: {}, y_range: {}, x_range{}'.format(frame_range, y_range, x_range))
            for i in range(frame_range[0], frame_range[1]):
                curr_frame = np.floor(self.file[i, y_range[0]:y_range[1], x_range[0]:x_range[1]]).astype(np.int16)
                f.write(curr_frame)
        print('Tiff has been saved to {}'.format(fname))

def from_slice(s: slice) -> Optional[np.ndarray]:
    """Creates an np.arange() array from a Python slice object.  Helps provide numpy-like slicing interfaces."""
    return np.arange(s.start, s.stop, s.step) if any([s.start, s.stop, s.step
                                                     ]) else None


def binned_mean(mov: np.ndarray, bin_size) -> np.ndarray:
    """Returns an array with the mean of each time bin (of size "bin_size")."""
    n_frames, Ly, Lx = mov.shape
    mov = mov[:(n_frames // bin_size) * bin_size]
    return mov.reshape(-1, bin_size, Ly, Lx).astype(np.float32).mean(axis=1)


@contextmanager
def temporary_pointer(file):
    """context manager that resets file pointer location to its original place upon exit."""
    orig_pointer = file.tell()
    yield file
    file.seek(orig_pointer)


# In[12]:


import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
from scipy.ndimage import maximum_filter1d, minimum_filter1d, gaussian_filter
from lxml import etree
import os

nplane = 1
reference_plane = 0
directory = '/Users/rishachakraborty/Downloads/Tseries74xystim/'
#stimtimedict, numiter, numstim, xmlfiletime, fluoresencedict, neuropildict, spikesdict, opsdict, databindict, statdict, stimcelldf, ROIdict = getdicts(directory, reference_plane)

#optional iteration through files within directory
#directory = '/Users/rishachakraborty/Downloads/XYSTIM/'
#keyword = 'xystim'
#files = [file for file in os.listdir(directory) if keyword in file]
#for k in range(len(files)):
    #directory = '/Users/rishachakraborty/Downloads/XYSTIM/' + files[k]
    


# In[ ]:


#draw total fluoresence, trace or dff across all frames

fig,axs = plt.subplots(1,numstim, figsize=(20,4))
for i in range(numstim):    
    closestROIindex = closestROI([stimcelldf[i][1], stimcelldf[i][2]],ROIdict)
    axs[i].plot(range(len(fluoresencedict[closestROIindex])), fluoresencedict[closestROIindex], alpha = 0.3)
    trace  = gettrace(closestROIindex, fluoresencedict, opsdict, neuropildict)
    axs[i].plot(range(len(trace)),trace)
    dff = getdff(trace, opsdict)
    axs[i].plot(range(len(dff)),dff)


# In[ ]:


#xystim

#timearray: multiplicative factor = .067 for 4 frame averaging and .267 for 2 frame averaging

tracestimavg = np.zeros(shape=(numstim,2,56))
counter = 0
frametimes = getframetimes((xmlfiletime['relativeTime']).to_numpy(),reference_plane,nplane)
fig, axs = plt.subplots(1,numstim, figsize=(20,4))
for i in range(len(stimtimedict)):
    if counter == numstim:
        counter = 0
    stimframe = closestframe(((stimtimedict[i][0])/1000), frametimes)
    startframe = closestframe(((stimtimedict[i][0])/1000) - 5, frametimes)
    avgframe1 = closestframe(((stimtimedict[i][0])/1000) - 2.5, frametimes)
    avgframe2 = closestframe(((stimtimedict[i][0])/1000) - 1.5, frametimes)
    endframe = closestframe(((stimtimedict[i][0])/1000) + 10, frametimes)
    timearray = [(frame - stimframe)*0.067 for frame in np.arange(startframe,endframe)]
    roivalue = list({row for row in stimcelldf if stimcelldf[row][0] == stimtimedict[i][2][0]})
    ROI = [stimcelldf[roivalue[0]][1],stimcelldf[roivalue[0]][2]]
    closestROIindex = closestROI(ROI,ROIdict)
    meanvalue = np.mean(fluoresencedict[closestROIindex][avgframe1:avgframe2])   
    tracestim = fluoresencedict[closestROIindex][startframe:endframe]
    tracestimadj = [(frame/meanvalue) for frame in tracestim]
    axs[counter].plot(timearray,tracestimadj, alpha = 0.75)
    #tracestimavg[counter,:] = tracestimavg[counter,:] + tracestim
    counter = counter + 1
#for i in range(numstim):
#    axs[i].plot(np.arange(-30,30,1),tracestimavg[i,:]/numiter, color = 'black')
#    axs[i].set(title = "z = " + str(round(stimcelldf[i][3], 3)))
    





# In[ ]:


#get average fluoresence of all stimmed ROIs across planes if multiplanestim
avef = np.empty((nplane, opsdict['nframes'], len(stimcelldf)))
avef[:] = np.nan

for ip in range(nplane):
    data = BinaryFile(Lx=opsdict['Lx'], Ly=opsdict['Ly'], filename= databindict[ip]).data    
    for i in range(len(stimcelldf)):
        avef[ip,:data.shape[0],i] = np.mean(data[:,statdict[closestROI([stimcelldf[i][1], stimcelldf[i][2]],ROIdict)]['ypix'],statdict[closestROI([stimcelldf[i][1], stimcelldf[i][2]],ROIdict)]['xpix']],1)    



# In[ ]:


#multiplanestim


meanarray = np.zeros(shape = (nplane,numstim,numiter))        

for k in range(nplane):
    fig, axs = plt.subplots(1,numstim, figsize=(20,4)) 
    frametimes = getframetimes((xmlfiletime['relativeTime']).to_numpy(),k,nplane)
    counter = 0
    for i in range(len(stimtimedict)):
        if counter == numstim:
            counter = 0
        startframe = closestframe(((stimtimedict[i][0])/1000) - 5, frametimes)
        endframe = closestframe(((stimtimedict[i][0])/1000) + 10, frametimes)
        stimframe = closestframe(((stimtimedict[i][0])/1000),frametimes)
        meanarray[k,counter,int(i/numstim)] = np.mean(avef[k,startframe:stimframe,counter])   
        tracestim = avef[k,startframe:endframe,counter]
        
        for j in range(len(stimtimedict[i][2])):
            roivalue = list({row for row in stimcelldict if stimcelldict[row][0] == stimtimedict[i][2][j]})
            ROI = stimcelldict[roivalue[0]][1]
            closestROIindex = closestROI(ROI,ROIdict)
            trace  = fluoresencedict[closestROIindex] - opsdict['neucoeff']*neuropildict[closestROIindex]
            tracestim = trace[startframe:endframe]
            tracestim = fluoresencedict[closestROIindex][startframe:endframe]
        axs[counter].plot(np.arange(-5,10,15/len(tracestim)),tracestim)
        counter = counter + 1
    
meanofmeans = np.mean(meanarray,axis=2)
stdofmeans = np.std(meanarray,axis=2)

fig2,axs2 = plt.subplots(1,nplane, figsize=(20,4)) 

for i in range(nplane):
    fig, axs1 = plt.subplots(1,numstim, figsize=(20,4)) 
    frametimes = getframetimes((xmlfiletime['relativeTime']).to_numpy(),i,nplane)
    counter = 0
    for j in range(len(stimtimedict)):
        if counter == numstim:
            counter = 0
        startframe = closestframe(((stimtimedict[j][0])/1000) - 5, frametimes)
        endframe = closestframe(((stimtimedict[j][0])/1000) + 10, frametimes)
        zfluor = np.zeros(shape = [numiter,endframe-startframe])
        trace = avef[i,startframe:endframe,counter]
        k = int(j/numstim)
        zfluor[k] = [(frame - meanofmeans[i,counter])/stdofmeans[i,counter] for frame in trace]
        axs1[counter].plot(np.arange(-5,10,15/len(zfluor[k])),zfluor[k])
        axs2[i].plot(np.arange(-5,10,15/len(np.mean(zfluor,axis=0))),np.mean(zfluor,axis=0))
        counter = counter + 1

