import numpy as np
import pandas as pd
import sknw_jwm as sknw
from skimage.filters import threshold_otsu,threshold_multiotsu
from skimage.morphology import skeletonize
from skimage.measure import label
from skimage.segmentation import clear_border
from skimage.io import imsave
from scipy.ndimage import binary_fill_holes,gaussian_laplace,binary_dilation
import cv2
from ast import literal_eval
import os
import itertools
from nd2reader import ND2Reader
from scipy.signal import find_peaks, peak_widths
from matplotlib import pyplot as plt
import warnings
warnings.filterwarnings('ignore')

def adaptive_threshold(img):
    img = cv2.normalize(img, None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8U)
    img = cv2.medianBlur(img,9)
    bw = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,15,3)
    bw = binary_fill_holes(np.invert(bw))
    return bw.astype('uint8')

def otsu_threshold_custom_bins(img,nbins = 100):
    counts, boundaries = np.histogram(img,nbins)
    bin_centers = boundaries[:-1]-np.divide(np.diff(boundaries[:-1])[0],2)
    
    weight1 = np.cumsum(counts,0)
    weight2 = np.flip(np.cumsum(np.flip(counts,[0]),0),[0])
    # class means for all possible thresholds
    mean1 = np.cumsum(counts*bin_centers,0)/weight1
    mean2 = np.flip(np.cumsum(np.flip(counts*bin_centers,[0]),0)/np.flip(weight2,[0]),[0])

    # Clip ends to align class 1 and class 2 variables:
    # The last value of ``weight1``/``mean1`` should pair with zero values in
    # ``weight2``/``mean2``, which do not exist.
    variance12 = weight1[:-1] * weight2[1:] * (mean1[:-1] - mean2[1:]) ** 2

    idx = np.argmax(variance12)
    threshold = bin_centers[idx]
    return threshold

def neighbors(shape):
    dim = len(shape)
    block = np.ones([3]*dim)
    block[tuple([1]*dim)] = 0
    idx = np.where(block>0)
    idx = np.array(idx, dtype=np.uint8).T
    idx = np.array(idx-[1]*dim)
    acc = np.cumprod((1,)+shape[::-1][:-1])
    return np.dot(idx, acc[::-1])

def count_skeleton_nodes(skel_coords):
    skel = np.array(skel_coords)[1:]
    skel[0] = skel[0] - skel.min(axis=1)[0] +5
    skel[1] = skel[1] - skel.min(axis=1)[1] +5
    skeleton = np.full([skel[0].max()+5,skel[1].max()+5],0)
    skeleton[skel[0],skel[1]] = 1 
    buf = np.zeros(tuple(np.array(skeleton.shape)+2), dtype=np.uint16)
    buf[tuple([slice(1,-1)]*buf.ndim)] = skeleton
    nbs = neighbors(buf.shape)
    acc = np.cumprod((1,)+buf.shape[::-1][:-1])[::-1]
    sknw.mark(buf, nbs)
    pts = np.array(np.where(buf.ravel()==2))[0]
    nodes = sknw.parse_nodes(buf,pts,nbs,acc)
    return len(nodes)

def label_unique_IDs(struc):
    cumulative_max_value = 0
    for idx,slice in enumerate(struc):
        slice = struc[idx]
        slice[slice > 0] + cumulative_max_value
        cumulative_max_value = cumulative_max_value + np.max(slice)
        struc[idx] = slice
    return struc

def parse_skeleton(sensor,skel_coords):
    '''
    This function applies the sknw package (https://github.com/jwmcca/sknw_edit) to trace the skeleton from start to finish. 
    Useful for measuring linescans of cells in downstream applications.

    Parameters
    ----------
    A binary image containing a skeleton

    Returns
    ----------
    A 2D numpy array. Array 1 has all x points, array 2 has all y points.
    '''
    skeleton = np.full(sensor,0)
    skeleton[skel_coords[-2],skel_coords[-1]] = 1 
    graph = sknw.build_sknw(skeleton)
    pts = graph[0].get(1)
    return np.column_stack(pts['pts'])

def check_in_focus(frame,coordinates):
    '''
    After labeling every cell and pulling their coordinates, this filtering step calculates a laplacian of gaussian of the input
    cell region of a fluorescence frame to make sure they are in focus or really contain a cell. I found cells out of focus have
    very low variance. I pick an arbitrary cutoff of 5 here.

    Parameters
    ----------
    [0] A frame from the stack of fluorescent images you are interrogating.
    [1] The (x,y) coordinates of the cell you're seeking.

    Returns
    ----------
    True/False whether the standard deviation of the pixel noise is less than 5.
    '''
    cell_image = frame[coordinates[-2].min()-10:coordinates[-2].max()+10,coordinates[-1].min()-10:coordinates[-1].max()+10]
    lap_gaus = gaussian_laplace(cell_image,sigma = 3)
    if np.std(lap_gaus) > 5:
        focus = True
    else:
        focus = False
    return focus

def check_for_masks_with_holes(cell_coord):
    cell = cell_coord[-2:]
    cell[0] = cell[0] - cell.min(axis=1)[0]
    cell[1] = cell[1] - cell.min(axis=1)[1]
    blank = np.full([cell[0].max()+1,cell[1].max()+1],0)
    blank[cell[0],cell[1]] = 1
    filled_I = binary_fill_holes(blank)
    inverted = np.invert(blank)
    holes_I = filled_I & inverted
    return holes_I.sum()

def get_indices_pandas(data):
    d = data.ravel()
    f = lambda x: np.unravel_index(x.index, data.shape)
    return pd.Series(d).groupby(d).apply(f)

def measure_thickness(sensor,cell_coord,skel):
    skel_only = np.full(sensor,0)
    skel_only[skel[-2],skel[-1]] = 1

    cell_only = np.full(sensor,False)
    cell_only[cell_coord[-2],cell_coord[-1]] = True
    # apply skeleton to select center line of distance
    distance = cv2.distanceTransform(cell_only.astype(np.uint8), distanceType=cv2.DIST_L2, maskSize=5).astype(np.float32)
    thickness = cv2.multiply(distance, skel_only.astype(np.float32))
    return thickness[skel_only!=0]

def nd2_to_array(images_path):
    """
    Produced by Alexandros Papagiannakis in the CJW lab

    This function is used to convert .nd2 images to numpy arrays.
    It also returms the image metadata.
    
    Parameters
    ----------
    image_path - string: the path of the .nd2 file

    Returns
    -------
    [0] the iteration axis - string ('c', 't', 'vc' or 'vct')
    [1] the .nd2 metadata and images (from the ND2_Reader pims module). This is a class object and has multuiple functions
    [2] a dictionary which contains the images as numpy arrays organized by:
        iteration_axis 't' - For fast time lapse (stream acquisition), key2: frame
        iteration_axis 'c' - For snapshots of a single XY position, key1: channel
        iteration_axis 'vc' - For snapshots of multiple XY positions, key1: position, key2: channel
        iteration_axis = 'vct' - For time lapse across different channels, key1: position, key2: channel, key3: time-point
    [3] channels: list of strings - each string represents the channel (e.g. ['Phase', 'mCherry', 'GFP', 'Phase_after'])
        If a certain wavelength (lambda) is used two times in the ND acquisition, then the second channel instance is referred to as '_after'
        An empty list is returned if no channels are selected.
    [4] the number of time-points - positive integer or zero if the Time label was not selected in the ND acquisition
    [5] The number of XY positions - positive integer or zero if the XY label was not selected in the ND acquisition
    
    Notes
    -----
    This function was adapted to include all possible channel, time-point, xy-position permutations in our image acquisition protocols in NIS elements (including the JOBS module)
    New permutations may need to be included for new image permutations.
    The iteration axis determines how the image dimensions are iterated and stored into dictionaries
    """
    # The path of the .nd2 file 
    images = ND2Reader(images_path)
    # "C:\Users\Alex\Anaconda3\Lib\site-packages\pims_nd2\nd2reader.py"
    # This path has been modified in lines 228 and 229 to accommodate the function.
    #print('metadata:',images.metadata)
    #print('dimensions:',images.sizes)
    
    scale = round(images.metadata['pixel_microns'],3)  # μm/px scale
    time_steps = images.get_timesteps()
    sensor = (images.sizes['x'], images.sizes['y'])
    channels = []
    if 'c' in images.sizes:
        # get the channels and frames from the .nd2 metadata
        chs = images.metadata['channels']
        channels = []
        
        for ch in chs:
            if ch in channels:
                channels.append(ch+'_after')
            else:
                channels.append(ch)
    # number_of_frames = images.metadata['sequence_count']
    iteration_axis = ''
    if 'v' in images.sizes and images.sizes['v'] > 1:
        iteration_axis += 'v'
        number_of_positions = images.sizes['v']
    if 'c' in images.sizes and images.sizes['c'] > 1:
        iteration_axis += 'c'
        number_of_channels = images.sizes['c']
    if 't' in images.sizes and images.sizes['t'] > 1:
        iteration_axis += 't'
        number_of_timepoints = images.sizes['t']
    # For a stream acquisition
    if iteration_axis == 't':
        image_arrays = {}
        number_of_positions = 0
        with images as frames:
            t = 0 # time point
            #print(frames)
            frames.iter_axes = iteration_axis
            for frame in frames:
                image_arrays[t] = np.array(frame)
                t += 1
        frames.close()
    # For snapshots at different channels
    elif iteration_axis == 'c':
        image_arrays = {}
        number_of_timepoints = 0
        number_of_positions = 0
        with images as frames:
            i = 0
            #print(frames)
            frames.iter_axes = iteration_axis
            for frame in frames:
                image_arrays[channels[i]] = np.array(frame)
                i += 1
        frames.close()
    # For snapshots at different XY positions for a single channel (this is how JOBS extracts the snapshots)
    elif iteration_axis == 'v':      
        image_arrays = {}
        number_of_timepoints = 0
        number_of_channels = 1
        with images as frames:
            i = 0
            #print(frames)
            frames.iter_axes = iteration_axis
            for frame in frames:
                image_arrays[i] = np.array(frame)
                i += 1
        frames.close()
    # For snapshots at different channels and XY positions
    elif iteration_axis == 'vc':
        image_arrays = {}
        number_of_timepoints = 0
        with images as frames:
            #print(frames)
            frames.iter_axes = iteration_axis
            pos = 0
            ch = 0
            image_arrays[pos] = {}
            for frame in frames:
                if ch < number_of_channels:
                    if pos < number_of_positions:
                        image_arrays[pos][channels[ch]] = np.array(frame)
                        ch+=1
                elif ch == number_of_channels:
                    pos += 1
                    image_arrays[pos] = {}
                    ch = 0
                    image_arrays[pos][channels[ch]] = np.array(frame)
                    ch+=1
        frames.close()
    # For snapshots at different channels and XY positions and timepoints
    elif iteration_axis == 'vt':
        image_arrays = {}
        with images as frames:
            #print(frames)
            frames.iter_axes = iteration_axis
            pos = 0
            tm = 0
            image_arrays[pos] = {}
            for frame in frames:
                if tm < number_of_timepoints:
                    image_arrays[pos][tm] = np.array(frame)
                    tm+=1
                elif tm == number_of_timepoints:
                    tm = 0
                    if pos < number_of_positions-1:
                        pos += 1
                        image_arrays[pos] = {}
                        image_arrays[pos][tm] = np.array(frame)
                        tm+=1             
        frames.close()
    # For snapshots at different channels and XY positions and timepoints
    elif iteration_axis == 'vct':
        image_arrays = {}
        with images as frames:
            #print(frames)
            frames.iter_axes = iteration_axis
            pos = 0
            ch = 0
            tm = 0
            image_arrays[pos] = {}
            image_arrays[pos][channels[ch]] = {}
            for frame in frames:
                if tm < number_of_timepoints:
                    image_arrays[pos][channels[ch]][tm] = np.array(frame)
                    tm+=1
                elif tm == number_of_timepoints:
                    tm = 0
                    if ch < number_of_channels-1:
                        ch += 1
                        image_arrays[pos][channels[ch]] = {}
                        image_arrays[pos][channels[ch]][tm] = np.array(frame)
                        tm+=1
                    elif ch == number_of_channels-1:
                        ch = 0
                        pos+=1
                        image_arrays[pos] = {}
                        image_arrays[pos][channels[ch]] = {}
                        image_arrays[pos][channels[ch]][tm] = np.array(frame)
                        tm+=1
        frames.close()
    # if no channels or time points are specified there should be only one image
    elif iteration_axis == '':
        number_of_timepoints = 0
        number_of_positions = 0
        with images as frames:
            for frame in frames:
                image_arrays = np.array(frame)
    
    return iteration_axis, images, image_arrays, channels, number_of_timepoints, number_of_positions, scale, time_steps, sensor

def polyfit2d(x, y, z, order):
        """
        Produced by Alexandros Papagiannakis in the CJW lab

        fits a quadratic surface fucntion
        used for background estimation
        """
        ncols = (order + 1)**2
        G = np.zeros((x.size, ncols))
        ij = itertools.product(range(order+1), range(order+1))
        for k, (i,j) in enumerate(ij):
            G[:,k] = x**i * y**j
        m, _, _, _ = np.linalg.lstsq(G, z, rcond=1)
        return m

def polyval2d(x, y, m):
    """
    Produced by Alexandros Papagiannakis in the CJW lab

    returns the values of the fitted quadratic surface function
    used for background estimation
    """
    order = int(np.sqrt(len(m))) - 1
    ij = itertools.product(range(order+1), range(order+1))
    z = np.zeros_like(x)
    for a, (i,j) in zip(m, ij):
        z += a * x**i * y**j
    return z

def back_sub(phase_image, signal_image, dilate=10):
        """
        Produced by Alexandros Papagiannakis in the CJW lab

        Subtracts an n_order second degree polynomial fitted to the non-cell pixels.
        The 2D polynomial surface is fitted to the non-cell pixels only.
            The order of the polynomial depends on whether there is uneven illumination or not
        The non-cell pixels are masked as thos below the otsu threshold estimated on the basis of the inverted phase image.

        Input:
            signal_image: numpy.array - the image to be corrected
            dilate: non-negative integer - the number of dilation rounds for the cell mask
            show: binary - True if the user wants to visualize the 2D surface fit
        Returns:
            [0] The average background
            [1] The background corrected image (after subtracting the 2D polynomial surface)
        """


        sensor = (phase_image.shape[0], phase_image.shape[1])

        # invert the image and apply an otsu threshold to separate the dimmest (or inversely brightest pixels) which correspond to the cells
        inverted_phase_image = 1/phase_image
        inverted_threshold = threshold_otsu(inverted_phase_image.ravel())

        phase_mask = inverted_phase_image > inverted_threshold

        # dilate the masked phase images
        threshold_masks_dil = binary_dilation(phase_mask, iterations=dilate)

        # select the range of the fit (across the entire sesnor) and the meshgrid
        # used in the polyval function
        xx, yy = np.meshgrid( np.linspace(0, sensor[0]-1, sensor[0]), np.linspace(0,sensor[1]-1, sensor[1]))


        zeros = np.nonzero(np.invert(threshold_masks_dil))
        xcoord = zeros[0]
        ycoord = zeros[1]
        zcoord = signal_image[xcoord,ycoord]


        # fit a polynomial surface function to the masked pixel intensities (excluding the cells)
        m = polyfit2d(np.array(xcoord),np.array(ycoord),np.array(zcoord),2)

        # re-estimate the background pixel intensities from the fitted function for the meshgrid (sensor area)
        zz = polyval2d(xx, yy, m)  # xx and yy are defined in line 287

        bkg_cor = signal_image - zz.T

        return bkg_cor

def simple_background_estimation(phase_image,signal_image,dilate = 10):
    sensor = (phase_image.shape[0], phase_image.shape[1])

    # invert the image and apply an otsu threshold to separate the dimmest (or inversely brightest pixels) which correspond to the cells
    inverted_phase_image = 1/phase_image
    inverted_threshold = threshold_otsu(inverted_phase_image.ravel())

    phase_mask = inverted_phase_image > inverted_threshold

    # dilate the masked phase images
    threshold_masks_dil = binary_dilation(phase_mask, iterations=dilate)


    # select the range of the fit (across the entire sesnor) and the meshgrid
    # used in the polyval function

    zeros = np.nonzero(np.invert(threshold_masks_dil))
    xcoord = zeros[0]
    ycoord = zeros[1]
    zcoord = signal_image[xcoord,ycoord]
    return np.mean(zcoord)


class borrelia_cell_segmentation:
    '''
    cell_segmentation is a class to segment Borrelia phase contrast images with a simple otsu.
        - You initiate the class by specifying your desired parameters, shown directly below. I filter cells by a minimum size in 
          pixel area, then cells in micron units. My default assumption is that cells must be at least 5 microns long, no larger
          than 0.55 microns wide, and the largest measured width cannot exceed 1um. You can also specify the pixel size of the 
          experiment and any adjustment to the otsu threshold.
        - After initiating, you can call the class to load the phase and signal images. It will perform the otsu threshold
          and produce a mask from there, screening obvious misses right away by the area threshold.
        - In a for loop in a primary script, you then screen all remaining masks in a for loop for the remaining criteria above.
          This function will out a "cell_masks" dictionary that contains all cell masks and their quantification.

    
    Made by Joshua McCausland in the CJW lab, 2023.
    '''
    def __init__(self,mimimum_size = 500,minimum_length = 5,maximum_width = 0.55,px_size = 0.065,
                 instantaneous_max_width = 1, back_sub = False,threshold = 'batch_otsu',thresh_adjust = 1,otsu_bins = 25,
                 remove_out_of_focus_cells = True,remove_linescans = False):
        args = {'minimum_size': mimimum_size,
                         'minimum_length': minimum_length,
                         'maximum_width': maximum_width,
                         'px_size': px_size,
                         'instantaneous_max_width': instantaneous_max_width,
                         'back_sub': back_sub,
                         'threshold': threshold,
                         'thresh_adjust': thresh_adjust,
                         'otsu_bins': otsu_bins,
                         'remove_out_of_focus_cells': remove_out_of_focus_cells,
                         'remove_linescans': remove_linescans}
        
        self.segmentation_params = args
        self.minimum_size = mimimum_size
        self.minimum_length = minimum_length
        self.maximum_width = maximum_width
        self.px_size = px_size
        self.thresh_adjust = thresh_adjust
        self.inst_max_width = instantaneous_max_width
        self.back_sub = back_sub
        self.otsu_bins = otsu_bins
        self.threshold_method = threshold
        self.remove_out_of_focus_cells = remove_out_of_focus_cells
        self.remove_linescans = remove_linescans
        self.df = pd.DataFrame()

    def load_parameters(self,file):
        self.segmentation_params = {}
        with open(file) as f:
            for line in f.readlines():
                line = line[0:-1]
                key,val = line.split(': ')
                try:
                    self.segmentation_params[f'{key}'] = literal_eval(val)
                except:
                    self.segmentation_params[f'{key}'] = val
        kwargs = self.segmentation_params
        self.minimum_size = kwargs['minimum_size']
        self.minimum_length = kwargs['minimum_length']
        self.maximum_width = kwargs['maximum_width']
        self.px_size = kwargs['px_size']
        self.thresh_adjust = kwargs['thresh_adjust']
        self.inst_max_width = kwargs['instantaneous_max_width']
        self.back_sub = kwargs['back_sub']
        self.otsu_bins = kwargs['otsu_bins']
        self.threshold_method = kwargs['threshold']
        self.remove_out_of_focus_cells = kwargs['remove_out_of_focus_cells']

    def load_dataframe(self,file):
        self.df = pd.read_pickle(file)

    def __px_size__(self):
        return self.px_size
    
    def test_thresholding_method(self,filename,phase_images,image_num=0,threshold = None,otsu_bins = None,thresh_adjust = None):        
        if threshold:
            self.threshold_method = threshold
                    
        if otsu_bins:
            self.otsu_bins = otsu_bins
                    
        if thresh_adjust:
            self.thresh_adjust = thresh_adjust
                          

        self.sensor = phase_images.shape
        img = phase_images[image_num]
        if self.threshold_method == 'batch_otsu':
            threshold = otsu_threshold_custom_bins(phase_images,nbins = self.otsu_bins)
            bw = phase_images < threshold * self.thresh_adjust
            bw = binary_fill_holes(bw[image_num])
        elif self.threshold_method == 'otsu':
            bw = np.zeros(img.shape).astype('int')
            threshold = threshold_otsu(img)
            bw = img < threshold * self.thresh_adjust
        elif self.threshold_method == 'multiotsu':
            threshold,_ = threshold_multiotsu(phase_images,nbins = self.otsu_bins)
            bw = phase_images < threshold * self.thresh_adjust
            bw = binary_fill_holes(bw[image_num])
        else:
            bw = np.zeros(img.shape).astype('int')
            bw = adaptive_threshold(img)
        
        bw = bw.astype('int')   
        ccs = label(bw)
        component_sizes = np.bincount(ccs.ravel())
        too_small = component_sizes < 500
        too_small_mask = too_small[ccs]
        ccs[too_small_mask] = 0

        ccs = clear_border(ccs,buffer_size = 5)

        titles = ['Phase_Image','Thresholded_Image','Initial_Screen']
        _,axs = plt.subplots(ncols=3,figsize = [15,5],layout = 'constrained')
        axs[0].imshow(img,cmap = 'gray')
        axs[1].imshow(bw,vmax = 1)
        axs[2].imshow(ccs,vmax = 1)

        for ax,title in zip(axs,titles):
            ax.set_title(title,weight = 'bold')
            ax.set_xticks([])
            ax.set_yticks([])
            ax.spines[['left','bottom','top','right']].set_visible(False)
        
        isExist = os.path.exists('Thresh_Testing')
        if not isExist:
            os.makedirs('Thresh_Testing')

        isExist = os.path.exists(f'Thresh_Testing/{filename}')
        if not isExist:
            os.makedirs(f'Thresh_Testing/{filename}')
        plt.savefig(f'Thresh_Testing/{filename}/{filename}_Image-{image_num:03}-Threshold_Test_{self.threshold_method}_Adjust-{self.thresh_adjust:.3f}_Bins-{self.otsu_bins:03}.pdf',dpi = 600)

    def __call__(self,filename,phase_images,threshold = None,thresh_adjust = None,otsu_bins = None,minimum_length = None):
        self.phase = phase_images
        self.filename = filename
        self.sensor = phase_images.shape
        self.segmentation_params[f'{filename}-sensor'] = self.sensor
        if thresh_adjust:
            self.thresh_adjust = thresh_adjust
            self.segmentation_params[f'{filename}-thresh_adjust'] = thresh_adjust

        if otsu_bins:
            self.otsu_bins = otsu_bins
            self.segmentation_params[f'{filename}-otsu_bins'] = otsu_bins

        if threshold:
            self.threshold_method = threshold 
            self.segmentation_params[f'{filename}-threshold'] = threshold
        
        if minimum_length:
            self.minimum_length = minimum_length
            self.segmentation_params[f'{filename}-minimum_length'] = minimum_length

        if self.threshold_method == 'batch_otsu':
            threshold = otsu_threshold_custom_bins(phase_images,nbins = self.otsu_bins)
            bw = phase_images < threshold * self.thresh_adjust
            for idx in np.arange(bw.shape[0]):
                bw[idx] = binary_fill_holes(bw[idx])
        elif self.threshold_method == 'otsu':
            bw = np.zeros(self.sensor).astype('int')
            for idx in np.arange(bw.shape[0]):
                threshold = threshold_otsu(phase_images[idx])                
                bw[idx] = phase_images[idx] < threshold * self.thresh_adjust
                bw[idx] = binary_fill_holes(bw[idx])
        elif self.threshold_method == 'multiotsu':
            threshold,_ = threshold_multiotsu(phase_images,nbins = self.otsu_bins)
            bw = phase_images < threshold * self.thresh_adjust
            for idx in np.arange(bw.shape[0]):
                bw[idx] = binary_fill_holes(bw[idx])
        else:
            bw = np.zeros(self.sensor).astype('int')
            for idx in np.arange(bw.shape[0]):
                bw[idx] = adaptive_threshold(phase_images[idx])

        bw = bw.astype('int')   
        
        ccs = label(bw)
        del bw
        cumulative_max_value = 1
        for image_slice in ccs:
            component_sizes = np.bincount(image_slice.ravel())
            too_small = component_sizes < self.minimum_size
            too_small_mask = too_small[image_slice]
            image_slice[too_small_mask] = 0
            for id in np.unique(image_slice[image_slice>0]):
                image_slice[image_slice == id] = cumulative_max_value
                cumulative_max_value += 1

        for idx in np.arange(ccs.shape[0]):
            ccs[idx] = clear_border(ccs[idx],buffer_size = 5)


        skeletons = skeletonize(ccs > 0).astype('int64')
        np.putmask(skeletons,skeletons,ccs)
        component_lengths = np.bincount(skeletons.ravel())  
        too_short = component_lengths*self.px_size <= self.minimum_length
        too_short_mask = too_short[skeletons]
        skeletons[too_short_mask] = 0

        cell_coords = get_indices_pandas(ccs)
        del ccs
        skel_coords = get_indices_pandas(skeletons)

        matches = skel_coords.index.intersection(cell_coords.index)
        self.skel_coords = skel_coords[matches[1:]]
        self.cell_coords = cell_coords[matches[1:]]

    def read_binary(self,binary_images,filename,phase_images):
        self.phase = phase_images
        self.filename = filename
        self.sensor = phase_images.shape
        bw = binary_images > 0
        bw = bw.astype('int')
        ccs = label(bw)
        del bw
        cumulative_max_value = 1
        for image_slice in ccs:
            for id in np.unique(image_slice[image_slice>0]):
                image_slice[image_slice == id] = cumulative_max_value
                cumulative_max_value += 1
        
        skeletons = skeletonize(ccs > 0).astype('int64')
        np.putmask(skeletons,skeletons,ccs)

        cell_coords = get_indices_pandas(ccs)
        del ccs
        skel_coords = get_indices_pandas(skeletons)

        matches = skel_coords.index.intersection(cell_coords.index)
        self.skel_coords = skel_coords[matches[1:]]
        self.cell_coords = cell_coords[matches[1:]]

    def screen_cells(self,signal_images):
        if not self.skel_coords.shape[0]:
            print('There are no cells to screen!')
            return
        temp_df = pd.DataFrame()
        temp_df['filename'] = pd.Series(np.repeat(self.filename,self.cell_coords.index.size))
        temp_df['CellID'] = self.cell_coords.index
        temp_df['CellCoord'] = temp_df.CellID.apply(lambda x: np.array(self.cell_coords[x]))
        temp_df['frame'] = temp_df.CellCoord.apply(lambda x: x[0][0]+1)
        temp_df['skel_coords'] = temp_df.CellID.apply(lambda x: np.array(self.skel_coords[x]))
        temp_df['CellLength'] = temp_df.skel_coords.apply(lambda x: x[0].size*self.px_size)
        temp_df = temp_df[temp_df.CellLength > self.minimum_length]
        if not self.remove_linescans:
            nodes = temp_df.skel_coords.apply(lambda x: count_skeleton_nodes(x))
            temp_df = temp_df[nodes == 2]
        thickness = temp_df.apply(lambda row: measure_thickness(self.sensor[1:],row['CellCoord'],row['skel_coords']),axis=1)
        temp_df['width'] = thickness.apply(lambda x: 2*np.mean(x)*self.px_size)
        max_width = thickness.apply(lambda x: 2*np.max(x)*self.px_size)
        if self.remove_linescans:
            temp_df = temp_df[(temp_df.width < self.maximum_width) & (temp_df.width > 0.1)]
        else:
            temp_df = temp_df[(temp_df.width < self.maximum_width) & (temp_df.width > 0.1) | (max_width < self.inst_max_width)]
        saturated =  temp_df.CellCoord.apply(lambda x: False if signal_images[x[0],x[1],x[2]].max() == 2**16-1 else True)
        temp_df = temp_df[saturated]
        if self.remove_out_of_focus_cells:
            check_if_in_focus = temp_df.apply(lambda row: check_in_focus(signal_images[row.frame-1],row.CellCoord),axis = 1)
            temp_df = temp_df[check_if_in_focus]
                
        
        '''
        if self.back_sub:
            print(f'Initializing background subtraction in {self.filename}...')
            bkg_sub = np.empty(signal_images.shape())
            for i in np.arange(bkg_sub.shape[0]):
                bkg_sub[i] = back_sub(self.phase[i],signal_images[i])
            signal_images = bkg_sub
        '''
        if self.back_sub:
            #bkg_imgs = []
            for i in np.arange(self.sensor[0]):
                bkg_val = simple_background_estimation(self.phase[i],signal_images[i])
                #bkg_imgs.append(signal_images[i] - bkg_val)
                signal_images[i] = signal_images[i] - bkg_val
            #signal_images = np.array(bkg_imgs)
            #del bkg_imgs
        del self.phase

        if not self.remove_linescans:
            temp_df['traced_skel_coords'] = temp_df.skel_coords.apply(lambda x: parse_skeleton(self.sensor[1:],x[1:]))
            linescan = temp_df.apply(lambda row: signal_images[np.repeat(row.frame-1,len(row.traced_skel_coords[0])),row.traced_skel_coords[0],row.traced_skel_coords[1]],axis = 1)
            if not linescan.shape[0]:
                print('There are no cells to screen!')
                return
            temp_df['linescan'] = temp_df.apply(lambda row: signal_images[np.repeat(row.frame-1,len(row.traced_skel_coords[0])),row.traced_skel_coords[0],row.traced_skel_coords[1]],axis = 1)
        temp_df['Mean_Intens'] = temp_df.CellCoord.apply(lambda x: signal_images[x[0],x[1],x[2]].mean())
        temp_df['Int_Intens'] = temp_df.CellCoord.apply(lambda x: signal_images[x[0],x[1],x[2]].sum())
        temp_df['Norm_Intens_Length'] = temp_df['Int_Intens'].div(temp_df['CellLength'])
        temp_df['Norm_Intens_Area'] = temp_df.apply(lambda row: row.Int_Intens/(row.CellCoord[0].shape[0]*self.px_size**2),axis = 1)
        self.df = pd.concat([self.df,temp_df])

    def archive_cells(self):
        if not self.skel_coords.shape[0]:
            print('There are no cells to screen!')
            return
        temp_df = pd.DataFrame()
        temp_df['filename'] = pd.Series(np.repeat(self.filename,self.cell_coords.index.size))
        temp_df['CellID'] = self.cell_coords.index
        temp_df['CellCoord'] = temp_df.CellID.apply(lambda x: np.array(self.cell_coords[x]))
        temp_df['frame'] = temp_df.CellCoord.apply(lambda x: x[0][0]+1)
        temp_df['skel_coords'] = temp_df.CellID.apply(lambda x: np.array(self.skel_coords[x]))
        temp_df['CellLength'] = temp_df.skel_coords.apply(lambda x: x[0].size*self.px_size)
        temp_df = temp_df[temp_df.CellLength > self.minimum_length]
        thickness = temp_df.apply(lambda row: measure_thickness(self.sensor[1:],row['CellCoord'],row['skel_coords']),axis=1)
        temp_df['width'] = thickness.apply(lambda x: 2*np.mean(x)*self.px_size)
        temp_df = temp_df[(temp_df.width < self.maximum_width) & (temp_df.width > 0.1)]
        self.df = pd.concat([self.df,temp_df])

    def save_binary(self):
        isExist = os.path.exists('Binary')
        if not isExist:
            os.makedirs('Binary')
        for file_key,file_grp in self.df.groupby('filename'):
            sensor = self.segmentation_params[f'{file_key}-sensor']
            bw = np.full(sensor,0)
            coords = pd.np.column_stack(file_grp.CellCoord)
            bw[coords[0],coords[1],coords[2]] = 1
            imsave(f'Binary/{file_key}-bw.tif',bw.astype('uint8'))

    def return_df(self,save_params=False):
        df = self.df
        if save_params:
            IsExists = os.path.exists('Segmentation_Params.txt')
            if IsExists:
                os.remove('Segmentation_Params.txt')
            with open('Segmentation_Params.txt', 'w') as f:
                for key,item in self.segmentation_params.items():
                    f.write(f'{key}: {item}')
                    f.write('\n')
        return df
    
    def make_demograph(self,color_map = 'ocean',lut_min = -2.5,lut_max = 3,figure_size = [5,3],savename = 'Cell',savedemo = 1):
        df = self.df
        df['sizes'] = df.linescan.apply(lambda x: x.shape[0])
        df = df.sort_values('sizes')
        df = df.reset_index()

        demo = np.empty([df.sizes.max()+1,df.shape[0]])
        demo[:] = np.nan
        half_width = np.ceil(df.sizes.max()/2).astype(int)
        for index,row in df.iterrows():
            linescan = row.linescan
            linescan_bottom = np.floor(len(linescan)/2).astype(int)
            linescan_top = np.ceil(len(linescan)/2).astype(int)
            demo[half_width-linescan_bottom:half_width+linescan_top,index] = (linescan-linescan.mean())/linescan.std()
        
        fig, ax = plt.subplots(figsize=figure_size,layout = 'constrained')
        cax = ax.imshow(demo,cmap=color_map,vmin = lut_min,vmax=lut_max, aspect="auto")
        midpoint = np.ceil((df.sizes.max()+1)/2)

        ytick_locations = [midpoint+midpoint,midpoint-midpoint, midpoint-midpoint/2, midpoint, midpoint+midpoint/2]
        ax.yaxis.set_ticks(ytick_locations,np.round((ytick_locations-midpoint)*self.px_size,2))
        ax.set_ylabel('Cell position ($\mu$m)')
        ax.set_xlabel('Cell number')
        ax.spines[['top','right']].set_visible(False)
        ax.spines[['bottom','left']].set_linewidth(1.2)

        fig.tight_layout(h_pad=2.5)
        fig.subplots_adjust(right=0.8)
        cbar_ax = fig.add_axes([0.85, 0.35, 0.015, 0.3])
        fig.colorbar(cax, cax=cbar_ax,label='Z Score')
        if savedemo: plt.savefig(f'{savename}-Demograph.pdf',dpi=600)
        return demo, fig, ax

    def analyze_linescan_signal(self,window_size = 10):
        df = self.df
        kernel = np.ones(window_size)/window_size
        df['smoothlinescan'] = df.linescan.apply(lambda x: np.convolve(x,kernel,mode='same'))
        df['peaks']  = df.smoothlinescan.apply(lambda x: find_peaks(x,prominence=100))
        df['proms'] = df.peaks.apply(lambda x: x[1]['prominences'])
        df['peakmax'] = df.proms.apply(lambda x: np.where(x == x.max())[0])
        df['peak'] = df.apply(lambda row: row.peaks[0][row.peakmax], axis = 1)
        df['peak_properties'] = df.apply(lambda row: peak_widths(row.smoothlinescan,row.peak,rel_height=0.5),axis = 1)
        df['xdata_peak'] = df.peak_properties.apply(lambda x: range(int(np.floor(x[2][0])),int(np.ceil(x[3][0]))))
        df['peak_width'] = df.apply(lambda row: row.peak_properties[0][0]*self.px_size if (row.smoothlinescan.max() > 2*row.smoothlinescan.mean()) else 0,axis = 1)
        df['fraction_peak_intensity'] = df.apply(lambda row: row.smoothlinescan[row.xdata_peak].sum()/row.smoothlinescan.sum() if (row.smoothlinescan.max() > 2*row.smoothlinescan.mean()) else 0,axis = 1)
        df['half_cell_difference'] = df.apply(lambda row: abs(row.smoothlinescan[row.peak[0]+1:-1].mean() - row.smoothlinescan[0:row.peak[0]-1].mean() if (row.smoothlinescan.max() > 2*row.smoothlinescan.mean()) else 0 ),axis = 1)
        df = df.drop(['peaks','proms','peakmax','peak','peak_properties','xdata_peak'],axis = 1)
        return df