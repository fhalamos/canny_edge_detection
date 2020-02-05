import numpy as np

"""
   Mirror an image about its border.

   Arguments:
      image - a 2D numpy array of shape (sx, sy)
      wx    - a scalar specifying width of the top/bottom border
      wy    - a scalar specifying width of the left/right border

   Returns:
      img   - a 2D numpy array of shape (sx + 2*wx, sy + 2*wy) containing
              the original image centered in its interior and a surrounding
              border of the specified width created by mirroring the interior
"""
def mirror_border(image, wx = 1, wy = 1):
   assert image.ndim == 2, 'image should be grayscale'
   sx, sy = image.shape
   # mirror top/bottom
   top    = image[:wx:,:]
   bottom = image[(sx-wx):,:]
   img = np.concatenate( \
      (top[::-1,:], image, bottom[::-1,:]), \
      axis=0 \
   )
   # mirror left/right
   left  = img[:,:wy]
   right = img[:,(sy-wy):]
   img = np.concatenate( \
      (left[:,::-1], img, right[:,::-1]), \
      axis=1 \
   )
   return img

"""
   Pad an image with zeros about its border.

   Arguments:
      image - a 2D numpy array of shape (sx, sy)
      wx    - a scalar specifying width of the top/bottom border
      wy    - a scalar specifying width of the left/right border

   Returns:
      img   - a 2D numpy array of shape (sx + 2*wx, sy + 2*wy) containing
              the original image centered in its interior and a surrounding
              border of zeros
"""
def pad_border(image, wx = 1, wy = 1):
   assert image.ndim == 2, 'image should be grayscale'
   sx, sy = image.shape
   img = np.zeros((sx+2*wx, sy+2*wy))
   img[wx:(sx+wx),wy:(sy+wy)] = image
   return img

"""
   Remove the border of an image.

   Arguments:
      image - a 2D numpy array of shape (sx, sy)
      wx    - a scalar specifying width of the top/bottom border
      wy    - a scalar specifying width of the left/right border

   Returns:
      img   - a 2D numpy array of shape (sx - 2*wx, sy - 2*wy), extracted by
              removing a border of the specified width from the sides of the
              input image
"""
def trim_border(image, wx = 1, wy = 1):
   assert image.ndim == 2, 'image should be grayscale'
   sx, sy = image.shape
   img = np.copy(image[wx:(sx-wx),wy:(sy-wy)])
   return img

"""
   Return an approximation of a 1-dimensional Gaussian filter.

   The returned filter approximates:

   g(x) = 1 / sqrt(2 * pi * sigma^2) * exp( -(x^2) / (2 * sigma^2) )

   for x in the range [-3*sigma, 3*sigma]
"""
def gaussian_1d(sigma = 1.0):
   width = np.ceil(3.0 * sigma)
   x = np.arange(-width, width + 1)
   g = np.exp(-(x * x) / (2 * sigma * sigma))
   g = g / np.sum(g)          # normalize filter to sum to 1 ( equivalent
   g = np.atleast_2d(g)       # to multiplication by 1 / sqrt(2*pi*sigma^2) )
   return g

"""
   CONVOLUTION IMPLEMENTATION (10 Points)

   Convolve a 2D image with a 2D filter.

   Requirements:

   (1) Return a result the same size as the input image.

   (2) You may assume the filter has odd dimensions.

   (3) The result at location (x,y) in the output should correspond to
       aligning the center of the filter over location (x,y) in the input
       image.

   (4) When computing a product at locations where the filter extends beyond
       the defined image, treat missing terms as zero.  (Equivalently stated,
       treat the image as being padded with zeros around its border).

   You must write the code for the nested loops of the convolutions yourself,
   using only basic loop constructs, array indexing, multiplication, and
   addition operators.  You may not call any Python library routines that
   implement convolution.

   Arguments:
      image  - a 2D numpy array
      filt   - a 1D or 2D numpy array, with odd dimensions
      mode   - 'zero': preprocess using pad_border or 'mirror': preprocess using mirror_border.

   Returns:
      result - a 2D numpy array of the same shape as image, containing the
               result of convolving the image with filt

   Reference
   https://www.pyimagesearch.com/2016/07/25/convolutions-with-opencv-and-python/
"""
def conv_2d(image, filt, mode='zero'):

  # make sure that both image and filter are 2D arrays
  assert image.ndim == 2, 'image should be grayscale'
  filt = np.atleast_2d(filt)

  # Dimensions of image and filter
  (iH, iW) = image.shape
  (kH, kW) = filt.shape

  # Allocate memory for the output image
  output = np.zeros((iH, iW), dtype="float32")

  # Dimensions of padding for image
  pad_x = (kW - 1) // 2
  pad_y = (kH - 1) // 2
  
  # Add padding
  if mode =='zero':
    image = pad_border(image, wx = pad_y, wy = pad_x)
  elif mode == 'mirror':
    image = mirror_border(image, wx = pad_y, wy = pad_x) 
  
  # loop over the input image, "sliding" the kernel across
  # each (x, y)-coordinate from left-to-right and top to
  # bottom

  # remember that now dimensions of image is (iH + pad_y*2, iW + pad_x*2)
  for y in np.arange(pad_y, iH + pad_y):
    for x in np.arange(pad_x, iW + pad_x):
      # extract the region of interest (ROI) of the image
      roi = image[y - pad_y:y + pad_y + 1, x - pad_x:x + pad_x + 1]
 
      # perform the actual convolution by taking the
      # element-wise multiplicate between the ROI and
      # the kernel, then summing the matrix
      k = (roi * filt).sum()
 
      # store the convolved value in the output (x,y)-
      # coordinate of the output image
      output[y - pad_y, x - pad_x] = k


  return output

"""
   GAUSSIAN DENOISING (5 Points)

   Denoise an image by convolving it with a 2D Gaussian filter.

   Convolve the input image with a 2D filter G(x,y) defined by:

   G(x,y) = 1 / sqrt(2 * pi * sigma^2) * exp( -(x^2 + y^2) / (2 * sigma^2) )

   You may approximate the G(x,y) filter by computing it on a
   discrete grid for both x and y in the range [-3*sigma, 3*sigma].

   See the gaussian_1d function for reference.

   Note:
   (1) Remember that the Gaussian is a separable filter.
   (2) Denoising should not create artifacts along the border of the image.
       Make an appropriate assumption in order to obtain visually plausible
       results along the border.

   Arguments:
      image - a 2D numpy array
      sigma - standard deviation of the Gaussian

   Returns:
      img   - denoised image, a 2D numpy array of the same shape as the input
"""
def denoise_gaussian(image, sigma = 1.0):
  filt = gaussian_1d(sigma)
  return conv_2d(conv_2d(image, filt, 'mirror'), np.transpose(filt), 'mirror')
  
  # Alternative approach
  # filt1 = gaussian_1d(sigma)
  # filt2 = gaussian_1d(sigma)
  # filt = np.matmul(filt1.T,filt2)
  # return conv_2d(image, filt, 'mirror')




"""
   MEDIAN DENOISING (5 Points)

   Denoise an image by applying a median filter.

   Note:
       Denoising should not create artifacts along the border of the image.
       Make an appropriate assumption in order to obtain visually plausible
       results along the border. No padding needed in median denosing.

   Arguments:
      image - a 2D numpy array
      width - width of the median filter; compute the median over 2D patches of
              size (2*width +1) by (2*width + 1)

   Returns:
      img   - denoised image, a 2D numpy array of the same shape as the input

  
  Reference used: https://stackoverflow.com/questions/18427031/median-filter-with-python-and-opencv
"""
def denoise_median(image, width = 1):

  (iH, iW) = image.shape

  # allocate memory for the output image
  output = np.zeros((iH, iW), dtype="float32")

  # copy the same pixels from the original image to output
  for y in range(iH):
    for x in range(iW):
        output[y,x]=image[y,x]


  for y in range(0, iH):
    for x in range(0,iW):
      #Careful not to overflow in the pixels index
      min_y = max(y - width, 0) 
      max_y = min(y + width + 1, iH)
      min_x = max(x-width,0)
      max_x = min(x+width+1,iW)

      #Get pixels sourranding point x,y
      members = image[min_y: max_y, min_x:max_x].ravel()
  
      members.sort()

      #in case of odd number of members
      if(len(members)%2==1):
        #int(len(members) / 2)) will be the index of the median term  
        median = members.item(int(len(members)/ 2))
      else: #This would be the case in edges, where there would be 4 pixels
        a = members.item(int(len(members)/2-1))
        b = members.item(int(len(members)/2))
        median = (a+b)/2

      output[y,x]=median

  return output



"""
   SOBEL GRADIENT OPERATOR (5 Points)
   Compute an estimate of the horizontal and vertical gradients of an image
   by applying the Sobel operator.
   The Sobel operator estimates gradients dx(horizontal), dy(vertical), of
   an image I as:

         [ 1  0  -1 ]
    dx = [ 2  0  -2 ] (*) I
         [ 1  0  -1 ]

         [  1  2  1 ]
    dy = [  0  0  0 ] (*) I
         [ -1 -2 -1 ]

   where (*) denotes convolution.
   Note:
      (1) Your implementation should be as efficient as possible.
      (2) Avoid creating artifacts along the border of the image.
   Arguments:
      image - a 2D numpy array
   Returns:
      dx    - gradient in x-direction at each point
              (a 2D numpy array, the same shape as the input image)
      dy    - gradient in y-direction at each point
              (a 2D numpy array, the same shape as the input image)
"""
def sobel_gradients(img):

  gx_1 = np.atleast_2d([-1, 0, 1])
  gx_2 = np.transpose(np.atleast_2d([1, 2, 1]))

  gy_1 = np.atleast_2d([-1, -2, -1])
  gy_2 = np.transpose(np.atleast_2d([1, 0, -1]))

  dx = conv_2d(conv_2d(img, gx_1, mode='mirror'), gx_2, mode='mirror')
  dy = conv_2d(conv_2d(img, gy_1, mode='mirror'), gy_2, mode='mirror')

  return dx, dy


"""
   NONMAXIMUM SUPPRESSION (10 Points)

   Nonmaximum suppression.

   Given an estimate of edge strength (mag) and direction (theta) at each
   pixel, suppress edge responses that are not a local maximum along the
   direction perpendicular to the edge.

   Equivalently stated, the input edge magnitude (mag) represents an edge map
   that is thick (strong response in the vicinity of an edge).  We want a
   thinned edge map as output, in which edges are only 1 pixel wide.  This is
   accomplished by suppressing (setting to 0) the strength of any pixel that
   is not a local maximum.

   Note that the local maximum check for location (x,y) should be performed
   not in a patch surrounding (x,y), but along a line through (x,y)
   perpendicular to the direction of the edge at (x,y).

   A simple, and sufficient strategy is to check if:
      ((mag[x,y] > mag[x + ox, y + oy]) and (mag[x,y] >= mag[x - ox, y - oy]))
   or
      ((mag[x,y] >= mag[x + ox, y + oy]) and (mag[x,y] > mag[x - ox, y - oy]))
   where:
      (ox, oy) is an offset vector to the neighboring pixel in the direction
      perpendicular to edge direction at location (x, y)

   Arguments:
      mag    - a 2D numpy array, containing edge strength (magnitude)
      theta  - a 2D numpy array, containing edge direction in [0, 2*pi)

   Returns:
      nonmax - a 2D numpy array, containing edge strength (magnitude), where
               pixels that are not a local maximum of strength along an
               edge have been suppressed (assigned a strength of zero)
"""
def nonmax_suppress(mag, theta):

  (iH, iW) = mag.shape

  #Our output will initially be full of zeros, and we will fill only pixels that are local maximums
  output = np.zeros((iH, iW))

  #ox and oy will be the x and y components of the unit vector in the gradient directionent
  #We will use this values to look at the neighbors of the pixel in the gradient direction
  #We know that pixels have discrete positions, so we will round this values so that
  #x + ox and y + oy are integers

  #Given the angle, we use cos and sin to find the x and y components of the gradient unit vector 
  #we are using rounding to find discrete values of ox and oy, and to determine if the components are strong enough to determine which pixels to look at
  ox = np.round(np.cos(theta)).astype(int)
  oy = np.round(np.sin(theta)).astype(int)

  for y in range(0,iH):
    for x in range(0,iW):

      # Point in the direction of the gradient
      neighbor_a_x = min(max(x+ox[y,x],0),iW-1)
      neighbor_a_y = min(max(y+oy[y,x],0),iH-1)
      # Point in the opposite direction of gradient
      neighbor_b_x = min(max(x-ox[y,x],0),iW-1)
      neighbor_b_y = min(max(y-oy[y,x],0),iH-1)
      
      neighbor_a_mag = mag[neighbor_a_y, neighbor_a_x] 
      neighbor_b_mag = mag[neighbor_b_y, neighbor_b_x] 

      #Now check if our point is a local maximum
      #We need the point to have magnitud as big as both neighbors,
      #and at least bigger than one      
      if ((mag[y,x] > neighbor_a_mag) and (mag[y,x] >= neighbor_b_mag) or 
        (mag[y,x] >= neighbor_a_mag) and (mag[y,x] > neighbor_b_mag)):
        #If the point is a local maximum, keep the value of the magnitude
        output[y,x] = mag[y,x]
        
  return output



"""
Given two edges, function returns if they are neighbors
"""
def edges_are_neighbors(strong_edge, weak_edge):
  (strong_y, strong_x) = strong_edge
  (weak_y, weak_x) = weak_edge

  if(abs(strong_x-weak_x)<=1 and abs(strong_y-weak_y)<=1):
    return True
  else:
    return False
"""
Checks if a given weak edge is in the direction of gradient of a strong edge
"""
def weak_edge_in_direction_of_strong(strong_edge, weak_edge, theta):
  (strong_y, strong_x) = strong_edge
  (weak_y, weak_x) = weak_edge

  angle = theta[strong_y, strong_x]

  ox = np.round(np.cos(angle)).astype(int)
  oy = np.round(np.sin(angle)).astype(int)

  if((strong_y+oy == weak_y and strong_x+ox == weak_x) 
    or((strong_y-oy == weak_y and strong_x-ox == weak_x))):
    return True
  else:
    return False


"""
   HYSTERESIS EDGE LINKING (10 Points)

   Hysteresis edge linking.

   Given an edge magnitude map (mag) which is thinned by nonmaximum suppression,
   first compute the low threshold and high threshold so that any pixel below
   low threshold will be thrown away, and any pixel above high threshold is
   a strong edge and will be preserved in the final edge map.  The pixels that
   fall in-between are considered as weak edges.  We then add weak edges to
   true edges if they connect to a strong edge along the gradient direction.

   Since the thresholds are highly dependent on the statistics of the edge
   magnitude distribution, we recommend to consider features like maximum edge
   magnitude or the edge magnitude histogram in order to compute the high
   threshold.  Heuristically, once the high threshod is fixed, you may set the
   low threshold to be propotional to the high threshold.

   Note that the thresholds critically determine the quality of the final edges.
   You need to carefully tuned your threshold strategy to get decent
   performance on real images.

   For the edge linking, the weak edges caused by true edges will connect up
   with a neighbouring strong edge pixel.  To track theses edges, we
   investigate the 8 neighbours of strong edges.  Once we find the weak edges,
   located along strong edges' gradient direction, we will mark them as strong
   edges.  You can adopt the same gradient checking strategy used in nonmaximum
   suppression.  This process repeats util we check all strong edges.

   In practice, we use a queue to implement edge linking.  In python, we could
   use a list and its fuction .append or .pop to enqueue or dequeue.

   Arguments:
     nonmax - a 2D numpy array, containing edge strength (magnitude) which is thined by nonmaximum suppression
     theta  - a 2D numpy array, containing edeg direction in [0, 2*pi)

   Returns:
     edge   - a 2D numpy array, containing edges map where the edge pixel is 1 and 0 otherwise.

  #Reference: http://justin-liang.com/tutorials/canny/#double-thresholding
"""
def hysteresis_edge_linking(nonmax, theta):#, consider_weak_edges): #Consider including 4th argument case we want to test how weak-edge-linking is working
  
  #First double thresholding
  high_threshold_ratio = 0.1
  low_threshold_ratio = 0.01

  high_threshold = np.max(nonmax)*high_threshold_ratio;
  low_threshold = high_threshold*low_threshold_ratio;

  (iH, iW) = nonmax.shape

  #Our output will initially be full of zeros, and we will mark with 1 strong edges
  output = np.zeros((iH, iW))

  strong_edges = list()
  weak_edges = list()

  for y in range(0,iH):
    for x in range(0,iW):
      if(nonmax[y,x]>=high_threshold):
        output[y,x]=1
        strong_edges.append((y,x))
      elif(nonmax[y,x]>=low_threshold):
        weak_edges.append((y,x))


  # if(consider_weak_edges):
  #Now we loop over all strong edges, for each we loop over the weak_edges and check if any of the close have the same direction
  for strong_edge in strong_edges:
    for weak_edge in weak_edges: #This loop is kind of inefficient. Wd be cool to only loop over weak_edges in the vecinity of the strong_edge. A dictionary of weak_edges indexed by their position could work
      if (edges_are_neighbors(strong_edge, weak_edge) and 
        weak_edge_in_direction_of_strong(strong_edge, weak_edge, theta)):
        output[weak_edge]=1

  return output

"""
   CANNY EDGE DETECTOR (5 Points)

   Canny edge detector.

   Given an input image:

   (1) Compute gradients in x- and y-directions at every location using the
       Sobel operator.  See sobel_gradients() above.

   (2) Estimate edge strength (gradient magnitude) and direction.

   (3) Perform nonmaximum suppression of the edge strength map, thinning it
       in the direction perpendicular to that of a local edge.
       See nonmax_suppress() above.

   (4) Compute the high threshold and low threshold of edge strength map
       to classify the pixels as strong edges, weak edges and non edges.
       Then link weak edges to strong edges

   Return the original edge strength estimate (max), the edge
   strength map after nonmaximum suppression (nonmax) and the edge map
   after edge linking (edge)

   Arguments:
      image    - a 2D numpy array

   Returns:
      mag      - a 2D numpy array, same shape as input, edge strength at each pixel
      nonmax   - a 2D numpy array, same shape as input, edge strength after nonmaximum suppression
      edge     - a 2D numpy array, same shape as input, edges map where edge pixel is 1 and 0 otherwise.
"""
def canny(image):

  dx, dy = sobel_gradients(image)

  mag = np.sqrt(np.square(dx)+np.square(dy))
  theta = np.arctan2(dy,dx)

  nonmax = nonmax_suppress(mag,theta)

  #only_strong_edge = hysteresis_edge_linking(nonmax, theta, False)
  edge = hysteresis_edge_linking(nonmax, theta)#, True)

  return mag, nonmax, edge#, only_strong_edge


# Extra Credits:
# (a) Improve Edge detection image quality (5 Points)
# (b) Bilateral filtering (5 Points)
# You can do either one and the maximum extra credits you can get is 5.
"""
    BILATERAL DENOISING (Extra Credits: 5 Points)
    Denoise an image by applying a bilateral filter
    Note:
        Performs standard bilateral filtering of an input image.
        Reference link: https://en.wikipedia.org/wiki/Bilateral_filter

        Basically, the idea is adding an additional edge term to Guassian filter
        described above.

        The weighted average pixels:

        BF[I]_p = 1/(W_p)sum_{q in S}G_s(||p-q||)G_r(|I_p-I_q|)I_q

        In which, 1/(W_p) is normalize factor, G_s(||p-q||) is spatial Guassian
        term, G_r(|I_p-I_q|) is range Guassian term.

        We only require you to implement the grayscale version, which means I_p
        and I_q is image intensity.

    Arguments:
        image       - input image
        sigma_s     - spatial param (pixels), spatial extent of the kernel,
                       size of the considered neighborhood.
        sigma_r     - range param (no normalized, a propotion of 0-255),
                       denotes minimum amplitude of an edge
    Returns:
        img   - denoised image, a 2D numpy array of the same shape as the input
"""

def denoise_bilateral(image, sigma_s=1, sigma_r=25.5):
    assert image.ndim == 2, 'image should be grayscale'
    ##########################################################################
    # TODO: YOUR CODE HERE
    raise NotImplementedError('denoise_bilateral')
    ##########################################################################
    return img
