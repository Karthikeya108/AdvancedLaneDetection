
**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

### Camera Calibration

The code for this step is contained in the method `calibrate_camera` (lines 31 to 56) in the file `lane_detector.py`.

The given set of sample chessboard images have 9 corners column-wise and 6 corners row-wise. 
I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

<img src="./output_images/undistort.jpg" height="400" width="1200">

I store the distortion co-effecients  and the camera matrix to be applied to the images from the drive.

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

I read the image into an array, and apply the `cv2.undistort()` function with `distortion co-effecients` and `camera matrix` computed in the first step. Following is an example of an image, before and after applying the `cv2.undistort()` function:

<img src="./output_images/undistort2.jpg" height="400" width="1200">

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used a combination of color and gradient thresholds to generate a binary image (thresholding steps at lines 58 through 85 in `lane_detector.py`).  Here's an example of my output for this step. 

<img src="./output_images/thresholding_pipeline_result.jpg" height="400" width="1200">

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `corners_unwarp()`, which appears in lines 87 through 103 in the file `lane_detector.py`.  The `corners_unwarp()` function takes as inputs an image (`img`), as well as camera matrix (`mtx`) and distortion co-efficients (`dist`).  I chose the hardcode the source and destination points form the top left to bottom left, clockwise:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 600, 450      | 300, 0        | 
| 685, 450      | 980, 0      |
| 1100, 720     | 980, 720      |
| 200, 720      | 360, 720        |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

<img src="./output_images/undistorted_warped_img.jpg" height="400" width="1200">

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

After having applied the perpective transform, I tried 2 methods to detect the lane pixels in the images. In the first method, I sliced the images horizontal sections and for each section applied convolution on a small window, across the section. Then identified the regions of lane lines with regions of high pixel values. With these pixel value postions, I fitted a polynomial using the `np.polyfit()` function. Using the co-effcients from the result, I computed the `x` values for given `y` poistion (0 to image height). The application of this technique can be visualized as follows:

<img src="./output_images/detect_lane_pixels_convolve.jpg" height="400" width="1200">

However, this method had some shortcomings. The technique worked for most parts of the project video, but had problems when there was different lighting conditions on the road and when the lane lines were discontinous. This issue can be visualized as follows:

<img src="./output_images/detect_lane_pixels_conv.jpg" height="600" width="1200">

Technique 1 in code is in the IPython Notebook (./AdvanceLandDetectionTechniques.ipynb) in code blocks 63, 64 and 65.

Then I resorted to the `sliding window` technique which was explained in the lectures. This technique also involved slicin the images and finding the high pixel value regions. However, for me this technique worked way better than the `convolution` technique. Following visualization illustrated the result.

<img src="./output_images/detect_lane_pixels_sw.jpg" height="600" width="1200">



#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I did this in lines # through # in my code in `my_other_file.py`

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in lines # through # in my code in `yet_another_file.py` in the function `map_lane()`.  Here is an example of my result on a test image:

![alt text][image6]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_video.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  
