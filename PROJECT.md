## Advanced Lane Finding Project

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.


## Camera Calibration

1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

### The code for the camera calibration is in Cell 2. 

I obtain the object points and corners from the Chessboard and use these to calibrate the camera. I then save it in a calibration file. I read the calibration file store the distortion coefficients in dst and mtx. 

Apply the distortion correction in Cell 3.

An example of a distortion corrected image is in Cell 4.


## Pipeline (single images)
1. Provide an example of a distortion-corrected image.

   See below.

2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image. Provide an example of a binary image result.

   I used a combination of color and gradient thresholds as in Cell 10. An example of this is in Cell 11.

3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed    image.

   I do this in Cell 12. 

   For Source and Destination points which I arrived at by some trial and error, I used:

   src = np.float32([[1270, 710],
                      [0,    710],
                      [546, 460],
                      [732, 460]])
   dst = np.float32([[1280, 720],
                      [0, 720],
                      [0, 0],
                      [1280, 0]])
                      
   Cell 14 shows an example of a transformed image using the source and destination points above. They are roughly parallel.
 
4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?
 
 
5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the        vehicle with respect to center.
 
 
6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.
 
 
 ## Pipeline (video)
1. Provide a link to your final video output. Your pipeline should perform reasonably well on the entire project video (wobbly      lines are ok but no catastrophic failures that would cause the car to drive off the road!).

  The final video ouput is stored in <p>Here's a <a href="./result.mp4">Link to my video result</a></p>


## Discussion
1. Briefly discuss any problems / issues you faced in your implementation of this project. Where will your pipeline likely fail?    What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.
