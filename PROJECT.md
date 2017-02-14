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


## How to Run this code.
 * Run python Advanced_Lane_Finding.py. This creates output in same directory called result.mp4
 * Also, I have the ipython notebook Advanced_Lane_Finding_Submission.ipynb
 * Images are stored in output_images

## Camera Calibration

1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

### The code for the camera calibration is in the iPython notebook in Cell 2. 

I obtain the object points and corners from the Chessboard and use these to calibrate the camera. I then save it in a calibration file. I read the calibration file store the distortion coefficients in dst and mtx. 

Apply the distortion correction in Cell 3.

An example of a distortion corrected image is in Cell 4.


## Pipeline (single images)
1. Provide an example of a distortion-corrected image.

Example is shown in Cell 5.

2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image. Provide an example of a binary image result.

   I used a combination of color ( S-channel) and gradient thresholds as in Cell 10. These were arrived at by         experimenting with different kernel and threshold values.
   
   An example of this is in Cell 11.

3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed    image.

   I do this in Cell 12. 

   For Source and Destination points which I arrived at by some trial and error, I used:
   ```
   
   src = np.float32([[1270, 710],
                      [0,    710],
                      [546, 460],
                      [732, 460]])
   dst = np.float32([[1280, 720],
                      [0, 720],
                      [0, 0],
                      [1280, 0]])
   ```                  
   This resulted in
   
   |Source    |  Destination   |
   |----------|----------------|
   |1270,710  |   1280,720     |
   |0,710     |   0,720        |
   |546,460   |   0,0          |
   |732,460   |   1280,0       |
                      
   Cell 14 shows an example of a transformed image using the source and destination points above. I verified that they are roughly parallel.
 
4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

   Much of the code is borrowed from the Udacity lecture notes. 
   
   The code is found in the function Lane_Finder in Cell 19.
   
   I compute the histogram of the bottom half of the image and identify the peaks in the left and right half of 
   the histogram. I used sliding windows and stepped through the windows identifying lane pixels and adding them to
   the left and right lane indicators. Then the left and right line pixel positions are extracted and lines are fit 
   using a second order polynomial.
   
   Once the lane is detected, I search within the region to obtain the next lane lines that are good and then fit the 
   lines. If the vehicle is offset is > .1, I recalculate the lines.

 
 
5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the  vehicle with respect to center.

   I calculate the radius of curvature in the function get_curvature which is in Cell 17. This code was borrowed from the Udacity lecture notes.
   ```
   y_eval = np.max(lefty)
   left_curverad = ((1 + (2*left_fit[0]*y_eval + left_fit[1])**2)**1.5) / np.absolute(2*left_fit[0])
   right_curverad = ((1 + (2*right_fit[0]*y_eval + right_fit[1])**2)**1.5) / np.absolute(2*right_fit[0])
   ```
   This gave me the radius in pixels which is then converted to meters in the following code:
   ```
   ym_per_pix = 30/720 # meters per pixel in y dimension
   xm_per_pix = 3.7/700 # meters per pixel in x dimension

   # Fit new polynomials to x,y in world space
   left_fit_cr = np.polyfit(lefty*ym_per_pix, leftx*xm_per_pix, 2)
   right_fit_cr = np.polyfit(righty*ym_per_pix, rightx*xm_per_pix, 2)
   # Calculate the new radii of curvature
   left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) /np.absolute(2*left_fit_cr[0])
   right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr
        
   ```  
   I calculate the position of the vehicle using the code in the function get_offset in Cell 18.
   
   I used the bottom left, right and bottom y co-ordinates and then convert it to meters to get the vehicle offset 
   from the center
   
 
 
6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.
 
 This is shown in Cell 23. Output is also stored in output_images.
 
 ## Pipeline (video)
1. Provide a link to your final video output. Your pipeline should perform reasonably well on the entire project video (wobbly      lines are ok but no catastrophic failures that would cause the car to drive off the road!).

   <p>Here's a <a href="./result.mp4">Link to my video result</a></p>


## Discussion
1. Briefly discuss any problems / issues you faced in your implementation of this project. Where will your pipeline likely fail?    What could you do to make it more robust?

  The most difficult part of this exercise was tuning the threshold and obtaining the source and destination points 
  for the Perspective transform. I already see that this did not perform well on the challenge video where there is 
  not adequate lighting. Also, if there are others cars in the lane, this approach would not work very well. I think 
  identifying and classifying objects such as other vehicles would make this more robust.
