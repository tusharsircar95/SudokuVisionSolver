# Sudoku Vision
Python implementation of a program using OpenCV that extracts a sudoku grid out of an image, reads the digits and finally solves the puzzle via backtracking

## Steps

- Image is read and the noise is smoothed out via gaussian blurring. After that it is thresholded via adaptive thresholding. This leaves us with a black(0) and white(255) image. The sudoku grid lines and the digits are white after this stage.

  - #### Original Image
  ![Original Image](https://github.com/tusharsircar95/SudokuVision/blob/master/Images/original.JPG)

  - ####  Smoothed Image
  ![Smoothed Image](https://github.com/tusharsircar95/SudokuVision/blob/master/Images/smoothed.JPG)

  - #### Image After Thresholding
  ![Thresholded Image](https://github.com/tusharsircar95/SudokuVision/blob/master/Images/thresholded.JPG)

- To extract the main sudoku grid, we assume that the sudoku is the biggest part of the image or more specifically, the bounding rectangle of the sudoku grid has the highest area. Now, flood filling algorithm is used to find connected components of white pixels. Here, given a white pixel, we do a breadth first style exploration of the image to find the white connected component containing this pixel. To calculate the area of the bounding rectangle, we simple find out the left-most,right-most, top-most and bottom-most white pixel in the component and then the area can be calculated easily.

  - #### Outer Grid Flood Filled In Grey
    ![Outer Grid Greyed](https://github.com/tusharsircar95/SudokuVision/blob/master/Images/floodFilled.JPG)

- The biggest connected component now corresponds to the outer grid lines of the sudoku. Making all other pixels black, leaves us with an image of the outer sudoku grid lines.

  - #### Outer Sudoku Grid Shown In White
  ![Sudoku Outer Grid In White](https://github.com/tusharsircar95/SudokuVision/blob/master/Images/outerGrid.JPG)


- Next, we figure out the corners points of the outer sudoku grid. To do this I have tried two approaches, one is relatively involved while the other is a crude approximation that works under some reasonable assumptions:

  - Hough Lines Approach
      Here we use the Hough transformation to find out parameters of the geometric line segments on the image. Then, the left-most line is taken to be the one which is almost vertical and has the minimum x-intercept. Similarly the right-most, top-most and bottom-most line can be obtained.
      Finally, the corner points are obtained as the intersection of the above lines. Eg. the top left point is the intersection of the left-most and top-most line
  
  - Crude Approximation
    This works if we assume the sudoku grid to be in the same plane and more or less straight (not rotated). We calculate Left to be the lowest x-coordinate of a white pixel, Top to be the lowest (more close to the top of the image) y-coordinate of a white pixel and so on in the thresholded image. Then the topLeft point becomes (Left,Top), bottomRight point becomes (Right,Bottom) and so on. 
     
- Once we have the corner points, we calculate the maximum side length of the 4-sided polygon formed by the corner points, this is approximately the side length of the sudoku square grid. To correct the possible skewed perspective of the image we map this image to a square image of side length calculated above by calculating the appropiate homography and using the warpPerspective function in openCV.
This leaves us with an image containing exactly the sudoku puzzle which can then be divided into cells (9x9) each representing a digit/blank.



- Now the grid lines of the sudoku are not needed and are just noise. Certain ways have been discussed in the next few steps to deal with those but it is better if we can eliminate those lines completely.

  - The outer grid lines is known to us via the flood filling we had done before. So that can be erased by simply flood filling with black pixels
  
  - To erase the inner grid lines, I have used a template matching approach. Here, the crosses of the inner grid lines are detected via template matching and then flood filling with black pixels is initiated at these points, which eliminates the inner grid lines to some extent. 

  - #### Final Sudoku Grid (after the above mentioned steps)
    ![Final Grid](https://github.com/tusharsircar95/SudokuVision/blob/master/Images/FinalGrid.JPG)


- We now come to the part where we recognize the digit/blank in a cell given the image of a cell. For this we use template matching. But before matching with templates, we preprocess the cell image. This involves the following steps:

  - The cell image may contain the sudoku grid lines along the boundaries. Few pixels along all boundaries are first clipped off. Then using the same flood fill algorithm described above the biggest white component is discovered which contains points roughly in the center of the image, this component corresponds to the digit. This works because if a digit is present, most probably it is the biggest white component in the cell and also if the grid lines form a bigger connected component than the digit also then also we know that it won't be lying in the center of the image. Everything besides this component is removed (made black).
   
  - Now the white component is centered in the image. To do this we calculate the displacement of the centroid of the component from the cell center and then displace each white pixel by that amount (Mean Displacement). This puts the digit at the center. Then, the digit is cropped out by removing all rows and columns that contain only black pixels.
  
    - #### Initial Cell Image
    
    ![Grid Cell With Noise](https://github.com/tusharsircar95/SudokuVision/blob/master/Images/cellWithNoise.JPG)
    
    - #### After Noise Removal And Centering
    
    ![Noise removed and centered](https://github.com/tusharsircar95/SudokuVision/blob/master/Images/noiseRemovedAndCentered.JPG)

  - This image is then placed at the center of a 30x30 black image and is then ready to go through template matching.
    - Digit Clipped Out
    
    ![Digit clipped out](https://github.com/tusharsircar95/SudokuVision/blob/master/Images/digitClip.JPG)
    
    - Digit Template Formed
    
    ![Digit Template Ready](https://github.com/tusharsircar95/SudokuVision/blob/master/Images/digitTemplateFormed.JPG)

  - Also, to check whether a given cell is blank, we pick a square in the center of the image (after step (a))and check whether the number of white pixels are less than a certain threshold, if yes, then the cell is marked as blank. Alternatively, we can even check if the percentage of white pixels in the biggest component is less than a certain threshold or not. Value of this threshold was obviously found experimentally.
 
- Once all digits are recognized we pass the sudoku grid into the solveSudoku function which sovles it via backtracking and returns the solution
  
  
# Some Fallbacks And Future Work

- Digit recognition part is not very accurate as template matching may not be the best way to go. A trained neural net model might be better, or maybe just better templates, as this part needs to be almost 100% accurate
- The Hough Lines approach can be exploited to recognize even a rotated sudoku grid image and correct the rotation by figuring out the angle of the extreme lines. The extreme lines here can be found by comparing the values of the x and y intercepts
  
  
  
  
  
