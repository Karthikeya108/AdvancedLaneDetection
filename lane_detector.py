import numpy as np
import cv2
from PIL import Image
import glob
from line import Line
import matplotlib.pyplot as plt

class LaneDetector:
    def __init__(self, sample_img_path):
        self.nx, self.ny = 9, 6
        self.sample_img = cv2.imread(sample_img_path)
        self.img_size = self.sample_img.shape[1::-1]
        self.mtx, self.dist = self.calibrate_camera()
        self.s_thresh=(170, 255)
        self.sx_thresh=(20, 100)
        self.img_src_points, self.warped_img, self.perspective_M, self.Minv = self.corners_unwarp(self.sample_img, self.mtx, self.dist)
        self.ploty = np.linspace(0, self.sample_img.shape[0]-1, num=self.sample_img.shape[0])
        self.y_eval =  np.max(self.ploty)
        # window settings
        self.window_width = 50 
        self.window_height = 80 # Break image into 9 vertical layers since image height is 720
        self.margin = 100
        self.ym_per_pix = 30/720
        self.xm_per_pix = 3.7/700
        self.car_position = 0.0
        self.left_lane = Line()
        self.right_lane = Line()
        self.radius_of_curvature = 0.0
        self.last_n_frames = 1
        
    def calibrate_camera(self):
        objp = np.zeros((self.ny*self.nx,3), np.float32)
        objp[:,:2] = np.mgrid[0:self.nx, 0:self.ny].T.reshape(-1,2)

        objpoints = []
        imgpoints = []
        cal_images_path = "./camera_cal/*jpg"
        cal_images = glob.glob(cal_images_path)

        for idx, fname in enumerate(cal_images):
            img = cv2.imread(fname)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Find the chessboard corners
            ret, corners = cv2.findChessboardCorners(gray, (self.nx, self.ny), None)

            # If found, add object points, image points
            if ret == True:
                objpoints.append(objp)
                imgpoints.append(corners)

        cal_img = cv2.imread('camera_cal/calibration1.jpg')
        cal_img_size = (cal_img.shape[1::-1])
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, cal_img_size, None, None)

        return mtx, dist
        
    def thresholding_pipeline(self, img, s_thresh=(170, 255), sx_thresh=(20, 100)):
        img = np.copy(img)
        # Convert to HLS color space
        hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
        h_channel = hls[:,:,0]
        l_channel = hls[:,:,1]
        s_channel = hls[:,:,2]

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Sobel x
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0) # Take the derivative in x
        abs_sobelx = np.absolute(sobelx) # Absolute x derivative to accentuate lines away from horizontal
        scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))

        # Threshold x gradient
        sxbinary = np.zeros_like(scaled_sobel)
        sxbinary[(scaled_sobel >= sx_thresh[0]) & (scaled_sobel <= sx_thresh[1])] = 1

        # Threshold color channel
        s_binary = np.zeros_like(s_channel)
        s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1
        # Stack each channel
        color_binary = np.dstack(( np.zeros_like(sxbinary), sxbinary, s_binary)) * 255

        combined_binary = np.zeros_like(sxbinary)
        combined_binary[(s_binary == 1) | (sxbinary == 1)] = 1

        return color_binary, combined_binary
    
    def corners_unwarp(self, img, mtx, dist):
        undist = cv2.undistort(img, mtx, dist, None, mtx)
        img_size = undist.shape[1::-1]
        src = np.float32([[600, 450], [685, 450], 
                          [1100, 720], [200, 720]])

        dst = np.float32([[300, 0], [980, 0], 
                          [980, 720], [300, 720]])

        M = cv2.getPerspectiveTransform(src, dst)
        Minv = cv2.getPerspectiveTransform(dst, src)
        warped = cv2.warpPerspective(img, M ,img_size, flags=cv2.INTER_LINEAR)

        cv2.polylines(img,np.int32([src]),True,(255,0,0),thickness=3)
        cv2.polylines(warped,np.int32([dst]),True,(255,0,0),thickness=3)

        return img, warped, M, Minv
        
    
    def detect_lane_pixels_sw(self, binary_warped):
        # Assuming you have created a warped binary image called "binary_warped"
        # Take a histogram of the bottom half of the image
        histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)
        # Create an output image to draw on and  visualize the result
        out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
        # Find the peak of the left and right halves of the histogram
        # These will be the starting point for the left and right lines
        midpoint = np.int(histogram.shape[0]//2)
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint

        # Choose the number of sliding windows
        nwindows = 9
        # Set height of windows
        window_height = np.int(binary_warped.shape[0]//nwindows)
        # Identify the x and y positions of all nonzero pixels in the image
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Current positions to be updated for each window
        leftx_current = leftx_base
        rightx_current = rightx_base
        # Set the width of the windows +/- margin
        margin = 100
        # Set minimum number of pixels found to recenter window
        minpix = 50
        # Create empty lists to receive left and right lane pixel indices
        left_lane_inds = []
        right_lane_inds = []
        
        if self.left_lane.current_fit and self.right_lane.current_fit:
            left_fit = self.left_lane.current_fit[-1]
            right_fit = self.right_lane.current_fit[-1]
            
            left_lane_inds = ((nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + 
            left_fit[2] - margin)) & (nonzerox < (left_fit[0]*(nonzeroy**2) + 
            left_fit[1]*nonzeroy + left_fit[2] + margin))) 

            right_lane_inds = ((nonzerox > (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + 
            right_fit[2] - margin)) & (nonzerox < (right_fit[0]*(nonzeroy**2) + 
            right_fit[1]*nonzeroy + right_fit[2] + margin)))  
        else:
            # Step through the windows one by one
            for window in range(nwindows):
                # Identify window boundaries in x and y (and right and left)
                win_y_low = binary_warped.shape[0] - (window+1)*window_height
                win_y_high = binary_warped.shape[0] - window*window_height
                win_xleft_low = leftx_current - margin
                win_xleft_high = leftx_current + margin
                win_xright_low = rightx_current - margin
                win_xright_high = rightx_current + margin
                # Draw the windows on the visualization image
                cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),
                (0,255,0), 2) 
                cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),
                (0,255,0), 2) 
                # Identify the nonzero pixels in x and y within the window
                good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
                (nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]
                good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
                (nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]
                # Append these indices to the lists
                left_lane_inds.append(good_left_inds)
                right_lane_inds.append(good_right_inds)
                # If you found > minpix pixels, recenter next window on their mean position
                if len(good_left_inds) > minpix:
                    leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
                if len(good_right_inds) > minpix:        
                    rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

            # Concatenate the arrays of indices
            left_lane_inds = np.concatenate(left_lane_inds)
            right_lane_inds = np.concatenate(right_lane_inds)

        # Extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds] 
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds] 

        # Fit a second order polynomial to each
        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)

        # Generate x and y values for plotting
        ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
        
        return left_fit, right_fit, left_fitx, right_fitx, ploty
    
    def get_lane_features(self, binary_warped):
        
        left_fit, right_fit, left_x_fitted, right_x_fitted, ploty = self.detect_lane_pixels_sw(binary_warped)
        
        if(len(self.left_lane.current_fit) >= self.last_n_frames):
            self.left_lane.current_fit.pop()
        self.left_lane.current_fit.append(left_fit)
        
        if(len(self.right_lane.current_fit) >= self.last_n_frames):
            self.right_lane.current_fit.pop()
        self.right_lane.current_fit.append(right_fit)
        
        self.left_lane.best_fit = np.mean( self.left_lane.current_fit, axis=0 )
        self.right_lane.best_fit = np.mean( self.right_lane.current_fit, axis=0 )

        for i in range(len(left_x_fitted)):

            if (right_x_fitted[i] < 0):
                right_x_fitted[i] = - right_x_fitted[i] + left_x_fitted[i] + 706
           
            if (right_x_fitted[i] - left_x_fitted[i]) < 700:
                right_x_fitted[i] = left_x_fitted[i] + 700
            
        if len(self.left_lane.recent_xfitted) > 1:
            self.left_lane.bestx = np.mean( self.left_lane.recent_xfitted, axis=0 )
            self.right_lane.bestx = np.mean( self.right_lane.recent_xfitted, axis=0 )  
            times = len(self.left_lane.recent_xfitted)
            self.left_lane.bestx = (self.left_lane.bestx + (times * left_x_fitted)) / (times+1)
            self.right_lane.bestx = (self.right_lane.bestx + (times * right_x_fitted)) / (times+1)

        if(len(self.left_lane.recent_xfitted) >= self.last_n_frames):
            self.left_lane.recent_xfitted.pop()
        self.left_lane.recent_xfitted.append(left_x_fitted)

        if(len(self.right_lane.recent_xfitted) >= self.last_n_frames):
            self.right_lane.recent_xfitted.pop()
        self.right_lane.recent_xfitted.append(right_x_fitted)
        
        if len(self.left_lane.recent_xfitted) < 2:
            self.left_lane.bestx = np.mean( self.left_lane.recent_xfitted, axis=0 )
            self.right_lane.bestx = np.mean( self.right_lane.recent_xfitted, axis=0 ) 

        left_fit_cr = np.polyfit(self.ploty * self.ym_per_pix, self.left_lane.bestx * self.xm_per_pix, 2)
        right_fit_cr = np.polyfit(self.ploty * self.ym_per_pix, self.right_lane.bestx * self.xm_per_pix, 2)
        # Calculate the new radii of curvature
        self.left_lane.radius_of_curvature = ((1 + (2 * left_fit_cr[0] * self.y_eval * self.ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
        self.right_lane.radius_of_curvature = ((1 + (2 * right_fit_cr[0] * self.y_eval * self.ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
        
        center_of_car = binary_warped.shape[1]//2
        height = binary_warped.shape[0]
        left_center = left_fit[0]*height**2 + left_fit[1]*height + left_fit[2]
        right_center = right_fit[0]*height**2 + right_fit[1]*height + right_fit[2]
        lane_center = (left_center + right_center) / 2
        self.car_position = self.xm_per_pix * (center_of_car - lane_center)

        return
    
    def add_text(self, img):
        font = cv2.FONT_HERSHEY_SIMPLEX
        pos1 = (10,50)
        pos2 = (10,110)
        fontScale = 1.5
        whiteColor = (255,255,255)
        lineType = 2

        self.radius_of_curvature = (self.left_lane.radius_of_curvature + self.right_lane.radius_of_curvature) / 2
        cv2.putText(img,'Radius of curvature: '+str(round(self.radius_of_curvature,2))+'(m)', pos1, font, fontScale, whiteColor, lineType)
        
        if (self.car_position <= 0.0):
            position = 'left'
        else:
            position = 'right'
        cv2.putText(img,'Vehicle is '+str(round(abs(self.car_position),2))+'m '+position+' of center', pos2, font, fontScale, whiteColor, lineType)
        
        return img
    
    def draw_lane_path(self, binary, undist):
        warp_zero = np.zeros_like(binary).astype(np.uint8)
        color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

        # Recast the x and y points into usable format for cv2.fillPoly()
        pts_left = np.array([np.transpose(np.vstack([self.left_lane.bestx, self.ploty]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([self.right_lane.bestx, self.ploty])))])
        pts = np.hstack((pts_left, pts_right))

        # Draw the lane onto the warped blank image
        cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))
        cv2.polylines(color_warp, np.int32([pts_left]),False,(0,0,255),thickness=20)
        cv2.polylines(color_warp, np.int32([pts_right]),False,(255,0,),thickness=20)

        # Warp the blank back to original image space using inverse perspective matrix (Minv)
        newwarp = cv2.warpPerspective(color_warp, self.Minv, (undist.shape[1], undist.shape[0])) 
        self.add_text(newwarp)
        # Combine the result with the original image
        result = cv2.addWeighted(undist, 1, newwarp, 0.3, 0)

        return result
        
    def detect_lane(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        undist = cv2.undistort(img, self.mtx, self.dist, None, self.mtx)
        color_binary, combined_binary = self.thresholding_pipeline(undist, self.s_thresh, self.sx_thresh)
        binary_warped = cv2.warpPerspective(combined_binary, self.perspective_M ,self.img_size, flags=cv2.INTER_LINEAR)
        self.get_lane_features(binary_warped)
        out = self.draw_lane_path(binary_warped, undist)
        
        return out