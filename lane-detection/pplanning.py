#Path Planning



# import calibrate
import draw_lane
import cv2
import numpy as np
from pathfinding.core.diagonal_movement import DiagonalMovement
from pathfinding.core.grid import Grid
from pathfinding.finder.a_star import AStarFinder
import pdb
from threshold_helpers import *
import matplotlib.pyplot as plt
import math
import pdb
# imag = cv2.imread('test_images/test5.jpg')
# imag_gray = cv2.cvtColor(imag, cv2.COLOR_BGR2GRAY)
# draw_lane.process_image(imag)
# cv2.imshow(imag)
# cv2.waitKey(0)
with open('test_dist_pickle.p', 'rb') as pick:
  dist_pickle = pickle.load(pick)

mtx = dist_pickle['mtx']
dist = dist_pickle['dist']
cap = cv2.VideoCapture('project_video.mp4')
image = cv2.imread('warped_5_final.jpg')


def draw_on_road(warped, left_fitx, left_yvals, right_fitx, right_yvals, ploty):
  #create img to draw the lines on
  warp_zero = np.zeros_like(warped).astype(np.uint8)
  color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

  #recast x and y into usable format for cv2.fillPoly
  pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
  pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
  # print('pts left', pts_left.shape, 'pts right', pts_right.shape)
  pts = np.hstack((pts_left, pts_right))

  #draw the lane onto the warped blank img
  cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))
  return color_warp
  # plt.imshow(color_warp)
  # plt.show()

#
lane = draw_lane.Lane()

def draw_visual(img,color_warp):
	img_size = (img.shape[1], img.shape[0])

	bot_width = .76
	mid_width = .08
	height_pct = .62
	bottom_trim = .935
	offset = img_size[0]*.25

	dst = np.float32([[img.shape[1]*(.5 - mid_width/2), img.shape[0]*height_pct], [img.shape[1]*(.5 + mid_width/2), img.shape[0]*height_pct],\
	 [img.shape[1]*(.5 + bot_width/2), img.shape[0]*bottom_trim], [img.shape[1]*(.5 - bot_width/2), img.shape[0]*bottom_trim]])
	src = np.float32([[offset, 0], [img_size[0] - offset, 0], [img_size[0] - offset, img_size[1]], [offset, img_size[1]]])

	# cv2.fillConvexPoly(image, src, 1)
	# plt.imshow(image)
	# plt.title('lines')
	# plt.show()
	Minv = cv2.getPerspectiveTransform(src, dst)

	#warp the blank back oto the original image using inverse perspective matrix
	newwarp = cv2.warpPerspective(color_warp, Minv, (img.shape[1], img.shape[0]))

	#combine the result with the original 
	result = cv2.addWeighted(img, 1, newwarp, 0.3, 0)
	# print('result shape', result.shape)
	# plt.imshow(result)
	# plt.show()
	return result

while(True):
    
    #
    
	ret, frame = cap.read()
   
    #frame = cv2.imread('intersect.png')
	undist_img = undist(frame, mtx, dist)
	# plt.imshow(undist_img)
	# plt.title('undist_img')
	# plt.show()

	# if want to perform mask, do it here
	# trapezoid = np.array([[570, 420], [160, 720], [1200, 720], [700, 420]], np.int32);
	# masked_image = region_of_interest(undist_img, [trapezoid])
	# plt.imshow(masked_image, cmap='gray')
	# plt.title('masked_image')
	# plt.show()
	#
	#Use Segnet Output Instead
	combo_image = combo_thresh(undist_img)
	#pdb.set_trace()
	# plt.imshow(combo_image, cmap='gray')
	# plt.title('combo_image')
	# plt.show()

	thresh_birdview = draw_lane.change_perspective(combo_image)
	left_fitx, lefty, right_fitx, righty, ploty, full_text = draw_lane.lr_curvature(thresh_birdview)
	#pdb.set_trace()

	#Path Planning
	matrix = cv2.resize(thresh_birdview, (100, 50))
	matrix[matrix>0] = 1
	grid = Grid(matrix=matrix.tolist())
	li = matrix.tolist()
	
	scfactor = 100.0/thresh_birdview.shape[1]
	start = grid.node(matrix.shape[1]/2, matrix.shape[0]-1)
	end_pts = [math.floor(0.5*scfactor*(left_fitx[0] +right_fitx[0]) ),0]
	#pdb.set_trace()
	#end = grid.node(matrix.shape[1]/2-5,0)

	end = grid.node(int(end_pts[0]), int(end_pts[1]))
	finder = AStarFinder(diagonal_movement=DiagonalMovement.always)
	path, runs = finder.find_path(start, end, grid)
	
	#Small frame to display
	bird_plan = draw_on_road(thresh_birdview, left_fitx, lefty, right_fitx, righty, ploty)
	bird_plan = cv2.resize(bird_plan, (100,50))

	#Draw Path
	for pt in path:
		#print pt
		cv2.circle(bird_plan,pt, 2, (0,0,255), -1)

	bird_plan_big = cv2.resize(bird_plan, (undist_img.shape[1],undist_img.shape[0]))
	result = draw_visual(undist_img,bird_plan_big)
	cv2.imshow('path',result)
	cv2.imshow('path2',bird_plan)
#	out.write(frame)
	#pdb.set_trace()
	#cv2.waitKey(0)
	
	if cv2.waitKey(1) & 0xFF == ord('q'):
	    break

#print('operations:', runs, 'path length:', len(path))
#print(grid.grid_str(path=path, start=start, end=end))
