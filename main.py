import cv2
import numpy as np
import queue
import math
from sklearn.neural_network import MLPClassifier
from sklearn.multiclass import OutputCodeClassifier
from sklearn.svm import LinearSVC
from sklearn.externals import joblib
from PIL import Image


# Fills connected components of white cells starting at the given point (seed) and returns the area
# of the filled portion
def customFloodFill(image,seed,color,areaType):
	q = queue.Queue()
	q.put(seed)
	rows,cols = np.shape(image)
	minX,minY,maxX,maxY = 20000,20000,0,0
	
	count = 0
	l = 1000
	while not q.empty():
		point = q.get()
		if image[point[0],point[1]] == 255:
			if point[0] < (rows/5):
				l = min(l,point[1])
			count = count + 1
			image[point[0],point[1]] = color
			for i in range(3):
				for j in range(3):
					x = point[0] + i - 1
					y = point[1] + j - 1
					if x >= 0 and y >= 0 and x < rows and y < cols:
						if image[x][y] == 255:
							q.put([x,y])
							minX = min(minX,x)
							minY = min(minY,y)
							maxX = max(maxX,x)
							maxY = max(maxY,y)
							
	boundingArea = (maxX-minX) * (maxY-minY)
	if areaType == 0:
		return image,boundingArea,l
	else: return image,count,l


# Extracts the outer grid of sudoku by finding the biggest white component using the flood fill algorithm	
def extractOuterGrid(img):
		rows,cols = np.shape(img)
		maxArea = 0
		point = [0,0]

		imgOriginal = img.copy()
		for i in range(rows):
			for j in range(cols):
				if img[i][j] == 255:
					img,area,dummy = customFloodFill(img,[i,j],100,0)
					if area > maxArea:
						maxArea = area
						point = [i,j]
					
		img = imgOriginal
		img,area,dummy = customFloodFill(img,[point[0],point[1]],100,0)	
		for i in range(rows):
			for j in range(cols):
				if img[i][j] == 100:
					img[i][j] = 255
				else: img[i][j] = 0
		return img,point
	
# Draws a line on the image given its parameters in normal form	
def drawLine(img,params):
	rho,theta = (params[0],params[1])
	imgX,imgY = np.shape(img)
	a = np.cos(theta)
	b = np.sin(theta)
	if b == 0:
		x1,y1,x2,y2 = rho,0,rho,imgY
		cv2.line(img,(x1,y1),(x2,y2),(0,0,0),1)
	else:
		x1,y1,x2,y2 = int(0),int((rho-(x1*a))/b),int(imgX),int((rho-(x2*a))/b)
		cv2.line(img,(x1,y1),(x2,y2),(0,0,0),1)
	return img
	
# Given a set of lines finds the lines at extreme ends (left,right,top,bottom)
def getExtremeLines(lines,imgX,imgY):
	threshold = 0.2
	leftExtreme = [0,0]
	rightExtreme = [0,0]
	topExtreme = [0,0]
	bottomExtreme = [0,0]
	maxX,maxY,minX,minY = 0,0,imgX,imgY

	for line in lines:
		for rho,theta in line:
			# Almost vertical line
			if (theta-0.0) < threshold or (np.pi-theta) < threshold:
				x = abs(rho)
				if x < minX:
					leftExtreme = [rho,theta]
					minX = x
				if x > maxX:
					rightExtreme = [rho,theta]
					maxX = x
			
			# Almost horizontal line
			if abs(theta-(np.pi/2)) < threshold:
				a = np.cos(theta)
				b = np.sin(theta)
				x = 0
				y = int((rho-(x*a))/b)
				
				if y < minY:
					topExtreme = [rho,theta]
					minY = y
				if y > maxY:
					bottomExtreme = [rho,theta]
					maxY = y
	return leftExtreme,rightExtreme,topExtreme,bottomExtreme
	
# Given the parameters of a line in normal form, returns two points on that line
def getTwoPoints(line,imgX,imgY):
	rho,theta = line[0],line[1]
	a,b = np.cos(theta),np.sin(theta)
	if b == 0:
		x1,y1,x2,y2 = rho,0,rho,imgY
		return (x1,y1),(x2,y2)
	else:
		x1,x2 = int(0),int(imgX)
		y1,y2 = int((rho-(x1*a))/b),int((rho-(x2*a))/b) 
		return (x1,y1),(x2,y2)

# Given the parameters of two lines in normal form, returns the intersection point of the two lines	
def getIntersectionPoint(l1,l2,imgX,imgY):
	
	p1,p2 = getTwoPoints(l1,imgX,imgY)
	q1,q2 = getTwoPoints(l2,imgX,imgY)
	
	x1 = p2[1] - p1[1]
	x2 = p1[0] - p2[0]
	x3 = x1*(p1[0]) + x2*(p1[1])
	
	y1 = q2[1] - q1[1]
	y2 = q1[0] - q2[0]
	y3 = y1*(q1[0]) + y2*(q1[1])

	dt = x1*y2 - x2*y1
	x = ((y2*x3) - (x2*y3))/dt
	y = ((x1*y3) - (y1*x3))/dt
	return (int(x),int(y))

# Returns the Euclidian norm distance between two points in 2D
def getDistance(p1,p2):
	return pow(pow((p1[0]-p2[0]),2) + pow((p1[1]-p2[1]),2),0.5)

# Removes border noise, checks if cell is a blank one and returns the cleaned cell image
def removeNoise(img):
	
	X,Y = np.shape(img)
	imgOriginal = img.copy()
	point = (0,0)
	maxPoints = 0
	
	for i in range(X):
		for j in range(Y):
			img,p,l = customFloodFill(img,(i,j),100,1)
			if p > maxPoints and l > 5:
				maxPoints = p
				point = (i,j)
	
	img = imgOriginal
	img,p,dummy = customFloodFill(img,point,100,1)
	# Blank cell
	percentageHighlight = ((p/(X*Y))*100)
	#print(percentageHighlight)
	#if percentageHighlight < 5:
	#	print('Blank\n')
	#	return img,False
		
	for i in range(X):
		for j in range(Y):
			if img[i][j] == 100:
				img[i][j] = 255
			else: img[i][j] = 0
			
	whiteCenterCount = 0
	X,Y = int(X/2),int(Y/2)

	for i in range(X-5,X+5):
		for j in range(Y-5,Y+5):
			if img[i][j] == 255:
				whiteCenterCount = whiteCenterCount + 1
	
	#showImage(img)
	#showImage(img[X-3:X+3,Y-3:Y+3])
	if whiteCenterCount < 2:
		return img,False
	return img,True

# Calculates scores using template matching	
def getMatchingScore(img,digit):
	score = (cv2.matchTemplate(img,cv2.imread('Templates/' + 'T'+str(digit) + '.jpg',0),cv2.TM_SQDIFF)/2000)
	return score

# Gets the best prediction of the digit in a cell using template matching
def getBestMatch(img):
	best = getMatchingScore(img,1)
	p = 1
	for i in range(2,10):
		k = getMatchingScore(img,i)
		if k < best:
			best = k
			p = i		
	return p
	
# Centers the image in cell using mean displacement
def centerDigit(img):
	xMean,yMean,count = 0,0,0
	(x,y) = np.shape(img)
	for i in range(x):
		for j in range(y):
			if img[i][j] == 255:
				xMean,yMean,count = (xMean+i),(yMean+j),(count+1)
	if count == 0:
		return img
		
	xMean,yMean = (xMean / count),(yMean / count)
	xDisp,yDisp = (xMean - (x/2)),(yMean - (y/2))
	
	newImg = np.zeros((x,y),np.uint8)
	for i in range(x):
		for j in range(y):
			if img[i][j] == 255:
				newImg[i-xDisp][j-yDisp] = 255
	return newImg
	
# Given the cropped out digit, places it on a black background for matching with templates
def getTestImage(img,size):	
	size = int(size)
	(x,y) = np.shape(img)
	left,right,bottom,top = x,0,y,0
	count = 0
	for i in range(x):
		for j in range(y):
			if img[i][j] == 255:
				left = min(left,i)
				right = max(right,i)
				top = max(top,j)
				bottom = min(bottom,j)

				count = count + 1
	if count == 0:
		return img
	
	img = img[left:right,bottom:top]
	cv2.imwrite('template.jpg',img)
	return img
	
# Divides the grid into 9x9 = 81 cells and does OCR on each after processing it
def displayCells(img,len):

	grid = np.zeros((9,9),np.uint8)
	len = int(len / 9)
	k = 3
	for i in range(9):
		for j in range(9):
			roi = img[ (i*len)+k:((i+1)*len)-k , (j*len)+k:((j+1)*len)-k ]
			roi,isDigit = removeNoise(roi)
			if isDigit:
				bg = np.zeros((60,60),np.uint8)
				x,y = np.shape(roi)
				bg[(30-math.ceil((x/2))):(30+math.floor((x/2))) ,  (30-math.ceil((y/2))):(30+math.floor((y/2))) ] = roi
				bg = centerDigit(bg.copy())
				fg = getTestImage(bg.copy(),20)
				
				w = min(30,np.shape(fg)[0])
				h = min(30,np.shape(fg)[1])
				fg = fg[0:w,0:h]
				
				bg = np.zeros((30,30),np.uint8)
				(fgX,fgY) = np.shape(fg)
				
				if fgX == 60:
					grid[i][j] = 0
					continue
					
				bg[15-math.ceil(fgX/2):15+math.floor(fgX/2) , 15-math.ceil(fgY/2):15+math.floor(fgY/2)] = fg
				
				prediction = getBestMatch(bg)
				grid[i][j] = prediction
				print(' ' + str(prediction) + ' ')
			else: 
				print(' 0 ')
				grid[i][j] = 0
	return grid
	
# Displays an image
def showImage(img,caption='image'):
	cv2.imshow(caption,img)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

# Matches a template of cross to detect inner grid lines and then removes them via flood filling
def removeInnerGridLines(img):
	template = cv2.imread('cross_template.jpg',0)
	(tx,ty) = np.shape(template)
	res = cv2.matchTemplate(img,template,cv2.TM_SQDIFF_NORMED)
	threshold = 0.1
	loc = np.where( res <= threshold)
	for pt in zip(*loc[::-1]):
		x = pt[0]
		y = pt[1]
		img,area,dummy = customFloodFill(img,(x + int(tx/2),y + int(ty/2)),0,0)
	return img
	
# Reads in image of sudoku and does processing
def performAnalysis(filename):
	
	# Read in an image in greyscale
	img = cv2.imread(filename,0)
	imgX,imgY = np.shape(img)
	imgOriginal = img
	
	# Add gaussian blurring
	img = cv2.GaussianBlur(img,(11,11),0)
	showImage(img)

	# Do adaptive thresholding
	img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY_INV,5,2)
	showImage(img)

	# Dilate using cross-type kernel
	kernel = np.ones((3,3),np.uint8)
	kernel[0,0] = 0
	kernel[0,2] = 0
	kernel[2,0] = 0
	kernel[2,2] = 0
	img = cv2.dilate(img, kernel, iterations=1)
	showImage(img)

	imgOriginal = img.copy()

	# Extract outer grid of sudoku
	img,point = extractOuterGrid(img)
	#showImage(img)

	# Reverse the dilation by eroding
	img = cv2.erode(img,kernel,iterations=1)
	#showImage(img)
	
	'''
	# Get corner points by complicated Hough transform algo
	
	# Get lines and then get the extreme left,right,top,bottom lines
	lines = cv2.HoughLines(img,1,np.pi/180,int(min(imgX,imgY)/2))
	leftExtreme,rightExtreme,topExtreme,bottomExtreme = getExtremeLines(lines,imgX,imgY)

	# Get corner points of the outer sudoku grid
	topLeft = getIntersectionPoint(leftExtreme,topExtreme,imgX,imgY)
	topRight = getIntersectionPoint(rightExtreme,topExtreme,imgX,imgY)
	bottomLeft = getIntersectionPoint(leftExtreme,bottomExtreme,imgX,imgY)
	bottomRight = getIntersectionPoint(rightExtreme,bottomExtreme,imgX,imgY)
	'''

	# Get corner points by basic algo
	l,r,t,b = 1000,0,10000,0
	for i in range(imgX):
		for j in range(imgY):
			if(img[i][j] == 255):
				l = min(l,j)
				r = max(r,j)
				t = min(t,i)
				b = max(t,i)
	topLeft = (l,t)
	topRight = (r,t)
	bottomLeft = (l,b)
	bottomRight = (r,b)
	
	'''			
	# Draw corner points on image
	cv2.circle(img,topLeft,5,(255,255,255),-1)		
	cv2.circle(img,topRight,5,(255,255,255),-1)		
	cv2.circle(img,bottomLeft,5,(255,255,255),-1)		
	cv2.circle(img,bottomRight,5,(255,255,255),-1)		
	#showImage(img)
	'''
	
	# Change perspective
	maxLength = int(max(getDistance(topLeft,topRight),getDistance(topLeft,bottomLeft),getDistance(topRight,bottomRight),getDistance(bottomLeft,bottomRight)))
	src = np.array([topLeft,topRight,bottomLeft,bottomRight])
	dst = np.array([[0,0],[maxLength-1,0],[0,maxLength-1],[maxLength-1,maxLength-1]])
	h,status = cv2.findHomography(src,dst)
	
	img,area,dummy = customFloodFill(imgOriginal,[point[0],point[1]],0,0)
	#showImage(img)
	img = cv2.warpPerspective(img,h,(maxLength,maxLength))
	#img = cv2.erode(img,kernel,iterations=1)
	#showImage(img)

	img = removeInnerGridLines(img)
	#showImage(img)
	sudoku = displayCells(img,maxLength)			
				
	cv2.imshow('imlage',img)
	cv2.waitKey(0)
	cv2.destroyAllWindows()
	
	return sudoku

from solveSudoku import solveSudoku	
sudoku = performAnalysis('sudoku.jpg')
print(sudoku)
solvedSudoku,flag = solveSudoku(sudoku)
print(solvedSudoku)

