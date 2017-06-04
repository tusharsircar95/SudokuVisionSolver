import math
import cv2
import numpy as np

def isPlacementPossible(grid,x,y,digit):
	inRow,inColumn,inBox = False,False,False
	for i in range(9):
		if i != y :
			if grid[x][i] == digit:
				inRow = True
				break
				
	if inRow == True:
		return False
		
	for i in range(9):
		if i != x :
			if grid[i][y] == digit:
				inColumn = True
				break
	if inColumn == True:
		return False
		
	p1 = 3 * math.floor(x/3)
	p2 = 3 * math.floor(y/3)
	
	for i in range(3):
		for j in range(3):
			if (p1+i) != x or (p2+j) != y:
				if grid[p1+i][p2+j] == digit:
					inBox = True
					break
	if inBox == True:
		return False
		
	return True
	
def getNextCell(x,y):
	if y < 8:
		return x,(y+1)
	return (x+1),0
	
def solveSudokuAux(grid,x,y):

	'''
		Returns true if a solution if found. 'grid' contains the solution
		
	'''
	# RECURSION BASE CASES
	# If cell is already filled
	if grid[x][y] != 0:
		# If sudoku is complete
		if x == 8 and y == 8:
			return True
		# If incomplete, move on to next cell in order
		x,y = getNextCell(x,y)
		return solveSudokuAux(grid,x,y)
		
	# If cell is blank
	for i in range(1,10):
		if isPlacementPossible(grid,x,y,i):
			grid[x][y] = i
			if x == 8 and y == 8:
				return True
			xNext,yNext = getNextCell(x,y)
			doesSolutionExist = solveSudokuAux(grid,xNext,yNext)
			if doesSolutionExist:
				return True
	grid[x][y] = 0
	return False
			
def solveSudoku(grid):
	#passesSanityChecks = doSanityCheck(grid)
	#if not passesSanityChecks:
	#	return grid,False
	doesSolutionExist = solveSudokuAux(grid,0,0)
	return grid,doesSolutionExist
	
	
#grid = np.zeros((9,9),np.uint8)
'''
for i in range(9):
	for j in range(9):
		grid[i][j] = input('Enter value for (' + str(i) + ',' + str(j) + ')')
'''
	
#solveSudoku(grid)