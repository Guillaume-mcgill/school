# imports 
import numpy as np

### DEFINE CONSTANTS ###

# For the logistic regression, let's convert the amino acids in vectors
# Vectors of length 21 (20 aa + 1 buffer for the end)

aa_encoding={'A' : [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
             'C' : [0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
             'D' : [0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
             'E' : [0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
             'F' : [0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
             'G' : [0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
             'H' : [0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
             'I' : [0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0],
             'K' : [0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0],
             'L' : [0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0],
             'M' : [0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0], 
             'N' : [0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0], 
             'P' : [0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0], 
             'Q' : [0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0], 
             'R' : [0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0], 
             'S' : [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0],
             'T' : [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0], 
             'V' : [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0], 
             'W' : [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0], 
             'Y' : [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0], 
             '-' : [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1]}		# this line here is the buffer


""" additional script file to train the model and store the results in the parameters.txt file"""

# define file relative directories
inputFile   = "../training_data/labels.txt"
outputFile  = "parameters.txt"


# algorithm that trains and gives the best weights and bias
# our logistic regression uses a Stochastic gradient descent, only using one training example at a time
# for our purposes, the output will be binary (y=1: alpha helix, y=0: beta-sheet)
class Logistic_Regression:
	def __init__(self, windowSize, stepSize):
		seed = 410 							        # Beautiful seed value :D
		np.random.seed(seed)
		self.stepSize = stepSize 
		self.w = np.random.rand(windowSize, 21)	       				# initializes the weight matrix: 20 amino acid & 1 buffer (in order for the window to work) 
		self.b = np.random.rand(1)[0]						# initialize a single value for b as the bias 
    
	def updateWeight(self, x, y):
		#print("\n x, y values: \n", x, y)
		alpha = (2*y-1)*(np.sum(x*self.w) + self.b)				# alpha allows to predict the binary output (alpha>0 ---> predict y=1; alpha<0 ---> predict y=0)
		
		gradient_w = np.divide((2*y-1), 1 + np.exp(alpha))*x			# w gradient 
		gradient_b = np.divide(2*y-1, 1 + np.exp(alpha))			# b gradient
		
		self.w += self.stepSize*gradient_w
		self.b += self.stepSize*gradient_b
		
    
    
# function to read the training data
def readAndFit(inputFile, windowSize):	     
	
	log_reg = Logistic_Regression(stepSize=0.01, windowSize=windowSize) 	# instantiate a logistic regression model
	
	# open the training file in reading mode 
	with open(inputFile, 'r') as trainingFile:
		iteration = 0
		while True:
			print("PROGRESS: ", (iteration/5326)*100,"% done")      # To show the progress of the algorithm in the total sequence lenght
			iteration +=1
			
			name = trainingFile.readline()
			sequence = trainingFile.readline()
			label = trainingFile.readline()
			if not label: break
			
			aminoSequence = list(sequence)				# For each input protein sequence, break into a list of single amino acids
			window_lateral = int(((windowSize)-1)/2)		        # Moving window across the sequence data
			
			
			for i in range(len(sequence)):
				x = []
				for j in range(windowSize):
					position = i - window_lateral + j				           # to get the position in the bigger array depending of where we are in the window
					if ((position<0) or (position>len(sequence)-2)):	   	           # if the window is outside of the amino acid region. -1 to adjust the sequence lenght and -1 for the condition to hold true (total -2 in the condition)
						x.append(aa_encoding.get('-'))
					else:
						x.append(aa_encoding.get(aminoSequence[position]))
						
				y = 1 if label[i]=='H' else 0
				log_reg.updateWeight(x=np.array(x), y=y)				           # update the values of the weights	
			
	return log_reg.w, log_reg.b


# write the training outputs in the parameters.txt file so that it can be used for prediction
def writeOutput(weights, bias, outputFile):
	with open(outputFile, 'w') as outputFile:
		outputFile.write(str(bias)+"\n")
		print("SHAPE OF THE WEIGHT MATRIX: ", weights.shape)
		for row in weights:
			for weight in row: 
				outputFile.write(str(weight)+"\n")		
	    
			
def main():
	weights, bias = readAndFit(inputFile=inputFile, windowSize=7)
	writeOutput(weights, bias, outputFile)

	
# run the main method
if __name__ == '__main__':
	main()
