# Python 3 script to edit for this project. 
# Note 1: Do not change the name of this file
# Note 2: Do not change the location of this file within the BIEN410_FinalProject package
# Note 3: This file can only read in "../input_file/input_file.txt" and "parameters.txt" as input
# Note 4: This file should write output to "../output_file/outfile.txt"
# Note 5: See example of a working SSPred.py file in ../scr_example folder

import numpy as np

# ACCESSIBLE READ/WRITE FILES:

inputFile       = "../input_file/infile.txt"
parameters      = "parameters.txt"
predictionFile	= "../output_file/outfile.txt"

# For the Naive Bayes approach:
# y=0:b-sheet, y=1:a-helix

windowSize=19
	
# dictionary to keep track of the conditional probabilities in the positive set at each position in the scanning window
positiveSet={'A' : np.array([0]*windowSize),
             'C' : np.array([0]*windowSize),
             'D' : np.array([0]*windowSize),
             'E' : np.array([0]*windowSize),
             'F' : np.array([0]*windowSize),
             'G' : np.array([0]*windowSize),
             'H' : np.array([0]*windowSize),
             'I' : np.array([0]*windowSize),
             'K' : np.array([0]*windowSize),
             'L' : np.array([0]*windowSize),
             'M' : np.array([0]*windowSize), 
             'N' : np.array([0]*windowSize), 
             'P' : np.array([0]*windowSize), 
             'Q' : np.array([0]*windowSize), 
             'R' : np.array([0]*windowSize), 
             'S' : np.array([0]*windowSize),
             'T' : np.array([0]*windowSize), 
             'V' : np.array([0]*windowSize), 
             'W' : np.array([0]*windowSize), 
             'Y' : np.array([0]*windowSize), 
             '-' : np.array([0]*windowSize)} # this line here is the buffer

# dictionary to keep track of the conditional probabilities in the negative set at each position in the scanning window
negativeSet={'A' : np.array([0]*windowSize),
             'C' : np.array([0]*windowSize),
             'D' : np.array([0]*windowSize),
             'E' : np.array([0]*windowSize),
             'F' : np.array([0]*windowSize),
             'G' : np.array([0]*windowSize),
             'H' : np.array([0]*windowSize),
             'I' : np.array([0]*windowSize),
             'K' : np.array([0]*windowSize),
             'L' : np.array([0]*windowSize),
             'M' : np.array([0]*windowSize), 
             'N' : np.array([0]*windowSize), 
             'P' : np.array([0]*windowSize), 
             'Q' : np.array([0]*windowSize), 
             'R' : np.array([0]*windowSize), 
             'S' : np.array([0]*windowSize),
             'T' : np.array([0]*windowSize), 
             'V' : np.array([0]*windowSize), 
             'W' : np.array([0]*windowSize), 
             'Y' : np.array([0]*windowSize), 
             '-' : np.array([0]*windowSize)} # this line here is the buffer


def readInput(inputFile):
	# Read the input data in a FASTA format and parse it into into a python dictionnary
	# dictionary in format {name (str):sequence (str)}     
	
	inputData = {}
	with open(inputFile, 'r') as f:
		while True:
			name = f.readline()
			sequence = f.readline()		
			if not sequence: break
			inputData.update({name.rstrip():sequence.rstrip()})		# convert to dictionary: rstrip removes all whitespaces from the end of a line
	return inputData


def read_NaiveBayes_Weights(parameters):
	
	with open(parameters, 'r') as f: 
		positive_prior = float(f.readline().rstrip())
		negative_prior = float(f.readline().rstrip())	
		
		positiveSet={}
		negativeSet={}
		for i in range(21):
			key = f.readline().rstrip()
			values = [0]*windowSize
			for j in range(windowSize):				# read all the likelihoods for each amino acid in the positive set 
				values[j] = float(f.readline().rstrip())
			positiveSet.update([(key,values)])			# update the positive dictionary with the appropriate values
				
		for i in range(21):
			key = f.readline().rstrip()
			values = [0]*windowSize
			for j in range(windowSize):				# read all the likelihoods for each amino acid in the negative set 	
				values[j] = float(f.readline().rstrip())
			negativeSet.update([(key,values)])			# update the negative dictionary with the appropriate values
	
	return positive_prior, negative_prior, positiveSet, negativeSet

def naive_Bayes_Predict(inputData, parameters):
	
	# fetch the training parameters used for prediction 
	positive_prior, negative_prior, positiveSet, negativeSet = read_NaiveBayes_Weights(parameters)

	predictions = {}
	iteration = 0 
	for sequenceName in inputData:
		
		# show the progress of the algorithm
		print("PROGRESS: ", (iteration/5326)*100,"% done")
		iteration +=1
		
		sequence = inputData[sequenceName]
		pred="" 
		
		window_lateral = int(((windowSize)-1)/2)
		aminoSequence = list(sequence)					# For each input protein sequence, break into a list of single amino acids

		for i in range(len(sequence)):		
			posterior_likelihood_prob = positive_prior/negative_prior 
			
			for j in range(windowSize):
				
				# position in the amino acid sequence relative to the window 
				position = i - window_lateral + j				                   # to get the position in the bigger array depending of where we are in the window
				
				#POSITIVE TRAINING SET
				if ((position<0) or (position>len(sequence)-2)):				   # if the window is outside of the amino acids region, then fetch the "-" position in the dictionary
					posterior_likelihood_prob *= positiveSet.get('-')[j]/negativeSet.get('-')[j]	        
				
				else:
					posterior_likelihood_prob *= positiveSet.get(aminoSequence[position])[j]/negativeSet.get(aminoSequence[position])[j]		   # if the index is not outside of the amino acid region, then update the array of the appropriate amino acid using the positive/negative dictionaries
							
			y = 'H' if posterior_likelihood_prob>1 else '-'
			pred += str(y)						# add the prediction for each amino acid		
		print("value of pred: ", pred)
		predictions.update({sequenceName:pred})
	return predictions	



def writeOutput(inputData,predictions,outputFile):
	
	# open the outputFile in writing mode 
	with open(outputFile, 'w') as file:
		for name in inputData:
			file.write(name+"\n")
			file.write(inputData[name]+"\n")
			file.write(predictions[name]+"\n")
	return


def main():
	
	# calls the appropriate methods in the correct order 
	inputData = readInput(inputFile)
	predictions = naive_Bayes_Predict(inputData, parameters)
	writeOutput(inputData, predictions, predictionFile)
	


# run the main method 
if __name__ == '__main__':
	main() 
