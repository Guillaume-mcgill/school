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

# DEFINE VARIABLES 

# dictionary of the amino acid preferences (used in the Naive Bayes approach)
aa_tendency = {'A':1, 'C':0, 'D':1, 'E':1, 'F':0, 'G':0, 'H':0, 'I':0, 'K':1, 'L':1, 'M':1, 'N':1, 'P':1, 'Q':1, 'R':1, 'S':0, 'T':0, 'V':0, 'W':0, 'Y':0}

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

def read_Logistic_Weights(parameters, windowSize):
	# each weight is on a new line, read it one by one and convert it to an array   
	
	weights = [[0]*21]*windowSize
	iteration = 0
	with open(parameters, 'r') as f:
		bias = f.readline()
		i=1
		while True:
			single_weight = f.readline()	
			if not single_weight: break
			print("iteration ",i)
			i+=1
			
			print(iteration//(windowSize))
			print(iteration%21)
			weights[iteration//21][iteration%21] = (float(single_weight.rstrip()))	   # convert to a weight matrix 
			iteration+=1
	return weights, bias

def read_NaiveBayes_Weights(parameters, windowSize):
	
	positive_set = [0]*windowSize		# will be used for the likelihood of each position to promote A-helix in the positive training set 
	negative_set = [0]*windowSize		# will be used for the likelihood of each position to promote A-helix in the negative training set 
	positive_prior = 0			# keeps track of the size of the positive training set 
	negative_prior = 0 			# keeps track of the size of the negative training set 	
	
	with open(parameters, 'r') as f: 
		positive_prior = float(f.readline().rstrip())
		negative_prior = float(f.readline().rstrip())
		for i in range(windowSize):
			print("value of i 1: ",i)
			positive_set[i] = float(f.readline().rstrip())
		for i in range(windowSize):
			print("value of i 2: ",i)
			negative_set[i] = float(f.readline().rstrip())
	
	return positive_set, negative_set, positive_prior, negative_prior

def naive_Bayes_Predict(inputData, parameters, windowSize):
	
	# fetch the training parameters used for prediction 
	positive_set, negative_set, positive_prior, negative_prior = read_NaiveBayes_Weights(parameters, windowSize)
	
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
				if ((position<0) or (position>len(sequence)-2)):				   # if the window is outside of the amino acids region, then x_n = 0 so predict accordingly for p(x_n|y=1) and p(x_n|y=0)
					posterior_likelihood_prob *= (1-positive_set[j])/(1-negative_set[j])	        
				
				elif (aa_tendency.get(sequence[position]) == 0): 	
					posterior_likelihood_prob *= (1-positive_set[j])/(1-negative_set[j])	   # if the amino acid doesn't favor an alpha helix, then x_n = 0 so predict accordingly
				else:
					posterior_likelihood_prob *= positive_set[j]/negative_set[j]		   # if the amino acid does FAVOR alpha helix, then x_n = 1, so predict accordingly 
							
			y = 'H' if posterior_likelihood_prob>1 else '-'
			pred += str(y)						# add the prediction for each amino acid		
		print("value of pred: ", pred)
		predictions.update({sequenceName:pred})
	return predictions	
	

def predict(inputData, weights, bias, windowSize):
		
	# fetch the parameters obtained from training here!
		
	predictions = {}
	iteration = 0 
	for sequenceName in inputData:
		print("PROGRESS: ", (iteration/5326)*100,"% done")
		iteration +=1
		sequence = inputData[sequenceName]
		pred="" 
		
		window_lateral = int(((windowSize)-1)/2)
		aminoSequence = list(sequence)						# For each input protein sequence, break into a list of single amino acids

		for i in range(len(sequence)):
			x = [0]*windowSize
			for j in range(windowSize):
				
				position = i - window_lateral + j				
				if ((position<0) or (position>len(sequence)-2)):	# if the window is outside of the amino acid region
					x[j] = aa_encoding.get('-')
				else:
					x[j] = aa_encoding.get(aminoSequence[position])
			
			yh = 1 if (np.sum(np.array(x)*np.array(weights)) + float(bias))>0 else 0			# wx+b >0 ---> predict y=1 (alpha helix) ; wx+b <0 ---> predict y=0 (beta sheet)
			struct = 'H' if yh==1 else '-'
			pred += struct										# add the prediction for each amino acid		
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
	
	windowSize = 9
	inputData = readInput(inputFile)
	
	# TO USE NAIVE BAYES, COMMENT THE TWO LINES BELOW
	weights, bias = read_Logistic_Weights(parameters, windowSize)
	predictions = predict(inputData, weights, bias, windowSize=windowSize)
	
	# TO USE NAIVE BAYES, UNCOMMENT THE TWO LINES BELOW
	#predictions = naive_Bayes_Predict(inputData, parameters, windowSize)
	writeOutput(inputData, predictions, predictionFile)
	


# run the main method 
if __name__ == '__main__':
	main()