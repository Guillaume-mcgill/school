# imports 
import numpy as np


# define the relative file directories
inputFile   = "../training_data/labels.txt"
outputFile  = "parameters.txt"

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
	

# to learn "patterns" (the likelihoods in each set and the prior probabilities) 
# the data is returned to write_Bayes_outputs to be written to the parameters.txt file
def fit_Naive_Bayes():

	# open the training file in reading mode 
	with open(inputFile, 'r') as trainingFile:
		
		# We assume y=1 for alpha helices and y=0 for beta sheets 
		num_positive = 0				# keeps track of the size of the positive training set 
		num_negative = 0 				# keeps track of the size of the negative training set 
		
		iteration = 0
		while True:
			# Keep track of the progression of the algorithm
			print("PROGRESS: ", (iteration/5326)*100,"% done")      
			iteration +=1
			
			# read the inputs line by line 
			name = trainingFile.readline()
			sequence = trainingFile.readline()
			label = trainingFile.readline()
			if not label: break
			
			aminoSequence = list(sequence)				# For each input protein sequence, break into a list of single amino acids
			window_lateral = int(((windowSize)-1)/2)		# lenght of the lateral side of the moving window
			
			for i in range(len(sequence)):
				
				# counts the number of alpha helices and beta sheets
				if label[i] == 'H':
					num_positive += 1
				else:
					num_negative += 1		
				
				# now look at the amino acids in the window to update the likelihoods
				for j in range(windowSize):
					 
					# position of the amino acid in the sequence based on the window index
					position = i - window_lateral + j
					
					#POSITIVE TRAINING SET
					if label[i] == 'H':
						if ((position<0) or (position>len(sequence)-2)):	   	        # if the window is outside of the amino acid region. -1 to adjust the sequence lenght and -1 for the condition to hold true (total -2 in the condition)
							positiveSet.get('-')[j] += 1					# add a count to the proper position in the positive set dictionary					
						else:
							positiveSet.get(aminoSequence[position])[j] += 1
								
					#NEGATIVE TRAINING SET 
					else: 
						if ((position<0) or (position>len(sequence)-2)):	   	           # if the window is outside of the amino acid region. -1 to adjust the sequence lenght and -1 for the condition to hold true (total -2 in the condition)
							negativeSet.get('-')[j] += 1							   
						else:
							negativeSet.get(aminoSequence[position])[j] += 1
		
		
		# divide by the number of occurences in each set to get the probability					
		for key, item in positiveSet.items():
			positiveSet[key] = item/num_positive			# calculates p(x1,x2,x3,...,xn | y=1) 
					
		for key, item in negativeSet.items():
			negativeSet[key] = item/num_negative			# calculates p(x1,x2,x3,...,xn | y=0)
						
		positive_prior = num_positive/(num_positive+num_negative)		# p(y=1)
		negative_prior = num_negative/(num_positive+num_negative)		# p(y=0)
			
	return positiveSet, negativeSet, positive_prior, negative_prior

	    
def write_Bayes_outputs(positive_set, negative_set, positive_prior, negative_prior):
	with open(outputFile, 'w') as f:
		f.write(str(positive_prior)+"\n")			# write prior prob of the positive set first
		f.write(str(negative_prior)+"\n")			# write prior prob of the negative set second
		for key, item in positive_set.items():
			f.write(str(key)+"\n")
			for index in item:				# then write down the keys and values of the positive set dictionary
				f.write(str(index)+"\n")
				
		for key, item in negative_set.items():
			f.write(str(key)+"\n")
			for index in item:				# then write down the keys and values of the negative set dictionary
				f.write(str(index)+"\n")		
			

def main():
	
	# runs the appropriate methods in correct order	
	positive_set, negative_set, positive_prior, negative_prior = fit_Naive_Bayes()
	write_Bayes_outputs(positive_set, negative_set, positive_prior, negative_prior)

	
# run the main method
if __name__ == '__main__':
	main()
