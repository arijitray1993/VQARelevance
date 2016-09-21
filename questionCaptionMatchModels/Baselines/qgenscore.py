import json
import numpy as np

from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

fileName=open("scores.json")

data=json.load(fileName)

tcount=0

unique_qi_applicable=dict()	
for entry in data.items():
	if entry[1]['label']!='':
		key=entry[1]['image']+'__'+entry[1]['question']
		if key in unique_qi_applicable:
			unique_qi_applicable[key]['lcounts'].append(int(entry[1]['label']))
			if len(unique_qi_applicable[key]['lcounts'])>0:
				if len(unique_qi_applicable[key]['lcounts'])==2:
					tcount+=1
					#print tcount
				if np.mean(unique_qi_applicable[key]['lcounts'])>=1.5:
					unique_qi_applicable[key]['label']=2
				else:
					unique_qi_applicable[key]['label']=1
		else:
			unique_qi_applicable[key]=dict()
			unique_qi_applicable[key]['lcounts']=[]
			unique_qi_applicable[key]['lcounts'].append(int(entry[1]['label']))
			unique_qi_applicable[key]['question']=entry[1]['question']
			unique_qi_applicable[key]['image']=entry[1]['image']
			unique_qi_applicable[key]['score']=entry[1]['score']
			if len(unique_qi_applicable[key]['lcounts'])>0:
				if np.mean(unique_qi_applicable[key]['lcounts'])>=1.5:
					unique_qi_applicable[key]['label']=2
				else:
					unique_qi_applicable[key]['label']=1

data=unique_qi_applicable

scores=[]
labels=[]

for key in data:
	if data[key]['label']!='':
		scores.append(float(data[key]['score']))
		labels.append(float(data[key]['label']))



scores=np.asarray(scores)
labels=np.asarray(labels)

scoretrain=scores[0:7196]
labelstrain=labels[0:7196]

scoretest=scores[7196:len(scores)]
labelstest=labels[7196:len(scores)]

scores=scoretrain
labels=labelstrain

# in the labels, 1=Relevant, 2=Irrelevant
labels=2-labels # Convert the labels to 0-Irrelevant, 1-Relevant
labelstest=2-labelstest

avg_score=np.mean(scores)

thres_scores=np.asarray([-10,-9,-8,-7,-6,-5.9,-5.7,-5.5,-5.3,-5.1,-5,-4.8,-4.6,-4.3,-4.25,-3,-2,-1,0]) # a bunch of thresholds to try

prec_scores_relevant=[]
prec_scores_irrelevant=[]

recall_scores_relevant=[]
recall_scores_irrelevant=[]

for thres in thres_scores: # let's just threshold with the average for now
	#PREC, RECALL for class "Relevant"	
	gt=labels>0.5
	pred=scores>thres #it seems that the scores are directly related to relevancy
	prec_scores_relevant.append(precision_score(gt,pred))
	recall_scores_relevant.append(recall_score(gt,pred))
	
	#PREC, RECALL for class "Irrelevant"
	gtir=labels<0.5
	predir=scores<thres 
	prec_scores_irrelevant.append(precision_score(gtir,predir))
	recall_scores_irrelevant.append(recall_score(gtir,predir))

recall_scores_relevant=np.asarray(recall_scores_relevant)
recall_scores_irrelevant=np.asarray(recall_scores_irrelevant)

norm_acc=(recall_scores_relevant + recall_scores_irrelevant)/2


print prec_scores_relevant
print recall_scores_relevant
print '\n'
print prec_scores_irrelevant
print recall_scores_irrelevant
print '\n'
print norm_acc


import numpy
import matplotlib.pyplot as plt
plt.plot(thres_scores,norm_acc)
#plt.axis([-10,-1,0,1])
plt.show()

thres_best=thres_scores[np.argmax(norm_acc)]

# test on test set ...
prec_scores_relevant=[]
prec_scores_irrelevant=[]

recall_scores_relevant=[]
recall_scores_irrelevant=[]

gt=labelstest>0.5
pred=scoretest>thres_best #the scores are directly related to relevancy
prec_scores_relevant.append(precision_score(gt,pred))
recall_scores_relevant.append(recall_score(gt,pred))

#PREC, RECALL for class "Irrelevant"
gtir=labelstest<0.5
predir=scoretest<thres_best 
prec_scores_irrelevant.append(precision_score(gtir,predir))
recall_scores_irrelevant.append(recall_score(gtir,predir))

recall_scores_relevant=np.asarray(recall_scores_relevant)
recall_scores_irrelevant=np.asarray(recall_scores_irrelevant)

norm_acc=(recall_scores_relevant + recall_scores_irrelevant)/2

print "Prec, Recall for True Premise"
print prec_scores_relevant
print recall_scores_relevant
print 'Prec Recall for False Premise'
print prec_scores_irrelevant
print recall_scores_irrelevant
print 'Norm Acc'
print norm_acc

print "best thresh based on train data: "+str(thres_best)


'''
plt.plot(prec_scores_irrelevant,recall_scores_irrelevant)
plt.show()
'''
