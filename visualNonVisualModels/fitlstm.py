
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import Embedding
from keras.layers import LSTM

from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score

#data= # a dict file max_len=i

execfile("preprocessdata.py")

X_train=np.asarray(question_tags)
Y_train=np.asarray(labels)

model = Sequential()
model.add(Embedding(vocab_size, vocab_size))
model.add(LSTM(output_dim=100, activation='sigmoid', inner_activation='hard_sigmoid'))
#model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='rmsprop')
'''
for epoch in range(1,10):
	for item,label in zip(X_train[1:100000], Y_train[1:100000]):
		model.fit(np.asarray([item]), np.asarray([[label]]), batch_size=1, nb_epoch=1)
'''
model.load_weights('lstmgenspecific0217.h5')

binpred=[]
for item in X_train[100001:131464]:
	pred=model.predict_proba(np.asarray([item]), batch_size=1)
	binpred.append(pred[0][0]>0.07)
	
binpred=np.asarray(binpred)

bintest=Y_train[100001:131464]>0.5

#a=accuracy_score(bintest,binpred)
#print "Accuracy : "+str(a)

#========== FOR CLASS Generic ====================
print "For Class Generic (LSTM)"
#Our Scores
p=precision_score(bintest,binpred)
print "precision: "+ str(p)
r1=recall_score(bintest,binpred)
print "recall: "+ str(r1)

#========== FOR CLASS Specific ===================
bintest_n=Y_train[100001:131464]<0.5
binpred_n=np.invert(binpred)

print "For Class Specific (LSTM)"
p=precision_score(bintest_n,binpred_n)
print "precision: "+ str(p)
r2=recall_score(bintest_n,binpred_n)
print "recall: "+ str(r2)

print "Normalized Accuracy : " + str((r1+r2)/2)




'''
print "-----TRAIN ACCURACY : ----------"

binpredtrain=[]
for item in X_train[1:100000]:
	pred=model.predict_proba(np.asarray([item]), batch_size=1)
	binpredtrain.append(pred[0][0]>0.07)
	
binpredtrain=np.asarray(binpredtrain)

bintrain=Y_train[1:100000]>0.5

a=accuracy_score(binpredtrain,bintrain)
print "Accuracy : "+str(a)

#========== FOR CLASS GENERAL ====================
print "For Class General (our Algorithm)"
#Our Scores
p=precision_score(binpredtrain,bintrain)
print "precision: "+ str(p)
r=recall_score(binpredtrain,bintrain)
print "recall: "+ str(r)
'''
