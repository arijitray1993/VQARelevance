require 'nn'
repl=require 'trepl'
---Load Data ---

scores=torch.load("scores_raw_allapplidata.t7")
scores2=torch.load("scores_raw_allapplidata2.t7")

dataset=torch.load("dataset_small.t7");
dataset2=torch.load("dataset_small_lost_correspondence.t7")

nim1=scores:size(1)
nim2=scores2:size(1)
nim=nim1+nim2
applicable=torch.DoubleTensor(nim);

for i=1,nim1 do
	local a;
	if dataset['applicable'][i][1]+dataset['applicable'][i][2]==0 then
		a=0.5;
	else
		a=dataset['applicable'][i][1]/(dataset['applicable'][i][1]+dataset['applicable'][i][2]);
	end
	applicable[i]=a;
end

for i=1,nim2 do
	local a;
	if dataset2['applicable'][i][1]+dataset2['applicable'][i][2]==0 then
		a=0.5;
	else
		a=dataset2['applicable'][i][1]/(dataset2['applicable'][i][1]+dataset2['applicable'][i][2]);
	end
	applicable[nim1+i]=a;
end

scoresfinal=torch.cat(scores,scores2,1)

scores=scoresfinal

trainlabels=torch.DoubleTensor(nim,1)

for i=1,nim do
	if applicable[i]>=0.5 then
		trainlabels[i]=1
	else
		trainlabels[i]=0
	end
end

traindata=scores

traindata=traindata[{{1,10792},{}}]
trainlabels=trainlabels[{{1,10792}}]

--repl()
--- Architecture ---

net=nn.Sequential()
net:add(nn.Linear(1000,100))
net:add(nn.ReLU())
net:add(nn.Linear(100,1))
net:add(nn.Sigmoid())

criterion=nn.BCECriterion()
normalizedaccuracy={0,0,0,0,0,0,0,0,0,0,0}
normalizedaccuracytrain={0,0,0,0,0,0,0,0,0,0,0}

for tcount=1,5 do
--	train_count=1

	ta=7195
----------- Uncomment below section to TRAIN PERCEPTRON -----------------------------------------
		for i=1,2000 do
			--repl()
			pred=net:forward(traindata[{{1,ta},{}}])
			--pred=pred:gt(0.5)
			--repl()
			err=criterion:forward(pred, trainlabels[{{1,ta},{}}])
--			print(err)
			net:zeroGradParameters()
	
			criterion_gradient=criterion:backward(pred, trainlabels[{{1,ta},{}}])
			net:backward(traindata[{{1,ta},{}}], criterion_gradient)

			net:updateParameters(0.01)
	
		end
--	torch.save("savedmodels/rawscoremodel" .. tcount .. ".net", net)
--------------------------------------------------------------------------------------------------

--	net=torch.load("savedmodels/rawscoremodel1.net")
	train_pred=net:forward(traindata[{{1,ta},{}}])
	testlabels=trainlabels[{{1,ta},{}}]
	count=0
	zcount=0
	onecount=0
	foundzcount=0
	foundonecount=0
	foundonewrong=0
	foundzwrong=0
	--repl()

	-- TRAINING ACCURACY -- 
	thresh=0.2
	for i=1,ta do
		if train_pred[i][1]>=thresh then
			if testlabels[i][1]==1 then
				count=count+1
			end
		else
			if testlabels[i][1]==0 then
				count=count+1
			end
		end

		if testlabels[i][1]==0 then
			zcount=zcount+1		
			if train_pred[i][1]<thresh then
				foundzcount=foundzcount+1
			else
				foundonewrong=foundonewrong+1
			end
		else
			onecount=onecount+1
			if train_pred[i][1]>=thresh then
				foundonecount=foundonecount+1
			else
				foundzwrong=foundzwrong+1
			end
		end 
	end

	recall_irr=foundzcount/zcount
	recall_r=foundonecount/onecount

	normalizedaccuracytrain[tcount]=normalizedaccuracytrain[tcount]+(recall_irr+recall_r)/2


	-- TESTING ACCURACY --
	train_pred=net:forward(traindata[{{ta+1,(#traindata)[1]},{}}])
	testlabels=trainlabels[{{ta+1,(#traindata)[1]},{}}]
	
	count=0
	zcount=0
	onecount=0
	foundzcount=0
	foundonecount=0
	foundonewrong=0
	foundzwrong=0

	thresh=torch.mean(train_pred)
	for i=1,((#traindata)[1]-ta-1) do
		if train_pred[i][1]>=thresh then
			if testlabels[i][1]==1 then
				count=count+1
			end
		else
			if testlabels[i][1]==0 then
				count=count+1
			end
		end

		if testlabels[i][1]==0 then
			zcount=zcount+1		
			if train_pred[i][1]<thresh then
				foundzcount=foundzcount+1
			else
				foundonewrong=foundonewrong+1
			end
		else
			onecount=onecount+1
			if train_pred[i][1]>=thresh then
				foundonecount=foundonecount+1
			else
				foundzwrong=foundzwrong+1
			end
		end 
	end

	accuracy=count/((#traindata)[1]-7195)
	-- FoR CLASS IRRELEVANT (CLASS 0) --
	precision_irr=foundzcount/(foundzcount+foundzwrong)
	recall_irr=foundzcount/zcount

	-- FOR CLASS RELEVANT (CLASS 1) --
	precision_r=foundonecount/(foundonecount+foundonewrong)
	recall_r=foundonecount/onecount

	print("Prec, Recall for Irrelevant:")
	print(precision_irr)
	print(recall_irr)

	print("Prec, Recall for Relevant:")
	print(precision_r)
	print(recall_r)

	print("")

	normalizedaccuracy[tcount]=normalizedaccuracy[tcount]+(recall_irr+recall_r)/2
--	train_count=train_count+1
	
end

print(torch.sum(normalizedaccuracy)/5)
repl()

require '../utils/metric'
print(metric.precision(-(entropy-expected_entropy),applicable:gt(0.5):double()))
print(metric.precision(-(entropy_model_averaging-entropy),applicable:gt(0.5):double()))
print(metric.precision(-(entropy),applicable:gt(0.5):double()))


