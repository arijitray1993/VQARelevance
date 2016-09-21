require 'nn'
repl=require 'trepl'

cmd = torch.CmdLine();
cmd:text('Compute entropies and write entropies and applicable to csv');
cmd:text('Options')
cmd:option('-data','dataset_small.t7','Dataset for testing');
cmd:option('-score','scores_allapplidata_dropout.t7','Score for testing');
cmd:option('-output','entropies.csv','Output filename');
params=cmd:parse(arg);
--print(params)


scores=torch.load(params.score);
scores2=torch.load("scores_allapplidata_dropout2.t7")
dataset=torch.load(params.data);
dataset2=torch.load("dataset_small_lost_correspondence.t7")

nim1=scores:size(1)
nim2=scores2:size(1)
nim=scores:size(1) + scores2:size(1);
entropy=torch.DoubleTensor(nim);
entropy_model_averaging=torch.DoubleTensor(nim);
expected_entropy=torch.DoubleTensor(nim);
applicable=torch.DoubleTensor(nim);
for i=1,nim1 do
	if i%10000==0 then
		print(i)
	end
	p_a_qi_theta=scores[{i,{},{}}];
	p_a_qi=p_a_qi_theta:mean(1);
	entropy[i]=-torch.cmul(p_a_qi,torch.log(p_a_qi)):sum();
	entropy_model_averaging[i]=-torch.cmul(p_a_qi,torch.log(p_a_qi_theta):mean(1)):sum();
	expected_entropy[i]=-torch.cmul(p_a_qi_theta,torch.log(p_a_qi_theta)):mean(1):sum();
	
	local a;
	if dataset['applicable'][i][1]+dataset['applicable'][i][2]==0 then
		a=0.5;
	else
		a=dataset['applicable'][i][1]/(dataset['applicable'][i][1]+dataset['applicable'][i][2]);
	end
	applicable[i]=a;
end

for i=1,nim2 do
	if i%10000==0 then
		print(i)
	end
	p_a_qi_theta=scores2[{i,{},{}}];
	p_a_qi=p_a_qi_theta:mean(1);
	entropy[nim1+i]=-torch.cmul(p_a_qi,torch.log(p_a_qi)):sum();
	entropy_model_averaging[nim1+i]=-torch.cmul(p_a_qi,torch.log(p_a_qi_theta):mean(1)):sum();
	expected_entropy[nim1+i]=-torch.cmul(p_a_qi_theta,torch.log(p_a_qi_theta)):mean(1):sum();
	
	local a;
	if dataset2['applicable'][i][1]+dataset2['applicable'][i][2]==0 then
		a=0.5;
	else
		a=dataset2['applicable'][i][1]/(dataset2['applicable'][i][1]+dataset2['applicable'][i][2]);
	end
	applicable[nim1+i]=a;
end


nu,ind=torch.sort(entropy,true);
nu,entropy_rank=torch.sort(ind);
nu,ind=torch.sort(entropy_model_averaging,true);
nu,entropy_model_averaging_rank=torch.sort(ind);
nu,ind=torch.sort(entropy-expected_entropy,true);
nu,model_entropy_rank=torch.sort(ind);

--repl()

--torch.save('entropies.t7',{H_a=entropy,E_a_E_w_log_p_a_given_w=entropy_model_averaging,E_w_H_a_given_w=expected_entropy});
--f=torch.DiskFile(params.output,'w'); 
--for i=1,nim,10 do
--	f:writeString(string.format('%d,%f,%f,%f,%f, %f\n',i,entropy[i],entropy_model_averaging[i],entropy[i]-expected_entropy[i],expected_entropy[i]-1.2*entropy_model_averaging[i], applicable[i]));
--end
--f:close();

------------------------------------------------------------ TRAINING TESTING CODE ----------------------------------------------------------------------------------
---------------------------------------------------------------------------------------------------------------------------------------------------------------------


normalizedaccuracy={0,0,0,0,0,0,0,0,0,0,0}
normalizedaccuracytrain={0,0,0,0,0,0,0,0,0,0,0}
train_count=1

for tcount=1,5 do
	
	-- DEFINE TRAIN DATA AND LABELS
	traindata=torch.DoubleTensor(nim,1)  
	traindata[{{},{1}}]=entropy
	--concat=torch.cat(entropy,expected_entropy,2)
	--traindata=torch.cat(concat,entropy_model_averaging,2)
	trainlabels=torch.DoubleTensor(nim,1)
	for i=1,nim do
		if applicable[i]>=0.5 then
			trainlabels[i]=1
		else
			trainlabels[i]=0
		end
	end
	--repl()

	------ RANDOMISE DATA AND LABELS -------------------

--	randind=torch.randperm(nim)

--	randtraindata=torch.DoubleTensor(nim,1)
--	randtrainlabels=torch.DoubleTensor(nim,1)

--	for i=1,nim do
--		randtraindata[{{i},{}}]=traindata[{{randind[i]},{}}]
--		randtrainlabels[i]=trainlabels[randind[i]]
--	end

--	traindata=randtraindata
--	trainlabels=randtrainlabels

	----------------------------------------------------

	--just to make sure we have same train test split numbers for both actual models and baselines.
	traindata=traindata[{{1,10792}}]
	trainlabels=trainlabels[{{1,10792}}]
	
	----------- TRAIN AND TEST -------------------------

	net=nn.Sequential()
	net:add(nn.Linear(1,3))
	net:add(nn.ReLU())
	net:add(nn.Linear(3,1))
	net:add(nn.Sigmoid())

	criterion=nn.BCECriterion()
	
	-- TRAIN PERCEPTRON --
	ta=7195
-----------------------------------------------------------------------Uncomment this section to train---------------	
	for i=1,3000 do
		pred=net:forward(traindata[{{1,ta},{}}])
		
		err=criterion:forward(pred, trainlabels[{{1,ta},{}}])
		--print(err)
		net:zeroGradParameters()

		criterion_gradient=criterion:backward(pred, trainlabels[{{1,ta},{}}])
		net:backward(traindata[{{1,ta},{}}], criterion_gradient)

		net:updateParameters(0.01)

	end
--	torch.save("savedmodels/entropymodel" .. tcount .. ".net", net)
---------------------------------------------------------------------------------------------------------------------

--	net=torch.load("savedmodels/entropymodel1.net")
	-- TRAINING ACCURACY -- 
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

	normalizedaccuracytrain[train_count]=normalizedaccuracytrain[train_count]+(recall_irr+recall_r)/2


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

	thresh=0.2
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

	--accuracy=count/2000
	-- FoR CLASS IRRELEVANT (CLASS 0) --
	precision_irr=foundzcount/(foundzcount+foundzwrong)
	recall_irr=foundzcount/zcount

	-- FOR CLASS RELEVANT (CLASS 1) --
	precision_r=foundonecount/(foundonecount+foundonewrong)
	recall_r=foundonecount/onecount

	--print("Accuracy:")
	--print(accuracy)

	print("Prec, Recall for Irrelevant:")
	print(precision_irr)
	print(recall_irr)

	print("Prec, Recall for Relevant:")
	print(precision_r)
	print(recall_r)
	
	
	normalizedaccuracy[train_count]=normalizedaccuracy[train_count]+(recall_irr+recall_r)/2
	train_count=train_count+1
end

normalizedaccuracy=torch.Tensor(normalizedaccuracy)
normalizedaccuracytrain=torch.Tensor(normalizedaccuracytrain)

normalizedaccuracy=torch.sum(normalizedaccuracy)/5
normalizedaccuracytrain=torch.sum(normalizedaccuracytrain)/5
print("Normalized Accuracy:")
print(normalizedaccuracy)
repl()
--require 'gnuplot'

--gnuplot.plot({normalizedaccuracy},{normalizedaccuracytrain})

--repl()

--require '../utils/metric'
--print(metric.precision(-(entropy-expected_entropy),applicable:gt(0.5):double()))
--print(metric.precision(-(entropy_model_averaging-entropy),applicable:gt(0.5):double()))
--print(metric.precision(-(entropy),applicable:gt(0.5):double()))


