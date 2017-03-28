require 'torch'
require 'nn'

-- Perceptron

-- inp = torch.randn(1,5)

-- mlp = nn.Sequential()

-- inputSize = 5

-- nclasses = 2

-- mlp:add(nn.Linear(inputSize , 1))
-- mlp:add(nn.Sigmoid())

-- print (mlp)

-- -- out = mlp:forward(torch.randn(1,5))
-- out = mlp:forward(inp)
-- print (out)


-- mlp2 = nn.Sequential()

-- inputSize = 5

-- nclasses = 2

-- mlp2:add(nn.Linear(inputSize , 2))
-- mlp2:add(nn.SoftMax())

-- print (mlp2)

-- -- out = mlp:forward(torch.randn(1,5))
-- out2 = mlp2:forward(inp)
-- print (out2)

-- generate dataset
dataset={};
function dataset:size() return 100 end -- 100 examples
for i=1,dataset:size() do 
	local input= torch.randn(2);     --normally distributed example in 2d
	local output= torch.Tensor(1);
	if input[1]*input[2]>0 then    --calculate label for XOR function
		output[1]=-1;
	else
		output[1]=1;
	end
	dataset[i] = {input, output};
end

mlp = nn.Sequential()

inputs=2; outputs=1; HUs=20;
mlp:add(nn.Linear(inputs,HUs))
mlp:add(nn.Tanh())
mlp:add(nn.Linear(HUs,outputs))

print (mlp)

criterion = nn.MSECriterion()  
trainer = nn.StochasticGradient(mlp, criterion)
trainer.learningRate = 0.01
trainer:train(dataset)

x = torch.Tensor(2)
x[1] =  0.5; x[2] =  0.5; print(mlp:forward(x))
x[1] =  0.5; x[2] = -0.5; print(mlp:forward(x))
x[1] = -0.5; x[2] =  0.5; print(mlp:forward(x))
x[1] = -0.5; x[2] = -0.5; print(mlp:forward(x))

