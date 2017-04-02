require 'rnn'

function build_network(inputSize, hiddenSize, outputSize)

   -- I0: The model architecture needs more work, why does having a linear layer as per
   -- the example in (http://arxiv.org/abs/1511.07889) cause a "size mismatch, m1: [10 x 512], m2: [10 x 512]" error?
   -- I1: add in a dropout layer
   rnn = nn.Sequential() 
   --:add(nn.Sequencer(nn.Linear(inputSize, hiddenSize))) 
   :add(nn.Sequencer(nn.LSTM(hiddenSize, hiddenSize)))
   :add(nn.Sequencer(nn.LSTM(hiddenSize, hiddenSize))) 
   :add(nn.Sequencer(nn.Linear(hiddenSize, outputSize))) 
   :add(nn.Sequencer(nn.LogSoftMax()))
   -- I1: Adding this line makes the loss oscillate a lot more during training, when according to 
   -- http://arxiv.org/abs/1409.2329 this should *help* model performance
   --rnn:getParameters():uniform(-0.1, 0.1)
   return rnn
end

-- Keep the input layer small so the model trains / converges quickly while training
local inputSize = 10
-- Most models seem to use 512 LSTM units in the hidden layers, so let's stick with this
local hiddenSize = 512
-- We want the network to classify the inputs using a one-hot representation of the outputs
local outputSize = 3

local rnn = build_network(inputSize, hiddenSize, outputSize)

--artificially small batchSize again for easy training
local batchSize=5
--and the same for dataset size
local dsSize=batchSize*2

inputs = {}

-- Build up our inputs and targets
-- I2, add code so that if --cuda supplied, these become CudaTensors
-- using the opt.XXX and 'require cunn'
-- I3 - replace this random data set with something more meaningful / learnable
-- and with a realistic testing and validation set
for i = 1, dsSize do
   table.insert(inputs, torch.randn(inputSize,hiddenSize))
end

-- *3 to get 3 sample labels..
-- I4: do these dims have to be like this as the Sequencer thinks it's doing a batch..
-- Intuitively I expected targets to be more like this commented-out line, but then we get
-- "bad argument #2 to '?' (out of range at..."
-- targets = torch.ceil(torch.rand(outputSize,dsSize)*3)
-- Same as I1 above (add :cuda()).
targets = torch.ceil(torch.rand(inputSize,hiddenSize)*3)

-- Decorate the regular nn Criterion with a SequencerCriterion as this simplifies training quite a bit
seqC = nn.SequencerCriterion(nn.ClassNLLCriterion())

local count = 0
local numEpochs=100
local start = torch.tic()

--Now let's train our network on the small, fake dataset we generated earlier
while numEpochs ~= 0 do
   rnn:training()
   count = count + 1
   out = rnn:forward(inputs)
   err = seqC:forward(out, targets)
   gradOut = seqC:backward(out, targets)
   rnn:backward(inputs, gradOut)
   if count % batchSize == 0 then
      local currT = torch.toc(start)
      print('loss', err .. ' in ', currT .. ' s')
      --TODO, make this configurable / reduce over time as the model converges
      rnn:updateParameters(0.05)
      -- I5: Are these steps necessary? Seem to make no difference to convergence if called or not
      -- Perhaps they are being called by 
      rnn:zeroGradParameters()
      rnn:forget()
      start = torch.tic()
   end
   -- I6: Make this configurable based on the convergence, so we keep going for bigger, more complex models until they are trained
   -- to an acceptable accuracy
   -- Also add in code to save out the model file to disk for evaluation / usage externally periodically
   if count % dsSize == 0 then
      numEpochs = numEpochs - 1
   end
end