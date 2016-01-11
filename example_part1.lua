require 'rnn'

cmd = torch.CmdLine()
cmd:text()
cmd:text('Simple LSTM example for the RNN library')
cmd:text()
cmd:text('Options')
cmd:option('-use_saved',false,'Use previously saved inputs and trained network instead of new')
cmd:text()

-- parse input params
opt = cmd:parse(arg)

-- Keep the input layer small so the model trains / converges quickly while training
local inputSize = 10
-- Larger numbers here mean more complex problems can be solved, but can also over-fit. 256 works well for now
local hiddenSize = 256
-- We want the network to classify the inputs using a one-hot representation of the outputs
local outputSize = 3

-- the dataset size is the total number of examples we want to present to the LSTM 
local dsSize=200

--We present the dataset to the network in batches where batchSize << dsSize
local batchSize=5
--And seqLength is the length of each sequence, i.e. the number of "events" we want to pass to the LSTM
--to make up a single example. I'd like this to be dynamic ideally for the YOOCHOOSE dataset..
local seqLength=5
-- number of target classes or labels, needs to be the same as outputSize above
-- or we get the dreaded "ClassNLLCriterion.lua:46: Assertion `cur_target >= 0 && cur_target < n_classes' failed. "
local nClass = 3

function build_data()
   local inputs = {}
   local targets = {}
   --Use previously created and saved data
   if opt.use_saved then
      inputs = torch.load('training.t7')
      targets = torch.load('targets.t7')
      rnn = torch.load('trained-model.t7')
   else
      for i = 1, dsSize do
         -- populate both tables to get ready for training
         table.insert(inputs, torch.randn(batchSize,inputSize))
         table.insert(targets, torch.LongTensor(batchSize):random(1,nClass))
      end
   end
   return inputs, targets
end

function build_network(inputSize, hiddenSize, outputSize)
   if opt.use_saved then
      rnn = torch.load('trained-model.t7')
   else
      rnn = nn.Sequential() 
      :add(nn.Sequencer(nn.Linear(inputSize, hiddenSize))) 
      :add(nn.Sequencer(nn.LSTM(hiddenSize, hiddenSize)))
      :add(nn.Sequencer(nn.LSTM(hiddenSize, hiddenSize))) 
      :add(nn.Sequencer(nn.Linear(hiddenSize, outputSize))) 
      :add(nn.Sequencer(nn.LogSoftMax()))
   end
   return rnn
end

function save(inputs, targets, rnn)
   -- Save out the tensors we created and the model itself so we can load it back in
   -- if -use_saved is set to true
   torch.save('training.t7', inputs)
   torch.save('targets.t7', targets)
   torch.save('trained-model.t7', rnn)
end

--two tables to hold the *full* dataset input and target tensors
local inputs, targets = build_data()
local rnn = build_network(inputSize, hiddenSize, outputSize)

-- Decorate the regular nn Criterion with a SequencerCriterion as this simplifies training quite a bit
-- SequencerCriterion requires tables as input, and this affects the code we have to write inside the training for loop
local seqC = nn.SequencerCriterion(nn.ClassNLLCriterion())

local start = torch.tic()

--Now let's train our network on the small, fake dataset we generated earlier
rnn:training()
--Feed our LSTM the dsSize examples in total, broken into batchSize chunks
for numEpochs=0,200,1 do
   local start = torch.tic()
   for offset=1,dsSize,batchSize+seqLength do
      -- We need to get a subset (of size batchSize) of the inputs and targets tables
      local batchInputs = {}
      local batchTargets = {}

      --start needs to be "2" and end "batchSize-1" to correctly index
      --all of the examples in the "inputs" and "targets" tables
      for i = 2, batchSize+seqLength-1,1 do
         table.insert(batchInputs, inputs[offset+i])
         table.insert(batchTargets, targets[offset+i])
      end
      out = rnn:forward(batchInputs)
      err = seqC:forward(out, batchTargets)
      gradOut = seqC:backward(out, batchTargets)
      rnn:backward(batchInputs, gradOut)
      --We update params at the end of each batch
      rnn:updateParameters(0.05)
      rnn:zeroGradParameters()
   end
   local currT = torch.toc(start)
   print('loss', err .. ' in ', currT .. ' s')
end
save(inputs, targets, rnn)
