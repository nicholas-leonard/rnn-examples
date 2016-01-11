Getting to grips with LSTM (part one)

# What is LSTM (Long Short Term Memory)

Have you got a data set which you are sure contains some really interesting temporal relationships? Do you suspect that if only you could exploit this knowledge you could improve your target predictor / classifier? If yes, then you probably already know about LSTM as a potential solution to your problem.

[LSTM](http://colah.github.io/posts/2015-08-Understanding-LSTMs/) (Long Short Term Memory) has a storied history. It dates back to 1997 so is not particularly new, yet it has only really come to prominence in the last two years (2013 - 2015) as:

1. LSTM has usurped previous state of the art approaches in applications such as handwriting recognition, phoneme recognition and more.

2. LSTM has replicated these successes in real-world applications (by contrast, many research ideas do not map across to industry use well, if at all).

I'm interested in LSTM for four reasons:

1. As noted above, it represents current state of the art for a wide range of problem domains / applications. I have a hunch that there is something intrinsic to LSTM which can port to other problem domains and add value by improving classification / prediction accuracy.

2. Ecommerce data has a strong temporal component - we denote web site users by some unique identifier (e.g. a cookie) and then we log their observable actions as a series of events on their timeline. Cause and effect is important, and we need something like LSTM to remember the important events and discard the less relevant examples. I work with Ecommerce data *a lot*, and not so much generating new sonnets or inception images :).

3. LSTM is essentially deep learning where the depth is created by unrolling the network through time.

4, LSTM is probably one of the main precursors to [RAM (Reasoning, Attention, Memory) networks](http://www.thespermwhale.com/jaseweston/ram/), and understanding / using LSTM can serve as a good base to apply RAM to target datasets. RAM is growing quickly..

# Motivation

Most of the data sets used in deep learning revolve around image and text corpuses. These sets and domains are very interesting, but they are too far away from ecommerce data to allow any useful conclusions or correlations to be drawn. So the goal for this series of posts is as follows:

1. Build up a good understanding of the [Torch7-based RNN framework](https://github.com/Element-Research/rnn/) by creating the simplest body of code which builds and trains an LSTM recurrent neural network.

2. Use these simple building blocks to construct and train a model which consumes and classifies an Ecommerce data set with strong temporal connections embedded in it.

3. Evaluate the model performance and apply well-known techniques to improve that initial performance.

# RNN Framework selection criteria

* Good Language support (debugger, editor)
* Speed of development
* Active community
* Good performance (training speed, data set scalability, accuracy / precision)
* Near transparent migration from CPU to GPU training
* Ability to modify the framework easily

The Torch + RNN layer we delve into here is by no means perfect, but it scores highly enough on these criteria to make it my choice. Realistically, it pays to have good familiarity with multiple frameworks. TensorFlow may also morph into some kind of generic back-end like gcc or clang where other ML frameworks output TF models for training. There's no reason to do that until TF demonstrates something superior - speed, ability to harness a cluster of nodes..

##Other candidates

There are multiple frameworks available that "do" LSTM, for example:

[Theano](http://deeplearning.net/software/theano/)
[Caffe](http://caffe.berkeleyvision.org/)
[Keras](http://keras.io/)
[Brainstorm](https://github.com/IDSIA/brainstorm)
[TensorFlow](https://www.tensorflow.org/)

I don't mind having to learn Lua or Python whereas a lot of researchers prefer the Python frameworks. I find the compile time overhead of Theano to be quite onerous so I prefer the interpreted nature of Torch7. Brainstorm is pretty new still and so is TensorFlow. I've played with Keras and it looks good, although I don't know that it lets me control enough directly.


#Torch and the RNN library

I don't propose to cover Torch itself in this post - there is simply too much to cover. Suffice it to say that the core Torch distro does not support LSTM or recurrent neural networks (RNNs) natively. It is possible of course to construct RNNs / LSTM using the Torch nngraph (specifically the gModule). Three specific implementations that I m aware of that take this approach are:

1. [Wojciech Zaremba's](https://github.com/wojzaremba) [LSTM code for the Penn TreeBank corpus](https://github.com/wojzaremba/lstm).
2. [Andrej Karpathy's](https://twitter.com/karpathy) LSTM ([char-rnn](https://github.com/karpathy/char-rnn)) code, built on Zaremba's code.
3. [Brendan Shillingford's](https://twitter.com/BrendanShilling) tutorials for Nando de Freitas' [ML course at Oxford](https://www.cs.ox.ac.uk/people/nando.defreitas/machinelearning/).

I find this code hard to understand / change so I was looking for a better abstraction of LSTM cells with the various forget gates "already baked in". That is the primary thing that brought me to [@nicholas-leonard's](https://github.com/nicholas-leonard) RNN addition to Torch.

##The RNN library

Now, the interesting thing about the [RNN library](http://arxiv.org/pdf/1511.07889v2.pdf) (and this in no small part prompted this sequence of posts) is that you don't need to use *all* of the RNN library to take advantage of LSTM. Therefore we will write the simplest code we can here and only add in the parts we need. One specific things we won't use just yet are:

1. The [dp package](https://github.com/nicholas-leonard/dp) (Experiment, DataSource etc.). It's powerful, but confusing..

# Structure of this tutorial

This tutorial builds as follows:

1. Post 1 (this post) - setting out the groundwork and building the base code for future posts.

2. Post 2 - Moving to use a real-world data set and to train on the GPU.

3. Post 3 - Performance tuning - how to improve the performance of the model by applying tried and tested optimisations / tricks of the trade when (a) engineering features / input layers for neural networks and (b) tuning the model itself.


# The core code

Ok, so our minimum requirements for the core code at this point are as follows:

1. Create the requisite data structures so we can train an LSTM on a synthetic data set using RNN.
2. Create an LSTM using the RNN library that is an adjunct to the main Torch repo.
3. Train the LSTM.
4. Save the trained LSTM and associated data structures to file so we know we can re-use trained models for prediction.
5. Load in the saved model and data sets to validate all of our work so far.

Without further ado, here is the code to do exactly that! There's not really much point in building it up slowly / line by line - hopefully the inline comments will make it clear exactly what each non-obvious line is doing. Obvious lines are self-documenting :)

(Thanks to [https://github.com/nicholas-leonard](@nicholas-leonard) for help in debugging this code as I wrote it)

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

The code is also [available here with better syntax highlighting](https://github.com/hsheil/rnn-examples).

##Running and testing the code

On our first run, the code minimises the loss / error in about 200 epochs and this takes about 6 minutes. With a working torch installation, you can test this as follows:

    time th example_part1.lua
    loss      3.3337339206784 in        1.0516710281372 s 
    loss      3.3362607493478 in        1.0598111152649 s 
    <snip/>
    loss      0.0011351864864528 in     1.1150119304657 s 
    loss      0.0011246778926528 in     1.0118091106415 s 
    loss      0.0011143412047407 in     1.12540102005 s   
    loss      0.0011041739914276 in     1.0564870834351 s 
    real    0m6.030s
    user    0m7.156s
    sys     0m1.202s

After you run the code you will see *.t7 files in the directory:

    ls -laht *.t7
    64 -rw-r--r--  1 hsheil  staff    31K 11 Jan 18:32 targets.t7
    44288 -rw-r--r--  1 hsheil  staff    22M 11 Jan 18:32 trained-model.t7
    216 -rw-r--r--  1 hsheil  staff   105K 11 Jan 18:32 training.t7

These files represent the trained model and inputs and targets - if we re-run the network again but load in these files, then the network *should* carry on training from where it left off:

    >th example_part1.lua -use_saved
    loss      0.0010941725196606 in     1.1108829975128 s 
    loss      0.0010843310458659 in     1.1420600414276 s 
    loss      0.0010746471778253 in     1.0338280200958 s

And it does! Note that we have simply over-trained our LSTM on the random problem - but it's hard at this point to get it to do well on a similarly generated random validation or test set. That will have to wait until part two, but we now have the plumbing code in place to perform our cross-validation in part two.


# Part two..

In the next part of this series of posts, we will:

1. Replace the play / synthetic data set with a real data set (this will expose a lot of "in the trenches" coding that is needed to make LSTM work), including having a separate validation and test set.

2. Add code so we can target dedicated hardware (efficiently) to speed up training - using CUDA and OpenCL. We can expect a 5x - 15x speedup running on GPUs, depending on how well we can keep the GPU fed with data without context-switching between it and the CPU.
