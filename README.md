# Extensive Vision Applications- Research Team
EVA is a learning community based out of bangalore focusing on AI. This was one of the several experimental projects tried by students.
https://sites.google.com/theschoolofai.in/theschoolofai/home


  
  
  
  # Description of approach:
  
  I have followed an upsample and downsample strategy to match Image size and channels across different layers.
  
  For eg: 
  Take DavidNet architecture as backbone
  
  It has 3 layers, (excluding initial conv layer).
  resolution and channles change as (32, 32, 64)-init layer> (16, 16, 128) > (8, 8, 256) > (4, 4, 512)
  
  
  for zeeconvolution we down and upsample this to get (8, 8, 128) > (8, 8, 256) > (8, 8, 512)
  
  Upon transposing and stiching layers together we get the image size as (896, 8, 8)
  
  With this as image we convolve multiple times, then go a GAP, concatenate this GAP with DavidNet Gap and feed it softmax 
  
  With MultiSoftmaxes:
  
  After each layer(could be customized), we do a gap on the davidnet backbone, at the same time, zeeconv is computed with layers just uptill now, then a gap on that zeeconv is computed.
  These 2 gaps are concatenated, and softmax cross entopy is calculated. This is done at the end of eah layer.
  
  A weighted sum of loss from each of these three layers are computed as final loss

# Important files
  
  Forgive my naming convention. I have called the network as zeedensenet
   
  Notebook to follow is in:
  
  ## Single Softmax:
  (just a single z-convolution layer with 2 kernels)
  https://github.com/selfishhari/eva_research_team4/blob/densenext/notebooks/testing_model_api_zeedensenext.ipynb
  
  2 layers of z-convolution with 16 and 32 kernels. Accuracy stuck at 87%. 
  https://github.com/selfishhari/eva_research_team4/blob/densenext/notebooks/testing_model_api_zeedensenext_2layers.ipynb
  This is the case everytime. 
  When I increase my zeedense layers the accuracy drops.
  However it reached 85% in just 3-4 epochs and then stagnates
  
  ## Multisoftmax:
  87%(voting accuracy-94%)
  Most updated notebook with visualizations:(87% just multisoftmax, voting accuracy-94%)
  
  https://github.com/selfishhari/eva_research_team4/blob/master/notebooks/zeedensenext_multisoft_501kparams_xaxis_noroots_withviz.ipynb
  ### Voting accuracy:
  Calculated by taking mode of predictions from different softmax layers. If all 3 gave different class softmax3 output is taken.
  Accuracy calculated by this approach gave a 7% lift.
  
  7M params:
  
  87%
  https://github.com/selfishhari/eva_research_team4/blob/master/notebooks/testing_model_api_zeedensenext_multisoft_7M_xaxis_roots.ipynb
  
  
  
  The model is in the below file:
  
  https://github.com/selfishhari/eva_research_team4/blob/master/src/zeedensenet.py
  
  
  The required model block (ZeeConvBlk) is in:
  https://github.com/selfishhari/eva_research_team4/blob/c252d5c288917b5a58f097b59ffa1eab8d0c4a04/src/model_blocks.py#L425 line number 425
  
  
  
  That's a huge file and I need to break it down. But for your reference start from the line number mentioned
  
  
  The training function is in:
  https://github.com/selfishhari/eva_research_team4/blob/c252d5c288917b5a58f097b59ffa1eab8d0c4a04/src/run_util.py#L147 line number:147
  
