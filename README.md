# eva_research_team4

# Important Instructions. These are best practices and must always be followed.

  1. Never make changes/push into the master/development branch.
  2. All changes must be made by creating a branch from development branch. "git checkout -b your_branch_name development"
  3. Do not create branches from master. Always use development.
  3. Once your changes are done, create a pull request to development and add everyone as reviewers. 
  4. Only after it is thoroughly tested in development, changes must be merged into master.
  
  
  # Instructions to Rohan to find the relevant code snippets(Convolving on z-direction)
  
  Forgive my naming convention. I have called the network as zeedensenet
   
  Notebook to follow is in:
  
  (just a single z-convolution layer with 2 kernels)
  https://github.com/selfishhari/eva_research_team4/blob/densenext/notebooks/testing_model_api_zeedensenext.ipynb
  
  2 layers of z-convolution with 16 and 32 kernels. Accuracy stuck at 87%. 
  https://github.com/selfishhari/eva_research_team4/blob/densenext/notebooks/testing_model_api_zeedensenext_2layers.ipynb
  This is the case everytime. 
  When I increase my zeedense layers the accuracy drops.
  However it reached 85% in just 3-4 epochs and then stagnated
  
  
  
  The model is in the below file:
  
  https://github.com/selfishhari/eva_research_team4/blob/densenext/src/zeedensenet.py
  
  
  The required model block (ZeeConvBlk) is in:
  https://github.com/selfishhari/eva_research_team4/blob/densenext/src/model_blocks.py line number 422
  
  That's a huge file and I need to break it down. But for your reference start from the line number mentioned
  
  
  The training function is in:
  https://github.com/selfishhari/eva_research_team4/blob/densenext/src/run_util.py line number:147
  
  
  # Description of approach:
  
  I have followed an upsample and downsample strategy to match Image size and channels across different layers.
  
  For eg: 
  Take DavidNet architecture
  
  It has 3 layers, (excluding initial conv layer).
  resolution and channles change as (32, 32, 64)-init layer> (16, 16, 128) > (8, 8, 256) > (4, 4, 512)
  
  
  for zeeconvolution we down and upsample this to get (8, 8, 128) > (8, 8, 256) > (8, 8, 512)
  
  Upon transposing and stiching layers together we get the image size as (896, 8, 8)
  
  With this as image we convolve multiple times, then go a GAP, concatenate this GAP with DavidNet Gap and feed it softmax 
