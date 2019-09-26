# eva_research_team4

# Important Instructions. These are best practices and must always be followed.

  1. Never make changes/push into the master/development branch.
  2. All changes must be made by creating a branch from development branch. "git checkout -b your_branch_name development"
  3. Do not create branches from master. Always use development.
  3. Once your changes are done, create a pull request to development and add everyone as reviewers. 
  4. Only after it is thoroughly tested in development, changes must be merged into master.
  
  
  # Instructions to Rohan to find the relevant code snippets
  Convolving on z-direction:
  
  Notebook is in:
  
  
  The model is in the below file:
  
  https://github.com/selfishhari/eva_research_team4/blob/densenext/src/zeedensenet.py
  
  
  The required model block (ConciseDenseBlk) is in:
  https://github.com/selfishhari/eva_research_team4/blob/densenext/src/model_blocks.py line number 422
  
  That's a huge file and I need to break it down. But for your reference start from the line number mentioned
  
  
  The training function is in:
  https://github.com/selfishhari/eva_research_team4/blob/densenext/src/run_util.py line number:147
