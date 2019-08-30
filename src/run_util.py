import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm_notebook as tqdm
import os, datetime, time

class Run():
  
  epochs = 24

  batch_size = 512

  max_lr = 0.4    

  min_lr = 0.0000001

  end_anneal_pc = 0.2

  max_mom = 0.9

  min_mom = 0.8

  wd = 0.00005

  batches_per_epoch = 50000//batch_size

  skip_testing_epochs = 0
  
  with tf.variable_scope("global", reuse=tf.AUTO_REUSE):

      global_step = tf.get_variable("global_step_variable", shape=(), dtype=tf.int16,
        initializer=tf.zeros_initializer)

  
  def initialize_everything(self, params, trn_data_supplier, tst_data_supplier):
    
      self.epochs = params["epochs"]

      self.batch_size = params["batch_size"]

      self.max_lr = params["max_lr"]    

      self.min_lr = params["min_lr"]

      self.end_anneal_pc = params["end_anneal_pc"]

      self.max_mom = params["max_mom"]

      self.min_mom = params["min_mom"]

      self.wd = params["wd"]

      self.batches_per_epoch = params["batches_per_epoch"]

      self.skip_testing_epochs = params["skip_testing_epochs"]
    
      self.comments = params["comments"]

      self.trn_data_supplier = trn_data_supplier

      self.tst_data_supplier = tst_data_supplier
  #### END OF INITIALIZATIONS ###
  #------------------------------------------------------------------------------------------------------------------------------#
      
  
  
  ############___________LR/MOM SCHEDULERS______________###############
  
  def lr_schedule(self, t):
    
        max_lr = self.max_lr
      
        min_lr = self.min_lr
        
        epochs = self.epochs
        
        perc_end = self.end_anneal_pc

        if max_lr * 0.1 < min_lr:

          break_lr = min_lr * 1.1

        else:

          break_lr = max_lr * 0.1

        lr = np.interp([t], [0, (epochs+1)//5, int((1-perc_end) * epochs), epochs+1], \
                       [0, max_lr, break_lr, min_lr])[0]
        
        #print(t, lr)

        return lr


  def mom_schedule(self, t):
    
        max_mom=self.max_mom
      
        min_mom=self.min_mom
        
        epochs=self.epochs

        mom = np.interp([t], [0, (epochs+1)//5, epochs], \
                       [max_mom, min_mom, max_mom])[0]
        
        #print(t, mom)
        return mom


  lr_func = lambda : Run.lr_schedule(Run, Run.global_step/Run.batches_per_epoch)/Run.batch_size

  mom_func = lambda : Run.mom_schedule(Run, Run.global_step/Run.batches_per_epoch)

  opt = tf.train.MomentumOptimizer(lr_func, momentum=mom_func, use_nesterov=True)
  #### END OF LR/MOM Schedulers ###
  #------------------------------------------------------------------------------------------------------------------------------#
      
  
  
  ############___________TRAIN______________###############
  
  def train(self, model , opt, lr_func, global_step, trn_data_supplier, tst_data_supplier,
            epochs = epochs, batch_size = batch_size, skip_testing_epochs = skip_testing_epochs,  log_results = False, verbose=True):
  
      

      t = time.time()
      
      tst_data = self.tst_data_supplier(0)
      
      len_test = tst_data[0].shape[0]

      test_set = tf.data.Dataset.from_tensor_slices(tst_data).batch(batch_size).prefetch(1)

      log_runs = {"epoch":[],"time":[], "lr":[], "train_acc":[], "test_acc":[], "train_loss":[], "test_loss":[]}

      t = time.time()

      print("****training starts****")

      for epoch in tqdm(range(epochs)):   
        
        trn_data = self.trn_data_supplier(epoch)
        
        len_train = trn_data[0].shape[0]

        train_set = tf.data.Dataset.from_tensor_slices(trn_data).shuffle(len_train).batch(batch_size).prefetch(1)

        train_loss = test_loss = train_acc = test_acc = 0.0

        tf.keras.backend.set_learning_phase(1)

        for (x, y) in train_set:

          with tf.GradientTape() as tape:

            loss, correct = model(x, y)

          var = model.trainable_variables

          grads = tape.gradient(loss, var)

          opt.apply_gradients(zip(grads, var), global_step=global_step)

          if epoch >= skip_testing_epochs:

            train_loss += loss.numpy()

            train_acc += correct.numpy()

        if epoch < skip_testing_epochs:

          print( epoch, time.time() - t)

          continue;

        tf.keras.backend.set_learning_phase(0)

        for (x, y) in test_set:

          loss, correct = model(x, y)

          test_loss += loss.numpy()

          test_acc += correct.numpy()

        lr_to_display = lr_func(epoch+1)
        
        if verbose:
          
          print('epoch:', epoch+1, 'lr:', lr_to_display, 'train loss: {0:.3f}'.format(train_loss / len_train), 'train acc:{0:.3f}'.format( train_acc / len_train),
              'val loss: {0:.3f}'.format( test_loss / len_test), 'val acc: {0:.3f}'.format(test_acc / len_test), 'time:{0:.3f}'.format(time.time() - t))

        if log_results:

          log_runs["epoch"] += [epoch+1]

          log_runs["lr"] += [lr_to_display]

          log_runs["time"] += [time.time() - t]

          log_runs["train_acc"] += [train_acc / len_train]

          log_runs["test_acc"] += [test_acc / len_test]

          log_runs["train_loss"] += [train_loss / len_train]

          log_runs["test_loss"] += [test_loss / len_test]
          
          ##---- END of EPOCH ##

      if log_results:    

        total_model_parameters = np.sum([np.prod(v.get_shape().as_list()) for v in model.trainable_variables])

        log_params = {"total_model_parameters":total_model_parameters, "epochs":epochs, "batch_size": batch_size, "max_lr": self.max_lr, "min_lr":self.min_lr,"anneal_perc":self.end_anneal_pc, 
                      "max_mom":self.max_mom,"min_mom":self.min_mom, "time_taken":time.time() - t, "train_acc":train_acc / len_train, "test_acc":test_acc / len_test, "wd":self.wd,
           "train_loss": train_loss / len_train, "test_loss":test_loss / len_test}

        self.logger(log_params, log_runs, model, comment= self.comments)

      return [model, time.time() - t, train_acc / len_train, test_acc / len_test, train_loss / len_train, test_loss / len_test]
    
  ############___________END OF TRAIN______________###############  
  #------------------------------------------------------------------------------------------------------------------------------#  
  
  
  
  ############___________LOGGER______________###############
    
  def logger(self, params, run_vals, model, comment="", logfilepath= "../data/run_logger.csv"):
  
      dt = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

      params["runs"] = run_vals

      params["log_time"] = dt

      params["comments"] = comment

      params = {k:[v] for k, v in params.items()}

      log_df = pd.DataFrame(params)

      print(log_df)

      if os.path.isfile(logfilepath):

        log_df.to_csv(logfilepath, mode="a", index=False, header=False)

      else:

        log_df.to_csv(logfilepath, index=False)

      #log_df.to_csv(logfilepath, index=False)
  ############___________END OF LOGGER______________###############  
  #------------------------------------------------------------------------------------------------------------------------------#      
  
  
      
  ############___________LR FINDER______________###############
      
      
  def lr_finder(self, model_class,
                lr_list=[0.001, 0.003, 0.005, 0.007, 0.008, 0.009, 0.01, 0.015, 0.02,0.03, 0.04, 0.05, 0.08, 
                       0.1, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 2, 3,4, 5, 6, 7, 8, 9, 10],               
              train_num_batches = 5, test_num_batches = 3, add_cutout = True, break_loss_factor=5):
  
      train_loss_list = []

      epoch = 0

      min_loss = 0

      #lr_list = np.interp(list(range(num_epochs)), [0, num_epochs], [0, .01])

      model = model_class()

      epoch_counter = 0

      global_step = self.global_step

      global_step.assign(0)

      def lr_finder_schedule(step,lr_list=lr_list):

            epoch = int(step)

            return lr_list[epoch]

      lr_func = lambda : lr_finder_schedule(global_step/train_num_batches)/self.batch_size

      mom_func = lambda : self.mom_schedule(self.global_step/train_num_batches)

      opt_lrfinder = tf.train.MomentumOptimizer(lr_func, momentum=mom_func, use_nesterov=True)


      def supply_lrfinder_train_images(epoch, train_num_batches=train_num_batches, batch_size=self.batch_size):

            runs = 10

            run = epoch % runs

            x_train_lr = (globals()['x_train%s' % epoch_num]).astype(np.float16)

            global y_train_org

            sample_idx = np.random.choice(y_train_lr.size, train_num_batches * batch_size, replace=False)

            x_train_lr = x_train_lr[sample_idx, :, :, :]

            y_train_lr = y_train_org[sample_idx]

            return (x_train_lr, y_train_lr)

      def supply_lrfinder_test_images(epoch, test_num_batches=test_num_batches, batch_size=self.batch_size):

            global x_test_org

            global y_test

            sample_idx = np.random.choice(y_test.size, test_num_batches * batch_size, replace=False)

            x_test_lr = x_test_org[sample_idx, :, :, :]

            y_test_lr = y_test[sample_idx]

            return (x_test_lr, y_test_lr)

      print("running lr_finder")

      for epoch in tqdm(range(len(lr_list))):

          epoch_counter += 1

          print(epoch_counter) 

          #model = DavidNet()

          model, time_for_epoch, train_acc, test_acc, train_loss, test_loss = \
          self.train(model=model, opt=opt_lrfinder, lr_func = lr_finder_schedule, global_step=global_step, epochs = 1, batch_size = self.batch_size, 
                trn_data_supplier=supply_lrfinder_train_images, tst_data_supplier=supply_lrfinder_test_images, skip_testing_epochs = self.skip_testing_epochs, log_results = False, verbose=False)

          if epoch_counter == 1:

            min_loss = train_loss

            train_loss_list += [train_loss]

          elif train_loss > break_loss_factor * min_loss:

            train_loss_list += [train_loss]

            if (epoch_counter) > 3:

              plt.plot(lr_list[:epoch_counter-3], train_loss_list[:-3])

            else:

              plt.plot(lr_list[:epoch_counter-1], train_loss_list[:-1])        


            plt.show()

            return [lr_list, train_loss_list]

          else:

            train_loss_list += [train_loss]

      plt.plot(lr_list, train_loss_list)

      plt.show()

      return [lr_list, train_loss_list]

  ############___________END OF LR FINDER______________###############  
  #------------------------------------------------------------------------------------------------------------------------------#  
  
  
  
  ############___________GRID SEARCH______________###############
  
  def grid_search(self, model_fn, params_tune_grid, trn_data_supplier, tst_data_supplier, choose_on="test_acc"):
    
      def dict_product(dicts):
    
        import itertools

        return (dict(zip(dicts, x)) for x in itertools.product(*dicts.values()))
  
      if choose_on == "test_acc" :
          
         min_val = 0
      else:
        
        min_val = 1000000000000000000000
        
      

      best_params = {}

      for params in dict_product(params_tune_grid):

        self.initialize_everything(params, trn_data_supplier, tst_data_supplier)

        model_tuned, time, train_acc, test_acc, train_loss, test_loss = self.run(model_fn, params, trn_data_supplier, tst_data_supplier)

        if choose_on == "test_acc":

          if(test_acc > min_val):

            min_val = test_acc

            best_params = params

        else:

          if(time < min_val):

            min_val = test_acc

            best_params = params

      return (best_params, min_val)
    
    
  ############___________END OF GRIDSEARCH______________###############  
  #------------------------------------------------------------------------------------------------------------------------------#  
  
  

  ############___________RUN______________###############
    
  def run(self, model_fn, params, trn_data_supplier, tst_data_supplier):
    
      model = model_fn()
      
      self.initialize_everything(params, trn_data_supplier, tst_data_supplier)
      
      global_step = self.global_step
      
      global_step.assign(0)#Reinitializing global step
      
      return self.train(model=model, opt=self.opt, lr_func = self.lr_schedule, global_step=global_step, epochs = self.epochs,
                        batch_size = self.batch_size, trn_data_supplier=self.trn_data_supplier,
                        tst_data_supplier=self.tst_data_supplier, skip_testing_epochs = self.skip_testing_epochs, 
                        log_results = True, verbose=True)
  ############___________END OF RUN______________###############  
  #------------------------------------------------------------------------------------------------------------------------------#  

    
