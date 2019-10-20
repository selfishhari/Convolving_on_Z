#from keras.applications import preprocess_input, decode_predictions

import tensorflow as tf
from keras.preprocessing import image
import numpy as np
import cv2
from keras import backend as K
from sklearn.metrics import confusion_matrix
from matplotlib import pyplot as plt
import seaborn as sn
import run_util
import pandas as pd



DATA_DIR = "../data/"

# Gradcam Function for a single RGB image
def applygradCAM_RGB(reqImage, model, layerName, No_of_Channel_in_Layer):
  x = image.img_to_array(reqImage)
  # expanding dimension for prediction
  x = np.expand_dims(x, axis=0)
  x = preprocess_input(x)
  
  # prediction of the image
  preds = model.predict(x)
  class_idx = np.argmax(preds[0])
  class_output = model.output[:, class_idx]
  
  # Getting the output of the last convolutional layer 
  last_conv_layer = model.get_layer(layerName)
  
  # Claculating the gradients
  grads = K.gradients(class_output, last_conv_layer.output)[0]
  pooled_grads = K.mean(grads, axis=(0, 1, 2))
  iterate = K.function([model.input], [pooled_grads, last_conv_layer.output[0]])

  pooled_grads_value, conv_layer_output_value = iterate([x])

  for i in range(No_of_Channel_in_Layer):
    conv_layer_output_value[:, :, i] *= pooled_grads_value[i]
  
  # Creating the heatmap
  heatmap = np.mean(conv_layer_output_value, axis = -1)
  heatmap = np.maximum(heatmap, 0)
  heatmap /= np.max(heatmap)
  
  # Resize heatmap to original image size
  heatmap = cv2.resize(heatmap, (reqImage.shape[1], reqImage.shape[0]))
  heatmap = np.uint8(255 * heatmap)
  heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
  superimposed_img = cv2.addWeighted(reqImage, 0.5, heatmap, 0.5, 0)
  return superimposed_img


# Gradcam Function for a single normalize image
def applygradCAM_normalize(reqImage, model, layerName, No_of_Channel_in_Layer):
  #x = image.img_to_array(reqImage)
  # expanding dimension for prediction
  x = np.expand_dims(x, axis=0)
  #x = preprocess_input(x)
  
  # prediction of the image
  preds = model.predict(x)
  class_idx = np.argmax(preds[0])
  class_output = model.output[:, class_idx]
  
  # Getting the output of the last convolutional layer 
  last_conv_layer = model.get_layer(layerName)
  
  # Claculating the gradients
  grads = K.gradients(class_output, last_conv_layer.output)[0]
  pooled_grads = K.mean(grads, axis=(0, 1, 2))
  iterate = K.function([model.input], [pooled_grads, last_conv_layer.output[0]])

  pooled_grads_value, conv_layer_output_value = iterate([x])

  for i in range(No_of_Channel_in_Layer):
    conv_layer_output_value[:, :, i] *= pooled_grads_value[i]
  
  # Creating the heatmap
  heatmap = np.mean(conv_layer_output_value, axis = -1)
  heatmap = np.maximum(heatmap, 0)
  heatmap /= np.max(heatmap)
  
  # Resize heatmap to original image size
  heatmap = cv2.resize(heatmap, (reqImage.shape[1], reqImage.shape[0]))
  heatmap = np.uint8(255 * heatmap)
  heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
  superimposed_img = cv2.addWeighted(image_reconstract(reqImage), 0.5, heatmap, 0.5, 0)
  return superimposed_img
  
### function to call accuracy and loss curve(works for keras model)
def plot_model_history(model_history):
    fig, axs = plt.subplots(1,2,figsize=(15,5))
    # summarize history for accuracy
    axs[0].plot(range(1,len(model_history.history['acc'])+1),model_history.history['acc'])
    axs[0].plot(range(1,len(model_history.history['val_acc'])+1),model_history.history['val_acc'])
    axs[0].set_title('Model Accuracy')
    axs[0].set_ylabel('Accuracy')
    axs[0].set_xlabel('Epoch')
    axs[0].set_xticks(np.arange(1,len(model_history.history['acc'])+1),len(model_history.history['acc'])/10)
    axs[0].legend(['train', 'val'], loc='best')
    # summarize history for loss
    axs[1].plot(range(1,len(model_history.history['loss'])+1),model_history.history['loss'])
    axs[1].plot(range(1,len(model_history.history['val_loss'])+1),model_history.history['val_loss'])
    axs[1].set_title('Model Loss')
    axs[1].set_ylabel('Loss')
    axs[1].set_xlabel('Epoch')
    axs[1].set_xticks(np.arange(1,len(model_history.history['loss'])+1),len(model_history.history['loss'])/10)
    axs[1].legend(['train', 'val'], loc='best')
    plt.show()



#function to fet accuracy(works for keras model)
def accuracy(test_x, test_y, model):
    result = model.predict(test_x)
    predicted_class = np.argmax(result, axis=1)
    true_class = np.argmax(test_y, axis=1)
    num_correct = np.sum(predicted_class == true_class) 
    accuracy = float(num_correct)/result.shape[0]
    return (accuracy * 100)



#function to get misclassified image index(for keras model)
def misclassified_index(model, test_images, test_labels):
  y_pred = run_util.predict(model, test_images)
  
  # getting index of all misclasified images
  img_index = []
  for i in range(1,test_images.shape[0]):
    if y_pred[i] != test_labels[i]:
        
      img_index.append(i)
    
  return img_index



# constract images from normalize value to RGB image
def image_reconstract(x):
  # normalize tensor: center on 0., ensure std is 0.1
  x -= x.mean()
  x /= (x.std() + 1e-5)
  x *= 0.1

  # clip to [0, 1]
  x += 0.5
  x = np.clip(x, 0, 1)

  # convert to RGB array
  x *= 255
  #x = x.transpose((1, 2, 0))
  x = np.clip(x, 0, 255).astype('uint8')
  return x

# create image gallary 5 in a row
def image_gallary(no_of_image, img, img_lbl, 
                  pred_lbl = None,
                  pred_lbl2=None,
                  prob1=None,
                  prob2=None,
                  denormalize=False,
                  train_std=np.array([62.99321928, 62.08870764, 66.70489964]),
                  train_mean = np.array([125.30691805, 122.95039414, 113.86538318]),
                  NUM_IMAGES_IN_ROW=5):
    
    
    
  
  
  if NUM_IMAGES_IN_ROW <= 5:
      
      fig_x = 12
      
      fig_y = 12
      
  else:
      
      fig_x = NUM_IMAGES_IN_ROW * 12 / 5
      
      fig_y = NUM_IMAGES_IN_ROW * 12 / 5
  
  
  #no_of_image: give no of image as multiple of 5
  if no_of_image<NUM_IMAGES_IN_ROW:
    row = 1
    col = no_of_image
    
    fig_x = fig_x * no_of_image / NUM_IMAGES_IN_ROW
    
    fig_y = fig_y * no_of_image / NUM_IMAGES_IN_ROW
  else:
    row = no_of_image//NUM_IMAGES_IN_ROW
    col = NUM_IMAGES_IN_ROW
  
  
  fig=plt.figure(figsize=(fig_x, fig_y))
  
  denormalize_fn = lambda x: ((x * train_std) + train_mean)
  
  for i in range(0,row*col):
      
    if denormalize == True:
        
        #print(i)
        
        img[i] = denormalize_fn(img[i])
        
    fig.add_subplot(row,col,i+1)
    
    plt.imshow(img[i].astype(np.uint8)) 
    
    if pred_lbl == None:
      plt.title('Actual: '+str(img_lbl[i]))
    elif pred_lbl2 == None:
      plt.title('Actual: '+str(img_lbl[i])+'\n Predicted: '+str(pred_lbl[i]))
      
    elif prob1 == None:        
        plt.title('Actual: '+str(img_lbl[i])+
                  '\n Predicted1: '+str(pred_lbl[i])+
                  '\n Predicted2: '+str(pred_lbl2[i]))
    elif prob2 == None:
        plt.title('Actual: '+str(img_lbl[i])+
                  '\n Predicted1: '+str(pred_lbl[i])+
                  '\n Predicted2: '+str(pred_lbl2[i])+
                  '\n Prob1: {0:.2f}'.format(prob1[i]))
        
    else:
        plt.title('Actual: '+str(img_lbl[i])+
                  '\n Predicted1: '+str(pred_lbl[i])+
                  '\n Predicted2: '+str(pred_lbl2[i])+
                  '\n Prob1: {0:.2f}'.format(prob1[i])+
                  '\n Prob2: {0:.2f}'.format(prob2[i]))
        
        
    plt.xticks([])
    plt.yticks([])
    
  plt.subplots_adjust(left=0.125, bottom=0.1, right=0.9, top=0.5, wspace=0.9, hspace=0.1)
  plt.show()


#create gradcam gallary with original image
def  gradCAM_galary_set(no_of_image, class_names, image_set, img_lbl, img_index, model, layerName, No_of_Channel_in_Layer):
  #no_of_image: give no of image as multiple of 5  
#   class_names = ['airplane','automobile','bird','cat','deer',
#                'dog','frog','horse','ship','truck']
  img_actual = [] 
  img_gradCam = [] 
  img_actual_value = []
  img_pred_value = []
  y_pred = model.predict(image_set)
  for i in range(1,no_of_image+1):
      img_req = np.array(image_set[img_index[i-1]]).astype(np.float16)
      img_req = image_reconstract(img_req)
      img_actual.append(img_req)
      img_req_grad = applygradCAM(image_set[img_index[i-1]], model, layerName, No_of_Channel_in_Layer)
      img_gradCam.append(img_req_grad)
      img_actual_value.append(class_names[img_lbl[img_index[i-1]]])
      img_pred_value.append(class_names[np.argmax(y_pred[img_index[i-1]])])
  if no_of_image<5:
    row = 1
    col = no_of_image
  else:
    row = no_of_image//5
    col = 5
  fig=plt.figure(figsize=(40,50))
  j=0
  for i in range(0,row*col):
    if i%5==0 and i!=0:
      j=2*i
    fig.add_subplot(row,col,j+1)
    plt.imshow(img_actual[i]) 
    plt.title('Fig:'+ str(i+1) +'; Actual: '+str(img_actual_value[i])+' & Predicted: '+str(img_pred_value[i]))
    plt.xticks([])
    plt.yticks([])
    fig.add_subplot(row,col,j+6)
    plt.imshow(img_gradCam[i])  
    plt.title('Fig:'+ str(i+1) +'; gradCAM O/P: ')
    plt.xticks([])
    plt.yticks([])
    j+=1
  plt.show()
  
def _get_batch_difference_df(model, imgs, ys, difference=True):
    
    """
    This function takes images and it's true labels.
    Computes multisoftmax probabilities.
    
    
    Returns a dataframe with images where there is a difference in prediction between two softmaxes.
    """
    
    model_out = model(imgs, ys, infer_multi=True)
    
    temp_df = pd.DataFrame(columns=[])
    
    temp_df["ys"] = ys
    
    temp_df["subset"] = False
    
    multi_accuracies = model_out[3]
    
    sm3_probs_all = multi_accuracies["sm3"]["prob"]
        
    sm3_class = tf.argmax(sm3_probs_all, axis =1)
        
    sm3_probs = tf.reduce_max(sm3_probs_all, axis=1)
    
    temp_df["sm3_class"] = sm3_class
    
    temp_df["sm3_probs"] = sm3_probs
    
    temp_df["sm3_correct"] = temp_df.ys == temp_df.sm3_class
    
    ###print("got sm3 set")
    
    if "sm1" in multi_accuracies.keys():
        
        sm1_probs_all = multi_accuracies["sm1"]["prob"]
        
        sm1_class = tf.argmax(sm1_probs_all, axis =1)
        
        sm1_probs = tf.reduce_max(sm1_probs_all, axis=1)
        
        temp_df["sm1_probs"] = sm1_probs
        
        temp_df["sm1_class"] = sm1_class
        
        temp_df["sm1_correct"] = temp_df.ys == temp_df.sm1_class
        
        if difference:
            
            temp_df["subset"] = (temp_df.sm1_correct != temp_df.sm3_correct) | (temp_df["subset"])
            
        else:
            
            temp_df["subset"] = True
            
        ##print("got sm1 set")
        
    if "sm2" in multi_accuracies.keys():
        
        sm2_probs_all = multi_accuracies["sm2"]["prob"]
        
        sm2_class = tf.argmax(sm2_probs_all, axis =1)
        
        sm2_probs = tf.reduce_max(sm2_probs_all, axis=1)
        
        temp_df["sm2_probs"] = sm2_probs
        
        temp_df["sm2_class"] = sm2_class
        
        temp_df["sm2_correct"] = temp_df.ys == temp_df.sm2_class
        
        if difference:
            
            temp_df["subset"] = (temp_df.sm2_correct != temp_df.sm3_correct) | (temp_df["subset"])
            
        else:
            
            temp_df["subset"] = True
        
        ##print("got sm2 set")
        
    #"""
    all_imgs = []
    
    imgs_sub = imgs.numpy()[temp_df["subset"].tolist(), :, :, :]
        
    for x in range(imgs_sub.shape[0]):
    
        all_imgs.append(imgs_sub[x,:,:,:])
        
    ##print("got images set")
    
        
    if difference:
            
        temp_df = temp_df.loc[temp_df["subset"], :]
        
    temp_df["imgs"] = all_imgs
    #"""
    
    return temp_df

def grab_different_imgs(model, dataset_supplier, difference=True):
    
    (data_set, len_data) = dataset_supplier(0)
    
    flag_first_batch = 1
    
    for (imgs, ys) in data_set:
        
        curr_df = _get_batch_difference_df(model, imgs, ys, difference)
        
        if flag_first_batch==1:
            
            complete_df = curr_df
            
        else:
            
            complete_df = pd.concat([complete_df, curr_df], axis=0, ignore_index=True)
            
        
        flag_first_batch = 0
    
    return complete_df


def plot_diff(df, sm_col="sm2_correct", main_col="sm3_correct",
              img_col="imgs",
              true_col="ys", 
              pred_col="sm2_class",
              pred_col2="sm3_class",
              prob_col1="sm2_probs",
              prob_col2="sm3_probs",
              ncols=5,
              denormalize=False
              ):
    """
    This function plots images where there was different predictions b/w the provided classes
    
    takes diff_df, main_col = sm3 class column, sm_col = smoftmax inconsideration column
    """
    
    class_names = ['airplane','automobile','bird','cat','deer',
               'dog','frog','horse','ship','truck']
    
    #Take only those rows where there is a difference
    
    sub_df = df.loc[df[main_col] != df[sm_col], :].reset_index(drop=True)
    
    #Separate where main layer is correct but not the sm layer, and vice-versa
    sub_df_main_correct = sub_df.loc[sub_df[main_col],:].reset_index(drop=True)
    
    sub_df_sm_correct = sub_df.loc[sub_df[sm_col],:].reset_index(drop=True)
    
    classwise_display([sub_df_main_correct, sub_df_sm_correct], 
                      img_col=img_col,
                      true_col=true_col, 
                      pred_col=pred_col, 
                      pred_col2=pred_col2,
                      prob_col1=prob_col1,
                      prob_col2=prob_col2,
                      ncols=ncols, 
                      class_map=class_names,
                      denormalize=denormalize,
                      message_list = ["images that are correct at last softmax but not at earlier ones",
                                      "images that are correct at earlier softmax but not at last softmax"])
    
    

def classwise_display(df_list, img_col, true_col, pred_col, 
                      ncols=5,
                      class_map=None,
                      pred_col2=None,
                      prob_col1=None,
                      prob_col2=None,
                      denormalize=False,
                      message_list = None):
    
    """
    Plots images in a grid class wise. Loops over each class in the df provided and plots
    
    Inputs:
        df_list - pandas DataFrame list, that has image array(numpy),truelabel, predicted label, etc..(refer other arguments)
        All dfs in the list must have similar structure(same column names and the format of values in them)
        
        img_col - column name of numpy array images
        
        true_col - column containing ground truth labels
        
        pred_col - column containing predicted labels
        
        ncols - number of images to display in a row
        
        class_map(required) - A list label names to be used to map the label numbers in true label and predicted label columns
        
        pred_col2(optional) - second predicted label column name
        
        prob_col1, prob_col2 (optional)- column names having probability values
        
        denormalize - Flag to denormalize image
    """
    
    all_classes = df_list[0].loc[:, true_col].unique()
    
    if message_list == None:
        
        message_list = [""]* len(df_list)
    
    for clss in all_classes:
        
        df_counter = 0
        
        for df in df_list:
            
            
            sub_df = df.loc[df[true_col]==clss, :].reset_index(drop=True)
            
            print("\nClass: {}-{}".format(class_map[clss], 
                  message_list[df_counter % len(df_list) ])+"\n")
            
            df_counter += 1
            
            if sub_df.shape[0] < 1:
                
                print("\n----NO IMAGES AVAIABLE FOR THIS SECTION-----")
                
                continue
            
            
            sub_df = sub_df.loc[list(range(ncols)),:].dropna()
            
            sub_df.sort_values(prob_col1, ascending=False, inplace=True)
            
            preds_mapped = [class_map[x] for x in sub_df.loc[:,pred_col].astype(int).tolist()]
            
            true_mapped = [class_map[x] for x in sub_df.loc[:,true_col].astype(int).tolist()]
            
            if pred_col2==None:
                preds_mapped2 = None
            else:
                preds_mapped2 = [class_map[x] for x in sub_df.loc[:,pred_col2].astype(int).tolist()]
                
            if prob_col1==None:
                prob1 = None
            else:
                prob1 = sub_df.loc[:,prob_col1].tolist()
                
            if prob_col2==None:
                prob2 = None
            else:
                prob2 = sub_df.loc[:,prob_col2].tolist()
            
            image_gallary(no_of_image = sub_df.shape[0], 
                          img = sub_df.loc[:,img_col].tolist(), 
                          img_lbl = true_mapped, 
                          pred_lbl = preds_mapped,
                          pred_lbl2=preds_mapped2,
                          prob1=prob1,
                          prob2=prob2,
                          denormalize=denormalize,
                          NUM_IMAGES_IN_ROW = ncols)
    
    
    return

def plot_good_and_worst(df, sm_col="sm2_correct",
              img_col="imgs",
              true_col="ys", 
              pred_col="sm2_class",
              prob_col="sm2_probs",
              ncols=5,
              denormalize=False,
              CLASSWISE_SELECT_TOP_IMAGES = 20
              ):
    """
    This function plots images which were classified correctly with high conf and classified wrong with high confidence(of being correct)
    
    takes diff_df, main_col = sm3 class column, sm_col = smoftmax inconsideration column
    """
    
    class_names = ['airplane','automobile','bird','cat','deer',
               'dog','frog','horse','ship','truck']
    
    #Take only those rows where there is a difference
    
    correct_df = df.loc[df[sm_col], :]
    
    high_conf_correct_df = correct_df.sort_values([pred_col, prob_col], ascending=False).groupby(pred_col).head(CLASSWISE_SELECT_TOP_IMAGES*20)
    
    incorrect_df = df.loc[~(df[sm_col]), :]
    
    high_conf_incorrect_df = incorrect_df.sort_values([pred_col, prob_col], ascending=False).groupby(pred_col).head(CLASSWISE_SELECT_TOP_IMAGES*20)
    
    print(high_conf_incorrect_df.shape, high_conf_correct_df.shape)
    
    classwise_display([high_conf_correct_df, high_conf_incorrect_df], 
                      img_col=img_col,
                      true_col=true_col, 
                      pred_col=pred_col,
                      pred_col2=pred_col,
                      prob_col1=prob_col,
                      ncols=ncols, 
                      class_map=class_names,
                      denormalize=denormalize,
                      message_list = ["Correct with high confidence", "Incorrect with high confidence"])
    



def plot_cm(all_y, all_preds, class_names=['airplane','automobile','bird','cat', 'deer','dog','frog','horse','ship','truck']):
    
    cm = confusion_matrix(all_y, all_preds)
    
    df_cm = pd.DataFrame(cm, index = [i for i in class_names],
                  columns = [i for i in class_names])
    
    plt.figure(figsize = (10,7))
    
    sn.heatmap(df_cm, annot=True)
    
    plt.show()
    
    return
