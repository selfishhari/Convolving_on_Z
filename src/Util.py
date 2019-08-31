from keras.applications import preprocess_input, decode_predictions
from keras.preprocessing import image
import numpy as np
import cv2
from keras import backend as K
from google.colab.patches import cv2_imshow
from matplotlib import pyplot as plt
%matplotlib inline

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
  y_pred = model.predict(test_images)
  # getting index of all misclasified images
  img_index = []
  for i in range(1,test_images.shape[0]):
    if np.argmax(y_pred[i])!=np.argmax(test_labels[i]):
      img_index.append(i)



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
def image_gallary(no_of_image, img, img_lbl, pred_lbl = None):
  #no_of_image: give no of image as multiple of 5
  if no_of_image<5:
    row = 1
    col = no_of_image
  else:
    row = no_of_image//5
    col = 5

  fig=plt.figure(figsize=(18,15))
  for i in range(0,row*col):
    fig.add_subplot(row,col,i+1)
    plt.imshow(img[i]) 
    if pred_lbl == None:
      plt.title('Actual: '+str(img_lbl[i]))
    else:
      plt.title('Actual: '+str(img_lbl[i])+' & Predicted: '+str(pred_lbl[i]))
    plt.xticks([])
    plt.yticks([])
  plt.show()


#create gradcam gallary with original image
def  gradCAM_galary_set(no_of_image, class_names image_set, img_lbl, img_index, model, layerName, No_of_Channel_in_Layer):
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
  
  