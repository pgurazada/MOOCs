import matplotlib.pyplot as plt
import seaborn as sns

#%matplotlib inline

sns.set_context('talk', font_scale=1.2)
sns.set_palette('gray')
sns.set_style('ticks', {'grid_color' : 0.6})

def plot_loss(fit_history, course_name):
    epochs = range(1, len(fit_history['binary_accuracy'])+1)
    
    plt.figure(figsize=(12, 6))
    
    plt.plot(epochs, fit_history['loss'], '--', label='Training loss')
    plt.plot(epochs, fit_history['val_loss'], '-', label='Validation loss')
    
    plt.title('Training and Validation loss \n(' + course_name + ')')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.show()

def plot_accuracy(fit_history, course_name):
    epochs = range(1, len(fit_history['binary_accuracy'])+1)
  
    plt.figure(figsize=(12, 6))
    
    plt.plot(epochs, fit_history['binary_accuracy'], '--', label='Training Accuracy')
    plt.plot(epochs, fit_history['val_binary_accuracy'], '-', label='Validation Accuracy')
    
    plt.title('Training and Validation accuracy \n(' + course_name + ')')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.show()

def plot_probs(pred_probs, course_name, data='training'):
    plt.figure(figsize=(12,6))
    
    plt.hist(pred_probs)
    
    plt.xlabel('Predicted probability')
    plt.ylabel('Count')
    plt.title('Distribution of predicted probabilities on the ' + data +  ' data \n (' + course_name + ')')
    
    plt.show()
