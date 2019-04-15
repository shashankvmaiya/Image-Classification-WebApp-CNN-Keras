# Web Application for MNIST Digit Classification using Convolutional Neural Network and Keras

Please visit [smaiya-mnist-cnn](https://smaiya-mnist-cnn.appspot.com/) for the Web Application. Below is a snippet from the web application. The application outputs the predicted digit. It also provides the confidence and other possibilities, if the confidence level of the prediction is < 99%
<src="./Images/results/web_app_4.png"/>
<src="./Images/results/web_app_9.png"/>

## Modeling
We train a handwritten digit classifier using the MNIST data set with Convolutional Neural Network using Keras with tensorflow backend. The model achieves an accuracy of 99.4%. The most common mis-classified digit are from 9 to 4. 
Below is the block diagram of the CNN that is used to train the model

<src="./Images/cnn_block_diagram.png"/>

## Model Performance
Below are few selected images for which the model correctly and incorrectly predicts the digit. We can see how 9 can be mistaken for a 4 even to a common person from these images

<src="./Images/results/model_perf.png"/>
