Download Link: https://assignmentchef.com/product/solved-cs725-assignment-2-feed-forward-neural-network
<br>
In this assignment, you will perform ​<strong>Classification</strong>​ by implementing ​<strong>Feed Forward Neural Network </strong>with loss (like cross entropy loss) using the method of gradient descent. We will use only simple Fully Connected Layers with different activation functions stacked one after the other. Training must be done using simple Stochastic Gradient Descent using mini batches. Since only classification tasks are considered, cross-entropy loss is used after applying the softmax function to the logits obtained from the network.

For more concepts, refer to the class notes/slides.

<strong> </strong>

<h1>LIST OF TASKS</h1>

Here we will enlist the tasks that you need to complete, along with a brief description of each. Please also look at the comments provided in each function that you need to implement for further specification of the input and output formats and shapes.

The general structure of the codebase is as follows. There is a class called called FullyConnectedLayer which represents one fully connected linear layer followed by a non-linear activation function which could be a ReLU or softmax layer. The NeuralNetwork class consists of a series of FullConnectedLayers stacked one after the other, with the output of the last layer representing a probability distribution over the classes for the given input. For this reason, the activation of the last function should always be the softmax function. Both these files are defined in nn.py.

In main.py, there are 2 tasks – taskXor and taskMnist – corresponding to the 2 datasets. In this, you need to define neural networks by adding fully connected layers. The code for the XOR dataset trains the model and prints the test accuracy at the end, while the code for the MNIST dataset trains the model and then uses the trained model to make predictions on the test set. Note that the answers to the test set have not been provided for the MNIST dataset.

<h2>Task 1</h2>

You need to implement the following functions in the FullyConnectedLayer class:

<ol>

 <li>__init__: Initialise the parameters (weights and biases) as needed. This is not graded, but necessary for the rest of the assignment</li>

 <li>relu_of_X: Return ReLU(X) where X is the input</li>

 <li>softmax_of_X: Return softmax(X) where X is the input. The output of this layer now represents a probability distribution over all the output classes</li>

 <li>forwardpass: Compute the forward pass of a linear layer making use of the above 2 functions. You can store information that you compute that will be needed in the backward pass in the variable self.data</li>

 <li>gradient_relu_of_X: Assume that the output and input are represented by X and Y, respectively such that Y = ReLU(X). This function should take as input dLoss/dY and return dLoss/dX.</li>

 <li>gradient_softmax_of_X: Like gradient_relu_of_X, this function takes as input dLoss/dY and should return dLoss/dX. You should try to work the gradient out on paper first and then try to implement it in the most efficient way possible. A “for” loop over the batch is an acceptable implementation for this subtask. [Hint: An output element y_j is not dependent on only x_j, so you may need to use the Jacobian Matrix here.]</li>

 <li>backwardpass: Implement the backward pass here, using the above 2 functions. This function should only compute the gradients and store them in the appropriate member variables (which will be checked by the autograder), and not update the parameters. The function should also return the gradient with respect to its input (dLoss/dX), taking the gradient with respect to its output (dLoss/dY) as an input parameter.</li>

 <li>updateWeights: This function uses the learning rate and the stored gradients to make actual updates.</li>

</ol>




<h2>Task 2</h2>

The NeuralNetwork class already has a defined __init__ function as well as a function to add layers to the network. You need to understand these functions and implement the following functions in the class:

<ol>

 <li>crossEntropyLoss: Computes the cross entropy loss using the one-hot encoding of the groundtruth label and the output of the model.</li>

 <li>crossEntropyDelta: Computes the gradient of the loss with respect to the model predictions P</li>

</ol>

i.e. d[crossEntropy(P, Y)] / dP, where Y refers to the ground-truth labels.

<ol>

 <li>train: This function should use the batch size, learning rate and number of epochs to implement the entire training loop. Make sure that the activation used in the last layer is the softmax function, so that the output of the model is a probability distribution over the classes. You can use the validation set to compute validation accuracy at different epochs (using the member functions of the NeuralNetwork class). Feel free to print different accuracies, losses and other statistics for debugging and hyperparameter tuning. It would, however, be preferable if you commented or deleted all the print commands in your final submission.</li>

</ol>




The train function will not be graded using the autograder. You will receive the marks for the datasets (Part B) only if you have satisfactorily implemented the train function.




There are also functions to make predictions and test accuracy that should not be modified.




<h2>Task 3</h2>

Finally, in main.py you need to define appropriate neural networks which give the best accuracy on both datasets. You also need to specify all the other hyperparameters like the batch size, learning rate and the number of epochs to train for. The relevant functions are: a. taskXor

<ol>

 <li>taskMnist</li>

 <li>preprocessMnist: Perform any preprocessing you wish to do on the data here. [Hint: Some minimal basic preprocessing is needed to train with stability]</li>

</ol>

Do not modify the code in the rest of the function since it will be used to generate your final predictions.







You are encouraged to make plots of the validation / training loss versus the number of epochs for different hyperparameters and note down any interesting or unusual observations. You can submit the same in a folder called “extra” in the main directory.

Tip: In case you are getting NaN as the loss value, ensure that if you are dividing by a variable that might be 0, add a small constant to it, i.e., 1/x -&gt; 1/(x + 1e-8)




<h1> Marks Distribution</h1>

<strong> </strong>​Students will be awarded marks based on the test cases that they pass from the autograder which we will use (Hidden Test Cases) to evaluate your code. However, you are also given one small test case (Visible Test Case) which you can evaluate on your own to get an understanding of how it will work, rest all test cases will not be shared with you. You can use the following command to evaluate your code on the testcase_01 file.

python3 autograder.py




Note: your autograder should be placed along with main.py. It will check your code for just one test case, we will be using different and more complex test cases to evaluate your submission later.

Autograder will evaluate following function of your program

<table width="383">

 <tbody>

  <tr>

   <td colspan="2" width="312">1. forwardpass</td>

   <td width="71">– 4 Marks</td>

   <td width="9"> </td>

  </tr>

  <tr>

   <td colspan="2" width="312">2. forwardpass + backwardpass</td>

   <td width="71">– 6 Marks</td>

   <td width="9"> </td>

  </tr>

  <tr>

   <td colspan="2" width="312">3. updateWeights</td>

   <td width="71">– 2 Marks</td>

   <td width="9"> </td>

  </tr>

  <tr>

   <td colspan="2" width="312">4. relu_of_X</td>

   <td width="71">– 1 Marks</td>

   <td width="9"> </td>

  </tr>

  <tr>

   <td colspan="2" width="312">5. gradient_relu_of_X</td>

   <td width="71">– 1 Marks</td>

   <td width="9"> </td>

  </tr>

  <tr>

   <td colspan="2" width="312">6. softmax_of_X</td>

   <td width="71">– 3 Marks</td>

   <td width="9"> </td>

  </tr>

  <tr>

   <td colspan="2" width="312">7. gradient_softmax_of_X</td>

   <td width="71">– 7 Marks</td>

   <td width="9"> </td>

  </tr>

  <tr>

   <td colspan="2" width="312">8. crossEntropyLoss</td>

   <td width="71">– 3 Marks</td>

   <td width="9"> </td>

  </tr>

  <tr>

   <td width="24">9.</td>

   <td width="288">crossEntropyDelta</td>

   <td colspan="2" width="80">– 3 Marks</td>

  </tr>

  <tr>

   <td width="24"> </td>

   <td width="288">Total Marks from autograder</td>

   <td colspan="2" width="80">– 30 Marks</td>

  </tr>

 </tbody>

</table>

Rest 20 Marks will be based on your accuracy on different datasets which is mentioned in the next section (Datasets).

<strong> </strong>

<strong> </strong>

<strong> </strong>

<h1>Datasets</h1>

For this assignment, you will be using two different datasets :

<ul>

 <li>XOR dataset (Toy dataset) – 7 Marks</li>

 <li>MNIST dataset – 13 Marks</li>

</ul>

The marks for each dataset will be given to you based on the accuracy you achieve on test data, which will be different for both datasets.

The formula we will use to award marks to students will be as follows:

<h1><em>Obtained</em> <em>Marks</em> = <sup>()</sup></h1>

An accuracy over 97.5% and 95% on XOR and MNIST respectively is satisfactory, and easily achievable in the given time constraints. If the time constraints are violated, no marks will be awarded for the question, so make sure you set hyperparameters such that training completes well within the specified time limit. We will be using the SL2 machines to evaluate submissions.

<h2>XOR Dataset</h2>

The input vector ​[x,y​X]​. The output ​ is a list of 2-dimensional vectors. Every example ​y​i corresponding to the i<sub>​                             </sub>th​<sub>​</sub><sub> example is either a 0 or 1. The labels follow</sub>X​ ​<sub>i</sub> is represented by a 2-dimensional<sub>​                                                                                                                                                                                                                      </sub><sub>                                    </sub>

XOR-like distribution. That is, the first and third quadrant have the same label (​y​i = 1) and the second<sub>​                     </sub> and fourth quadrants have the same label (​y​i = 0) as shown in the figure below:<sub>​     </sub>







There are a total of 10000 points, and the training, validation and test splits contains 7000, 2000 and 1000 points respectively. As discussed in class, the decision boundaries can be learnt exactly for this dataset using a Neural Network. Hence, if your implementation is correct, the accuracy on the train, validation and test sets should be close to 100%

<h2>MNIST Dataset</h2>

We use the <u>​</u><a href="http://yann.lecun.com/exdb/mnist/">MNIST data set</a><u>​</u> which contains a collection of handwritten numerical digits (0-9) as 28×28-sized binary images. Therefore, input ​X​ is represented as a vector of size 784 and the number of output classes is 10 (1 for each digit). In this case, the features are the grayscale image values at each of the pixels in the image. These images have been size-normalised and centred in a fixed-size image. MNIST provides a total 70,000 examples, divided into a test set of 10,000 images and a training set of 60,000 images. In this assignment, we will carve out a validation set of 10,000 images from the MNIST training set, and use the remaining 50,000 examples for training.

Simple feedforward neural networks (consisting of fully connected layers separated by non-linear activation functions) can be used to achieve a fairly high accuracy (even over 97%), but achieving this accuracy might require some careful tuning of the hyperparameters like the number of layers, number of hidden nodes and the learning rate.

Here are a few examples from the dataset:




<strong>Have Fun ! </strong>