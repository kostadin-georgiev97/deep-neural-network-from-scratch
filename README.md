# Deep Neural Network From Scratch
Implementation of DNN with Early Stopping from scratch in Python. The evaluation was done on two simple datasets (Blobs and Moons) and on one more challenging dataset (Fashion-MNIST).

## Results

| Dataset | Model | n_hidden | max_iterations | early_stopping | early_stopping_interval | Accuracy |
| ------- | ----- | -------- | -------------- | -------------- | ----------------------- | -------- |
| Blobs | LogisticRegressionSGD | N/A |  10,000 |  False | N/A | 100.0 |
| Blobs | ShallowNeuralNetworkSGD | (10,) | 10,000 | False |  N/A | 100.0 |
| Moons | LogisticRegressionSGD | N/A |  10,000 |  False |  N/A | 90.7 |
| Moons | ShallowNeuralNetworkSGD | (20,) | 200,0000 |  False |  N/A | 97.3 |
| Fashion-MNIST | ShallowNeuralNetworkSGD | (8,) | 100 epochs |  False |  N/A | 99.1 |
| Fashion-MNIST | DeepNeuralNetworkSGD | (8,4) | 500 epochs | True | 100 | 98.8 |

From the results of the different models on the different datasets, I have made the following observations:

1. The Blobs dataset is very easy because it is linearly separable. That is why it was very easy to achieve 100% accuracy regardless of the model. I would expect to get the same performance even with classical transparent Machine Learning techniques.

2. The Moons dataset appeared to be a bit more difficult. Because it isn't linearly separable it is impossible to achieve good results with the first model. We can see a considerable improvement of nearly 7% by using the Shallow Neural Network. I suppose with better param tuning even higher results might be possible but I didn't want to spend unnecessary time on the problem.

3. The Fashion-MNIST dataset is indeed much more complex, but due to the large number of available samples and the even distribution of classes the Shallow Neural Network was able to achieve a very high accuracy of 99.1%. However, by looking at the learning curve could see how the performance on the validation set plateaued really fast in the first couple of epochs. There is no guarantee how well is the model generalized, so that may lead to potentially worse performance in the real-world. On the other hand, DeepNeuralNetworkSGD with Early Stopping achieved a little lower accuracy by just a 0.3 difference. Nevertheless, it could still potentially outperform the former approach in a real-world scenario because it is supposed to be better generalized.
