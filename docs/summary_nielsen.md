# Neural Networks and Deep Learning
M. A. Nielsen  
*Summary by Yvan Richard*

## Table of Contents

1. [Using Neural Nets to Recognize Handwritten Digits](#1-using-neural-nets-to-recognize-handwritten-digits)

## 1. Using Neural Nets to Recognize Handwritten Digits

### 1.1. Introduction

The ability that humans have to recognize handwritten digits or more
generally to *classify* the objects they are seeing into
distinct categories with an extremely high accuracy is the typical kind
of task that might be puzzling for a computer program to reproduce.

Naively, we could encode hard-coded patterns into a program that will then
be "able" to identify them on multiple instances of the data set. However, this
approach would be extremely tedious and our results' sensitivity to different
handwriting styles quite significant.

Neural networks, a biologically inspired algorithm, proceed differently. They
use few hundreds of **training instances** (i.e. labeled instances of handwritten digits) to **train** themselves to make accurate classification prediction. 

> **Personal Note to the Reader**   
>
> In machine learning, we might face a great variety of tasks. Two common tasks we often encounter are so-called **regression** or **classification** tasks. In a regression (e.g. linear regression with Ordinary Least Squares) we predict a *numerical* outcomes while in classification we predict a *categorical* outcomes (e.g. dog or cat). The task we tackle in this first chapter is a multi-classification tasks (as oposed to a binary classification task where we only have two labels).

Hence, in this first chapter we will explain how they are structured and how they can be implemented as a computer program.

### 1.2. Perceptrons

A perceptron can be considered as one of the building blocks of a larger
neural network. Basically, a perceptron takes a real valued vector $\mathbf{x}$ as input and produces a single binary output $y$:

![Perceptron](/docs/images_doc/perceptron.png)

As illustrated in this image, the input vector is then reduced to a scalar through a weighted sum with weight vector $\mathbf{w}$:

$$
\sum_{i = 0}^{n} x_i w_i = \mathbf{x}^T\mathbf{w}
$$

In general, we might have a **bias** by introducing a dummy input in $\mathbf{x}$
where $x_0 = 1$, as displayed in the above scheme. Once the weighted sum has been computed, we transform it with an **activation** function. In the case of the
perceptron, that activation function is a step function:

$$
y = 
\begin{cases}
0 \quad \text{ if } \quad \mathbf{x}^T\mathbf{w} \leq 0 \\
1 \quad \text{ else }
\end{cases}
$$

This is a very decision making model. By varying the weights vector $\mathbf{w}$ and the threshold (above $0$ is classicaly used), we can obtain different decision results. Here, we note that the higher the **bias** the more likely we will trigger our perceptron to activate/fire (i.e. output $1$). Now, how
are we supposed to know which weights to input, what magnitude the bias should take or where to set the threshold to make accurate decisions? Well, this is exactly where **learning algorithms** are needed.

### 1.3. Sigmoid Neurons

As we just discovered, we must devise a learning algorithm to tune the weights, bias, and threshold of our neural network. In this section, we add a different building block of neural networks: **sigmoid neurons**. They function excatly as
perceptrons but their *activation function* is not the step function anymore but
the **real valued sigmoid function**:

$$
\sigma(z) = \frac{1}{1 + e^{-z}}, \qquad \sigma : \mathbb{R} \to (0,1).
$$

This is a mapping of the real line onto the real interval $(0, 1)$. In our case,
we have:

$$
z = \sum_{i = 0}^{n} w_i x_i, \quad x_0 = 1, w_0 = b
$$

or expressed differently:

$$
z = b + \sum_{i = 1}^{n} x_i w_i
$$

When we look at the shape of the sigmoid function:

![Sigmoid Function](/images/sigmoid_function.png)

we realize that it could assimilated to a smoothed version of the step function.
When the weighted sum $z \to - \infty$ we obtain $0$ and when $z \to \infty$ we have $1$. This smoother change implies that the gradient of the output with respect to $\mathbf{w}$ where $w_0 = b$ is lower, thus the change is less abrupt than when we use the step function. (This is also an issue that trigger the *vanishing gradient problem*). The gradient is:

$$
\begin{align*}
\nabla_{\mathbf{w}} \sigma(z) 
&= \nabla_{\mathbf{w}} \left( \frac{1}{1 + e^{-\mathbf{w}^T \mathbf{x}}} \right) \\
&= \frac{\mathbf{x} e^{-\mathbf{w}^T \mathbf{x}}}{\left(1 + e^{-\mathbf{w}^T \mathbf{x}}\right)^2} \\
&= \left(\frac{1}{1 + e^{-\mathbf{w}^T \mathbf{x}}}\right)\left(1 - \frac{1}{1 + e^{-\mathbf{w}^T \mathbf{x}}}\right)\mathbf{x} \\
&= \sigma(z)\bigl(1 - \sigma(z)\bigr)\,\mathbf{x}.
\end{align*}
$$

### 1.4. The Architecture of Neural Networks

In this section, we introduce the useful terminology to understand neural
networks. Generally, a neural network is represented as this:

![neural net schema](/docs/images_doc/nn_schema.png)

The leftmost layer where each node value is given by the component of $\mathbf{x}$
is called the **input layer**. The rightmost layer contains the output of our neural network; this is the **output layer**. The middle layers are generally 
referred to as **hidden layers**.

Usually, the design of input and output layers is quite straightforward based on
the task we want to accomplish. However, the **width** (number of nodes in each hidden layer) and the **depth** (number of hidden layers) is quite hard to fine tune (these are referred to as some **hyperparameters** of our neural network).

Furthermore, in our approach of NNs, we consider them as being **feedorward** neural networks. This means there are no loops in the network: information is always fed forward, never fed back.

### 1.5. A Simple Network to Classify Handwritten Digits

In this first version of the program, we focus on classifying single digits
and not a string of digits (this would involve a segmentation step first).
To solve our problem, the structure of our neural network will be:

$$
(784) \rightarrow (15) \rightarrow (10)
$$

where each bracket indicates a layer and the number of nodes in it. In the input layer, we have $784$ input nodes since the images we will classify are black and white (i.e. take values between $0$ and $255$) and formatted as $28 \times 28$ pixels so $28^2 = 784$ pixels in total. In the hidden layer, we might experiment with various values of $n$. For the output layer, we number the output neurons from $0$ through $9$, and figure out which neuron has the highest activation value. If that neuron is, say, neuron number $6$, then our network will guess that the input digit was a $6$.

### 1.6. Learning with Gradient Descent

Now that we have a design for our neural network, how can it learn to recognize digits? The first thing we'll need is a data set to learn from - a so-called training data set. We'll use the `mnist` data set. Here are a few samples:

![mnist samples](/images/mnist_samples.png)

The reader can found the code used to load the `mnist` data set and generate the
above plot in the notebook [load_mnis.ipynb](/notebooks/data/load_mnist.ipynb).

**The Output of our Neural Network**

Once we feed (forward) our NN with the different inputs, we will obtain, in the output layer something like this:

$$
\hat y = 
\begin{bmatrix}
y_0 \\
y_1 \\
y_2 \\
y_3 \\
y_4 \\
y_5 \\
y_6 \\
y_7 \\
y_8 \\
y_9
\end{bmatrix}
$$

and, if the class is $i$, we want $\hat y_i$ to be as high as possible and all the others $\hat y_j$, where $j \neq i$ to be as low as possible. I use a hat on top of $y$ to indicate that this is an *estimated* vector.

**Evaluating the Output**

Once we obtained an estimate of the output vector $\hat y$, we want
to compare it to the correct output vector $y$. This one is a binary vector that could be equal to:

$$
y^T =
\begin{bmatrix}
0 & 0 & 1 & 0 & 0 & 0 & 0 & 0 & 0 & 0
\end{bmatrix}
$$

meaning here that the true class, the **label** is $2$. Now, to quantify how our estimated vector $\hat y$ performed on a particular instance of the data, we will use a **cost function** $C$ (sometimes called loss function) that takes high value if our estimation $\hat y$ is bad and the value $0$ if it is perfect. Our cost function will be the traditional **mean squared error** (MSE) function:

$$
C(y^{(i)}, \hat y^{(i)}) = \frac{1}{n} \sum_{i = 0}^{n} \left( \lVert y^{(i)} - \hat y^{(i)} \rVert \right)^2
$$

here, the index $^{(i)}$ indicates that this is the instance $i$ of the data, not the $i$-th component of the vector and $\lVert v \rVert$
is simply the euclidean norm of the vector $v$.

Now, the optimization problem we face is:

$$
\min_{\theta} \; C(\theta)
= \min_{W,\, b} \; \frac{1}{n} \sum_{i=1}^{n} \lVert y^{(i)} - \hat y^{(i)}(\theta) \rVert^2 .
$$

where $\theta = (W, b)$, $b$ is the bias, and $W$ is the weight matrix of the neural network.
When training our NN with mean-squared error, we compute the cost for each image, average over the training batch, and adjust all weights and biases jointly.

We use the MSE function because it is a quadratic function continuously differentiable with a unique minimum. These are nice properties for what is coming next.
This choice might not be the best but it is sufficient for know.

**Gradient Descent**

We know from calculus that the gradient of a function $\nabla f$ indicates the direction of the steepest ascent. For instance:

$$
f(x, y) = x^2 + y^2
$$

we compute the gradient as:

$$
\nabla f =
\begin{bmatrix}
2x \\
2y
\end{bmatrix}
$$

and if we are at $(2, 2)$ in the $(x, y)$ plane, then, we know that the direction of the steepest ascent is given by:

$$
\nabla f (1, 1) =
\begin{bmatrix}
4 \\
4
\end{bmatrix}
= v_1
$$

To get the steepest descent, we just take the negative of the gradient (recall that we want to minimize our cost function, i.e. go towards the minimum region).
Visually, I simulated this with a heatmap (code in [GD.ipynb](/notebooks/theory/GD.ipynb)):

![](/images/gradient_descent_heatmap.png)

Here we are taking the negative of the gradient and pointing in the direction of the steepest descent. To minimize our cost function $C$ in function of $\theta$ we are basically going to use the same methodology only this time we scale with a scalar $\eta$ the gradient to adjust the magnitude of our steps towards a minimum.
$\eta$ is the **learning rate** of our NN and it is also a hyperparameter. At this point, it should be mentioned that the literature covers many ways in which we should perform gradient descent. We limit ourselves to this approach for the moment being.

As for now, it is important to understand that we are going update $\theta$ whith this algorithm:

$$
\theta_{t + 1} = \theta_t - \eta \cdot \nabla_{\theta} C(\theta_t)
$$

where $\theta_{t + 1}$ is the "updated" version of $\theta_t$. In the simple gradient descent approach we have:

$$
\theta_{t+1} = \theta_t - \eta\, \frac{1}{n}\sum_{i=1}^n \nabla_\theta C^{(i)}
$$

for each time step / updates. This means that we compute the gradient $\nabla_\theta C^{(i)}$ for each training instance $(i)$.
Unfortunately, when the number of training inputs is very large this can take a long time, and learning thus occurs slowly. Furthermore, this approach is risky since it easily ends up in a local minima instead of a global one.
Below is a solution generally favored to the classic gradient descent approach.

**Stochastic Gradient Descent**

Unlike full-batch gradient descent—where the gradient is computed using the entire training set—**stochastic gradient descent (SGD)** updates the parameters using only a randomly selected subset of the data at each iteration. This reduces computational cost per update and introduces beneficial noise that can help escape shallow minima. When the subset size $m$ satisfies $1 < m < n$, we speak of **mini-batch gradient descent**.

Suppose we update the parameter vector $\theta$ over $T$ epochs. At iteration $t$, the procedure is:

1. **Random mini-batch selection**  
   Shuffle the dataset at the start of each epoch and select a subset of $m$ training instances,  

$$
X_t = \{x_t^{(1)},\ldots,x_t^{(m)}\}.
$$  

   Here $m \le n$ is the **batch size**, and each $x_t^{(i)}$ is a single training example.

2. **Mini-batch gradient computation**  
   Estimate the gradient of the cost using only the mini-batch:

$$
\nabla C(\theta_t)
= \frac{1}{m}\sum_{i=1}^{m} 
\nabla_{\theta} C\bigl(x_t^{(i)}, \theta_t \bigr).
$$

3. **Parameter update rule**  
   Update the parameters in the direction of steepest descent with learning rate \$\eta$:

$$
\theta_{t+1}
= \theta_t - \eta\,\nabla C(\theta_t).
$$

4. **Iteration / early stopping**  
   Repeat until $t = T$, or stop earlier if a convergence criterion is satisfied  
   (e.g., $\|\nabla C(\theta_t)\|$ becomes small or validation loss stops improving).

This stochastic approximation makes each update cheaper and often improves generalization by preventing the optimizer from settling too quickly into sharp minima. In practice, stochastic gradient descent is a commonly used and powerful technique for learning in neural networks.

### 1.7. Implementing our Network to Classify Digits

In this section, we will present the implementation of a program that relies on neural networks and stochastic gradient descent to recognize handwritten digits from the `mnist` data set. I try to consistently use the following notation:

- $k$: lowercase letters are scalar.
- $\mathbf{v}$: lowercase letters in bold are vectors.
- $A$: uppercase letters are matrices

Specifically for neural networks ($i \in \{1, \cdots, n\}$) :

- $\mathbf{b}^{(i)}$: bias vector for layer $i$
- $\mathbf{x}$: input vector without bias
- $\mathbf{a}^{(i)}$: vector for hidden layer $i$
- $W^{(i)}$: weight matrix connecting layers $(i - 1), i$
- $b_j^{(i)}$: is the bias for neuron $j$ in layer $i$
- $w_{jk}^{(i)}$: weight connecting neuron $k$ in layer $(i - 1)$ and neuron $j$ in layer $i$.
- $\mathbf{\hat y}, \mathbf{y}$: estimated output vector, output vector


#### The Foundations of the `Network` Class

With the concepts we have discovered above, we are already able to code the structure (i.e. member variables) of a
class called `Network` (for those reading these lines, I recommend checking what *OOP* is if the concept of class is unclear).
Then, we will also code some methods for this class.

First, we create the skeleton of the `Network` class member variable:

```python
class Network:
    """
    A simple neural network class inspired by Michael Nielsen's implementation.
    """
    # constructor: take numpy array as input for the layers' size
    def __init__(self, sizes):
        self.num_layers_ = len(sizes)
        self.sizes_ = sizes

        # randomly generated biases (from the standard Gaussian distribution)
        # bias generated for the hidden layers and output layer
        self.bias_ = [np.random.randn(y, 1) for y in sizes[1:]] 

        # generate the weights
        self.weights_ = [np.random.randn(y, x) for x, y
                            in zip(sizes[:-1], sizes[1:])]
```

As one can see, several member variables are initialized in our `Network` object.
We pass to the constructor a numpy array named `sizes` as an argument. Then, we
register the number of layers in `num_layers_` and the number of neurnones in each layer
in `sizes_`. Once this is done, we need to generate (with list comprehension) the bias vectors for each layer except the input layer. We proceed to create the `weights_` in roughly the same way. At this stage, it is important to note that for instance:

```python
net = Network([784, 15, 10])
W_1 = net.weights_[0]
```

In this context, $W^{(1)}$ is going to be a matrix of weights connecting the input layer $\mathbf{x}$ and the first hidden layer $\mathbf{a}^{(1)}$. For example, if the input layer $\mathbf{x}$ has $4$ neurons and the first hidden layer $\mathbf{a}^{(1)}$ has $3$
neurons, like on this schema I realized:

![neural net schema](/docs/images_doc/nn_schema_yvan.jpeg)


we have:

$$
W^{(1)} =
\begin{bmatrix}
w_{11} & w_{12} & w_{13} & w_{14} \\
w_{21} & w_{22} & w_{23} & w_{24} \\
w_{31} & w_{32} & w_{33} & w_{34}
\end{bmatrix}
$$

so we see that the first column of $W^{(1)}$ are the weights that connect the first neuron in the input layer (indexed as layer $0$ in my notation) to the first hidden layer
$\mathbf{a}^{(1)}$. This means that:

$$
\mathbf{a^{(1)}} = 
\begin{bmatrix}
a_1^{(1)}  \\
a_2^{(1)}  \\
a_3^{(1)} \\
\end{bmatrix}
= W^{(1)}\mathbf{x} + b^{(1)}
$$

where we have:

$$
a_1^{(1)} = \sum_{i = 1}^{4} w_{i,1}x_i + b^{(1)}_1
$$

This means that the first node of the first hidden layer: $a_1^{(1)}$, is a weighted sum
of each neuron from the input layer $\mathbf{x}$ plus the bias. As you can see, one might need a little bit of time to accomodate to the notation. Then, once this has been established, the output layer ${\mathbf{y}}$ is obtained through:

$$
\mathbf{y} = \sigma\bigl(W^{(2)} \mathbf{a}^{(1)} + \mathbf{b}^{(2)}\bigr)
$$

where $\sigma(\cdot)$ is applied elementwise (we vectorize the function). In the output
layer, the sigmoid can be replaced by a softmax function. This mechanism is exactly what our `feedforward()`
method of the class `Network` will accomplish:

```python
# feedforward method
def feedforward(self, a):
   """Return the output of the network if ``a`` is input."""
   for W, b in zip(self.weights_, self.bias_):
      a = W @ a + b
      a = utils.sigmoid(a)

   return a
```

Once that this method has been created, we build one of the piece of the learning algorithm, the method `SGD`

```python
# Stochastic Gradient Descent method
def SGD(self, training_data, epochs, mini_batch_size, eta, test_data=None):
"""Train the neural network using mini-batch stochastic
gradient descent.  The ``training_data`` is a list of tuples
``(x, y)`` representing the training inputs and the desired
outputs.  The other non-optional parameters are
self-explanatory.  If ``test_data`` is provided then the
network will be evaluated against the test data after each
epoch, and partial progress printed out.  This is useful for
tracking progress, but slows things down substantially."""
if test_data: n_test = len(test_data)
n = len(training_data)
for j in range(epochs):
   # shuffle the training data before creating mini-batches
   rd.shuffle(training_data)
   
   # create the mini-batches
   mini_batches = [
         training_data[k:k+mini_batch_size]
         for k in range(0, n, mini_batch_size)]
   
   # update the parameters for each mini-batch
   for mini_batch in mini_batches:
         self.update_mini_batch(mini_batch, eta)
   
   # evaluate the network on the test data
   if test_data:
         print(f"Epoch {j}: {self.evaluate(test_data)} / {n_test}")
         print(f"Epoch {j} complete")

```

The attentive reader will note that we are using two other methods in this function: `evaluate()` and `update_mini_batch()`.
To understand how the second method is coded, we need to wait until the next chapter on backpropagation. However, the implementation of `evaluate()` is fairly simple. We have the test data populated of tupples `(x, y)` with the features (i.e. the $784$ input pixels, and the target $y$, which is an int corresponding to the label of the digit). Hence, we implement the function as:

```python
# evaluate
    def evaluate(self, test_data):
        """
        Return the number of test inputs for which the neural
        network outputs the correct result. Note that the neural
        network's output is assumed to be the index of whichever
        neuron in the final layer has the highest activation.
        """
        # compute the test_results (x, y) is a tupple
        test_results = [(np.argmax(self.feedforward(x)), y) for (x, y) in test_data]

        # compute the number of correct predictions
        return sum(int(x == y) for (x, y) in test_results)
```

Up to this point, we have already made quite some progress but we are not there yet. We still need to have an algorithm able to correctly update the weights and biases of the Network based on the cost function. This is the focus of the second chapter.


## 2. The Backpropagation Algorithm

In this section, we uncover how the backpropagation algorithm. Subsequently, we will be able to implement the methods that are missing to our class `Network` for tackling the classification problem we face.




## References

- Nielsen, M. A. (2015). *Neural Networks and Deep Learning*. Determination Press. http://neuralnetworksanddeeplearning.com/