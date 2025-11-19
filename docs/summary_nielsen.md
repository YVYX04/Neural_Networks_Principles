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

## References

- Nielsen, M. A. (2015). *Neural Networks and Deep Learning*. Determination Press. http://neuralnetworksanddeeplearning.com/