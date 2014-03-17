Restricted Boltzmann Machines
===

This is a simple Python implementation of Restricted Boltzmann Machine (RBM). 
It handles Numpy arrays as well as Scipy sparse matrices. 
I am using the moment method to update the weights.

To use this module, make sure the following Python libraries are installed:
<ul>
    <li>Numpy</li>
    <li>Scipy</li>
    <li>PIL</li>
    <li>Matplotlib</li>
</ul> 
    
Given an input vector X, the RBM algorithm learns a set of internal values H that would produce the data X.


    
<h3>Sample Features Learned:</h3>
The following is an example of Weights rbm library learned from grayscaled Galaxy images from the <a href="http://www.kaggle.com/c/galaxy-zoo-the-galaxy-challenge">Kaggle Galaxy Zoo challenge</a>:

<img src="https://raw2.github.com/ramarlina/RBM/master/images/rbm_results_5000_784x322.png"/>

<h3>Deep Neural Networks</h3>
This module can be used to build deep neural networks by training one layer after another.

The following example shows how to train two consecutive layers of a neural network using an arbitrary data vector X:
    
    # import the rbm library
    import rbm
    
    # The first step is to initialize the RBM:
    r1 = rbm.RBM(<input_size>, <output_size>)
    
    # Next train the RBM layer using the "fit" method:
    r1.fit(X, max_epochs=1000)
    

    # Call "activate" to get the output of a given layer
    V = r1.activate(X)
    
    # Now, a second layer can be built from the output of the first layer:
    r2 = rbm.RBM(V.shape[1], <output_size>)
    r2.fit(V, max_epochs=1000)
    
<img src="https://docs.google.com/drawings/d/1judnrIeLx5cz6P4sGIwDPRKqL1-lCvFFQyExAkAXlf8/pub?w=480&amp;h=360">


