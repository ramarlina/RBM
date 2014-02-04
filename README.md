RBM
===

This is a simple Python implementation of Restricted Boltzmann Machine. 
It handles Numpy arrays as well as Scipy sparse matrices. 
I am using the moment method to update the weights.

To use this module, make sure the following Python libraries are installed:
    - Numpy
    - Scipy
    - PIL
    - Matplotlib
    
Example: The following example shows how rbm.py can be used to train two consecutive layers of a neural network using an arbitrary data vector X:
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
    
