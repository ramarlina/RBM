import rbm
import numpy


def load_data():
    return numpy.random.uniform(0, 1, (1000, 20))
    
if __name__=="__main__":
    X = load_data()
    # create the restricted boltzmann machine trainer
    r = rbm.RBM(20, 30)
    # train the RBM, run 1000 iterations
    r.fit(X, 1000)
     