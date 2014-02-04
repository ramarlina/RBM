import numpy 
from scipy.sparse.csr import csr_matrix   
from scipy import sparse as S
from matplotlib import pyplot as plt 
from PIL import Image
 

def normalize(x):
    min_val = numpy.min(x)
    max_val = numpy.max(x)
    return (x-min_val)/(max_val-min_val)
 
def make_tile(X, img_shape, tile_shape, tile_spacing=(0, 0)):
    out_shape = [(ishp + tsp) * tshp - tsp for ishp, tshp, tsp
                      in zip(img_shape, tile_shape, tile_spacing)]
    H, W = img_shape
    Hs, Ws = tile_spacing
    out_array = numpy.zeros(out_shape, dtype='uint8')
    for tile_row in xrange(tile_shape[0]):
      for tile_col in xrange(tile_shape[1]):
          if tile_row * tile_shape[1] + tile_col < X.shape[0]: 
              img = normalize(X[tile_row * tile_shape[1] + tile_col].reshape(img_shape))
              out_array[
                  tile_row * (H+Hs): tile_row * (H + Hs) + H,
                  tile_col * (W+Ws): tile_col * (W + Ws) + W
                  ] \
                  = img * 255
    return out_array

 
def sigmoid(x):
    return 1.0/(1+numpy.exp(-x))   
    
def rbm_sigmoid(x):
    return x*(x>0).astype("i")
    
def grad_sigmoid(x):
    return x*(1-x) 

def pick(y, n):
    idx = numpy.arange(len(y))
    numpy.random.shuffle(idx)
    return idx[:n]

class RBM():
    def __init__(self, num_visible=None, num_hidden=None, W=None, learning_rate = 0.1):
        self.learning_rate = learning_rate 
        if W == None:
            self.W =  numpy.random.uniform(-.1,0.1,(num_visible,  num_hidden)) / numpy.sqrt(num_visible + num_hidden)
            self.W = numpy.insert(self.W, 0, 0, axis = 1)
            self.W = numpy.insert(self.W, 0, 0, axis = 0)
        else:
            self.W=W
        self.E = []
        self.momentum = 0.5
        self.last_change = 0
        self.last_update = 0
        self.cd_steps = 1
        self.epoch = 0
        self.dropout = 0 
        
    def CD(self, phase):
        if (phase == 0):      
            self.S_hidden = self.S_visible.dot(self.W)
            self.S_hidden = sigmoid(self.S_hidden)
            #fix bias
            self.S_hidden[:,0]  = 1.0 
            
        if (phase == 1):
            self.S_visible = self.S_hidden.dot(self.W.T)        
            self.S_visible = sigmoid(self.S_visible)
            self.S_visible[:,0]  = 1.0
            
    def fit(self, Input, max_epochs = 1, batch_size=100):  
        if isinstance(Input, S.csr_matrix):
            bias = S.csr_matrix(numpy.ones((Input.shape[0], 1))) 
            csr = S.hstack([bias, Input]).tocsr()
        else:
            csr = numpy.insert(Input, 0, 1, 1) 
        for epoch in range(max_epochs): 
            idx = numpy.arange(csr.shape[0])
            numpy.random.shuffle(idx)
            idx = idx[:batch_size] 
            data = csr[idx] 
                   
            self.S_visible = data
            self.CD(0)
            pos_hidden_states = self.S_hidden > numpy.random.uniform(0,1, self.S_hidden.shape)
            pos_associations = data.T.dot(self.S_hidden)

            for i in range(self.cd_steps): 
                self.CD(1)
                neg_visible_probs = self.S_visible 
                self.CD(0) 
              
            neg_associations = neg_visible_probs.T.dot(self.S_hidden)

            # Update weights.
            self.last_change = self.learning_rate * ((pos_associations - neg_associations) / batch_size)
            w_update = self.learning_rate * ((pos_associations - neg_associations) / batch_size) 
            total_change = numpy.sum(numpy.abs(w_update))
            #if total_change > 10:
            #    w_update *= 3.0/total_change
            self.W += self.momentum * self.last_change  + w_update
  
            #self.cd_steps=self.epoch/30000 + 1
            
            Error = numpy.mean((data - neg_visible_probs)**2)**0.5
            self.E.append(Error)
            self.epoch += 1
            print "Epoch %s: RMSE = %s; ||W||: %6.1f; Sum Update: %f" % (epoch, Error, numpy.sum(numpy.abs(self.W)), total_change)  
        return self

    def learning_curve(self):
        plt.ion()
        plt.show()
        plt.plot(numpy.array(self.E))

    def view(self):
        x = int(numpy.ceil((self.W.shape[0]-1)**0.5))
        n = int(numpy.ceil(self.W.shape[1]**0.5))
        W = make_tile(self.W[1:].T, img_shape= (x,x), tile_shape=(n,n))
        filename="rbm/img/rbm_results_%d_%dx%d.png"%(self.epoch, (self.W.shape[0]-1), self.W.shape[1])
        img = Image.fromarray(W)
        img.save(filename)  
        plt.figure()
        plt.imshow(W, cmap='gray')
        plt.ion()
        plt.show()
        
    def sample(self, prob):
        return (prob > numpy.random.uniform(0,1, prob.shape)).astype("i")
     
    def get_hidden_prob(self, X):
        if isinstance(X, S.csr_matrix):
            bias = S.csr_matrix(numpy.ones((X.shape[0], 1))) 
            csr = S.hstack([bias, X]).tocsr()
        else:
            csr = numpy.insert(X, 0, 1, 1)
        p = rbm_sigmoid(csr.dot(self.W)) 
        p[:,0]  = 1.0 
        return p

    def get_visible_prob(self, H):
        if H.shape[1] == self.W.shape[0]:
            if isinstance(H, S.csr_matrix):
                bias = S.csr_matrix(numpy.ones((H.shape[0], 1))) 
                csr = S.hstack([bias, H]).tocsr()
            else:
                csr = numpy.insert(H, 0, 1, 1)
        else:
            csr = H
        p = rbm_sigmoid(csr.dot(self.W.T)) 
        p[:,0] = 1
        return p
         