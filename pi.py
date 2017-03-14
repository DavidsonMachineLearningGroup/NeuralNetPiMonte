import tensorflow as tf;
import numpy as np;

BATCH_SIZE = 32
NUM_EPOCHS = 10000000; # for now, each batch is an epoch
layer1_hidden_numnodes = 2;
layer2_hidden_numnodes = 3;
num_randinputs = 2;
num_pythagorean_results = 1;

class pinet_class:
    def __init__(self, sess=None):
        self.inputs = tf.placeholder(tf.float32, shape=(BATCH_SIZE, num_randinputs));
        self.pythagorean_labels = tf.placeholder(tf.float32, shape=(BATCH_SIZE, num_pythagorean_results));
        self.fc_layers ()
        self.parameters = []
        
        # Mean squared error
        self.pythagorean_cost = tf.reduce_sum(tf.pow(self.pythagorean_outputneuron-self.pythagorean_labels, 2))/(BATCH_SIZE);
        self.pythagorean_optimizer = tf.train.GradientDescentOptimizer(0.001).minimize(self.pythagorean_cost);
        
    def fc_layers(self):
        with tf.name_scope('opt1/hidden1') as scope:
          self.weights1 = tf.Variable(tf.truncated_normal([num_randinputs, layer1_hidden_numnodes]))
          self.biases1 = tf.Variable(tf.zeros([layer1_hidden_numnodes]))
          self.hidden1 = tf.nn.relu(tf.matmul(self.inputs, self.weights1) + self.biases1)
          
        with tf.name_scope('opt1/hidden2') as scope:
          self.weights2 = tf.Variable(tf.truncated_normal([layer1_hidden_numnodes, layer2_hidden_numnodes]))
          self.biases2 = tf.Variable(tf.zeros([layer2_hidden_numnodes]))
          self.hidden2 = tf.nn.relu(tf.matmul(self.hidden1, self.weights2) + self.biases2)
          
        with tf.name_scope('opt1/pythagorean_linear') as scope:
          self.weights3 = tf.Variable(tf.truncated_normal([layer2_hidden_numnodes, num_pythagorean_results]))
          self.biases3 = tf.Variable(tf.zeros([num_pythagorean_results]))
          self.pythagorean_outputneuron = tf.matmul(self.hidden2, self.weights3) + self.biases3;
          self.hidden3 = tf.nn.relu (self.pythagorean_outputneuron);
          
def GenBatch_pythagorean ():
   # generate 2 random numbers (uniform distribution) between 0 and 1 for all batch slots
   inputs = np.reshape (np.random.uniform (0, 1, size=BATCH_SIZE*num_randinputs), (BATCH_SIZE,num_randinputs));
   results = np.array ([ (a**2.0+b**2.0)**(1.0/2) for (a,b) in inputs]).reshape (BATCH_SIZE, num_pythagorean_results); # calculate c in: a^2+b^2=c^2
   return (inputs, results);
          
with tf.Session() as sess:
   pinet = pinet_class (sess);
   tf.initialize_all_variables().run()
   
   pinum = 0;
   pidenom = 0;
   
   for step in range (NUM_EPOCHS):
      batch_data, batch_labels = GenBatch_pythagorean ()
      feed_dict = {pinet.inputs : batch_data, pinet.pythagorean_labels : batch_labels}
      _, loss, pythagorean_answers = sess.run ([pinet.pythagorean_optimizer, pinet.pythagorean_cost, pinet.pythagorean_outputneuron], feed_dict=feed_dict)
      
      piprob_labels = (pythagorean_answers<=1).astype (np.int32).reshape (BATCH_SIZE,);
      pinum += sum(piprob_labels);
      pidenom += len (piprob_labels);
      
      if (step % 1000 == 0):
         print ("Step %d: Pythagorean Loss is %f") % (step, loss);
         print ("Step %d: Manual montecarlo count is %d div %d = %f") % (step, pinum, pidenom, pinum * 1.0 / pidenom * 4);
         
print ("done")