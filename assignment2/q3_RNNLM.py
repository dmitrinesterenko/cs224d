import getpass
import sys
import time

import numpy as np
from copy import deepcopy

from utils import calculate_perplexity, get_ptb_dataset, Vocab
from utils import ptb_iterator, sample

import tensorflow as tf
from tensorflow.contrib.seq2seq import sequence_loss
from model import LanguageModel
from q2_initialization import xavier_weight_init

# Let's set the parameters of our model
# http://arxiv.org/pdf/1409.2329v4.pdf shows parameters that would achieve near
# SotA numbers

# The paper mentions dropout = 0.5, minibatch = 20, 39 epochs with a learning
# rate of 1 decreased by 1.2 each epoch after the first 6
# Medium LSTM has 650 units per layer with params [-0.5, 0.5]
# Large LSTM has 1500 units per layer with params [-0.4, 0.4]
# Used gradient clipping
# learning rate of 1 decreased by 1.15 each epoch after the first 14
# They achieve perplexity of < 100 even on the non-regularized LSTMs
# and 68.7 on an averaged result of 38 large regularized LSTMs
# mention of unrolling for 20 steps (sequence length)
class Config(object):
  """Holds model hyperparams and data information.

  The config class is used to store various hyperparameters and dataset
  information parameters. Model objects are passed a Config() object at
  instantiation.
  """
  batch_size = 20 #64
  embed_size = 50
  hidden_size = 650
  num_steps = 10
  max_epochs = 39 #16
  early_stopping = 10
  dropout = 0.5 #0.9
  lr = 0.001
  # Tried a larger learning rate and on the test set get to loss of 0 pretty quickly (within 10
  # epochs but the validation set explodes so we're skipping the good min)
  #lr = 0.1

class RNNLM_Model(LanguageModel):

  def print_graph(self):
    for i in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='RNNLM'):
        print(i)

  def load_data(self, debug=False):
    """Loads starter word-vectors and train/dev/test data."""
    print("loading data")
    self.vocab = Vocab()
    self.vocab.construct(get_ptb_dataset('train'))
    self.encoded_train = np.array(
        [self.vocab.encode(word) for word in get_ptb_dataset('train')],
        dtype=np.int32)
    self.encoded_valid = np.array(
        [self.vocab.encode(word) for word in get_ptb_dataset('valid')],
        dtype=np.int32)
    self.encoded_test = np.array(
        [self.vocab.encode(word) for word in get_ptb_dataset('test')],
        dtype=np.int32)
    if debug:
      num_debug = 1024
      self.encoded_train = self.encoded_train[:num_debug]
      self.encoded_valid = self.encoded_valid[:num_debug]
      self.encoded_test = self.encoded_test[:num_debug]

  def add_placeholders(self):
    """Generate placeholder variables to represent the input tensors

    These placeholders are used as inputs by the rest of the model building
    code and will be fed data during training.  Note that when "None" is in a
    placeholder's shape, it's flexible
    Adds following nodes to the computational graph.
    (When None is in a placeholder's shape, it's flexible)

    input_placeholder: Input placeholder tensor of shape
                       (None, num_steps), type tf.int32
    labels_placeholder: Labels placeholder tensor of shape
                        (None, num_steps), type tf.float32
    dropout_placeholder: Dropout value placeholder (scalar),
                         type tf.float32

    Add these placeholders to self as the instance variables

      self.input_placeholder
      self.labels_placeholder
      self.dropout_placeholder

    (Don't change the variable names)
    """
    ### YOUR CODE HERE
    print("adding placeholders")
    self.input_placeholder = tf.placeholder(tf.int32, shape=(None, self.config.num_steps))
    # changing the dtype of labels to be tf.int32 because the sequence_loss
    # function requirese the labels to be of that type (true/false?)
    self.labels_placeholder = tf.placeholder(tf.int32, shape=(None, self.config.num_steps))
    self.dropout_placeholder = tf.placeholder(tf.float32)
    ### END YOUR CODE

  def add_embedding(self):
    """Add embedding layer.

    Hint: This layer should use the input_placeholder to index into the
          embedding.
    Hint: You might find tf.nn.embedding_lookup useful.
    Hint: You might find tf.split, tf.squeeze useful in constructing tensor inputs
    Hint: Check the last slide from the TensorFlow lecture.
    Hint: Here are the dimensions of the variables you will need to create:

      L: (len(self.vocab), embed_size)

    Returns:
      inputs: List of length num_steps, each of whose elements should be
              a tensor of shape (batch_size, embed_size).
    """
    print("embedding")
    with tf.variable_scope('embedding') as scope:
        # The embedding lookup is currently only implemented for the CPU
        with tf.device('/cpu:0'):
          ### YOUR CODE HERE
          embeddings = tf.get_variable("embeddings", \
                shape=(len(self.vocab), \
                self.config.embed_size))
          embeddings = tf.random_uniform([len(self.vocab), \
            self.config.embed_size], -1,1)
          inputs = tf.reshape(tf.nn.embedding_lookup(embeddings, \
            self.input_placeholder), (self.config.num_steps, self.config.batch_size * self.config.embed_size))
          scope.reuse_variables()
          ### END YOUR CODE
          return inputs

  def weight_init(self, name, shape):
      weight = self.xavier_initializer(shape, name)

      return weight

  def bias_init(self, name, shape):
      return tf.get_variable(name, shape, \
          tf.float32)

  def add_projection(self, rnn_outputs):
    """Adds a projection layer.

    The projection layer transforms the hidden representation to a distribution
    over the vocabulary.

    Hint: Here are the dimensions of the variables you will need to
          create

          U:   (hidden_size, len(vocab))
          b_2: (len(vocab),)

    Args:
      rnn_outputs: List of length num_steps, each of whose elements should be
                   a tensor of shape (batch_size, embed_size).
    Returns:
      outputs: List of length num_steps, each a tensor of shape
               (batch_size, len(vocab)
    """
    print("projecting")
    ### YOUR CODE HERE
    with tf.variable_scope("projection") as scope:
        weights = self.weight_init("weights", \
            (self.config.hidden_size, len(self.vocab)))
        biases = self.bias_init("biases", len(self.vocab))
        outputs = []
        for i in xrange(self.config.num_steps):
            outputs.append(tf.matmul(rnn_outputs[i], weights) + biases)
        scope.reuse_variables()
    ### END YOUR CODE
    return outputs

  def add_loss_op(self, output):
    """Adds loss ops to the computational graph.

    Hint: Use tensorflow.python.ops.seq2seq.sequence_loss to implement sequence loss.

    Args:
      output: A tensor of shape (None, self.vocab)
        Actually according to the docs this should be of the shape below
        ([batch_size x sequence_length x logits] tensor)
        https://www.tensorflow.org/api_docs/python/tf/contrib/seq2seq/sequence_loss
        But based on our result of the  projection operation we should also try
        sequence_length x batch_size x logits
    Returns:
      loss: A 0-d tensor (scalar)
    """
    ### YOUR CODE HERE
    print("calculating loss")
    with tf.variable_scope("loss_op") as scope:
        labels = tf.reshape(self.labels_placeholder, [self.config.num_steps, self.config.batch_size])
        weights = tf.ones(shape=tf.shape(labels), dtype=tf.float32, name="weights")
        loss = sequence_loss(logits=output, targets=labels, weights=weights, name="sequence_loss")
        tf.summary.scalar("loss", loss)
        scope.reuse_variables()
    ### END YOUR CODE
    return loss

  def add_training_op(self, loss):
    """Sets up the training Ops.

    Creates an optimizer and applies the gradients to all trainable variables.
    The Op returned by this function is what must be passed to the
    `sess.run()` call to cause the model to train. See

    https://www.tensorflow.org/versions/r0.7/api_docs/python/train.html#Optimizer

    for more information.

    Hint: Use tf.train.AdamOptimizer for this model.
          Calling optimizer.minimize() will return a train_op object.

    Args:
      loss: Loss tensor, from cross_entropy_loss.
    Returns:
      train_op: The Op for training.
    """
    ### YOUR CODE HERE
    print("training")
    with tf.variable_scope("training") as scope:
        train_op = tf.train.AdamOptimizer(self.config.lr).minimize(loss)
        scope.reuse_variables()
    ### END YOUR CODE
        return train_op

  def __init__(self, config):
    print("initializing")
    self.config = config
    self.xavier_initializer = xavier_weight_init()
    # Set debug=True to only grab 1024 words for train, validation and test
    self.load_data(debug=False)
    self.add_placeholders()
    self.inputs = self.add_embedding()
    self.rnn_outputs = self.add_model(self.inputs)
    self.outputs = self.add_projection(self.rnn_outputs)

    # We want to check how well we correctly predict the next word
    # We cast o to float64 as there are numerical issues at hand
    # (i.e. sum(output of softmax) = 1.00000298179 and not 1)
    self.predictions = [tf.nn.softmax(tf.cast(o, 'float64')) for o in self.outputs]
    output = tf.reshape(self.outputs, [self.config.num_steps, self.config.batch_size, len(self.vocab)])

    self.calculate_loss = self.add_loss_op(output)
    self.train_step = self.add_training_op(self.calculate_loss)




  def add_model(self, inputs):
    """Creates the RNN LM model.

    In the space provided below, you need to implement the equations for the
    RNN LM model. Note that you may NOT use built in rnn_cell functions from
    tensorflow.

    Hint: Use a zeros tensor of shape (batch_size, hidden_size) as
          initial state for the RNN. Add this to self as instance variable

          self.initial_state

          (Don't change variable name)
    Hint: Add the last RNN output to self as instance variable

          self.final_state

          (Don't change variable name)
    Hint: Make sure to apply dropout to the inputs and the outputs.
    Hint: Use a variable scope (e.g. "RNN") to define RNN variables.
    Hint: Perform an explicit for-loop over inputs. You can use
          scope.reuse_variables() to ensure that the weights used at each
          iteration (each time-step) are the same. (Make sure you don't call
          this for iteration 0 though or nothing will be initialized!)
    Hint: Here are the dimensions of the various variables you will need to
          create:

          H: (hidden_size, hidden_size)
          I: (embed_size, hidden_size)
          b_1: (hidden_size,)

    Args:
      inputs: List of length num_steps, each of whose elements should be
              a tensor of shape (batch_size, embed_size).
    Returns:
      outputs: List of length num_steps, each of whose elements should be
               a tensor of shape (batch_size, hidden_size)
    """
    ### YOUR CODE HERE
    print("modeling")
    with tf.variable_scope("model") as scope:
        hidden_weights = self.weight_init("hidden_weights", (self.config.hidden_size, self.config.hidden_size))
        weights = self.weight_init("weights", (self.config.embed_size, self.config.hidden_size))
        biases = self.bias_init("biases", (self.config.hidden_size))
        self.initial_state = tf.zeros((self.config.batch_size, self.config.hidden_size))
        h_t = self.initial_state
        rnn_outputs = []

        inputs = tf.nn.dropout(inputs, self.config.dropout)

    for i in xrange(self.config.num_steps):
        shaped_input = tf.reshape(inputs[i], \
                (self.config.batch_size, self.config.embed_size))
        h_t = tf.sigmoid(tf.matmul(h_t, hidden_weights) \
                + tf.matmul(shaped_input, weights) + biases)
        tf.summary.histogram('h_t', h_t)
        tf.Print(i, [shaped_input, h_t] , message="Step")
        rnn_outputs.append(h_t)
        scope.reuse_variables()

    self.final_state = h_t
    rnn_outputs = tf.nn.dropout(rnn_outputs, self.config.dropout)
    ### END YOUR CODE
    return rnn_outputs


  def run_epoch(self, session, data, epoch=0, train_op=None, verbose=10):
    config = self.config
    dp = config.dropout
    if not train_op:
      train_op = tf.no_op()
      dp = 1
    total_steps = sum(1 for x in ptb_iterator(data, config.batch_size, config.num_steps))
    total_loss = []
    state = self.initial_state.eval()

    for step, (x, y) in enumerate(
      ptb_iterator(data, config.batch_size, config.num_steps)):
      # We need to pass in the initial state and retrieve the final state to give
      # the RNN proper history
      feed = {self.input_placeholder: x,
              self.labels_placeholder: y,
              self.initial_state: state,
              self.dropout_placeholder: dp}
      loss, state, summary = session.run(
          [self.calculate_loss, self.final_state, train_op], feed_dict=feed)
      total_loss.append(loss)

      if verbose and step % verbose == 0:
           ## Logging
           summaries = tf.summary.merge_all()
           train_writer = tf.summary.FileWriter("./logs", session.graph)
           print("Loss is {}".format(loss))

           print('\r{} / {} : pp = {}'.format(
                step, total_steps, np.exp(np.mean(total_loss))))
           #sys.stdout.flush()
       #if verbose:
        #sys.stdout.write('\r')
    return np.exp(np.mean(total_loss))

def generate_text(session, model, config, starting_text='<eos>',
                  stop_length=5, stop_tokens=None, temp=0.2):
  """Generate text from the model.

  Hint: Create a feed-dictionary and use sess.run() to execute the model. Note
        that you will need to use model.initial_state as a key to feed_dict
  Hint: Fetch model.final_state and model.predictions[-1]. (You set
        model.final_state in add_model() and model.predictions is set in
        __init__)
  Hint: Store the outputs of running the model in local variables state and
        y_pred (used in the pre-implemented parts of this function.)

  Args:
    session: tf.Session() object
    model: Object of type RNNLM_Model
    config: A Config() object
    starting_text: Initial text passed to model.
  Returns:
    output: List of word idxs
  """
  state = model.initial_state.eval()
  # Imagine tokens as a batch size of one, length of len(tokens[0])
  tokens = [model.vocab.encode(word) for word in starting_text.split()]
  for i in xrange(stop_length):
    ### YOUR CODE HERE
    for step, (x, y) in enumerate(
      ptb_iterator(tokens, config.batch_size, config.num_steps)):
        #train_pp = model.run_epoch(
        #      session, tokens,
        #      train_op=model.train_step, verbose=100)
        #import pdb; pdb.set_trace()

        train_op = tf.no_op()
        feed = {model.input_placeholder: x,
                model.labels_placeholder: y,
                model.initial_state: state,
                model.dropout_placeholder: 1}

        loss, state, summary = session.run(
              [model.calculate_loss, model.final_state, train_op], feed_dict=feed)


        y_pred = state
    ### END YOUR CODE
    try:
        next_word_idx = sample(y_pred[0], temperature=temp)
        #print(model.vocab.decode(next_word_idx)) # for checking
        tokens.append(next_word_idx)
    except ValueError: 
        print("Exception on sum(y_pred[0][:-1])>1 {}".format(sum(y_pred[0][:-1])))
    if stop_tokens and model.vocab.decode(tokens[-1]) in stop_tokens:
      break
  output = [model.vocab.decode(word_idx) for word_idx in tokens]
  return output

def generate_sentence(session, model, config, *args, **kwargs):
  """Convenice to generate a sentence from the model."""
  return generate_text(session, model, config, *args, stop_tokens=['<eos>'], **kwargs)

def test_RNNLM():
  config = Config()
  gen_config = deepcopy(config)

  gen_config.batch_size = gen_config.num_steps = 1

  # We create the training model and generative model
  with tf.variable_scope('RNNLM') as scope:
    model = RNNLM_Model(config)
    # This instructs gen_model to reuse the same variables as the model above
    scope.reuse_variables()
    model.print_graph()
    gen_model = RNNLM_Model(gen_config)

  init = tf.global_variables_initializer()
  saver = tf.train.Saver()

  with tf.Session() as session:
    best_val_pp = float('inf')
    best_val_epoch = 0

    session.run(init)
    def generate_sentence_local():
        """Generate a sentence given local params"""
        print('Generating a sentence ... ')
        starting_text = 'in moscow'
        print ' '.join(generate_sentence(
              session, gen_model, gen_config, starting_text=starting_text, temp=0.2))

    #generate_sentence_local()
 
    for epoch in xrange(config.max_epochs):
      print 'Epoch {}'.format(epoch)
      start = time.time()
      ###
      train_pp = model.run_epoch(
          session, model.encoded_train,
          train_op=model.train_step, verbose=1000)
      valid_pp = model.run_epoch(session, model.encoded_valid, epoch, verbose=200)

      ## Logging
      summaries = tf.summary.merge_all()
      train_writer = tf.summary.FileWriter("./logs", session.graph)

      ## Sanity Output
      print 'Training perplexity: {}'.format(train_pp)
      print 'Validation perplexity: {}'.format(valid_pp)
       
      if valid_pp < best_val_pp:
        best_val_pp = valid_pp
        best_val_epoch = epoch
        saver.save(session, './ptb_rnnlm.weights_{}'.format(config.embed_size))
      if epoch - best_val_epoch > config.early_stopping:
        print 'I am stopping early'
        break
      print 'Total time: {}'.format(time.time() - start)
    
    # TODO move to a separate file so that experiments can be run after training without retraining
    saver.restore(session, './ptb_rnnlm.weights_{}'.format(config.embed_size))
    test_pp = model.run_epoch(session, model.encoded_test)
    print '=-=' * 5
    print 'Test perplexity: {}'.format(test_pp)
    print '=-=' * 5
    
    # Generate a sentence
    for t in [0.2, 0.5, 1.0, 1.2]: 
        print("Temperature (diversity) is {}".format(t))
       
        starting_text = 'in moscow'
        while starting_text:
          generate_sentence_local()
          starting_text = raw_input('> ')

if __name__ == "__main__":
    test_RNNLM()
