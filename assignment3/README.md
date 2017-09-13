### Learning
With annealing even after epoch 17 the results were ~.64 on validation.
When using Adam do not use learning rate annealing. It doesn't make sense since Adam is one of the second order optimizer that use a moving m and v during optimization.

Embed size 70, learning rate .001 and constant, no annealing

Training acc (only root node): 0.927142857143
Validation acc (only root node): 0.5

Test set acc: 0.535

(Almost as good as guessing on validation and test mean, however really good at training. So we are overfitting the data)

Embed size 35, learning rate .001 

Validation acc gets to be .71 after epoch 11, however drops off to .56 by epoch 13 and the result test acc is .64
