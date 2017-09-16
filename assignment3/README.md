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

Embed size 35, learning rate .001, l2 reg 0.02

Validation acc gets to be .63 acc after epoch 4. Fast results not as good as our priors however

Test set acc: 0.675

Embed size 350, lr .001, l2 .05

Training acc (only root node): 0.954285714286
Validation acc (only root node): 0.71
[[ 326.   24.]
 [   8.  342.]]
[[ 48.   2.]
 [ 27.  23.]]

stopped at 4

Training time: 90.9717063189 mns

Test
=-=-=
Test acc: 0.745

Embed size 140, lr .001 with annealing, l2 0.2

Training acc (only root node): 0.928571428571
Validation acc (only root node): 0.71
[[ 319.   31.]
 [  19.  331.]]
[[ 49.   1.]
 [ 28.  22.]]
epoch time 934.882878065 sec., time left 5.97286283208 hrs

stopped at 7

Training time: 127.54835122 mns
Test
=-=-=
Test acc: 0.695


Embed size 350, lr 0.001 without annealing, l2 0.5 -- It actually forgets things that it learned :(

Training acc (only root node): 0.925714285714
Validation acc (only root node): 0.57
[[ 315.   35.]
 [  17.  333.]]
 [[ 39.  11.]
  [ 32.  18.]]
  epoch time 1146.48967886 sec., time left 7.32479517049 hrs
  epoch 8


Training acc (only root node): 0.814285714286
Validation acc (only root node): 0.51
[[ 329.   21.]
 [ 109.  241.]]
 [[ 49.   1.]
  [ 48.   2.]]
  epoch time 1183.56133604 sec., time left 7.23287483136 hrs
  epoch 9
  )
