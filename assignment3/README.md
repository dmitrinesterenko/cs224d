### Learning
With annealing even after epoch 17 the results were ~.64 on validation.
When using Adam do not use learning rate annealing. It doesn't make sense since Adam is one of the second order optimizer that use a moving m and v during optimization.
