#TODO

- [] Add regularization
- [] Add dropout

# Running time comparisons with AWS p2 instance
## 1 Epoch
Training loss: 0.402585595846595846
Training acc: 0.874025763551
Validation loss: 0.291980147362

[[41939   151   105   219   345]
 [  403  1439    45   132    75]
 [  485    53   612    70    48]
 [  886   153    57   843   153]
 [  948    81    18   147  1955]]
Tag: O - P 0.9391 / R 0.9808
Tag: LOC - P 0.7666 / R 0.6872
Tag: MISC - P 0.7312 / R 0.4826
Tag: ORG - P 0.5974 / R 0.4030
Tag: PER - P 0.7589 / R 0.6208
Total time: 105.241132975
Test
=-=-=

## Another restart gives a better total time

[[42050   129    99   228   253]
 [  214  1683    23   120    54]
 [  244    35   905    56    28]
 [  586   100    40  1269    97]
 [  513    39    11    99  2487]]
Tag: O - P 0.9643 / R 0.9834
Tag: LOC - P 0.8474 / R 0.8037
Tag: MISC - P 0.8395 / R 0.7137
Tag: ORG - P 0.7161 / R 0.6066
Tag: PER - P 0.8520 / R 0.7898
Total time: 85.1370520592
Epoch 5
Training loss: 0.0791734457016457016
Training acc: 0.974315026446
Validation loss: 0.22555449605

## A test run with dropout and regularization
Epoch 0
Training loss: 0.446270912886912886
Training acc: 0.867474376415
Validation loss: 0.322937309742

[[41772   167   226   276   318]
 [  311  1389    56   197   141]
 [  366    46   737    83    36]
 [  853   114    76   849   200]
 [  957    80    27   211  1874]]
Tag: O - P 0.9438 / R 0.9769
Tag: LOC - P 0.7734 / R 0.6633
Tag: MISC - P 0.6569 / R 0.5812
Tag: ORG - P 0.5254 / R 0.4058
Tag: PER - P 0.7295 / R 0.5951
Total time: 152.228859901
Test
=-=-=

