from utils import ptb_iterator, sample

train_data = [i for i in range(1024)]
#num_steps is how many things in a sequence do we grab from data.
#If you want to grab 10 words then num_steps is 10
for batch in ptb_iterator(train_data, batch_size=2, num_steps=1):
    print("Batch")
    x, y = batch
    print(x, y)
