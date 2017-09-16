## CS224D Deep Learning for NLP

### Assignment 1
The contents of the problem set for assignment 1 are
[here](http://cs224d.stanford.edu/assignment1/index.html)
with the PDF of the assignment also included in this repository as
assignment1.pdf

#### Running
This uses python 2.7. Run the excercises by using python q{1,2,3}_{problem}.py,
basically execute any code by passing the file to the python interpreter.

#### Tests
Some of the functionality has tests but some is still missing.

I think there is a large chunk of q3_word2vec.py that can benefit from sanity
tests.

### Assignment 2

#### Running
This assignment benefits from using GPUs so there are instructions on spinning
up a spot instance on AWS. I use a p2.8xlarge instance which as of June 04th
$7.20 per hour. Spot instance makes it possible to run this same instance type
for $1.3 per hour.

To make a request for a spot instance run `./aws_setup/setup.sh` currently this
assumes my laptop and uses an AWS profile that I had setup for my personal AWS
account. You would need to change things around and this probably won't travel
as well because the AWS AMI that I am using is in my own account. This doesn't
travel well at the moment.

config.json is a saved file from my first spot instance request using the AWS
management console. The saved config.json and setup.sh scripts in the aws_setup
folder make it much easier for me to re-request an instance next time I want to
run a training.

Once instance is up there will be a cs224 folder in the home folder.
`git pull origin master` and you have the latest code. Execute with `python
q3_RNNLM.py`

### Support Scripts for AWS
./aws_setup contains support scripts for creatin AWS spot instance
infrastructure

source ./aws_setup/commands.sh to have some basic commands that perform:
`describe` the environment
`setup` a new spot instance
`login` to the instance
`terminate` it when you are done;

#### Login
When running N multiple instances choose which one to login to by providing an
instance number from 0-N.
`./aws_setup/login.sh 0` to login to the 1st instance

### Assignment 3
The best executions were trained in 90 minutes on p2.xlarge and had accuracy of
.745 on validation.

Updating the embed size up to 2800 from the starting 35 did little to change the
resulting accuracy.

The l2 regularization was as it should be super important in getting good
validation results.

The training set and the validation set accuracies were still wide apart, the
training set accuracy would frequently be above %92

All of the results of running Assignment 3 and individual write ups are in
assignment3/output and [assignment3/README.md](assignment3/README.md)

