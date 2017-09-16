FN=$(date +%Y%m%d%H%M)
nohup python ./rnn.py > output/$FN &
tail -f output/$FN
