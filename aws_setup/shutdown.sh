#!/bin/bash
# Save the results
function save(){
    CURRENT_BRANCH=$(git rev-parse --abbrev-ref HEAD)
    git add .
    git commit -m "Saving the results of the run"
    git push origin $CURRENT_BRANCH
}

function test_save(){
    DN=$(date +%Y%m%d%H%M)
    touch $DN
    save
}

function get_latest_data(){
    FILE=$(ls -ht ./assignment3/output/)[0]
}

function mail(){
    echo "mailing $(get_latest_data)"
}

function shutdown(){
    sudo shutdown -h 2 "System is done for the day, shutdown in 2"
}

save
mail
shutdown

