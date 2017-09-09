#!/bin/bash
describe ()
{
    STATUS=$(aws ec2 describe-spot-instance-requests --profile dmitripersonal)
    echo $STATUS
}

active (){
    ACTIVE=$(aws ec2 describe-spot-instance-requests --profile dmitripersonal --filter Name="state",Values="active")
    echo $ACTIVE
}

active_ids (){
    ACTIVE_IDS=$(aws ec2 describe-spot-instance-requests --profile dmitripersonal --filter Name="state",Values="active" | jq .SpotInstanceRequests.Status.Code)
    echo $ACTIVE_IDS

}

