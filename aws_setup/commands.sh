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

active_id (){
   $INSTANCE_ID=0
    ACTIVE_ID=$(aws ec2 describe-spot-instance-requests --profile dmitripersonal --filter Name="state",Values="active" | jq .SpotInstanceRequests[$INSTANCE_ID].Status.Code)
    echo $ACTIVE_ID

}

