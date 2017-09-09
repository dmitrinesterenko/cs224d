#!/bin/bash
SPOT_NUMBER=0
INSTANCE_ID=$(aws ec2 describe-spot-instance-requests --profile dmitripersonal | jq -r .SpotInstanceRequests[$SPOT_NUMBER].InstanceId)
echo "Terminating $INSTANCE_ID"
aws ec2 terminate-instances --instance-ids $INSTANCE_ID --profile dmitripersonal
