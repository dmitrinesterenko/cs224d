#!/bin/bash
INSTANCE_ID=$(aws ec2 describe-spot-instance-requests --profile dmitripersonal | jq -r .SpotInstanceRequests[0].InstanceId)
echo "Terminating $INSTANCE_ID"
aws ec2 terminate-instances --instance-ids $INSTANCE_ID --profile dmitripersonal
