#!/bin/bash
source "./aws_setup/commands.sh"
INSTANCE_ID=$1
STATUS=$(active | jq .SpotInstanceRequests[$INSTANCE_ID].Status.Code)
echo $STATUS

until [ "$STATUS" == "\"fulfilled\"" ]; do
    STATUS=$(active | jq .SpotInstanceRequests[$INSTANCE_ID].Status.Code)
    echo -n "."
done

INSTANCE_ID=$(active | jq -r .SpotInstanceRequests[$INSTANCE_ID].InstanceId)
echo $INSTANCE_ID
PUBLIC_IP=$(aws ec2 describe-instances --instance-id $INSTANCE_ID --profile dmitripersonal | jq -r .Reservations[0].Instances[0].PublicIpAddress)
echo "SSHing you into $PUBLIC_IP"
ssh -i ~/.docker/machine/machines/aws01/id_rsa ec2-user@$PUBLIC_IP
# get console output (for fun)
# aws ec2 get-console-output --instance-id i-02cc73d176af0df10 --profile dmitripersonal
