#!/bin/bash
STATUS=$(aws ec2 describe-spot-instance-requests --profile dmitripersonal | jq .SpotInstanceRequests[0].Status.Code)
echo $STATUS
if [ $STATUS != "fulfilled" ]; then
    print "Not yet fulfilled"
fi

until [ $STATUS != "fulfilled" ]; do
    STATUS=$(aws ec2 describe-spot-instance-requests --profile dmitripersonal | jq .SpotInstanceRequests[0].Status.Code)
    echo -n "."
done
INSTANCE_ID=$(aws ec2 describe-spot-instance-requests --profile dmitripersonal | jq -r .SpotInstanceRequests[0].InstanceId)
echo $INSTANCE_ID
PUBLIC_IP=$(aws ec2 describe-instances --instance-id $INSTANCE_ID --profile dmitripersonal | jq -r .Reservations[0].Instances[0].PublicIpAddress)
echo "SSHing you into $PUBLIC_IP"
ssh -i ~/.docker/machine/machines/aws01/id_rsa ec2-user@$PUBLIC_IP
# get console output (for fun)
# aws ec2 get-console-output --instance-id i-02cc73d176af0df10 --profile dmitripersonal
