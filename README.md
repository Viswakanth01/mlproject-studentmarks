# mlproject-studentmarks

## Creating  Environment

'''
conda create -p ml
'''

'''
conda activate ml/
'''
## Setting up the requirements
'''
add and update reqirements.txt and setup files
'''

## Creating Flask app
'''
After creating app
Create templates folder and then add the required html files
'''
# Dockers
'''
Install Dockers Desktop based on the respective OS
Create docker hub account
'''
## TO create Dcoker image
'''
Create Dockerfile
'''
'''
docker build -t viswakanthk/student-performance .
'''
## To run as countainer
'''
docker run -p 5000:5000 viswakanthk/student-performance
'''

# CI/CD Pipeline
## pre-requisite
'''
Sign-in account in aws 
Sign-in account in aws console
'''
'''
create .github/workflow folder and add main.yaml file
'''
'''
commit all changes to github
'''

## In aws console  
'''
Create IAM user
Create ECR repository
Create EC2
'''
# Docker Setup In EC2 commands to be Executed
'''
#optinal

sudo apt-get update -y

sudo apt-get upgrade

#required

curl -fsSL https://get.docker.com -o get-docker.sh

sudo sh get-docker.sh

sudo usermod -aG docker ubuntu

newgrp docker
'''
# Configure EC2 as self-hosted runner:
'''
In Github Actions create self-hosted runner
'''
# Setup github secrets:
'''
AWS_ACCESS_KEY_ID=

AWS_SECRET_ACCESS_KEY=

AWS_REGION = 

AWS_ECR_LOGIN_URI = 

ECR_REPOSITORY_NAME =
'''