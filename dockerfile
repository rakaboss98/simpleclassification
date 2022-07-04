FROM ubuntu
RUN pip3 install torch torchvision boto3
COPY training_script.py /usr/bin/train
RUN chmod 775 usr/bin/train
EXPOSE 8080