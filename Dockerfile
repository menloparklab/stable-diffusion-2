# Must use a Cuda version 11+
FROM pytorch/pytorch:1.13.0-cuda11.6-cudnn8-runtime

WORKDIR /

# Install git
RUN apt-get update && apt-get install -y git build-essential

# Install python packages
RUN conda install xformers -c xformers/label/dev

# Install python packages
RUN pip3 install --upgrade pip
ADD requirements.txt requirements.txt
RUN pip3 install -r requirements.txt

# We add the banana boilerplate here
ADD server.py .
EXPOSE 8000

# Add your huggingface auth key here
ARG HF_AUTH_TOKEN
ENV HF_AUTH_TOKEN=$HF_AUTH_TOKEN

# Add your model weight files 
# (in this case we have a python script)
ADD download.py .
RUN python3 download.py

# Add your custom app code, init() and inference()
ADD app.py .

CMD python3 -u server.py
