# This is an example docker file that you can use to build your own docker
# image and then sync it with your docker repo

# Base Image on DRL image
FROM mitdrl/ubuntu:latest

# Set the working directory
WORKDIR /src
COPY *.yml /src

# Update the conda base environment
RUN git config --global url."https://".insteadOf git://
RUN conda env update --name base --file environment.yml

RUN rm /etc/apt/sources.list.d/cuda.list
# RUN rm /etc/apt/sources.list.d/nvidia-ml.list
RUN apt-get update
RUN DEBIAN_FRONTEND=noninteractive TZ=America/New_York apt-get -y install tzdata
#RUN apt-get install -y libgl1-mesa-dev
RUN apt install -y libglew-dev
RUN apt-get install -y xorg-dev libglu1-mesa-dev
RUN apt-get -y install cmake
# Update something to the bashrc (/etc/bashrc_skipper) to customize your shell
RUN echo -e "alias py='python'" >> /etc/bashrc_skipper

# Switch to src directory
# WORKDIR /src

# Copy your code into the docker that is assumed to live in . (on machine)
COPY ./ /src
RUN apt install nano
RUN pip install -e .
#CMD [echo "HI"; python examples/attic/gym_test.py]
#RUN ls examples/attic/
