# Use an official Python runtime as a parent image
FROM tensorflow/tensorflow

# Set the working directory
WORKDIR /project

# Copy pipenv files into working directory
COPY $PWD/code/Pipfile /project
COPY $PWD/code/Pipfile.lock /project

# User configuration - override with --build-arg
ARG user=jtr008
ARG group=stud
ARG uid=1000
ARG gid=1000

# Some debs want to interact, even with apt-get install -y, this fixes it
ENV DEBIAN_FRONTEND=noninteractive
ENV HOME=/project

# Install needed programs from apt-get
RUN apt-get update && apt-get install -y sudo screen git cmake texlive-xetex

# Install any needed packages from apt
RUN pip3 install --trusted-host pypi.python.org --upgrade pip
RUN pip3 install --trusted-host pypi.python.org pipenv
RUN pipenv lock -r > requirements.txt
RUN pip3 install --trusted-host pypi.python.org -r requirements.txt --ignore-installed

# Configure user
RUN groupadd -g $gid $user
RUN useradd -u $uid -g $gid $user
RUN usermod -a -G sudo $user
RUN passwd -d $user

# Make ports available to the world outside this container
EXPOSE 1337
EXPOSE 1338
