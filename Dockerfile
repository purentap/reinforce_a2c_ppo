FROM debian:bookworm

# install debian packages
ENV DEBIAN_FRONTEND noninteractive
RUN apt-get update -qq \
 && apt-get install --no-install-recommends -y \
    # install essentials
    build-essential \
    g++ \
    git \
    openssh-client \
    wget \
    ca-certificates \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/*

RUN wget https://repo.anaconda.com/miniconda/Miniconda3-4.7.12.1-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh && \
    /opt/conda/bin/conda clean -tipsy && \
    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
    echo "conda activate base" >> ~/.bashrc
ENV PATH /opt/conda/bin:$PATH

RUN /opt/conda/bin/conda install python=3.7

RUN /opt/conda/bin/conda install nodejs

RUN /opt/conda/bin/conda upgrade -n base conda

RUN /opt/conda/bin/conda install conda-forge::swig

# Copy the requirements text into the container at /
ADD /requirements.txt /

# install dependencies
RUN pip install -r requirements.txt

# Create a directory to mount homework at run 
RUN mkdir hw2

# Set the working directory to /hw2
WORKDIR /hw2

EXPOSE 8889

# Install the package and run ipython in no-browser mode
CMD ["sh", "-c", "pip install -e . && jupyter notebook --ip=0.0.0.0 --no-browser --allow-root --port=8889"]