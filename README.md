# Master's thesis in machine learning

This repository holds the Python and LaTeX code for my master's thesis in machine learning at UiB ([INF399](https://www.uib.no/en/course/INF399)), with the title: "Analysis of Word Embeddings: A Clustering and Topological Approach". The master's thesis was supervised by [Nello Blaser](https://www.uib.no/personer/Nello.Blaser) and was based on the topic from "[Topology of encodings](https://www.uib.no/en/rg/ml/128703/available-masters-thesis-topics-machine-learning#topology-of-encodings)". The finalized version of the thesis [can be found here](https://github.com/JonasTriki/masters-thesis-ml/blob/master/Master's%20Thesis%20in%20ML%20-%20Jonas%20Folkvord%20Triki.pdf).

## Reposotory structure

The repository is structed as follows:

- The `code` directory holds all the relevant code used to perform the analysis of word embeddings, as well as some plots used in the thesis.
- The `custom_figures` directory holds custom figures included in the background chapter of the thesis.
- The `outline` directory contains the LaTeX outline of the master's thesis. Note that the information here is outdated.
- The `thesis` directory holds the LaTeX code used to generate the master's thesis PDF.

## Running the code

First, make sure Docker is installed and up to date. Then, from the root of the repository, run the Makefile to create up an ubuntu environment:

```bash
make DETACHED=1
```

The `DETACHED=1` argument allows the Docker container to be deattached from the terminal and you can access the docker container at any time by running:

```bash
docker exec -it jtr008-docker-container /bin/bash
```

To enable GPUs, perform the following commands instead:

```bash
make -f Makefile-gpu DETACHED=1
docker exec -it jtr008-docker-container-gpu /bin/bash
```
