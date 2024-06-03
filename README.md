# Learning Task Representations from In-Context Learning

## Getting Started

Install the dependencies using Conda. You may need to adjust the environment YAML file (such as the name, Python version, etc.) depending on your setup.
```
conda env create -f environment.yml
conda activate icl-task-repr
```

**Note:** Our method works in two modalities: numeric functions (synthetic tasks) and language. For functions, we use the [code](https://github.com/dtsip/in-context-learning) from [Garg et al.](https://arxiv.org/abs/2208.01066), while the [code](https://github.com/ericwtodd/function_vectors) of [Todd et al.](https://functions.baulab.info/) is utilized for the linguistic tasks. Therefore, we examine the requirements for functions and language separetely. 


## Synthetic Tasks
