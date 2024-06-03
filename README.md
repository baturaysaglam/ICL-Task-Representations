# Learning Task Representations from In-Context Learning

## Getting Started

### 1. Dependencies
Install the dependencies using Conda. You may need to adjust the environment YAML file (such as the name, Python version, etc.) depending on your setup.
```
conda env create -f environment.yml
conda activate icl-task-repr
```

### 2. Use the forked ``transformers`` package 

A couple of lines have been added to [``modeling_gpt2.py``](https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py) to track the attention output for computing/adding the LTV. This won't effect the operation of the package. 
```
cd ICL-Task-Representations
cd transformers
pip install .
```

**Note:** Our method works on two modalities: numeric functions (synthetic tasks) and language. For functions, we use the [code](https://github.com/dtsip/in-context-learning) from [Garg et al.](https://arxiv.org/abs/2208.01066), while the [code](https://github.com/ericwtodd/function_vectors) of [Todd et al.](https://functions.baulab.info/) is utilized for the linguistic tasks. So we examine the requirements for functions and language separetely. 


## Synthetic Tasks
### 0. Load the pretrained GPT-2 models

