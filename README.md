# LLM-Quantum-Computer ֎ QAOA-GPT

- This model is designed to serve the purpose of efficient generation of adaptive and regular Quantum Approximate Optimization Algorithm Circuits. This approach leverages Generative Pretrained Transformers (GPT) to directly synthesize quantum circuits for solving quadratic unconstrained binary optimization problems, and demonstrate it on the MaxCut problem on graphs.
---

## Overview

- The simulation framework creates a dataset of graph-circuit samples and graph embeddings. The dataset is tokenized and processed by nanoGpt. After the model is trained we sample a new circuit that corresponds to an input graph. The new circuit is a complete ansatz with already defined operators and parameters that can solve the maxcut problem if the input graph without the need of optimization iterations. Read the [report](qaoa-gpt-project.pdf) for more details. 
---

## How to Run the Simulation

1. **Important Requirements:**  
   numpy==1.26.4 \
   matplotlib==3.10.1 \
   qiskit==0.39.4 \
   qiskit-aer==0.11.2 \
   torch==2.7.0 \
   torchvision==0.22.0 \
   tqdm==4.67.1

2. **Configurations:**
   - Adjust the training dataset parameters (graph size, number of samples) in gpt_token.py
   - Adjust the n_dim, max_iters etc. in nanoGpt/config/train_maxcut.py
   - Configure manual.py after the sample.py is executed

4. **Execute the Simulation:**
   
   Generate the training dataset: graph_circuit_tokens.txt and feather.txt
   ```bash
   python gpt_token.py
   ```
   Tokenize the txt files. Creates .bin .npy and .kpl files which are the input to the transformer
   ```bash
   python nanoGpt/data/maxcut/prepare.py
   ```
   ```bash
   cd nanoGp
   ```
   Training process. The configurations are fixed in the config.train_maxcut.py file. Set device to 'cpu' if nvidia gpu does not exist in your machine
   ```bash
   python train.py config/train_maxcut.py –device=cpu
   ```
   Generate new graph. The new graphs are generated from the Barabasi-Albert model BA(n, m) n = Number of nodes, m = Number of edges to attach from a new node to existing nodes
   ```bash
   python sample.py
   ```
   Copy the generated circuit's data and the BA graph to manual.py  
   ```bash
   cd ..
   python manual.py
   ```
