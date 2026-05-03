# CG Task
* Deadline: Mon, 18 May 23:59
* Submission (report) format: PDF or Jupyter NB
* Jupyter NB to complete: `notebooks/cg/cg-linear-regression-stud.ipynb`
* **All original provided materials are in `02_MPW-CG` folder**
* **All resulting files are in `notebooks/cg/` and `src/cg/`**

## Goal
Complete and experiment the implementation of a computational graph and of a training loop for linear 
regression task. 

## Requirements on report
- _Clear, concise, and well-structured_,
- Includes all _relevant details about your experiments_: 
  - the **_hyperparameters_** you used, 
  - the **_results_** you obtained,
  - any **_insights or conclusions_** you can draw from your experiments.
- **_States group members_** in the beginning of the report.

## Objectives
### Mandatory (main) objectives
a) Understand the concepts of backpropagation and of forward and backward passes in the
context of a computational graph.

b) Implement new nodes in a computational graph.

c) Implement a training loop with stochastic and batch approaches

d) Reimplement and experiment with different optimization algorithms and compare their
performance.

### Optional objectives
**Pick at least 2 of the following optional objectives** to further explore and analyze the performance
of your models :
1. Investigate the use of 2nd order model instead of the simple linear model.
2. Re-implement and experiment with more advanced optimizers such as RMSProp, Nesterov
3. or Adam.
4. Implement an early stopping strategy in your training loop.
5. Implement a Learning Rate Decay on Plateau strategy in the training loop.
6. Normalize the input data with a zero norm approach and compare to your experiments 
without normalization.

------
The whole text below was given in the original provided `README.md` file.

------
# Simple Computational Graph

Minimal educational computational graph for scalar forward and backward
passes, used in the linear regression notebook.

## Project Files

- `cgnodes.py`: Node and graph implementation.
- `cg-linear-regression.ipynb`: Notebook to complete.
- `../data/lausanne-appart.csv`: Dataset used by the notebook.

## Setup

```bash
uv sync
```

## API Quick Reference

### Core classes

- `CompGraph(in_nodes, out_nodes)`: wraps graph execution.
- `ValueNode(v=None)`: scalar value and gradient carrier.
- `MultiplyNode(x1, x2, out)`: computes `x1 * x2`.
- `AddNode([a, b, ...], out)`: computes sum of inputs.
- `SquareNode(a, out)`: computes `a^2`.

### Execution lifecycle

1. Build `ValueNode` objects for inputs, parameters, and outputs.
2. Build operator nodes to connect the graph.
3. Create `CompGraph(input_nodes, output_nodes)`.
4. Call `reset_values()` before each sample/batch step.
5. Call `forward([...])` to compute output values.
6. Call `backward()` to propagate gradients to leaf/input nodes.

## Notes

- This code is intentionally simple and scalar-based for teaching.
- Operator nodes push values forward as soon as all required inputs are ready.
- Binary operator nodes (`MultiplyNode`) expose a `get_parent_values()` helper that returns parent scalars in a fixed, named order, keeping `forward()` and `backward()` free of raw index accesses.
- `CompGraph.backward()` starts from each output with upstream gradient `1.0`.
