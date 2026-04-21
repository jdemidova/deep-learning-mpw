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
