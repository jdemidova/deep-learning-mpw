from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Optional


class CompGraph:
    """A simple computational graph entry point.

    It manages input/output ValueNode objects and orchestrates full
    forward/backward passes.
    """

    def __init__(self, in_nodes: list[ValueNode], out_nodes: list[ValueNode]):
        """Create a graph wrapper around input and output value nodes.

        Args:
            in_nodes: Ordered list of external input placeholders.
            out_nodes: Ordered list of nodes considered graph outputs.
        """
        self.in_nodes = in_nodes
        self.out_nodes = out_nodes
        self.forwarded = False
        self.validate_graph()

    def validate_graph(self):
        """Validate that graph boundaries are made of ValueNode objects."""
        # all input nodes should be ValueNode nodes
        for node in self.in_nodes:
            if not isinstance(node, ValueNode):
                raise Exception("Input node of CompGraph is not a ValueNode")
        # all output nodes should be ValueNode nodes
        for node in self.out_nodes:
            if not isinstance(node, ValueNode):
                raise Exception("Output node of CompGraph is not a ValueNode")

    def forward(self, input_values: list[float]):
        """Run a forward pass from graph inputs to outputs.

        Args:
            input_values: Scalar values fed to input nodes in the same
                order as ``in_nodes``.

        Raises:
            Exception: If number of provided values does not match graph
                input count.
        """
        if len(input_values) != len(self.in_nodes):
            raise Exception(
                "Can't forward: number of input differs to number "
                "of input nodes"
            )
        for i, in_node in enumerate(self.in_nodes):
            in_node.receive_parent_value(input_values[i])
            in_node.forward()
        self.forwarded = True

    def backward(self):
        """Run backward pass from graph outputs with unit upstream gradient.

        Raises:
            Exception: If called before at least one successful forward pass.
        """
        if not self.forwarded:
            raise Exception("Can't backward, you need to call forward first")
        for node in self.out_nodes:
            node.backward(1.0)

    def reset_values(self):
        """Reset cached values recursively so a new pass can be executed."""
        for node in self.in_nodes:
            node.reset_values()
        self.forwarded = False


class MetaNode(ABC):
    """Base abstraction for all graph nodes.

    A node keeps references to upstream ``parents`` and downstream
    ``children``. Implementations are responsible for tracking when inputs
    are ready (``input_ready``) and for propagating values/gradients.
    """

    def __init__(self):
        self.parents: list[MetaNode] = []
        self.children: list[MetaNode] = []
        self.input_ready: bool = False

    def add_child(self, node: MetaNode):
        """Register a downstream child node.

        This helper only updates the current node and does not update
        ``node.parents``. Prefer ``connect_to`` for bidirectional linking.
        """
        self.children.append(node)

    def add_parent(self, node: MetaNode):
        """Register an upstream parent node.

        This helper only updates the current node and does not update
        ``node.children``. Prefer ``connect_to`` for bidirectional linking.
        """
        self.parents.append(node)

    def connect_to(self, node: MetaNode):
        """Connect this node to another node.

        The connection is created in both directions:
        ``self -> node`` and ``node <- self``.
        """
        self.children.append(node)
        node.parents.append(self)

    @abstractmethod
    def forward(self):
        """Compute local output and push it to children when ready."""
        pass

    @abstractmethod
    def receive_parent_value(self, v):
        """Receive one upstream value from a parent node."""
        pass

    @abstractmethod
    def reset_values(self):
        """Clear cached forward/backward state for a new pass."""
        pass

    @abstractmethod
    def backward(self, grad_z):
        """Propagate gradient from downstream to upstream parents."""
        pass


class ValueNode(MetaNode):
    """A node that stores scalar values and propagated gradients."""

    def __init__(self, v: Optional[float] = None):
        """Create a value node.

        Args:
            v: Optional initial value. If provided, the node starts as ready.
        """
        super().__init__()
        self.v: Optional[float] = None
        self.grad_v: Optional[float] = None
        if v is not None:
            self.v = v
            self.input_ready = True

    def receive_parent_value(self, v):
        """Store incoming value and mark this node ready."""
        self.v = v
        self.input_ready = True

    def set_grad_value(self, grad_v):
        """Set gradient explicitly.

        This is mainly a helper for manual experiments/tests.
        """
        self.grad_v = grad_v

    def reset_values(self):
        """Clear value and gradient, then recursively reset descendants."""
        self.v = None
        self.grad_v = None
        self.input_ready = False
        for node in self.children:
            node.reset_values()

    def forward(self):
        """Forward stored value to all children.

        Raises:
            Exception: If value is missing when forward is requested.
        """
        if self.v is None:
            raise Exception(
                "Forward not possible as no value set in this ValueNode"
            )
        for node in self.children:
            node.receive_parent_value(self.v)
            node.forward()

    def backward(self, grad_z):
        """Accumulate gradient and pass it to parents.

        A ``ValueNode`` can receive multiple downstream contributions;
        those are summed in ``grad_v``.
        """
        if self.grad_v is None:
            self.grad_v = grad_z
        else:
            self.grad_v += grad_z
        for node in self.parents:
            node.backward(grad_z)


class MultiplyNode(MetaNode):
    """A binary multiplication node: z = x1 * x2.

    Semantic role-based: x1 and x2 roles are explicit at
    construction, making gradient flow deterministic and unambiguous.
    """

    def __init__(self, x1: ValueNode, x2: ValueNode, out: ValueNode):
        """Create a multiplication operator node.

        Args:
            x1: First multiplicand node.
            x2: Second multiplicand node.
            out: Output node receiving ``x1 * x2``.
        """
        super().__init__()
        # parents[0] is always x1, parents[1] is always x2
        x1.connect_to(self)
        x2.connect_to(self)
        self.connect_to(out)
        self._received_count = 0

    def get_parent_values(self) -> tuple[float, float]:
        """Return parent values in the fixed ``(x1, x2)`` order."""
        return self.parents[0].v, self.parents[1].v

    def receive_parent_value(self, v: float):
        """Record input arrival from a parent.

        The scalar itself is not stored here because it is read from parent
        nodes during forward/backward; only readiness state is tracked.
        """
        del v  # value is read from parents[].v in forward/backward
        if self._received_count >= 2:
            raise Exception(
                "This node accepts 2 inputs that are already filled"
            )
        self._received_count += 1
        if self._received_count == 2:
            self.input_ready = True

    def reset_values(self):
        """Reset readiness counters and recursively reset descendants."""
        self._received_count = 0
        self.input_ready = False
        for node in self.children:
            node.reset_values()

    def forward(self):
        """Compute product and push to children once both inputs are ready."""
        if self.input_ready:
            x1_val, x2_val = self.get_parent_values()
            z = x1_val * x2_val
            for node in self.children:
                node.receive_parent_value(z)
                node.forward()

    def backward(self, grad_z):
        """Apply product rule and route gradients to both parents."""
        x1_val, x2_val = self.get_parent_values()
        self.parents[0].backward(grad_z * x2_val)
        self.parents[1].backward(grad_z * x1_val)


class AddNode(MetaNode):
    """A variadic addition node: z = sum(x_i)."""

    def __init__(self, in_nodes: list[ValueNode], out_node: ValueNode):
        """Create an addition operator with an arbitrary number of inputs.

        Args:
            in_nodes: List of input addend nodes.
            out_node: Output node receiving the sum.
        """
        super().__init__()
        # build connections to ValueNode objects
        for node in in_nodes:
            node.connect_to(self)
        self.connect_to(out_node)
        # initialize internal values - here the input is a list of values
        self.inputs: list[float] = []

    def receive_parent_value(self, v: float):
        """Append one addend and mark ready when all addends are present."""
        self.inputs.append(v)
        if len(self.inputs) == len(self.parents):
            self.input_ready = True
        elif len(self.inputs) > len(self.parents):
            raise Exception('All inputs are already set')

    def reset_values(self):
        """Clear cached addends and recursively reset descendants."""
        self.inputs = []
        self.input_ready = False
        for node in self.children:
            node.reset_values()

    def forward(self):
        """Compute sum of inputs and push result to children."""
        if self.input_ready:
            s = sum(self.inputs)
            for node in self.children:
                node.receive_parent_value(s)
                node.forward()

    def backward(self, grad_z):
        """Distribute same upstream gradient to all addends."""
        grad_x = 1.0 * grad_z
        # this value is distributed back to all input values
        for node in self.parents:
            node.backward(grad_x)


class SquareNode(MetaNode):
    """A unary square node: z = x^2."""

    def __init__(self, in_node: ValueNode, out_node: ValueNode):
        """Create a square operator node.

        Args:
            in_node: Input node ``x``.
            out_node: Output node receiving ``x^2``.
        """
        super().__init__()
        # build connections to ValueNode objects
        in_node.connect_to(self)
        self.connect_to(out_node)
        # initialize internal values - here the input is single value
        self.x: float = None

    def receive_parent_value(self, v: float):
        """Store input value and mark node as ready."""
        self.x = v
        self.input_ready = True

    def reset_values(self):
        """Clear cached input and recursively reset descendants."""
        self.x = None
        self.input_ready = False
        for node in self.children:
            node.reset_values()

    def forward(self):
        """Compute square and propagate result to children."""
        if self.input_ready:
            z = self.x * self.x
            for node in self.children:
                node.receive_parent_value(z)
                node.forward()

    def backward(self, grad_z):
        """Apply derivative of square function: d(x^2)/dx = 2x."""
        grad_x = 2 * self.x * grad_z
        for node in self.parents:
            node.backward(grad_x)


# ######  TO DO  #########
    # class MSELossNode(MetaNode):
    # Advice: mimic the MultiplyNode and adapt the forward() and
    # backward() methods
