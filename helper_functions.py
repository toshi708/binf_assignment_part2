"""Helper functions for HW3"""
import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt
from matplotlib.axes import Axes


class Node:
    def __init__(
        self,
        name: str,
        left: "Node",
        left_distance: float,
        right: "Node",
        right_distance: float,
        confidence: float = None,
    ):
        """A node in a binary tree produced by neighbor joining algorithm.

        Parameters
        ----------
        name: str
            Name of the node.
        left: Node
            Left child.
        left_distance: float
            The distance to the left child.
        right: Node
            Right child.
        right_distance: float
            The distance to the right child.
        confidence: float
            The confidence level of the split determined by the bootstrap method.
            Only used if you implement Bonus Problem 1.

        Notes
        -----
        The current public API needs to remain as it is, i.e., don't change the
        names of the properties in the template, as the tests expect this kind
        of structure. However, feel free to add any methods/properties/attributes
        that you might need in your tree construction.

        """
        self.name = name
        self.left = left
        self.left_distance = left_distance
        self.right = right
        self.right_distance = right_distance
        self.confidence = confidence


def neighbor_joining(distances: np.ndarray, labels: list) -> Node:
  # start as collection of nodes with no children
  nodes = []
  for label in labels:
    nodes.append(Node(label, None, 0, None, 0))

  distances = distances.copy()

  while len(nodes) > 2:
    n = len(nodes)

    # total distance from each node to all others
    row_sums = []
    for i in range(n):
      row_sums.append(sum(distances[i]))

    # build the Q matrix, find the pair with smallest value
    Q = np.zeros((n, n))

    for i in range(n):
      for j in range(n):
        if i == j:
          Q[i, j] = float("inf")
        else:
          Q[i, j] = ((n - 2) * distances[i][j]- row_sums[i] - row_sums[j])

    i, j = np.unravel_index(np.argmin(Q), Q.shape)

    # get the branch lenghts and join the pair together
    distance_i = (distances[i][j] / 2 + (row_sums[i] - row_sums[j]) / (2 * (n - 2)))
    distance_j = distances[i][j] - distance_i

    new_node = Node(f"Node{n}", nodes[i], distance_i, nodes[j], distance_j)

    remaining_nodes = []
    remaining_indices = []
    for k in range(n):
      if k != i and k != j:
        remaining_nodes.append(nodes[k])
        remaining_indices.append(k)

    new_distances = []
    for a in remaining_indices:
      row = []
      for b in remaining_indices:
         row.append(distances[a][b])
      new_distances.append(row)

    # get the distances between new node and remaining nodes
    new_row = []
    for k in remaining_indices:
       distance = (distances[i][k] + distances[j][k] - distances[i][j]) / 2
       new_row.append(distance)

    remaining_nodes.append(new_node)
    for row in new_distances:
      row.append(0)
    new_distances.append([0] * len(remaining_nodes))

    for k in range(len(new_row)):
      new_distances[k][-1] = new_row[k]
      new_distances[-1][k] = new_row[k]

    nodes = remaining_nodes
    distances = new_distances

  # calculate half distance between two remaining nodes and return root node
  half_distance = distances[0][1] / 2
  root = Node("Root", nodes[0], half_distance, nodes[1], half_distance)
  return root

def plot_nj_tree(tree: Node, ax: Axes = None, **kwargs) -> None:
  if ax is None:
    _, ax = plt.subplots(figsize=(10, 6))

  # go through stack and collect all the leaves (no children)
  leaves = []
  stack = [tree]
  while stack:
    node = stack.pop()

    if node.right is not None:
      stack.append(node.right)

    if node.left is not None:
      stack.append(node.left)


    if node.left is None and node.right is None:
      leaves.append(node.name)

  # give each leaf a vertical position
  positions = {}
  for i, name in enumerate(leaves):
    positions[name] = i

  # go through tree calculating each node horizontal position
  stack = [(tree, 0)]
  while stack:
    node, x = stack.pop()

    # if its a leaf, then draw it
    if node.left is None and node.right is None:
      ax.text(x, positions[node.name], str(node.name), va="center")
      continue

    # calculate children's horizontal position from parent's position and branch length
    child_positions = []
    if node.left is not None:
      left_x = x + node.left_distance
      stack.append((node.left, left_x))
      child_positions.append((left_x, node.left))

    if node.right is not None:
      right_x = x + node.right_distance
      stack.append((node.right, right_x))
      child_positions.append((right_x, node.right))

    # draw branch from current node to each child.
    for child_x, child in child_positions:
      # leaves use assigned vertical position, internal nodes calculate so branching part sits at middle
      if child.left is None and child.right is None:
        child_y = positions[child.name]
      else:
        child_leaves = []
        search = [child]

        while search:
          current = search.pop()

          if current.left is None and current.right is None:
            child_leaves.append(positions[current.name])
          else:
            if current.left is not None:
              search.append(current.left)
            if current.right is not None:
              search.append(current.right)

        child_y = sum(child_leaves) / len(child_leaves)

      ax.plot([x, child_x], [child_y, child_y], color="black")

    # if current node has two children draw vertical line to show diverging branches
    if len(child_positions) == 2:
      ys = []

      for child_x, child in child_positions:
        search = [child]
        child_leaves = []

        while search:
          current = search.pop()
          if current.left is None and current.right is None:
            child_leaves.append(positions[current.name])
          else:
            if current.left is not None:
              search.append(current.left)
            if current.right is not None:
              search.append(current.right)

        ys.append((sum(child_leaves) / len(child_leaves)))

      ax.plot([x, x], [min(ys), max(ys)], color="black")

  return ax


def _find_a_parent_to_node(tree: Node, node: Node) -> tuple:
    """Utility function for reroot_tree"""
    stack = [tree]

    while len(stack) > 0:

        current_node = stack.pop()
        if node.name == current_node.left.name:
            return current_node, "left"
        elif node.name == current_node.right.name:
            return current_node, "right"

        stack += [
            n for n in [current_node.left, current_node.right] if n.left is not None
        ]

    return None



def _remove_child_from_parent(parent_node: Node, child_location: str) -> None:
    """Utility function for reroot_tree"""
    setattr(parent_node, child_location, None)
    setattr(parent_node, f"{child_location}_distance", 0.0)


def reroot_tree(original_tree: Node, outgroup_node: Node) -> Node:
    """A function to create a new root and invert a tree accordingly.

    This function reroots tree with nodes in original format. If you
    added any other relational parameters to your nodes, these parameters
    will not be inverted! You can modify this implementation or create
    additional functions to fix them.

    Parameters
    ----------
    original_tree: Node
        A root node of the original tree.
    outgroup_node: Node
        A Node to set as an outgroup (already included in a tree).
        Find it by it's name and then use it as parameter.

    Returns
    -------
    Node
        Inverted tree with a new root node.
    """
    tree = deepcopy(original_tree)

    parent, child_loc = _find_a_parent_to_node(tree, outgroup_node)
    distance = getattr(parent, f"{child_loc}_distance")
    _remove_child_from_parent(parent, child_loc)

    new_root = Node("new_root", parent, distance / 2, outgroup_node, distance / 2)
    child = parent

    while tree != child:
        parent, child_loc = _find_a_parent_to_node(tree, child)

        distance = getattr(parent, f"{child_loc}_distance")
        _remove_child_from_parent(parent, child_loc)

        empty_side = "left" if child.left is None else "right"
        setattr(child, f"{empty_side}_distance", distance)
        setattr(child, empty_side, parent)

        if tree.name == parent.name:
            break
        child = parent

    other_child_loc = "right" if child_loc == "left" else "left"
    other_child_distance = getattr(parent, f"{other_child_loc}_distance")

    setattr(child, f"{empty_side}_distance", other_child_distance + distance)
    setattr(child, empty_side, getattr(parent, other_child_loc))

    return new_root

def sort_children_by_leaves(tree: Node) -> None:
  stack = [tree]

  while len(stack) > 0:
    node = stack.pop()

    if node is None:
      continue

    left_leaf_count = 0
    left_stack = [node.left]

    while len(left_stack) > 0:
      current = left_stack.pop()

      if current is None:
        continue
      if current.left is None and current.right is None:
        left_leaf_count = left_leaf_count + 1
      else:
        left_stack.append(current.left)
        left_stack.append(current.right)

    right_leaf_count = 0
    right_stack = [node.right]
    while len(right_stack) > 0:
      current = right_stack.pop()

      if current is None:
        continue
      if current.left is None and current.right is None:
        right_leaf_count = right_leaf_count + 1
      else:
        right_stack.append(current.left)
        right_stack.append(current.right)

    if left_leaf_count > right_leaf_count:
      tmp = node.left
      node.left = node.right
      node.right = tmp

      tmp = node.left_distance
      node.left_distance = node.right_distance
      node.right_distance = tmp

    if node.left is not None:
      stack.append(node.left)

    if node.right is not None:
      stack.append(node.right)    

def plot_nj_tree_radial(tree: Node, ax: Axes = None, **kwargs) -> None:
    """A function for plotting neighbor joining phylogeny dendrogram
    with a radial layout.

    Parameters
    ----------
    tree: Node
        The root of the phylogenetic tree produced by `neighbor_joining(...)`.
    ax: Axes
        A matplotlib Axes object which should be used for plotting.
    kwargs
        Feel free to replace/use these with any additional arguments you need.

    Example
    -------
    >>> import matplotlib.pyplot as plt
    >>>
    >>> tree = neighbor_joining(distances)
    >>> fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 8))
    >>> plot_nj_tree_radial(tree=tree, ax=ax)
    >>> fig.savefig("example_radial.png")

    """
    raise NotImplementedError()