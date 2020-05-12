## Summary
Nifty project I did for an algorithms class and cleaned up. I'll probably fiddle around with it some more for fun later. The project tries to find approximations of two combined NP hard problems: minimum routing cost tree and dominating set. The results are trees which span or are adjacent to every vertex which minimize average distance between every vertex. The main application for this type of algorithm is designing communication networks with fast communication between every node. Hopefully I didn't break anything
## Files
- solver.py: Main file which contains the solving algorithms.
- utils.py: Checks validity and computes cost of solutions.
- parse.py: Parses input and output graphs.
- generator.py: Original graph generator I made to generate some test inputs.
- examples:A couple of examples of inputs and outputs
## Solvers
The main methods used were shortest path tree algorithms, minimum spanning tree algorithms, or variations thereof using a heuristic based on a [paper](https://doi.org/10.1016/j.comnet.2008.08.013) by Campos and Ricardo. The heuristic is used as a replacement for the costs of edges in Dijkstra's algorithm and accounts for the weight of edges, degree of vertices, average distance from vertices added so far and median distance to all vertices. Generally shortest path trees performed best, and it was [proven](https://doi.org/10.1137/0601008) that this approach finds a minimum routing cost tree at most twice the cost of the optimal, though this might not transfer to the dominating set part of the problem.

The resulting spanning trees are then greedily trimmed by removing leaves which reduce the cost of the tree. Trees were also pre-trimmed by removing vertices with high distance from the targets of Dijkstra's. Several other algorithms focused on finding good dominating sets first instead, including randomized ones. The evolutionary algorithm uses a similar structure but randomly adds and removes leaves from a tree and reweights edges multiplicatively.

The heuristic was trained on a randomly selected subset inputs for faster training, though likely less accurate. Training on the whole input space was tried but it was slow enough to quickly deter that. Another loss that was tried was counting improvement on a set of basic shortest path trees because it allowed for consideration of trees that different significantly from regular shortest path trees.


