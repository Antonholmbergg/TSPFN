"""The sampling algorithm:
Start with an MLP architecture and dropp weights from it. 

Sample dataest with k features and n samples (n, k) do:

1. sample a number of MLP layers l from p(l) and nodes per layer h from p(h).
2. Sample weights for the "edges" E_i,j from p_w(.)
3. drop a random set of edges. Set weights to 0?  # torch.nn.utils.prune.random_structured
4. sample a set of k feature nodes and a label node
5. sample noise distributions from a p(eps) from p(p(eps)). Is this the biases?
6. sample activation functions
fix SCM
for each member of the dataset (n times):
1. sample noise variables from eps_i from the sampled distributions
2. compute the value 
"""
import networkx as nx
import numpy as np

from tspfn.data import PriorHyperParameters


class SCM:
    def __init__(self, prior_hp: PriorHyperParameters):
        self.graph = nx.gnr_graph(n=prior_hp.n_nodes_total, p=prior_hp.redirection_probablility, seed=prior_hp.rng.integers(10_000, 100_000))


if __name__ == '__main__':
    gnr_graph = SCM()
    print(f"\nIs the graph a Directed Acyclic Graph (DAG)? {nx.is_directed_acyclic_graph(gnr_graph)}")

    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(8, 6))
    pos = nx.spring_layout(gnr_graph, seed=42)
    nx.draw(gnr_graph, pos, with_labels=True, node_color='lightblue',
            edge_color='gray', arrows=True, arrowsize=10, node_size=700)
    plt.title("Generated GNR Graph")
    fig.savefig("gnr_graph.png")




# if __name__ == "__main__":
#     scm = SCM(PriorHyperParameters())
#     print(scm.forward(torch.Tensor([1] * scm.layers[0].in_features)))
