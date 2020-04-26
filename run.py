import argparse
import numpy as np
import networkx as nx
import pandas as pd
from gensim.models import Word2Vec

from settings import DATA_DIR, NAMESPACES
from node2vec import Graph
from go_utils import get_GO_dag, filter_GO_dag_by_namespace


def parse_args():
    '''
    Parses arguments.
    '''

    parser = argparse.ArgumentParser(description="Run node2vec on a specified Gene Ontology.")

    parser.add_argument("namespace", choices=NAMESPACES,
                        help='Namespace to extract from the ontology')
    parser.add_argument('--dimensions', type=int, default=128,
                        help='Number of dimensions. Default is 128.')
    parser.add_argument('--window-size', type=int, default=7,
                        help='Context size for optimization. Default is 7.')
    parser.add_argument('--iter', default=5, type=int,
                        help='Number of epochs in SGD. Default is 5.')
    parser.add_argument('--workers', type=int, default=8,
                        help='Number of parallel workers. Default is 8.')
    parser.add_argument('--p', type=float, default=1,
                        help='Return hyperparameter. Default is 1.')
    parser.add_argument('--q', type=float, default=1,
                        help='Inout hyperparameter. Default is 1.')

    return parser.parse_args()


def create_graph(args, edges):
    '''
    Reads the input network in networkx.
    '''

    # use undirected graph
    nx_G = nx.Graph(edges)
    for edge in nx_G.edges():
        nx_G[edge[0]][edge[1]]['weight'] = 1

    # calculate node2vec parameters
    walk_length = nx.diameter(nx_G)
    degrees = dict(nx_G.degree())
    mean_degree = np.mean(list(degrees.values()))
    num_walks = int(10 * mean_degree)
    print(f"num_walks: {num_walks} walk_length: {walk_length}")

    G = Graph(nx_G, is_directed=False, p=args.p, q=args.q)
    return G, num_walks, walk_length


def learn_embeddings(args, walks):
    '''
    Learn embeddings by optimizing the Skipgram objective using SGD.
    '''
    walks = [list(map(str, walk)) for walk in walks]

    model = Word2Vec(
        walks,
        size=args.dimensions,
        window=args.window_size,
        min_count=0,
        sg=1,
        workers=args.workers,
        iter=args.iter)

    filename = f"{args.namespace}_emb_{args.dimensions}.txt"
    model.wv.save_word2vec_format(DATA_DIR / filename)


def run(args):
    '''
    Pipeline for representational learning for all nodes in a graph.
    '''
    GO_dag = get_GO_dag()
    edges = filter_GO_dag_by_namespace(GO_dag, args.namespace)
    G, num_walks, walk_length = create_graph(args, edges)
    G.preprocess_transition_probs()
    walks = G.simulate_walks(num_walks, walk_length)
    learn_embeddings(args, walks)


if __name__ == "__main__":
    args = parse_args()
    run(args)