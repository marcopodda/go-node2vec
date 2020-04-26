import sys
import argparse
import gensim

from umap import UMAP
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE

from matplotlib import pyplot as plt

from settings import NAMESPACES


REDUCERS = {"tsne": TSNE, "umap": UMAP}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--namespace", choices=NAMESPACES, default="molecular_function",
                        help='Namespace to extract from the ontology')
    parser.add_argument('--reducer', choices=REDUCERS.keys(), default="tsne")
    return parser.parse_args()


def parse_embedding_matrix(filename):
    model = gensim.models.KeyedVectors.load_word2vec_format(filename, binary=False)
    return model.index2word, model.vectors


def get_reduced_embeddings(data, reducer_name="umap"):
    assert reducer_name in REDUCERS.keys()
    data = StandardScaler().fit_transform(data)
    reducer_class = REDUCERS[reducer_name]
    reducer = reducer_class()
    embeddings = reducer.fit_transform(data)
    return embeddings


def plot_embeddings(embeddings, index=None, annotations=None, title="title"):
    plt.scatter(embeddings[:,0], embeddings[:,1], s=1, marker=".")
    if index is not None and annotations is not None:
        for key in annotations:
            idx = index.index(str(key))
            plt.annotate(annotations[key], (embeddings[idx, 0], embeddings[idx, 1]))
    plt.title(title)
    plt.show()


if __name__ == "__main__":
    args = parse_args()

    filename = f"DATA/{args.namespace}_emb.txt"
    index, data = parse_embedding_matrix(filename)
    embeddings = get_reduced_embeddings(data, reducer_name=args.reducer)

    # this annotation works with namespace == molecular_function
    # change it to something sensible in case you use any other namespace.
    annotations = {
        18722: "1-phenanthrol sulfotransferase activity",
        18723: "2-phenanthrol sulfotransferase activity",
        18727: "3-phenanthrol sulfotransferase activity"
    }

    plot_embeddings(embeddings, index=index, annotations=annotations, title=args.namespace)

