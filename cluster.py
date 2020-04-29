import numpy as np
import numpy_indexed as npi
from sklearn.preprocessing import StandardScaler
import hdbscan as hdb
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt

from settings import DATA_DIR, get_or_create_dir, NAMESPACES
from visualize import parse_embedding_matrix

import seaborn as sns

sns.set_context('poster')
sns.set_style('white')
sns.set_color_codes()
plot_kwds = {'alpha': 0.5, 's': 1, 'linewidths': 0, 'marker': '.'}

PLOT_DIR = get_or_create_dir("PLOT")
dimensions = 128


def main():
    for namespace in NAMESPACES:
        # Load data
        filename = DATA_DIR / f"{namespace}_emb_{dimensions}.txt"
        index, data = parse_embedding_matrix(filename)

        # Normalization was not usefull since embeddings are somehow
        # already normalized (was tested but not was included in the
        # final version).
        #scaler = StandardScaler()
        #std_data = scaler.fit_transform(data)
        std_data = data

        # Projecting data in 2d for visualization pourpose.
        projection = TSNE().fit_transform(std_data)
        plt.scatter(*projection.T, **plot_kwds)

        # Clustering with HDBSCAN (density based algorithm) tested some distance measure and paprameters.
        clusterer = hdb.HDBSCAN(
            min_samples=15,
            min_cluster_size=2,
            metric='euclidean',
            cluster_selection_method='leaf',
            cluster_selection_epsilon=0.5)
        clusterer.fit(std_data)

        #Plot the clustering in 2d and save it in pdf. Visualization can be improved.
        color_palette = sns.color_palette('Paired', len(clusterer.labels_))
        cluster_colors = [color_palette[x] if x >= 0 else (0.5, 0.5, 0.5) for x in clusterer.labels_]
        cluster_member_colors = [sns.desaturate(x, p) for x, p in zip(cluster_colors, clusterer.probabilities_)]
        plt.scatter(*projection.T, c=cluster_member_colors, **plot_kwds)
        plt.title(namespace)
        plt.savefig(PLOT_DIR / f"{namespace}.pdf")

        #Packing and saving CLustering information for future use.
        clustering_data = np.column_stack((index,clusterer.labels_,clusterer.probabilities_))
        np.savetxt(
            fname=DATA_DIR / f"{namespace}_clsing_{dimensions}.txt",
            X=clustering_data,
            fmt='%s',header=f"{len(index)} {dimensions}",
            delimiter=' ',
            comments='')

        unique = np.unique(clustering_data[:, 1:2])
        f_handle = open(DATA_DIR / f"{namespace}_cls_{dimensions}.txt", 'a')
        f_handle.write(f"{len(unique)} {dimensions}\n")

        for element in unique:
            present = clustering_data[:,1]==element
            tmp_cluster = np.extract(present,clustering_data[:,0])
            hd=[element,len(tmp_cluster)]
            clus = np.append(hd,tmp_cluster)
            f_handle.write(f" {clus[0]}")
            for i in range(1,len(clus)):
                f_handle.write(f" {clus[i]}")
            f_handle.write("\n")

        f_handle.close()

if __name__ == "__main__":
    main()