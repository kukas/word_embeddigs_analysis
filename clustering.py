import streamlit as st
import pandas as pd
import numpy as np
import csv
import sys
import os
import random
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from sklearn.decomposition import FastICA

import plotly.express as px

random.seed(42)

# read the tsv file into a list of lists
@st.cache
def load_data():
    vocab = []
    embeddings = []
    with open("nmt-en-dec-512.txt") as f:
        for line in f:
            row = line.split("\t")
            vocab.append(row[0])
            embeddings.append(row[1:])
            assert len(row) == 513
    assert len(vocab) == len(embeddings)
    assert len(vocab) == 5000

    embeddings = np.array(embeddings, dtype=np.float32)

    return vocab, embeddings


@st.cache
def run_tsne(embeddings, perplexity, metric):
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        metric=metric,
        n_iter=2000,
        n_iter_without_progress=300,
        random_state=42,
        verbose=1,
    )
    embeddings_2d = tsne.fit_transform(embeddings)
    return embeddings_2d


st.title("Assignment 3 - Word Clustering and Component Analysis")
st.subheader("Jiří Balhar")

data_load_state = st.text("Loading data...")
vocab, embeddings = load_data()
data_load_state.text("Loading data...done!")

st.subheader("T-SNE visualization of word embeddings")

# perplexity = st.slider(
#     "Select the perplexity", min_value=1, max_value=100, value=30, step=10
# )
perplexity = st.selectbox("Select the perplexity", (1, 30, 100), index=1)

metric = st.selectbox("Select the metric", ("euclidean", "cosine"), index=1)

tsne_embeddings = run_tsne(embeddings, perplexity, metric)

# show scatter plot of word embeddings
def show_clusters(embeddings, labels, clusters=None, title=None):
    if clusters is not None:
        # convert to categorical values for plotly
        clusters = [str(c) for c in clusters]
    fig = px.scatter(
        embeddings,
        x=0,
        y=1,
        hover_name=labels,
        color=clusters,
        title=title,
        labels={"0": "dim. 1", "1": "dim. 2", "color": "Cluster"},
    )
    st.plotly_chart(fig)


"""
By experimenting with different perplexity values and metrics, we have found good results for perplexity=30 and metric=cosine.

*Note that the t-SNE visualization is interactive, so you can hover over the points to see the word labels.*
"""

show_clusters(tsne_embeddings, vocab, title="t-SNE visualization of word embeddings")

st.subheader("Clustering of word embeddings")
# preset = st.radio("Select a clustering preset", ("Comedy", "Drama", "Documentary"))

"""
Here we can select the clustering method and the number of clusters. For agglomerative clustering it is also possible to select the linkage strategy. The summary of the results is below the interactive plot.
"""

method = st.selectbox(
    "Select the clustering method",
    ("k-means", "gaussian-mixture", "agglomerative-clustering"),
)
# n_clusters = st.slider(
#     "Select the number of clusters", min_value=2, max_value=100, value=10, step=1
# )
n_clusters = st.selectbox(
    "Select the number of clusters",
    (3,5,10),
)
linkage_strategy = None
if method == "agglomerative-clustering":
    linkage_strategy = st.selectbox(
        "Select the linkage strategy", ("ward", "single", "complete")
    )


@st.cache
def get_clusters(method, n_clusters, embeddings, linkage_strategy=None):
    if method == "k-means":
        kmeans = KMeans(n_clusters=n_clusters, random_state=42).fit(embeddings)
        clusters = kmeans.labels_
    elif method == "gaussian-mixture":
        gmm = GaussianMixture(n_components=n_clusters, random_state=42).fit(
            embeddings
        )
        clusters = gmm.predict(embeddings)
    elif method == "agglomerative-clustering":

        agg = AgglomerativeClustering(
            n_clusters=n_clusters, linkage=linkage_strategy
        ).fit(embeddings)
        clusters = agg.labels_
    return clusters


clusters = get_clusters(method, n_clusters, tsne_embeddings, linkage_strategy)
silhouette = silhouette_score(tsne_embeddings, clusters)

method_name = method
if linkage_strategy is not None:
    method_name = f"{method} ({linkage_strategy})"
show_clusters(
    tsne_embeddings,
    vocab,
    clusters,
    f"{method_name} clustering, n_clusters={n_clusters}, silhouette={silhouette:.3f}",
)

# save the results to a csv file
if os.path.exists("clustering_results.csv"):
    results = pd.read_csv("clustering_results.csv")
else:
    results = pd.DataFrame(
        columns=[
            "tsne_perplexity",
            "tsne_metric",
            "method",
            "n_clusters",
            "linkage_strategy",
            "silhouette",
        ]
    )
# add the results to the dataframe
results = results.append(
    {
        "tsne_perplexity": perplexity,
        "tsne_metric": metric,
        "method": method,
        "n_clusters": n_clusters,
        "linkage_strategy": linkage_strategy,
        "silhouette": silhouette,
    },
    ignore_index=True,
)
# delete duplicates
results = results.drop_duplicates(["method", "n_clusters", "linkage_strategy"])

# sort the results by silhouette score
results = results.sort_values(by="silhouette", ascending=False)

results.to_csv("clustering_results.csv", index=False)

st.subheader("Results")
st.write(results)

"""
We have found the best silhouette score for the k-means algorithm with 10 clusters. The silhouette score is for this setting is 0.414. 
High silhouette score means that the clusters are well separated from each other and that the clusters are dense.

Visually inspecting the outputs of the algorithms, we see that the algorithms produce meaningful clusters. For low number of clusters 
we see fuzzy separation into 1) common nouns, 2) proper nouns, pronouns, numbers and 3) verbs, adjectives. For higher number of clusters 
we see some clusters with clear common characteristic such as the cluster of names of people at the top of the point cloud. For some clusters it is harder to find a common characteristic.

Among the algorithms, we see one outlier in agglomerative clustering with single linkage strategy. This 
setting seems to be better suited for larger number of clusters because the clusters it produces are very small. Other than that, the
other algorithms produce similar results. 

For k-means we see the sharp boundaries between the clusters which is expected because the algorithm partitions the Euclidean data space 
into Voronoi cells. Especially for small number of clusters the boundary therefore does not follow the shape of the clusters.

In this regard the agglomerative clustering with ward linkage strategy seems to produce more natural clusters.
"""

st.subheader("Principal Component Analysis")

pca = PCA()
pca.fit(embeddings)
pca_embeddings = pca.transform(embeddings)

# select the first two principal components
pca_embeddings_2d = pca_embeddings[:, :2]

# pca_embeddings

show_clusters(
    pca_embeddings_2d,
    vocab,
    clusters,
    f"PCA with {method_name} clusters, n_clusters={n_clusters}, silhouette={silhouette:.3f}",
)

"""
Here we can see the PCA visualization of the word embeddings. The visualized clusters are the same as in the t-SNE visualization. We can therefore
compare the two visualizations. As we can see the PCA visualization does not show the clusters in the original data space. The clusters
are not well separated in the projected space. This is expected because PCA is a linear projection and the clusters are not linearly separable in our case.

On the other hand, we can look for more general patterns in the data. Therefore if we set the number of clusters to 3, we can see that
the first principal component separates the words nouns and non-nouns.
The second principal component separates the nouns into common nouns and proper nouns. 
"""

"""
Here we can again select the clustering method and the number of clusters.
"""

method = st.selectbox(
    "Select the clustering method in the PCA space",
    ("k-means", "gaussian-mixture", "agglomerative-clustering"),
)
# n_clusters = st.slider(
#     "Select the number of clusters", min_value=2, max_value=100, value=10, step=1
# )
n_clusters = st.selectbox(
    "Select the number of clusters in the PCA space",
    (3,5,10),
)
linkage_strategy = None
if method == "agglomerative-clustering":
    linkage_strategy = st.selectbox(
        "Select the linkage strategy.", ("ward", "single", "complete")
    )



clusters = get_clusters(method, n_clusters, pca_embeddings_2d, linkage_strategy)
silhouette = silhouette_score(pca_embeddings_2d, clusters)

method_name = method
if linkage_strategy is not None:
    method_name = f"{method} ({linkage_strategy})"
show_clusters(
    pca_embeddings_2d,
    vocab,
    clusters,
    f"{method_name} clustering, n_clusters={n_clusters}, silhouette={silhouette:.3f}",
)

# save the results to a csv file
if os.path.exists("clustering_pca_results.csv"):
    results = pd.read_csv("clustering_pca_results.csv")
else:
    results = pd.DataFrame(
        columns=[
            "method",
            "n_clusters",
            "linkage_strategy",
            "silhouette",
        ]
    )
# add the results to the dataframe
results = results.append(
    {
        "method": method,
        "n_clusters": n_clusters,
        "linkage_strategy": linkage_strategy,
        "silhouette": silhouette,
    },
    ignore_index=True,
)
# delete duplicates
results = results.drop_duplicates(["method", "n_clusters", "linkage_strategy"])

# sort the results by silhouette score
results = results.sort_values(by="silhouette", ascending=False)

results.to_csv("clustering_pca_results.csv", index=False)

st.subheader("Results")
st.write(results)

"""
Here we see the results for clustering in the PCA space. We can see that the highest silhouette score 
is for k-means with 3 clusters and the score is higher than in the t-SNE space. The reason for this might be 
that the clustering here is done in the first two dimensions of the PCA space. The first principal components capture the most variance in the data. And so it is easier to form more dense clusters in the first two dimensions.

On the other hand, we can see that because we discard the other principal components, we lose some information about the data. 
The results for higher number of clusters seem not to be as meaningful as in the original space. This is also reflected in the silhouette scores, 
we can see that now the best score is for 3 clusters while in the original space the best score was for 10 clusters.
"""


def visualize_component(embeddings, vocab, i, explained_variance_ratio=None, algoname="PCA"):
    # plt.figure(figsize=(20, 5))
    # (n, bins, patches) = plt.hist(embeddings, bins=100)
    # plt.title(f"PCA component {i}")
    # plt.show()


    # create the bins
    counts, bins = np.histogram(embeddings, bins=100)

    # get the words in each bin
    bin_words = []
    for j in range(len(bins) -1):
        bin_words.append([])
        for k in range(len(embeddings)):
            if embeddings[k] >= bins[j] and embeddings[k] < bins[j + 1]:
                bin_words[j].append(vocab[k])

    # set the x ticks to be one random word sampled from each bin
    single_bin_words = []
    multiple_bin_words = []
    for j, bin_value in zip(range(len(bin_words)), bins):
        # random sample from bin_words[j]
        if len(bin_words[j]) == 0:
            single_bin_words.append(f"{bin_value:.2f}")
            multiple_bin_words.append("")
            continue
        word = random.sample(bin_words[j], 1)[0]
        single_bin_words.append(f"{word}, {bin_value:.2f}")
        more_words = set([word]).union(set(random.sample(bin_words[j], min(5, len(bin_words[j])))))
        more_words = ", ".join(more_words)
        multiple_bin_words.append(more_words)
    # single_bin_words.append("")
    # plt.xticks(bins, single_bin_words, rotation=90)


    bins = 0.5 * (bins[:-1] + bins[1:])

    if explained_variance_ratio is None:
        explained_variance_ratio = ""
    else:
        explained_variance_ratio = f", explained variance = {explained_variance_ratio*100:.1f}%"


    fig = px.bar(y=single_bin_words, x=counts, text=multiple_bin_words, title=f"{algoname} component {i+1}"+explained_variance_ratio, labels={'y':f'{i+1}. component', 'x':'number of words'}, height=800,)
    fig.update_traces(hovertemplate='sampled words: %{text}')
    fig.update_layout(hovermode="y")
    st.plotly_chart(fig)
    # fig = px.bar(df, y='pop', x='country', text='pop')Principal
    # fig.update_traces(texttemplate='%{text:.2s}', textposition='outside')
    # fig.update_layout(uniformtext_minsize=8, uniformtext_mode='hide')
    # fig.show()

st.subheader("Visualization of the principal components")

"""
Here we show sampled words from each bin in the histogram of the first five principal components. The plot shows the number of words in each bin and on hover we show the sampled words from each bin.
"""

visualize_component(pca_embeddings[:, 0], vocab, 0, pca.explained_variance_ratio_[0])

"""
The first principal component seems to correspond to noun-ness of the word. We can see that the words with the highest values are nouns and the words with the lowest values are adjectives or verbs.
"""

visualize_component(pca_embeddings[:, 1], vocab, 1, pca.explained_variance_ratio_[1])

"""
For the second PC we see that the words with the highest values are names of people or generally human and nonhuman "actors" - such as King, merchant, troll, elf.
"""

visualize_component(pca_embeddings[:, 2], vocab, 2, pca.explained_variance_ratio_[2])

"""
The third PC is interesting, it seems that the high values correspond to words that relate to movement such as jumping, stepped, squeezed, flipped, falls. 
On the other end we see many words that relate to emotional reactions - pleased, amused, despair, anxious, complained, etc.
"""

visualize_component(pca_embeddings[:, 3], vocab, 3, pca.explained_variance_ratio_[3])

"""
Here I am not sure, the positive words seem to be more narrative and expressive and the negative words seem to be more formal.
"""

visualize_component(pca_embeddings[:, 4], vocab, 4,  pca.explained_variance_ratio_[4])

"""
The interpretion is getting harder and harder. It seems to me that the positive words are about communication and formal social interaction (greeted, convince, interview, instructed)
"""

st.subheader("Independent Component Analysis")

"""
Lastly, we also compute the Independent Component Analysis (ICA) of the embeddings. 
The distributions have a bell shape but some of them are not symmetric and have longer tails. The principle behind ICA is to find independent, non-gaussian distributions, that make up the observed signal.
To be sure, we can use a statistical test and check if the distribution is normal.
"""

ica = FastICA(n_components=50, random_state=42)
ica.fit(embeddings)
ica_embeddings = ica.transform(embeddings)

comments = [
    "The first dimension seems to relate to movement again",
    "The second dimension seems to be about communication and social interaction",
    "The third dimension separates out adverbs",
    "The fourth dimension has materials and objects on the negative side",
    "The fifth dimension separates past tense verbs",
]

from scipy import stats
for i in range(5):
    visualize_component(ica_embeddings[:, i], vocab, i, algoname="ICA")


    x = ica_embeddings[:, i]
    k2, p = stats.normaltest(x)
    alpha = 1e-3
    st.write(comments[i])
    if p < alpha:  # null hypothesis: x comes from a normal distribution
        st.write(f"The null hypothesis can be rejected, the distribution is not normal, p = {p}")
    else:
        st.write("The null hypothesis cannot be rejecteed, the distribution may be normal")

