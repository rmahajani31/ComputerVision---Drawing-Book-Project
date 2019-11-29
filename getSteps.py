from sklearn.cluster import AgglomerativeClustering
import numpy as np

def clusterShapes(X):
    clustering = AgglomerativeClustering(n_clusters=5)
    clustering.fit(X)
    return clustering.labels_
