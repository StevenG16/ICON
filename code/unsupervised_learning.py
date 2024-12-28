from data_preprocessing import query_to_dataset, data_files, queries
from sklearn.cluster import MiniBatchKMeans
import data_visualization


def test_clustering(dataset, num_clusters):
    kmeans = MiniBatchKMeans(n_clusters=num_clusters)
    kmeans.fit(dataset)
    return (kmeans.cluster_centers_)



def elbow_test(dataset, max_clusters):
    inertias = []
    for i in range(1, max_clusters+1):
            kmeans = MiniBatchKMeans(n_clusters=i)
            kmeans.fit(dataset)
            inertias.append(kmeans.inertia_)
    return inertias


def test_main():
    max_clusters = 20
    dataset = query_to_dataset(data_files["small_prolog"], queries["factors_all"])
    inertias = elbow_test(dataset, max_clusters)
    data_visualization.plot_elbow(max_clusters, inertias)

def main():
    
    dataset = query_to_dataset(data_files["small_prolog"], queries["factors_all"])
    num_clusters = 16
    centers = test_clustering(dataset, num_clusters)
    i = 1
    
    for center in centers:
        img_name = f"cluster_center{i}"
        data_visualization.plot_factors(center, img_name)
        i += 1 
    




if __name__ == "__main__":
    test_main()
