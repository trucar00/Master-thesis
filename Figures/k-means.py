from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

X,y = make_blobs(n_samples = 300, n_features = 2, centers = 2, random_state = 42)
#print(X)
print(y)
plt.figure(figsize=(6,6))
plt.scatter(X[:,0],X[:,1], c="black")
plt.xticks([])   # remove x-axis ticks
plt.yticks([])   # remove y-axis ticks
plt.margins(x=0.02)
plt.savefig("k_means_init.pdf", bbox_inches="tight", pad_inches=0.05)
plt.show()

scaler = StandardScaler()
X = scaler.fit_transform(X)

km = KMeans(n_clusters=2, random_state=0, n_init=1)

y2 = km.fit_predict(X)

print(y2)

colors = ["blue", "orange"]

plt.figure(figsize=(6,6))
plt.scatter(
    X[:,0],
    X[:,1],
    c=[colors[i] for i in y2]
)

""" plt.scatter(
    km.cluster_centers_[:,0],
    km.cluster_centers_[:,1],
    color="red",
    marker="x",
    s=200,
    linewidths=3
) """
plt.xticks([])   # remove x-axis ticks
plt.yticks([])   # remove y-axis ticks
plt.margins(x=0.02)
plt.savefig("k_means_cluster.pdf", bbox_inches="tight", pad_inches=0.05)
plt.show()