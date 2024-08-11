from sklearn.datasets import fetch_olivetti_faces
from sklearn.utils import check_random_state
import matplotlib.pyplot as plt

# Load faces data
faces_dataset = fetch_olivetti_faces(shuffle=True, random_state=check_random_state(0))
dataset_faces = faces_dataset.data

def plot_gallery(title, images, n_row=2, n_col=5):
    plt.figure(figsize=(2. * n_col, 2.26 * n_row))
    plt.suptitle(title, size=16)

    for i, comp in enumerate(images):
        plt.subplot(n_row, n_col, i + 1)
        vmax = max(comp.max(), -comp.min())
        plt.imshow(comp.reshape((64, 64)), cmap=plt.cm.gray, interpolation='nearest', vmin=-vmax, vmax=vmax)
        plt.xticks(())
        plt.yticks(())
    
    plt.subplots_adjust(0.01, 0.05, 0.99, 0.94, 0.04, 0.)

plot_gallery("First centered Olivetti faces", dataset_faces[:10])
plt.show()
