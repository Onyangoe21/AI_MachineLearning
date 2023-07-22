from skimage import data, segmentation, color
from skimage.future import graph
from matplotlib import pyplot as plt
from PIL import Image
from skimage import io

path='/home/user/Desktop/Image1.jpeg'
img = io.imread(path)

labels1 = segmentation.slic(img, compactness=50, n_segments=5000,
                        start_label=1)
out1 = color.label2rgb(labels1, img, kind='avg', bg_label=0)
out1 = Image.fromarray(out1, 'RGB')
out1.save('/home/user/Desktop/1.png')
g = graph.rag_mean_color(img, labels1, mode='similarity')
labels2 = graph.cut_normalized(labels1, g)
out2 = color.label2rgb(labels2, img, kind='avg', bg_label=0)
out2 = Image.fromarray(out2, 'RGB')
out2.save('/home/user/Desktop/2.png')

fig, ax = plt.subplots(nrows=2, sharex=True, sharey=True, figsize=(6, 8))

ax[0].imshow(out1)
ax[1].imshow(out2)

for a in ax:
    a.axis('off')

plt.tight_layout()
plt.show()