import matplotlib.pyplot as plt
import numpy as np
from keras import models

model = models.load_model('data/out/model.h5')
print(model.summary())

# img = visualize_activation(model, layer_idx=-1, filter_indices=[0])
# plt.imshow(img)

# Extracts the outputs of the layers
layer_outputs = [layer.output for layer in model.layers]
# Creates a model that will return these outputs, given the model input
activation_model = models.Model(inputs=model.input, outputs=layer_outputs)

# Load Sample Image
file = 'data/out/img/002_END_F7_0_0.csv'
img_tensor = np.loadtxt(file, delimiter=',')
img_tensor = np.expand_dims(img_tensor, axis=0)
img_tensor /= 255
activations = activation_model.predict(img_tensor)
cnn_l1 = activations[0][0].transpose()
cnn_l2 = activations[3][0].transpose()
print(cnn_l1.shape)
print(cnn_l2.shape)

plt.matshow(cnn_l1[:, :120], cmap='viridis')
plt.matshow(cnn_l2[:, :60], cmap='viridis')
plt.show()
