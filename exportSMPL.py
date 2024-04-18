from inference import get_model
import numpy as np

model, _, _, _, _, _, mesh_model = get_model()

# Export Face of Mesh Model
np.save("SMPL.npy", mesh_model.face)
