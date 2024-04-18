from inference import get_model
import torch

model, _, _, _, _, _, _ = get_model()

scripted_model = torch.jit.script(model)
scripted_model.save('GTRS.pt')