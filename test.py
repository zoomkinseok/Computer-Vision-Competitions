# import os
# import torch
# from util import dataload
# from models import Covnet
# import torchvision
# import pandas as pd
#
# def load_model():
#     model = None
#
#     # Write your load function for classifier model here
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     model = Covnet()
#     dir = os.path.dirname(__file__)
#     model_dir = os.path.join(dir, 'model')
#     model.load_state_dict(torch.load(model_dir, map_location=device))
#     return model
#
# def evaluation(test_data, model):
#     model.eval()
#     outputs = model(test_data)
#     _, predicted = torch.max(outputs.data, 1)
#     model_output = predicted.cpu().numpy()
#
#     return model_output
# #
# # samples, labels = iter(dataloader).next()
# # plt.figure(figsize=(16,24))
# # grid_imgs = torchvision.utils.make_grid(samples[:24])
# # np_grid_imgs = grid_imgs.numpy()
# # plt.imshow(np.transpose(np_grid_imgs, (1,2,0)))
#
#
# validation_loss_list = []
# _,_, test_set = dataload()
# test_dataset = torch.utils.data.DataLoader(dataset=test_set, batch_size=32, shuffle=False)