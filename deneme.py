import torch 

# check if CUDA is available
train_on_gpu = torch.cuda.is_available()
print(train_on_gpu)

print([torch.cuda.device(i) for i in range(torch.cuda.device_count())])