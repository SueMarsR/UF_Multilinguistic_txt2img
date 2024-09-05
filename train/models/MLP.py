import torch
import torch.nn.functional as F
import torch.utils.checkpoint


class MLP(torch.nn.Module):
    def __init__(self, input_dim=1024, output_dim=512, dtype=torch.float16):
        super(MLP, self).__init__()
        # if dtype == "fp32":
        #     self.dtype = torch.float32
        # elif dtype == "fp16":
        #     self.dtype = torch.float16
        # elif dtype == "bf16":
        #     self.dtype = torch.bfloat16
        self.dtype = dtype
        self.fc1 = torch.nn.Linear(input_dim, input_dim * 2).to(self.dtype)
        self.fc2 = torch.nn.Linear(input_dim * 2, output_dim).to(self.dtype)

    def forward(self, x):
        b, n, m = x.shape  # 1,77,1024
        x = x.reshape(-1, m)  # 77, 1024
        # print("x.shape: ", x.shape, x.dtype)
        dout = F.relu(self.fc1(x.to(self.dtype)))  # 77, 2048
        dout = F.relu(self.fc2(dout))  # 77, 512
        dout = dout.reshape(b, n, -1)
        return dout.to(self.dtype)

    def save_pretrained(self, path):
        pass
