import torch
import os
# from .MLP import MLP

class AdaptiveTextEncoder(torch.nn.Module):
    # def __init__(self, text_encoder, input_dim, output_dim):
    def __init__(self, text_encoder, linear, text_adapter):
        super(AdaptiveTextEncoder, self).__init__()
        self.text_encoder = text_encoder
        # self.linear = MLP(input_dim=input_dim, output_dim=output_dim)  # 实例化linear
        self.linear = linear
        self.text_adapter = text_adapter
        

    def forward(self, input_ids, dtype):
        encoder_outputs = self.text_encoder(input_ids.long())[0]
        hidden_states_outputs = self.linear(encoder_outputs.to(dtype))
        adapter_outputs = self.text_adapter(hidden_states_outputs).to(dtype)
        return hidden_states_outputs, adapter_outputs
    
    def save_pretrained(self, save_directory):
        """
        Save the model's state_dict to the specified directory.
        """
        if not os.path.exists(save_directory):
            os.makedirs(save_directory)
        torch.save(self.state_dict(), os.path.join(save_directory, "text_encoder.pth"))