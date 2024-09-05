import torch

class CombinedModel(torch.nn.Module):
    def __init__(self, text_encoder, text_adapter):
        super(CombinedModel, self).__init__()
        self.text_encoder = text_encoder
        self.text_adapter = text_adapter

    def forward(self, inputs):
        encoder_outputs = self.text_encoder(inputs)
        adapter_outputs = self.text_adapter(encoder_outputs)
        return adapter_outputs
