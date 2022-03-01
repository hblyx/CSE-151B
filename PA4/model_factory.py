import torch
import torch.nn as nn
import torchvision.models as models


################################################################################
# CSE 151B: Programming Assignment 4
# Code snippet by Ajit Kumar, Savyasachi
# Updated by Rohin
# Winter 2022
################################################################################

# Build and return the model here based on the configuration.
def get_model(config_data, vocab):
    hidden_size = config_data['model']['hidden_size']
    embedding_size = config_data['model']['embedding_size']
    model_type = config_data['model']['model_type']

    # You may add more parameters if you want

    raise NotImplementedError("Model Factory Not Implemented")


class baseline_encoder(nn.Module):
    def __init__(self, embed_size):
        super(baseline_encoder, self).__init__()

        self.resnet = models.resnet50(pretrained=True)
        # freeze
        for param in self.resnet.parameters():
            param.requires_grad_(False)
        # replace the classifier with a fc embedding layer
        self.resnet.fc = nn.Linear(in_features=self.resnet.fc.in_features, out_features=embed_size)

    def forward(self, image):
        return self.resnet(image)


class baseline_decoder(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, device, num_layers=2):
        super(baseline_decoder, self).__init__()

        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        self.device = device

        self.word_embed = nn.Embedding(self.vocab_size, self.embed_size)
        self.LSTM = nn.LSTM(
            input_size=self.embed_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True
        )

        self.out = nn.Linear(in_features=self.hidden_size, out_features=self.vocab_size)

    def forward(self, x, captions):
        self.batch_size = x.shape[0]
        self.hidden = self.init_hidden(self.batch_size)
        caption_embed = self.word_embed(captions[:, :-1])
        caption_embed = torch.cat((x.unsqueeze(dim=1), caption_embed), 1)

        output, self.hidden = self.LSTM(caption_embed, self.hidden)
        output = self.out(output)

        return output

    def init_hidden(self, batch_size):
        return (torch.zeros(self.num_layers, batch_size, self.hidden_size).to(self.device),
                torch.zeros(self.num_layers, batch_size, self.hidden_size).to(self.device))

    def predict(self, feature, max_len=20):  # take batch output from the output layer to convert it to caption
        out = []
        batch_size = feature.shape[0]
        hidden = self.init_hidden(batch_size)

        while True:
            x, hidden = self.LSTM(feature, hidden)
            outputs = self.out(x)
            outputs = outputs.squeeze(1)
            _, max_idx = torch.max(outputs, dim=1)
            out.append(max_idx.cpu().numpy()[0].item())

            if (max_idx == 2 or len(out) >= max_len):
                break

            feature = self.word_embed(max_idx)
            feature = inputs.unsqueeze(1)

            return final_output