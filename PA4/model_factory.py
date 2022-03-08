import torch
import torch.nn as nn
import torchvision.models as models


################################################################################
# CSE 151B: Programming Assignment 4
# Code snippet by Ajit Kumar, Savyasachi
# Updated by Rohin
# Winter 2022
################################################################################

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

    def predict(self, feature, deterministic=True, temp=0.0, max_len=20):  # take batch output from the output layer
        # to convert it to caption
        out = []
        batch_size = feature.shape[0]
        hidden = self.init_hidden(batch_size)

        while True:
            x, hidden = self.LSTM(feature, hidden)
            outputs = self.out(x)
            outputs = outputs.squeeze(1)

            if deterministic:
                max_idx = outputs.argmax(dim=1)
            else:
                # Stochastic
                p = nn.functional.softmax(outputs / temp)
                max_idx = torch.multinomial(p, 1)
                max_idx = max_idx.reshape(-1)

            batch_idx = max_idx.cpu().tolist()  # the i-th word_idx for all feature in batch

            out.append(batch_idx)

            if (len(out) >= max_len):
                break

            feature = self.word_embed(max_idx)
            feature = feature.unsqueeze(1)

        out = torch.transpose(torch.tensor(out), 0, 1).tolist()  # change it to (n, t)

        return out


class RNN_decoder(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, device, num_layers=2):
        super(RNN_decoder, self).__init__()

        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        self.device = device

        self.word_embed = nn.Embedding(self.vocab_size, self.embed_size)
        self.RNN = nn.RNN(
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

        output, self.hidden = self.RNN(caption_embed, self.hidden)
        output = self.out(output)

        return output

    def init_hidden(self, batch_size):
        return torch.zeros(self.num_layers, batch_size, self.hidden_size).to(self.device)  # RNN use H_0 only

    def predict(self, feature, deterministic=True, temp=0.0, max_len=20):  # take batch output from the output layer to
        # convert it to caption
        out = []
        batch_size = feature.shape[0]
        hidden = self.init_hidden(batch_size)

        while True:
            x, hidden = self.RNN(feature, hidden)
            outputs = self.out(x)
            outputs = outputs.squeeze(1)

            if deterministic:
                max_idx = outputs.argmax(dim=1)
            else:
                # Stochastic
                p = nn.functional.softmax(outputs / temp)
                max_idx = torch.multinomial(p, 1)
                max_idx = max_idx.reshape(-1)

            batch_idx = max_idx.cpu().tolist()  # the i-th word_idx for all feature in batch

            out.append(batch_idx)

            if len(out) >= max_len:
                break

            feature = self.word_embed(max_idx)
            feature = feature.unsqueeze(1)

        out = torch.transpose(torch.tensor(out), 0, 1).tolist()  # change it to (n, t)

        return out


class arch2_decoder(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, device, num_layers=2):
        super(arch2_decoder, self).__init__()

        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        self.device = device

        self.word_embed = nn.Embedding(self.vocab_size, self.embed_size)
        self.LSTM = nn.LSTM(
            input_size=self.embed_size * 2,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True
        )

        self.out = nn.Linear(in_features=self.hidden_size, out_features=self.vocab_size)

    def forward(self, x, captions):
        self.batch_size = x.shape[0]
        self.hidden = self.init_hidden(self.batch_size)

        # adding <pad> at the beginning, <pad> is 0 in vocabulary
        pads = torch.zeros((self.batch_size, 1), dtype=torch.int64).to(self.device)
        captions = torch.cat((pads, captions), dim=1)

        caption_embed = self.word_embed(captions[:, :-1])
        # instead of concat the image feature as the first input, concat it into all inputs
        caption_embed = torch.cat((x.unsqueeze(dim=1).repeat(1, caption_embed.size()[1], 1), caption_embed), 2)

        output, self.hidden = self.LSTM(caption_embed, self.hidden)
        output = self.out(output)

        return output

    def init_hidden(self, batch_size):
        return (torch.zeros(self.num_layers, batch_size, self.hidden_size).to(self.device),
                torch.zeros(self.num_layers, batch_size, self.hidden_size).to(self.device))

    def predict(self, feature, deterministic=True, temp=0.0, max_len=20):  # take batch output from the output layer
        # to convert it to caption
        out = []
        batch_size = feature.shape[0]
        hidden = self.init_hidden(batch_size)

        # initialize <pad> as the first input
        pads = torch.zeros((batch_size, 1), dtype=torch.int64).to(self.device)
        pad_embed = self.word_embed(pads)
        inputs = torch.cat((feature, pad_embed), 2)


        while True:
            x, hidden = self.LSTM(inputs, hidden)
            outputs = self.out(x)
            outputs = outputs.squeeze(1)

            if deterministic:
                max_idx = outputs.argmax(dim=1)
            else:
                # Stochastic
                p = nn.functional.softmax(outputs / temp)
                max_idx = torch.multinomial(p, 1)
                max_idx = max_idx.reshape(-1)

            batch_idx = max_idx.cpu().tolist()  # the i-th word_idx for all feature in batch

            out.append(batch_idx)

            if (len(out) >= max_len):
                break

            inputs = self.word_embed(max_idx)
            inputs = inputs.unsqueeze(1)
            inputs = torch.cat((feature, inputs), 2)


        out = torch.transpose(torch.tensor(out), 0, 1).tolist()  # change it to (n, t)

        return out
