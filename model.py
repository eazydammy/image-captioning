import torch
import torch.nn as nn
import torchvision.models as models


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features
    

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super(DecoderRNN, self).__init__()
        
        # define embedding
        self.embedding = nn.Embedding(vocab_size, embed_size)
        
        # define LSTM 
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        
        # define fully-connected layer for model output
        self.fc = nn.Linear(hidden_size, vocab_size)
        
        # initialize model weights
        self.w_init()
    
    def w_init(self):
        # initialize embedding weights
        self.embedding.weight.data.uniform_(-0.1,0.1)
        
        # initialize fully-connected layer weights
        nn.init.xavier_normal_(self.fc.weight)
        self.fc.weight.data.uniform_(-0.1,0.1)
        self.fc.bias.data.fill_(0.01)
        
        # initialize bias for all forget gates to 1. to improve performance
        # Source: https://discuss.pytorch.org/t/set-forget-gate-bias-of-lstm/1745/4
        for names in self.lstm._all_weights:
            for name in filter(lambda n: "bias" in n,  names):
                bias = getattr(self.lstm, name)
                n = bias.size(0)
                start, end = n//4, n//2
                bias.data[start:end].fill_(1.)
    
    def forward(self, features, captions):
        # embed captions
        captions_embed = self.embedding(captions[:,:-1]) # discard <end> tag in caption
        
        # concat features with captions embed
        embeddings = torch.cat((features.unsqueeze(1), captions_embed), 1)  
        
        # pass embeddings through LSTMs
        lstm_outputs, _ = self.lstm(embeddings)
        
        # pass LSTM outputs through output FC layer
        outputs = self.fc(lstm_outputs)
        
        return outputs

    def sample(self, inputs, states=None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        pass