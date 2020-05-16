import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn.utils.rnn import pack_padded_sequence

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
        super().__init__()
        self.hidden_size = hidden_size
        self.word_embeddings = nn.Embedding(vocab_size, embed_size)
        self.num_layers = num_layers
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.hidden_to_tags = nn.Linear(hidden_size, vocab_size)

    def init_hidden(self, batch_size):
        # (n_layers, batch_size, hidden_dim)
        return (torch.zeros((self.num_layers, batch_size, self.hidden_size), device=device),
                torch.zeros((self.num_layers, batch_size, self.hidden_size), device=device))
    
    def forward(self, features, captions):        
        batch_size = features.shape[0]
        captions = captions[:, :-1]
        embeds = self.word_embeddings(captions)
        features = features.unsqueeze(1)
        join_image_text = torch.cat((features, embeds), 1)
        self.hidden = self.init_hidden(batch_size)
        lstm_out, _ = self.lstm(join_image_text, self.hidden)
        tag_outputs = self.hidden_to_tags(lstm_out)
                
        return tag_outputs

    def sample(self, inputs, states=None, max_len=20):
        '''
        accepts pre-processed image tensor (inputs) and returns predicted sentence
        (list of tensor ids of length max_len)
        '''
        current_embeds = inputs
        output = []
        batch_size = inputs.shape[0]
        
        if not states:
            hidden = self.init_hidden(batch_size)
        else:
            hidden = states
            
        for word_idx in range(0, max_len):
            lstm_out, hidden = self.lstm(current_embeds, hidden)
            tag_outputs = self.hidden_to_tags(lstm_out)
            tag_outputs = tag_outputs.squeeze(1)
            next_word = torch.argmax(tag_outputs, dim = 1)
            
            output.append(next_word.cpu().numpy()[0].item())
            
            if (next_word == 1):
                # <end> word, do not continue predicting
                break
            
            current_embeds = self.word_embeddings(next_word)
            current_embeds = current_embeds.unsqueeze(1)
            
        return output