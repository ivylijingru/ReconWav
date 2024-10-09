"""
Convert a tensor with shape [batch_size, seq_len_latent, shape_latent]
We want to convert to shape [batch_size, seq_len_feature, shape_feature]

What we can do is:
    First interpolate seq_len_latent to seq_len_feature (e.g. adaptive average pooling)
    Then do self-attention on latent embedding
    (This is similar to what FastSpeech is doing after )

Alternative:
    We use cross-attention and predict the features recursively
    First do self-attention on latent embedding
    (This idea tends to formulate the problem similar to translation)

TODO: WIP; will first implement a shorter version of things
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class basicMLP(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim,
        target_seq_len, # this guy is around 344 for 4 seconds
    ) -> None:
        super().__init__()
        self.target_seq_len = target_seq_len
        self.hidden = nn.Linear(input_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, output_dim)
        # TODO: we can consider cutting the mel spectrogram (maybe)
        self.interpolation_layer = nn.AdaptiveAvgPool1d(self.target_seq_len)
        dropout_prob = 0.5 # TODO: hardcode
        self.dropout = nn.Dropout(p=dropout_prob)
    
    def forward(self, x):
        x = x.transpose(-2, -1) # [batch_size, shape_latent, seq_len_latent]
        x = self.interpolation_layer(x) # [batch_size, shape_latent, seq_len_feature]
        x = x.transpose(-2, -1) # # [batch_size, seq_len_feature, shape_latent]

        # TODO: consider other methods other than MLP
        x = self.hidden(x)
        x = self.dropout(x)
        x = F.relu(x)

        output = self.output(x)
        return output.transpose(-2, -1)


if __name__ == "__main__":
    input_dim = 128
    hidden_dim = 512
    output_dim = 80
    target_seq_len = 344

    model = basicMLP(input_dim, hidden_dim, output_dim, target_seq_len)
    x = torch.zeros(16, 4, 128)
    print(model(x).shape)