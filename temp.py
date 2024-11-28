import torch
import torch.nn as nn
import math
import logging
from typing import Optional

logger = logging.getLogger(__name__)


# position wise encoding
class PositionalEncodingComponent(nn.Module):
    '''
    Class to encode positional information to tokens.
    For future, I want that this class to work even for sequences longer than 5000
    '''

    def __init__(self, hid_dim, dropout=0.2, max_len=5000):
        super().__init__()

        assert hid_dim % 2 == 0  # If not, it will result error in allocation to positional_encodings[:,1::2] later

        self.dropout = nn.Dropout(dropout)

        self.positional_encodings = nn.Parameter(torch.zeros(1, max_len, hid_dim), requires_grad=False)
        # Positional Embeddings : [1,max_len,hid_dim]

        pos = torch.arange(0, max_len).unsqueeze(1)  # pos : [max_len,1]
        div_term = torch.exp(-torch.arange(0, hid_dim, 2) * math.log(
            10000.0) / hid_dim)  # Calculating value of 1/(10000^(2i/hid_dim)) in log space and then exponentiating it
        # div_term: [hid_dim//2]

        self.positional_encodings[:, :, 0::2] = torch.sin(pos * div_term)  # pos*div_term [max_len,hid_dim//2]
        self.positional_encodings[:, :, 1::2] = torch.cos(pos * div_term)

    def forward(self, x):
        # TODO: update this for very long sequences
        x = x + self.positional_encodings[:, :x.size(1)].detach()
        return self.dropout(x)


# feed forward
class FeedForwardComponent(nn.Module):
    '''
    Class for pointwise feed forward connections
    '''

    def __init__(self, hid_dim, pf_dim, dropout):
        super().__init__()

        self.dropout = nn.Dropout(dropout)

        self.fc1 = nn.Linear(hid_dim, pf_dim)
        self.fc2 = nn.Linear(pf_dim, hid_dim)

    def forward(self, x):
        # x : [batch_size,seq_len,hid_dim]
        x = self.dropout(torch.relu(self.fc1(x)))

        # x : [batch_size,seq_len,pf_dim]
        x = self.fc2(x)

        # x : [batch_size,seq_len,hid_dim]
        return x


# multi headed attention
class MultiHeadedAttentionComponent(nn.Module):
    '''
    Multiheaded attention Component.
    '''

    def __init__(self, hid_dim, n_heads, dropout):
        super().__init__()

        assert hid_dim % n_heads == 0  # Since we split hid_dims into n_heads

        self.hid_dim = hid_dim
        self.n_heads = n_heads  # no of heads in 'multiheaded' attention
        self.head_dim = hid_dim // n_heads  # dims of each head

        # Transformation from source vector to query vector
        self.fc_q = nn.Linear(hid_dim, hid_dim)

        # Transformation from source vector to key vector
        self.fc_k = nn.Linear(hid_dim, hid_dim)

        # Transformation from source vector to value vector
        self.fc_v = nn.Linear(hid_dim, hid_dim)

        self.fc_o = nn.Linear(hid_dim, hid_dim)

        self.dropout = nn.Dropout(dropout)

        # Used in self attention for smoother gradients
        self.scale = nn.Parameter(torch.sqrt(torch.FloatTensor([self.head_dim])), requires_grad=False)

    def forward(self, query, key, value, mask: Optional[torch.Tensor] = None):
        # query : [batch_size, query_len, hid_dim]
        # key : [batch_size, key_len, hid_dim]
        # value : [batch_size, value_len, hid_dim]

        batch_size = query.shape[0]

        # Transforming quey,key,values
        Q = self.fc_q(query)
        K = self.fc_k(key)
        V = self.fc_v(value)

        # Q : [batch_size, query_len, hid_dim]
        # K : [batch_size, key_len, hid_dim]
        # V : [batch_size, value_len,hid_dim]

        # Changing shapes to acocmadate n_heads information
        Q = Q.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        K = K.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        V = V.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)

        # Q : [batch_size, n_heads, query_len, head_dim]
        # K : [batch_size, n_heads, key_len, head_dim]
        # V : [batch_size, n_heads, value_len, head_dim]

        # Calculating alpha
        score = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.scale
        # score : [batch_size, n_heads, query_len, key_len]

        if mask is not None:
            score = score.masked_fill(mask == 0, -1e10)

        alpha = torch.softmax(score, dim=-1)
        # alpha : [batch_size, n_heads, query_len, key_len]

        # Get the final self-attention  vector
        x = torch.matmul(self.dropout(alpha), V)
        # x : [batch_size, n_heads, query_len, head_dim]

        # Reshaping self attention vector to concatenate
        x = x.permute(0, 2, 1, 3).contiguous()
        # x : [batch_size, query_len, n_heads, head_dim]

        x = x.view(batch_size, -1, self.hid_dim)
        # x: [batch_size, query_len, hid_dim]

        # Transforming concatenated outputs
        x = self.fc_o(x)
        # x : [batch_size, query_len, hid_dim]

        return x, alpha


# EncodingLayer
class EncodingLayer(nn.Module):
    '''
    Operations of a single layer. Each layer contains:
    1) multihead attention, followed by
    2) LayerNorm of addition of multihead attention output and input to the layer, followed by
    3) FeedForward connections, followed by
    4) LayerNorm of addition of FeedForward outputs and output of previous layerNorm.
    '''

    def __init__(self, hid_dim, n_heads, pf_dim, dropout):
        super().__init__()

        self.self_attn_layer_norm = nn.LayerNorm(hid_dim)  # Layer norm after self-attention
        self.ff_layer_norm = nn.LayerNorm(hid_dim)  # Layer norm after FeedForward component

        self.self_attention = MultiHeadedAttentionComponent(hid_dim, n_heads, dropout)
        self.feed_forward = FeedForwardComponent(hid_dim, pf_dim, dropout)

        self.dropout = nn.Dropout(dropout)

    def forward(self, src, src_mask=None):
        # src : [batch_size, src_len, hid_dim]
        # src_mask : [batch_size, 1, 1, src_len]

        # get self-attention
        _src, _ = self.self_attention(src, src, src, src_mask)

        # LayerNorm after dropout
        src = self.self_attn_layer_norm(src + self.dropout(_src))
        # src : [batch_size, src_len, hid_dim]

        # FeedForward
        _src = self.feed_forward(src)

        # layerNorm after dropout
        src = self.ff_layer_norm(src + self.dropout(_src))
        # src: [batch_size, src_len, hid_dim]

        return src


class AudioRepresentations(nn.Module):
    '''
    Group of layers that give final audio representation for cross attention

    The class get an input of size [batch_size,max_audio_len]
    we split the max_audio_len by audio_split_samples.
    Example: if the input was [10,60000] and audio_split_samples as 1000
    then we split the input as [10,60,1000]
    '''

    def __init__(self, audio_split_samples, hid_dim, n_layers, n_heads, pf_dim, dropout, max_length):
        super().__init__()

        # Used for splitting the original signal
        self.audio_split_samples = audio_split_samples

        # Transform input from audio_split_dim to hid_dim
        self.transform_input = nn.Linear(audio_split_samples, hid_dim)

        self.pos_embedding = PositionalEncodingComponent(hid_dim, dropout, max_length)

        self.layers = nn.ModuleList([EncodingLayer(hid_dim, n_heads, pf_dim, dropout) for _ in range(n_layers)])

        self.dropout = nn.Dropout(dropout)

        self.scale = nn.Parameter(torch.sqrt(torch.FloatTensor([hid_dim])), requires_grad=False)

    def forward(self, audio):
        # You don't need mask for audio in attention because that padded
        # audio : [batch_size, max_audio_len]

        assert audio.shape[1] % self.audio_split_samples == 0

        batch_size = audio.shape[0]
        audio = audio.reshape(batch_size, -1, self.audio_split_samples)
        # audio : [batch_size, src_len , audio_split_samples]

        audio_embeddings = self.transform_input(audio) * self.scale
        # audio embeddings : [batch_size, src_len, hid_dim]

        # TODO: find better ways to give positional information. Here it is giving each audio_split_sample chunk same
        #  positional embedding
        audio = self.pos_embedding(audio_embeddings)
        # audio : [batch_size, src_len, hid_dim]

        for layer in self.layers:
            audio = layer(audio)
        # audio : [batch_size, src_len, hid_dim]

        return audio


class TextRepresentations(nn.Module):
    """
    Group of layers that give final text representation for cross attention
    """

    def __init__(self, input_dim, hid_dim, n_layers, n_heads, pf_dim, dropout, text_pad_index, max_length=5000):
        super().__init__()

        self.tok_embedding = nn.Embedding(input_dim, hid_dim)
        self.pos_embedding = PositionalEncodingComponent(hid_dim, dropout, max_length)

        # encoder layers
        self.layers = nn.ModuleList([EncodingLayer(hid_dim, n_heads, pf_dim, dropout) for _ in range(n_layers)])

        self.dropout = nn.Dropout(dropout)

        self.scale = nn.Parameter(torch.sqrt(torch.FloatTensor([hid_dim])), requires_grad=False)

        self.text_pad_index = text_pad_index

    def create_text_mask(self, text):
        # masks padded values of text

        # text : [batch_size, src_len]
        text_mask = (text != self.text_pad_index).unsqueeze(1).unsqueeze(2)

        return text_mask

    def forward(self, text):
        # text : [batch_size, src_len]

        text_mask = self.create_text_mask(text)
        # text_mask : [batch_size,1,1,src_len]

        batch_size = text.shape[0]
        src_len = text.shape[1]

        tok_embeddings = self.tok_embedding(text) * self.scale

        # token plus position embeddings
        text = self.pos_embedding(tok_embeddings)

        for layer in self.layers:
            text = layer(text, text_mask)
        # src : [batch_size, src_len, hid_dim]

        return text


class ImageRepresentations(nn.Module):
    def __init__(self, hid_dim, n_layers, n_heads, pf_dim, dropout, channels, pool = 'cls',
                 image_size = (256, 256), patch_size = (16, 16), **kwargs):
        super().__init__()
        image_height, image_width = image_size
        patch_height, patch_width = patch_size

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'


        self.to_patch = lambda x: x.view(-1, channels, image_width // patch_width, patch_width, \
                image_height // patch_height, patch_height).permute(0,2,4,1,3,5).reshape(-1, num_patches, patch_dim)
        self.to_patch_embedding = nn.Sequential(
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, hid_dim),
            nn.LayerNorm(hid_dim),
        )

        self.pos_embedding = PositionalEncodingComponent(hid_dim, dropout, num_patches + 1)

        self.cls_token = nn.Parameter(torch.randn(1, 1, hid_dim))
        self.dropout = nn.Dropout(dropout)

        # encoder layers
        self.layers = nn.ModuleList([EncodingLayer(hid_dim, n_heads, pf_dim, dropout) for _ in range(n_layers)])

        self.to_latent = nn.Identity()

        self.hid_dim = hid_dim
        self.pool = pool

    def forward(self, image):

        x = self.to_patch(image)
        x = self.to_patch_embedding(x)
        batch_size, n, _ = x.shape

        cls_tokens = self.cls_token.expand(batch_size, 1, self.hid_dim)
        x = torch.cat((cls_tokens, x), dim=1)
        x = self.pos_embedding(x)
        x = self.dropout(x)

        for layer in self.layers:
            x = layer(x)

        # x = x.mean(dim=1) if self.pool == 'mean' else x[:, 0]
        return self.to_latent(x)

# Cross Attention Layer
class CrossAttentionLayer(nn.Module):
    '''
    This layer takes input the audio and text representations after they have been
    passed through their respective Encoding layers.
    The text representations will act as query
    the audio representations will be key and values.
    So this will take most important features from text representation based on the
    attention between audio and the text features.
    '''

    def __init__(self, hid_dim, n_heads, pf_dim, dropout):
        super().__init__()

        self.self_attn_layer_norm = nn.LayerNorm(hid_dim)  # Layer norm after self-attention
        self.ff_layer_norm = nn.LayerNorm(hid_dim)  # Layer norm after FeedForward component

        self.self_attention = MultiHeadedAttentionComponent(hid_dim, n_heads, dropout)
        self.feed_forward = FeedForwardComponent(hid_dim, pf_dim, dropout)

        self.dropout = nn.Dropout(dropout)

    def forward(self, text, audio):
        # text : [batch_size, text_len, hid_dim]
        # audio : [batch_size, audio_len, hid_dim

        # get self-attention
        _text, _ = self.self_attention(text, audio, audio) # query, key, value

        # LayerNorm after dropout
        text = self.self_attn_layer_norm(text + self.dropout(_text))
        # text : [batch_size, text_len, hid_dim]

        # FeedForward
        _text = self.feed_forward(text)

        # layerNorm after dropout
        text = self.ff_layer_norm(text + self.dropout(_text))
        # text: [batch_size, text_len, hid_dim]

        return text


# Model
class Model(nn.Module):
    """
    Model class
    We will use <sos> token for prediction of classes
    """

    def __init__(self, audio_split_samples, hid_dim, audio_representation_layers, n_heads, pf_dim, dropout, max_length \
                 , len_text_vocab, text_pad_index, text_representation_layers, image_representation_layers,\
                 channels1, image_size1, patch_size1, channels2, image_size2, patch_size2, cross_attention_layers, \
                 output_dim,config):
        super().__init__()
        self.audio_representations = AudioRepresentations(audio_split_samples, hid_dim, audio_representation_layers,
                                                          n_heads, pf_dim, dropout, max_length)
        self.text_representations = TextRepresentations(len_text_vocab, hid_dim, text_representation_layers, n_heads,
                                                        pf_dim, dropout, text_pad_index, max_length)

        self.image_representations1 = ImageRepresentations(hid_dim, image_representation_layers, n_heads, pf_dim, dropout,
                                                    channels1, image_size = image_size1, patch_size = patch_size1)

        self.image_representations2 = ImageRepresentations(hid_dim, image_representation_layers, n_heads, pf_dim,
                                                           dropout, channels2, image_size=image_size2, patch_size=patch_size2)

        self.cross_attention = [[CrossAttentionLayer(hid_dim, n_heads, pf_dim, dropout) for _ in range(4)]\
                                      for _ in range(cross_attention_layers)]

        self.feed_forward = nn.Linear(hid_dim, output_dim)
        self.loss = nn.L1Loss()

        self.config = config

    def forward(self, audio, text, image1, image2, label):
        # audio : [batch_size, max_audio_len]
        # text : [batch_size, src_len]

        audio = self.audio_representations(audio)
        # audio : [batch_size, audio_len, hid_dim] where audio_len= max_audio_len/audio_split_samples

        text = self.text_representations(text)
        # text : [batch_size, src_len, hid_dim]

        image1 = self.image_representations1(image1)
        # image1: [batch_size, num_patches, w, h]

        image2 = self.image_representations2(image2)
        # image2: [batch_size, num_patches, w, h]

        for layer in self.cross_attention:
            text = layer[0](text, image2)
            image1 = layer[1](image1, text)
            audio = layer[2](audio, image1)
            image2 = layer[3](image2, audio)

        pred_token = image2[:, -1, :]
        # pred_token : [batch_size, hid_dim]

        output = self.feed_forward(pred_token)


        loss = self.loss(output, label)

        return {'loss': loss}


model = Model(100,18,1,6,20,0.9,1000,100,torch.zeros([300]),2,2,3,(128,128),(32,32),3,(256,256),(16,16),2,1,[])
a = torch.rand([5,3,256,256]).float()
b = torch.arange(245760).reshape(5,3,128,128).float()
c = torch.randn([5,1000])
d = torch.randint(0,99,[5,300])
out = model(c,d,b,a,torch.tensor([4.4,9.8,33333,-3423.8,23423.3]))
print(out)