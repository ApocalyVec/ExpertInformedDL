import torch
from torch import nn
from timm.models.vision_transformer import VisionTransformer, Block, Attention

class ExtensionTimmViT(nn.Module):
    def __init__(self, vision_transformer: VisionTransformer, num_dim_fixation=2, num_lstm_layers=2, rnn_hidden_dim=64, fixation_conditioned=True):
        """
        Composite class extending the VisionTransformer from timm
        the lstm encodes the fixation sequence
        Parameters
        ----------
        vision_transformer
        rnn_hidden_dim
        fixation_conditioned: use the fixation sequence to condition the input
        """
        super().__init__()
        self.vision_transformer = vision_transformer
        self.depth = len(vision_transformer.blocks)
        self.num_classes = vision_transformer.num_classes  # copy the num classes from the VisionTransformer but not using its mlp

        self.lstm_hidden_dim = rnn_hidden_dim
        self.num_dim_fixation = num_dim_fixation
        self.num_lstm_layers = num_lstm_layers
        self.fixation_conditioned = fixation_conditioned

        # wrap the original blocks with our implementation to access the attention activation
        for i in range(len(self.vision_transformer.blocks)):
            self.vision_transformer.blocks[i] = ExpertTimmViTBlock(self.vision_transformer.blocks[i])

        # add the lstm layer to process the fixation sequence
        self.lstm = nn.LSTM(self.num_dim_fixation, self.lstm_hidden_dim, num_layers=self.num_lstm_layers, bidirectional=False, batch_first=True)

        if self.fixation_conditioned:
            self.vision_transformer.head = self.head = nn.Linear(
                self.vision_transformer.embed_dim + self.lstm_hidden_dim * self.num_lstm_layers,
                self.num_classes) if self.num_classes > 0 else nn.Identity()
        else:
            self.vision_transformer.head = self.head = nn.Linear(self.vision_transformer.embed_dim, self.num_classes) if self.num_classes > 0 else nn.Identity()

    def forward_features(self, img, collapse_attention_matrix=True, *args, **kwargs):
        x = self.vision_transformer.patch_embed(img)
        x = self.vision_transformer._pos_embed(x)
        x = self.vision_transformer.norm_pre(x)
        # if self.grad_checkpointing and not torch.jit.is_scripting():
        #     x = checkpoint_seq(self.blocks, x)
        # else:
        x = self.vision_transformer.blocks(x)
        attention = self.vision_transformer.blocks[-1].attention  # we keep the attention activation of the last layer

        x = self.vision_transformer.norm(x)
        try:
            assert attention is not None
        except AssertionError:
            raise ValueError("the attention activation in forward_features is none, check your depth parameter")
        return x, attention

    def forward(self, img, fixation_sequence):
        x, attention = self.forward_features(img)
        attention = attention[:, :, 1:, 1:]
        attention = attention / torch.sum(attention, dim=3, keepdim=True)
        attention = torch.sum(attention, dim=2)

        x = self.vision_transformer.forward_head(x, pre_logits=True)  # set pre_logits to true, don't apply the classification head
        if self.fixation_conditioned:
            rnn_outputs, hiddens = self.lstm(fixation_sequence)
            hidden, cell = hiddens
            x = torch.concat([x] + [hidden[i, :, :] for i in range(self.num_lstm_layers)], dim=1)
        x = self.head(x)
        return x, attention

    def get_grid_size(self):
        return self.vision_transformer.patch_embed.grid_size

    # TODO add test function, without fixation sequence as in ExpertAttentionViT.test()

class ExpertTimmViTBlock(nn.Module):
    def __init__(self, block: Block):
        super().__init__()
        self.block = block
        self.attn = ExpertTimmViTAttention(self.block.attn)
        self.attention = None

    def forward(self, x):
        x, _ = self.attn(x)  # TODO
        x = x + self.block.drop_path1(self.block.norm1(x))
        x = x + self.block.drop_path2(self.block.norm2(self.block.mlp(x)))
        return x

class ExpertTimmViTAttention(nn.Module):
    def __init__(self, attention: Attention):
        super().__init__()
        self.attention = attention

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.attention.qkv(x).reshape(B, N, 3, self.attention.num_heads, C // self.attention.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.attention.scale
        attn = attn.softmax(dim=-1)
        attn = self.attention.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.attention.proj(x)
        x = self.attention.proj_drop(x)
        return x, attn
