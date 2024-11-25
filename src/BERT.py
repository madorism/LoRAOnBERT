import torch.nn as nn
import torch
import torch.nn.functional as F
import math
from dataclasses import dataclass
from typing import Tuple, TypeVar
from transformers import BertForSequenceClassification as HFBertForSequenceClassification


@dataclass
class BertConfig:
    """
    Configuration class for BERT model.

    Attr:
        max_position_embeddings (int, optional, defaults to 512) — The maximum sequence length that this model might ever be used with.
        vocab_size (int, optional, defaults to 30522) — Vocabulary size of the BERT model.

        hidden_size (int, optional, defaults to 768) — Dimensionality of the encoder layers and the pooler layer.

        num_hidden_layers (int, optional, defaults to 12) — Number of hidden layers in the Transformer encoder.
        num_attention_heads (int, optional, defaults to 12) — Number of attention heads for each attention layer in the Transformer encoder.
        hidden_dropout_prob (float, optional, defaults to 0.1) — The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
        attention_probs_dropout_prob (float, optional, defaults to 0.1) — The dropout ratio for the attention probabilities.

        intermediate_size (int, optional, defaults to 3072) — Dimensionality of the “intermediate” (often named feed-forward) layer in the Transformer encoder.

        n_classes (int) — Integer representing the number of classes in the taske
        layer_norm_eps (float, optional, defaults to 1e-12) — The epsilon used by the layer normalization layers.
        pad_token_id (int) — Integer representing the token id for the padding token.

        is_decoder (bool, optional, defaults to False) — Whether the model is used as a decoder or not.

    """

    max_position_embeddings : int = 512
    vocab_size: int = 30522

    hidden_size: int = 768

    num_hidden_layers: int = 12
    num_attention_heads : int = 12
    hidden_dropout_prob : int = 0.1
    attention_probs_dropout_prob : int = 0.1

    intermediate_size: int = hidden_size * 4

    num_classes: int = 2
    layer_norm_eps: float = 1e-12
    pad_token_id: int = 103

    is_decoder : bool = False


class MultiHeadSelfAttention(nn.Module):
  """
  A class for Multi-head Self Attention block (hugging face like)
  """

  def __init__(self, config: BertConfig):
    """
    Constructor
      Args:
      -config: Configuration class for BERT model.
    """
    super(MultiHeadSelfAttention, self).__init__()
    assert config.hidden_size % config.num_attention_heads == 0

    self.num_heads = config.num_attention_heads
    self.head_size = config.hidden_size // config.num_attention_heads

    self.query = nn.Linear(config.hidden_size, config.hidden_size)
    self.key = nn.Linear(config.hidden_size, config.hidden_size)
    self.value = nn.Linear(config.hidden_size, config.hidden_size)

    self.att_dropout = nn.Dropout(config.attention_probs_dropout_prob)

    self.out = nn.Linear(config.hidden_size, config.hidden_size)

  def forward(self, x:torch.Tensor,  mask: torch.Tensor=None):
    """
    Take the input X, compute the self-attention score and, produce representation and mask it if needed

    Inputs:
    - x: Tensor of the shape BxLxC, where B is the batch size, L is the sequence length,
    and C is the input dimension in previous representative space
    - mask: Tensor indicating the mask position
    """
    B,L,C = x.shape

    # Query, Key, Value
    Q = self.query(x)
    K = self.key(x)
    V = self.value(x)

    # Split heads
    Q = Q.reshape(B, L, self.num_heads, self.head_size).permute(0, 2, 1, 3)
    K = K.reshape(B, L, self.num_heads, self.head_size).permute(0, 2, 1, 3)
    V = V.reshape(B, L, self.num_heads, self.head_size).permute(0, 2, 1, 3)

    # scale dot product score
    dot_product = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_size ** 0.5)

    # mask
    if mask is not None:
      dot_product = dot_product.masked_fill(mask == 0, -1e9)


    # get attented context
    score = F.softmax(dot_product, dim=-1)
    score = self.att_dropout(score)

    attention = torch.matmul(score, V)
    attention = attention.permute(0,2,1,3).contiguous().view(B, L, C)

    # Linear map to another representative space
    representation = self.out(attention)

    return representation



class FeedForward(nn.Module):
    """
    A feedforward network with two hidden layers
    """
    def __init__(self, config: BertConfig):
        """
        Constructor
          Args:
          -config: Configuration class for BERT model.
        """
        super(FeedForward, self).__init__()

        self.linear1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.gelu = nn.GELU()
        self.linear2 = nn.Linear(config.intermediate_size, config.hidden_size)


    def forward(self, x: torch.Tensor):
      """
        feed-forward with non-linear map

        Inputs:
        - x: input tensor of the shape BxLxC

        Return:
        - x: Tensor of the shape of BxLxC
        """
      x = self.linear1(x)
      x = self.gelu(x)
      x = self.linear2(x)
      return x



class EncoderCell(nn.Module):
    """
    A single cell (unit) for the Transformer encoder.
    """
    def __init__(self, config: BertConfig):
        """
        Constructor
          Args:
          -config: Configuration class for BERT model.
        """
        super(EncoderCell, self).__init__()

        self.multi_head_attention = MultiHeadSelfAttention(config)
        self.layer_norm1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout1 = nn.Dropout(config.hidden_dropout_prob)

        self.feed_forward = FeedForward(config)
        self.layer_norm2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout2 = nn.Dropout(config.hidden_dropout_prob)


    def forward(self, x:torch.Tensor,  mask: torch.Tensor=None):
        """
        forward a attention cell
        Inputs:
        - x: input tensor of the shape BxLxC
        - mask: mask Tensor for prediction use

        Return:
        - x: Tensor of the shape of BxLxC, which is the output of the encoder, a representation vector contatning rich information
        """

        # post-layer normalization and Self-Attention with residual connection

        attention_output = self.multi_head_attention(x, mask)
        attention_output = self.dropout1(attention_output)
        x = x + attention_output
        x = self.layer_norm1(x)


        # post-layer normalization and Feed Forward with residual connection

        ff_output = self.feed_forward(x)
        ff_outut = self.dropout2(ff_output)

        x = x + ff_output
        x = self.layer_norm2(x)


        return x



class BertEncoder(nn.Module):
  """
  Encoder part of BERT model
  """
  def __init__(self, config: BertConfig):
    """
    Constructor
      Args:
      -config: Configuration class for BERT model.
    """
    super(BertEncoder, self).__init__()

    self.layers = nn.ModuleList([EncoderCell(config) for _ in range(config.num_hidden_layers)])

  def forward(self, x: torch.Tensor, mask: torch.Tensor=None):
    """
    Inputs:
    - x: input tensor of the shape BxLxC
    - mask: mask Tensor for prediction use

    Return:
    - x: Tensor of the shape of BxLxC, going through num-cells Encoder block for representation extraction
    """

    for encoder_cell in self.layers:
      x = encoder_cell(x, mask)

    return x



class BertPooler(nn.Module):
    """
    Pooler of a Bert model,
    it takes the CLS token (added to the first position in a sentence) representation vector
    from encoder output
    """

    def __init__(self, config: BertConfig):
        super(BertPooler, self).__init__()

        self.linear = nn.Linear(config.hidden_size, config.hidden_size)
        self.tanh = nn.Tanh()

    def forward(self, encoder_out):
        """
        Inputs:
        - encoder_out: output representation from encoder
                       of the shape BxLxC

        Return:
        - out: the first token's representation vector in each sentence
               of the shape B*L
        """
        pool_first_token = encoder_out[:, 0]
        out = self.linear(pool_first_token)
        out = self.tanh(out)
        return out


class BertEmbeddings(nn.Module):
    def __init__(self, config):
        super(BertEmbeddings, self).__init__()
        self.config = config
        self.word_embed = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.position_embed = nn.Embedding(config.max_position_embeddings , config.hidden_size)
        self.token_type_embed = nn.Embedding(2, config.hidden_size)
        self.layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        # position ids (used in the pos_emb lookup table) that we do not want updated through backpropogation
        self.register_buffer("position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)))

    def forward(self, input_ids, token_type_ids=None):
        word_embedding = self.word_embed(input_ids)
        pos_embedding = self.position_embed(self.position_ids)
        if  token_type_ids is not None:
          type_embedding= self.token_type_embed(token_type_ids)
          embedding = word_embedding + pos_embedding + type_embedding
        else:
          embedding = word_embedding + pos_embedding

        embedding = self.layernorm(embedding)
        embedding = self.dropout(embedding)
        return embedding



class BertModel(nn.Module):
    def __init__(self, config):
        super(BertModel, self).__init__()

        self.config = config
        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config)
        self.pooler = BertPooler(config)

    def forward(self, input_ids, token_type_ids=None, att_mask=None):
        embeddings = self.embeddings(input_ids, token_type_ids)
        out = self.encoder(embeddings, att_mask)
        pooled_out = self.pooler(out)

        return out, pooled_out



class BertClassifier(nn.Module):
    def __init__(self, config):
        super(BertClassifier, self).__init__()

        self.config = config
        self.bert = BertModel(config)
        self.classifier = nn.Linear(config.hidden_size, config.num_classes)

    def forward(self, input_ids, token_type_ids=None, att_mask=None):
        _, pooled_out = self.bert(input_ids, token_type_ids, att_mask)
        logits = self.classifier(pooled_out)

        return logits

    @classmethod
    def from_pretrained(cls, model_name_or_path, config):
      print("Loading weights from pretrainged language model:")
      model_hf = HFBertForSequenceClassification.from_pretrained(pretrained_model_name_or_path=model_name_or_path, num_labels=config.num_classes)
      sd_hf = model_hf.state_dict()
      sd_keys_hf = sd_hf.keys()
      
      
      model = cls(config)
      sd = model.state_dict()
      sd_keys = sd.keys()

      my_str_list = []
      hf_str_list = []

      hf_embedding_parts = ['word_embeddings.weight','position_embeddings.weight','token_type_embeddings.weight','LayerNorm.weight','LayerNorm.bias']
      my_embedding_parts = ['word_embed.weight','position_embed.weight','token_type_embed.weight','layernorm.weight','layernorm.bias']

      for hf_part,my_part in zip(hf_embedding_parts, my_embedding_parts):
        hf_str = 'bert.embeddings.'+ hf_part
        my_str = 'bert.embeddings.' + my_part
        hf_str_list.append(hf_str)
        my_str_list.append(my_str)

      hf_bottom_parts = ['bert.pooler.dense.weight', 'bert.pooler.dense.bias']
      my_bottom_parts = ['bert.pooler.linear.weight', 'bert.pooler.linear.bias']

      for hf_part,my_part in zip(hf_bottom_parts, my_bottom_parts):
        hf_str = hf_part
        my_str = my_part
        hf_str_list.append(hf_str)
        my_str_list.append(my_str)

      att_parts = ['query', 'key', 'value']
      paras = ['.weight', '.bias']
      hf_layer_parts = ['.attention.output.dense','.attention.output.LayerNorm', '.intermediate.dense',
                  '.output.dense', '.output.LayerNorm']
      my_layer_parts = ['.multi_head_attention.out', '.layer_norm1', '.feed_forward.linear1',
                  '.feed_forward.linear2', '.layer_norm2']


      for i in range(config.num_hidden_layers):

        for part in att_parts:
          for para in paras:
            hf_str = 'bert.encoder.layer.' + str(i) +'.attention.self.'+part+para
            my_str = 'bert.encoder.layers.' + str(i) +'.multi_head_attention.'+part+para
            hf_str_list.append(hf_str)
            my_str_list.append(my_str)

        for hf_part, my_part in zip(hf_layer_parts, my_layer_parts):
          for para in paras:
            hf_str = 'bert.encoder.layer.' + str(i) + hf_part + para
            my_str = 'bert.encoder.layers.' + str(i) + my_part + para
            hf_str_list.append(hf_str)
            my_str_list.append(my_str)

      with torch.no_grad():
        for my_str, hf_str in zip(my_str_list, hf_str_list):
          sd[my_str].copy_(sd_hf[hf_str])

      return model


class BertRegressor(nn.Module):
    def __init__(self, config):
        super(BertRegressor, self).__init__()

        self.config = config
        self.bert = BertModel(config)
        self.regressor = nn.Linear(config.hidden_size, 1)

    def forward(self, input_ids, token_type_ids=None, att_mask=None):
        encoder_out, pooled_out = self.bert(input_ids, token_type_ids, att_mask)

        mean_hidden = torch.mean(encoder_out, dim=1)

        logits = self.regressor(mean_hidden)

        return logits

    @classmethod
    def from_pretrained(cls, model_name_or_path, config):
      print("import pretrained weights:")
      model_hf = HFBertForSequenceClassification.from_pretrained(pretrained_model_name_or_path=model_name_or_path, num_labels=config.num_classes)
      sd_hf = model_hf.state_dict()
      sd_keys_hf = sd_hf.keys()


      model = cls(config)
      sd = model.state_dict()
      sd_keys = sd.keys()

      my_str_list = []
      hf_str_list = []

      hf_embedding_parts = ['word_embeddings.weight','position_embeddings.weight','token_type_embeddings.weight','LayerNorm.weight','LayerNorm.bias']
      my_embedding_parts = ['word_embed.weight','position_embed.weight','token_type_embed.weight','layernorm.weight','layernorm.bias']

      for hf_part,my_part in zip(hf_embedding_parts, my_embedding_parts):
        hf_str = 'bert.embeddings.'+ hf_part
        my_str = 'bert.embeddings.' + my_part
        hf_str_list.append(hf_str)
        my_str_list.append(my_str)

      hf_bottom_parts = ['bert.pooler.dense.weight', 'bert.pooler.dense.bias']
      my_bottom_parts = ['bert.pooler.linear.weight', 'bert.pooler.linear.bias']

      for hf_part,my_part in zip(hf_bottom_parts, my_bottom_parts):
        hf_str = hf_part
        my_str = my_part
        hf_str_list.append(hf_str)
        my_str_list.append(my_str)

      att_parts = ['query', 'key', 'value']
      paras = ['.weight', '.bias']
      hf_layer_parts = ['.attention.output.dense','.attention.output.LayerNorm', '.intermediate.dense',
                  '.output.dense', '.output.LayerNorm']
      my_layer_parts = ['.multi_head_attention.out', '.layer_norm1', '.feed_forward.linear1',
                  '.feed_forward.linear2', '.layer_norm2']


      for i in range(config.num_hidden_layers):

        for part in att_parts:
          for para in paras:
            hf_str = 'bert.encoder.layer.' + str(i) +'.attention.self.'+part+para
            my_str = 'bert.encoder.layers.' + str(i) +'.multi_head_attention.'+part+para
            hf_str_list.append(hf_str)
            my_str_list.append(my_str)

        for hf_part, my_part in zip(hf_layer_parts, my_layer_parts):
          for para in paras:
            hf_str = 'bert.encoder.layer.' + str(i) + hf_part + para
            my_str = 'bert.encoder.layers.' + str(i) + my_part + para
            hf_str_list.append(hf_str)
            my_str_list.append(my_str)

      with torch.no_grad():
        for my_str, hf_str in zip(my_str_list, hf_str_list):
          sd[my_str].copy_(sd_hf[hf_str])

      return model