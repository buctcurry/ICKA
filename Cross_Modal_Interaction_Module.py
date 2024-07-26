import torch
from torch import nn
from torchcrf import CRF
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F
from my_bert.file_utils import cached_path, WEIGHTS_NAME, CONFIG_NAME
import copy
import json
import logging
import math
import os
import shutil
import tarfile
import tempfile
import sys
from typing import Optional, Tuple, List
from sparsemax import Sparsemax

PRETRAINED_MODEL_ARCHIVE_MAP = {
    'bert-base-uncased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased.tar.gz",
    'bert-large-uncased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-uncased.tar.gz",
    'bert-base-cased': "bert_model/bert-base-cased/",
    'bert-large-cased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-cased.tar.gz",
    'bert-base-multilingual-uncased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-multilingual-uncased.tar.gz",
    'bert-base-multilingual-cased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-multilingual-cased.tar.gz",
    'bert-base-chinese': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-chinese.tar.gz"
}

BERT_CONFIG_NAME = 'bert_config.json'

def gelu(x):
    """Implementation of the gelu activation function.
        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
        Also see https://arxiv.org/abs/1606.08415
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))
def swish(x):
    return x * torch.sigmoid(x)


logger = logging.getLogger(__name__)
ACT2FN = {"gelu": gelu, "relu": torch.nn.functional.relu, "swish": swish}

class BertConfig(object):
    """Configuration class to store the configuration of a `BertModel`.
    """
    def __init__(self,
                 vocab_size_or_config_json_file,
                 hidden_size=768,
                 num_hidden_layers=12,
                 num_attention_heads=12,
                 intermediate_size=3072,
                 hidden_act="gelu",
                 hidden_dropout_prob=0.1,
                 attention_probs_dropout_prob=0.1,
                 max_position_embeddings=512,
                 type_vocab_size=2,
                 initializer_range=0.02,
                 layer_norm_eps=1e-12):
        """Constructs BertConfig.

        Args:
            vocab_size_or_config_json_file: Vocabulary size of `inputs_ids` in `BertModel`.
            hidden_size: Size of the encoder layers and the pooler layer.
            num_hidden_layers: Number of hidden layers in the Transformer encoder.
            num_attention_heads: Number of attention heads for each attention layer in
                the Transformer encoder.
            intermediate_size: The size of the "intermediate" (i.e., feed-forward)
                layer in the Transformer encoder.
            hidden_act: The non-linear activation function (function or string) in the
                encoder and pooler. If string, "gelu", "relu" and "swish" are supported.
            hidden_dropout_prob: The dropout probabilitiy for all fully connected
                layers in the embeddings, encoder, and pooler.
            attention_probs_dropout_prob: The dropout ratio for the attention
                probabilities.
            max_position_embeddings: The maximum sequence length that this model might
                ever be used with. Typically set this to something large just in case
                (e.g., 512 or 1024 or 2048).
            type_vocab_size: The vocabulary size of the `token_type_ids` passed into
                `BertModel`.
            initializer_range: The sttdev of the truncated_normal_initializer for
                initializing all weight matrices.
            layer_norm_eps: The epsilon used by LayerNorm.
        """
        if isinstance(vocab_size_or_config_json_file, str) or (sys.version_info[0] == 2
                        and isinstance(vocab_size_or_config_json_file, unicode)):
            with open(vocab_size_or_config_json_file, "r", encoding='utf-8') as reader:
                json_config = json.loads(reader.read())
            for key, value in json_config.items():
                self.__dict__[key] = value
        elif isinstance(vocab_size_or_config_json_file, int):
            self.vocab_size = vocab_size_or_config_json_file
            self.hidden_size = hidden_size
            self.num_hidden_layers = num_hidden_layers
            self.num_attention_heads = num_attention_heads
            self.hidden_act = hidden_act
            self.intermediate_size = intermediate_size
            self.hidden_dropout_prob = hidden_dropout_prob
            self.attention_probs_dropout_prob = attention_probs_dropout_prob
            self.max_position_embeddings = max_position_embeddings
            self.type_vocab_size = type_vocab_size
            self.initializer_range = initializer_range
            self.layer_norm_eps = layer_norm_eps
        else:
            raise ValueError("First argument must be either a vocabulary size (int)"
                             "or the path to a pretrained model config file (str)")

    @classmethod
    def from_dict(cls, json_object):
        """Constructs a `BertConfig` from a Python dictionary of parameters."""
        config = BertConfig(vocab_size_or_config_json_file=-1)
        for key, value in json_object.items():
            config.__dict__[key] = value
        return config

    @classmethod
    def from_json_file(cls, json_file):
        """Constructs a `BertConfig` from a json file of parameters."""
        with open(json_file, "r", encoding='utf-8') as reader:
            text = reader.read()
        return cls.from_dict(json.loads(text))

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"

    def to_json_file(self, json_file_path):
        """ Save this instance to a json file."""
        with open(json_file_path, "w", encoding='utf-8') as writer:
            writer.write(self.to_json_string())

class BertPreTrainedModel(nn.Module):
    """ An abstract class to handle weights initialization and
        a simple interface for dowloading and loading pretrained models.
    """
    def __init__(self, config, *inputs, **kwargs):
        super(BertPreTrainedModel, self).__init__()
        if not isinstance(config, BertConfig):
            raise ValueError(
                "Parameter config in `{}(config)` should be an instance of class `BertConfig`. "
                "To create a model from a Google pretrained model use "
                "`model = {}.from_pretrained(PRETRAINED_MODEL_NAME)`".format(
                    self.__class__.__name__, self.__class__.__name__
                ))
        self.config = config

    def init_bert_weights(self, module):
        """ Initialize the weights.
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, BertLayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *inputs, **kwargs):
        """
        Instantiate a BertPreTrainedModel from a pre-trained model file or a pytorch state dict.
        Download and cache the pre-trained model file if needed.

        Params:
            pretrained_model_name_or_path: either:
                - a str with the name of a pre-trained model to load selected in the list of:
                    . `bert-base-uncased`
                    . `bert-large-uncased`
                    . `bert-base-cased`
                    . `bert-large-cased`
                    . `bert-base-multilingual-uncased`
                    . `bert-base-multilingual-cased`
                    . `bert-base-chinese`
                - a path or url to a pretrained model archive containing:
                    . `bert_config.json` a configuration file for the model
                    . `pytorch_model.bin` a PyTorch dump of a BertForPreTraining instance
                - a path or url to a pretrained model archive containing:
                    . `bert_config.json` a configuration file for the model
                    . `model.chkpt` a TensorFlow checkpoint
            from_tf: should we load the weights from a locally saved TensorFlow checkpoint
            cache_dir: an optional path to a folder in which the pre-trained models will be cached.
            state_dict: an optional state dictionnary (collections.OrderedDict object) to use instead of Google pre-trained models
            *inputs, **kwargs: additional input for the specific Bert class
                (ex: num_labels for BertForSequenceClassification)
        """
        state_dict = kwargs.get('state_dict', None)
        kwargs.pop('state_dict', None)
        cache_dir = kwargs.get('cache_dir', None)
        kwargs.pop('cache_dir', None)
        from_tf = kwargs.get('from_tf', False)
        kwargs.pop('from_tf', None)

        if pretrained_model_name_or_path in PRETRAINED_MODEL_ARCHIVE_MAP:
            archive_file = PRETRAINED_MODEL_ARCHIVE_MAP[pretrained_model_name_or_path]
        else:
            archive_file = pretrained_model_name_or_path
        # redirect to the cache, if necessary
        try:
            resolved_archive_file = cached_path(archive_file, cache_dir=cache_dir)
        except EnvironmentError:
            logger.error(
                "Model name '{}' was not found in model name list ({}). "
                "We assumed '{}' was a path or url but couldn't find any file "
                "associated to this path or url.".format(
                    pretrained_model_name_or_path,
                    ', '.join(PRETRAINED_MODEL_ARCHIVE_MAP.keys()),
                    archive_file))
            return None
        if resolved_archive_file == archive_file:
            logger.info("loading archive file {}".format(archive_file))
        else:
            logger.info("loading archive file {} from cache at {}".format(
                archive_file, resolved_archive_file))
        tempdir = None
        if os.path.isdir(resolved_archive_file) or from_tf:
            serialization_dir = resolved_archive_file
        else:
            # Extract archive to temp dir
            tempdir = tempfile.mkdtemp()
            logger.info("extracting archive file {} to temp dir {}".format(
                resolved_archive_file, tempdir))
            with tarfile.open(resolved_archive_file, 'r:gz') as archive:
                archive.extractall(tempdir)
            serialization_dir = tempdir
        # Load config
        config_file = os.path.join(serialization_dir, CONFIG_NAME)
        if not os.path.exists(config_file):
            # Backward compatibility with old naming format
            config_file = os.path.join(serialization_dir, BERT_CONFIG_NAME)
        config = BertConfig.from_json_file(config_file)
        logger.info("Model config {}".format(config))
        # Instantiate model.
        model = cls(config, *inputs, **kwargs)
        if state_dict is None and not from_tf:
            weights_path = os.path.join(serialization_dir, WEIGHTS_NAME)
            state_dict = torch.load(weights_path, map_location='cpu')
        if tempdir:
            # Clean up temp dir
            shutil.rmtree(tempdir)
        if from_tf:
            # Directly load from a TensorFlow checkpoint
            weights_path = os.path.join(serialization_dir, TF_WEIGHTS_NAME)
            return load_tf_weights_in_bert(model, weights_path)
        # Load from a PyTorch state_dict
        old_keys = []
        new_keys = []
        for key in state_dict.keys():
            new_key = None
            if 'gamma' in key:
                new_key = key.replace('gamma', 'weight')
            if 'beta' in key:
                new_key = key.replace('beta', 'bias')
            if new_key:
                old_keys.append(key)
                new_keys.append(new_key)
        for old_key, new_key in zip(old_keys, new_keys):
            state_dict[new_key] = state_dict.pop(old_key)

        missing_keys = []
        unexpected_keys = []
        error_msgs = []
        # copy state_dict so _load_from_state_dict can modify it
        metadata = getattr(state_dict, '_metadata', None)
        state_dict = state_dict.copy()
        if metadata is not None:
            state_dict._metadata = metadata

        def load(module, prefix=''):
            local_metadata = {} if metadata is None else metadata.get(prefix[:-1], {})
            module._load_from_state_dict(
                state_dict, prefix, local_metadata, True, missing_keys, unexpected_keys, error_msgs)
            for name, child in module._modules.items():
                if child is not None:
                    load(child, prefix + name + '.')
        start_prefix = ''
        if not hasattr(model, 'bert') and any(s.startswith('bert.') for s in state_dict.keys()):
            start_prefix = 'bert.'
        load(model, prefix=start_prefix)
        if len(missing_keys) > 0:
            logger.info("Weights of {} not initialized from pretrained model: {}".format(
                model.__class__.__name__, missing_keys))
        if len(unexpected_keys) > 0:
            logger.info("Weights from pretrained model not used in {}: {}".format(
                model.__class__.__name__, unexpected_keys))
        if len(error_msgs) > 0:
            raise RuntimeError('Error(s) in loading state_dict for {}:\n\t{}'.format(
                               model.__class__.__name__, "\n\t".join(error_msgs)))
        return model


class BertModel(BertPreTrainedModel):
    """BERT model ("Bidirectional Embedding Representations from a Transformer").

    Params:
        config: a BertConfig class instance with the configuration to build a new model

    Inputs:
        `input_ids`: a torch.LongTensor of shape [batch_size, sequence_length]
            with the word token indices in the vocabulary(see the tokens preprocessing logic in the scripts
            `extract_features.py`, `run_classifier.py` and `run_squad.py`)
        `token_type_ids`: an optional torch.LongTensor of shape [batch_size, sequence_length] with the token
            types indices selected in [0, 1]. Type 0 corresponds to a `sentence A` and type 1 corresponds to
            a `sentence B` token (see BERT paper for more details).
        `attention_mask`: an optional torch.LongTensor of shape [batch_size, sequence_length] with indices
            selected in [0, 1]. It's a mask to be used if the input sequence length is smaller than the max
            input sequence length in the current batch. It's the mask that we typically use for attention when
            a batch has varying length sentences.
        `output_all_encoded_layers`: boolean which controls the content of the `encoded_layers` output as described below. Default: `True`.

    Outputs: Tuple of (encoded_layers, pooled_output)
        `encoded_layers`: controled by `output_all_encoded_layers` argument:
            - `output_all_encoded_layers=True`: outputs a list of the full sequences of encoded-hidden-states at the end
                of each attention block (i.e. 12 full sequences for BERT-base, 24 for BERT-large), each
                encoded-hidden-state is a torch.FloatTensor of size [batch_size, sequence_length, hidden_size],
            - `output_all_encoded_layers=False`: outputs only the full sequence of hidden-states corresponding
                to the last attention block of shape [batch_size, sequence_length, hidden_size],
        `pooled_output`: a torch.FloatTensor of size [batch_size, hidden_size] which is the output of a
            classifier pretrained on top of the hidden state associated to the first character of the
            input (`CLS`) to train on the Next-Sentence task (see BERT's paper).

    Example usage:
    ```python
    # Already been converted into WordPiece token ids
    input_ids = torch.LongTensor([[31, 51, 99], [15, 5, 0]])
    input_mask = torch.LongTensor([[1, 1, 1], [1, 1, 0]])
    token_type_ids = torch.LongTensor([[0, 0, 1], [0, 1, 0]])

    config = modeling.BertConfig(vocab_size_or_config_json_file=32000, hidden_size=768,
        num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072)

    model = modeling.BertModel(config=config)
    all_encoder_layers, pooled_output = model(input_ids, token_type_ids, input_mask)
    ```
    """
    def __init__(self, config):
        super(BertModel, self).__init__(config)
        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config)
        self.pooler = BertPooler(config)
        self.apply(self.init_bert_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, output_all_encoded_layers=True):
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        # We create a 3D attention mask from a 2D tensor mask.
        # Sizes are [batch_size, 1, 1, to_seq_length]
        # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
        # this attention mask is more simple than the triangular masking of causal attention
        # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype) # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        embedding_output = self.embeddings(input_ids, token_type_ids)
        encoded_layers = self.encoder(embedding_output,
                                      extended_attention_mask,
                                      output_all_encoded_layers=output_all_encoded_layers)
        sequence_output = encoded_layers[-1]
        pooled_output = self.pooler(sequence_output)
        if not output_all_encoded_layers:
            encoded_layers = encoded_layers[-1]
        return encoded_layers, pooled_output

class BertEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings.
    """
    def __init__(self, config):
        super(BertEmbeddings, self).__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=0)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_ids, token_type_ids=None):
        seq_length = input_ids.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        words_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = words_embeddings + position_embeddings + token_type_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class BertEncoder(nn.Module):
    def __init__(self, config):
        super(BertEncoder, self).__init__()
        layer = BertLayer(config)
        self.layer = nn.ModuleList([copy.deepcopy(layer) for _ in range(config.num_hidden_layers)])

    def forward(self, hidden_states, attention_mask, output_all_encoded_layers=True):
        all_encoder_layers = []
        for layer_module in self.layer:
            hidden_states = layer_module(hidden_states, attention_mask)
            if output_all_encoded_layers:
                all_encoder_layers.append(hidden_states)
        if not output_all_encoded_layers:
            all_encoder_layers.append(hidden_states)
        return all_encoder_layers

class BertLayer(nn.Module):
    def __init__(self, config):
        super(BertLayer, self).__init__()
        self.attention = BertAttention(config)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(self, hidden_states, attention_mask):
        attention_output = self.attention(hidden_states, attention_mask)
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output


class BertAttention(nn.Module):
    def __init__(self, config):
        super(BertAttention, self).__init__()
        self.self = BertSelfAttention(config)
        self.output = BertSelfOutput(config)

    def forward(self, input_tensor, attention_mask):
        self_output = self.self(input_tensor, attention_mask)
        attention_output = self.output(self_output, input_tensor)
        return attention_output

class BertSelfAttention(nn.Module):
    def __init__(self, config):
        super(BertSelfAttention, self).__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads))
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # attention_probs = Sparsemax(dim=-1)(attention_scores)
        
        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        return context_layer


class BertLayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super(BertLayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias


class BertOutput(nn.Module):
    def __init__(self, config):
        super(BertOutput, self).__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertIntermediate(nn.Module):
    def __init__(self, config):
        super(BertIntermediate, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        if isinstance(config.hidden_act, str) or (sys.version_info[0] == 2 and isinstance(config.hidden_act, unicode)):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class BertSelfOutput(nn.Module):
    def __init__(self, config):
        super(BertSelfOutput, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertCoAttention(nn.Module):
    def __init__(self, config):
        super(BertCoAttention, self).__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads))
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, s1_hidden_states, s2_hidden_states, s2_attention_mask):

        mixed_query_layer = self.query(s1_hidden_states)
        mixed_key_layer = self.key(s2_hidden_states)
        mixed_value_layer = self.value(s2_hidden_states)



        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)



        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        attention_scores = attention_scores + s2_attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()

        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        return context_layer


class BertCrossAttention(nn.Module):
    def __init__(self, config):
        super(BertCrossAttention, self).__init__()
        self.self = BertCoAttention(config)
        self.output = BertSelfOutput(config)

    def forward(self, s1_input_tensor, s2_input_tensor, s2_attention_mask):
        s1_cross_output = self.self(s1_input_tensor, s2_input_tensor, s2_attention_mask)
        attention_output = self.output(s1_cross_output, s1_input_tensor)
        return attention_output


class BertCrossAttentionLayer(nn.Module):
    def __init__(self, config):
        super(BertCrossAttentionLayer, self).__init__()
        self.attention = BertCrossAttention(config)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(self, s1_hidden_states, s2_hidden_states, s2_attention_mask):
        attention_output = self.attention(s1_hidden_states, s2_hidden_states, s2_attention_mask)
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output


class BertCrossEncoder(nn.Module):
    def __init__(self, config, layer_num):
        super(BertCrossEncoder, self).__init__()
        layer = BertCrossAttentionLayer(config)
        self.layer = nn.ModuleList([copy.deepcopy(layer) for _ in range(layer_num)])

    def forward(self, s1_hidden_states, s2_hidden_states, s2_attention_mask, output_all_encoded_layers=True):
        all_encoder_layers = []
        for layer_module in self.layer:
            s1_hidden_states = layer_module(s1_hidden_states, s2_hidden_states, s2_attention_mask)
            if output_all_encoded_layers:
                all_encoder_layers.append(s1_hidden_states)
        if not output_all_encoded_layers:
            all_encoder_layers.append(s1_hidden_states)
        return all_encoder_layers

class BertPooler(nn.Module):
    def __init__(self, config):
        super(BertPooler, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output

class BertSelfEncoder(nn.Module):
    def __init__(self, config):
        super(BertSelfEncoder, self).__init__()
        layer = BertLayer(config)
        self.layer = nn.ModuleList([copy.deepcopy(layer) for _ in range(1)])

    def forward(self, hidden_states, attention_mask, output_all_encoded_layers=True):
        all_encoder_layers = []
        for layer_module in self.layer:
            hidden_states = layer_module(hidden_states, attention_mask)
            if output_all_encoded_layers:
                all_encoder_layers.append(hidden_states)
        if not output_all_encoded_layers:
            all_encoder_layers.append(hidden_states)
        return all_encoder_layers


class cross_attention_Y(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        is_decoder: bool = False,
        bias: bool = True,
    ):
        super(cross_attention_Y, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads

        if (self.head_dim * num_heads) != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim}"
                f" and `num_heads`: {num_heads})."
            )
        self.scaling = self.head_dim ** -0.5
        self.is_decoder = is_decoder
        self.relu = torch.nn.ReLU()
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
            self,
            hidden_states: torch.Tensor,
            key_value_states: Optional[torch.Tensor] = None,
            past_key_value: Optional[Tuple[torch.Tensor]] = None,
            attention_mask: Optional[torch.Tensor] = None,
            layer_head_mask: Optional[torch.Tensor] = None,
            output_attentions: bool = False,
            neg_type: bool = False,
            tau=1.0,
            prior_score=None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """Input shape: Batch x Time x Channel"""
        # if key_value_states are provided this layer is used as a cross-attention layer
        # for the decoder
        is_cross_attention = key_value_states is not None
        bsz, tgt_len, _ = hidden_states.size()
        # get query proj
        query_states = self.q_proj(hidden_states) * self.scaling
        # get key, value proj
        if is_cross_attention and past_key_value is not None:
            # reuse k,v, cross_attentions
            key_states = past_key_value[0]
            value_states = past_key_value[1]
        elif is_cross_attention:
            # cross_attentions
            key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
            value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
        elif past_key_value is not None:
            # reuse k, v, self_attention
            key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
            value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)
        else:
            # self_attention
            key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
            value_states = self._shape(self.v_proj(hidden_states), -1, bsz)

        if self.is_decoder:
            # if cross_attention save Tuple(torch.Tensor, torch.Tensor) of all cross attention key/value_states.
            # Further calls to cross_attention layer can then reuse all cross-attention
            # key/value_states (first "if" case)
            # if uni-directional self-attention (decoder) save Tuple(torch.Tensor, torch.Tensor) of
            # all previous decoder key/value_states. Further calls to uni-directional self-attention
            # can concat previous decoder key/value_states to current projected key/value_states (third "elif" case)
            # if encoder bi-directional self-attention `past_key_value` is always `None`
            past_key_value = (key_states, value_states)

        proj_shape = (bsz * self.num_heads, -1, self.head_dim)
        query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
        key_states = key_states.view(*proj_shape)
        value_states = value_states.view(*proj_shape)
        src_len = key_states.size(1)
        attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
        if attn_weights.size() != (bsz * self.num_heads, tgt_len, src_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz * self.num_heads, tgt_len, src_len)}, but is {attn_weights.size()}"
            )
        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, tgt_len, src_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, tgt_len, src_len)}, but is {attention_mask.size()}"
                )
            # attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
            attention_mask = attention_mask.expand(-1, self.num_heads, -1, -1)
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights.masked_fill(attention_mask, -10000.0)
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
        if neg_type:
            attn_weights = 1.0 - torch.nn.functional.softmax(attn_weights / tau, dim=-1)
        else:
            attn_weights = torch.nn.functional.softmax(attn_weights / tau, dim=-1)
        if prior_score is not None:
            prior_score = prior_score.repeat(self.num_heads, 1, 1)
            attn_weights = attn_weights + prior_score
        if layer_head_mask is not None:
            if layer_head_mask.size() != (self.num_heads,):
                raise ValueError(
                    f"Head mask for a single layer should be of size {(self.num_heads,)}, but is {layer_head_mask.size()}"
                )
            attn_weights = layer_head_mask.view(1, -1, 1, 1) * attn_weights.view(bsz, self.num_heads, tgt_len,
                                                                                 src_len)
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
        if output_attentions:
            # this operation is a bit awkward, but it's required to
            # make sure that attn_weights keeps its gradient.
            # In order to do so, attn_weights have to be reshaped
            # twice and have to be reused in the following
            attn_weights_reshaped = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights_reshaped.view(bsz * self.num_heads, tgt_len, src_len)
        else:
            attn_weights_reshaped = None
        attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
        attn_output = torch.bmm(attn_probs, value_states)
        if attn_output.size() != (bsz * self.num_heads, tgt_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, tgt_len, self.head_dim)}, but is {attn_output.size()}"
            )
        attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
        attn_output = attn_output.transpose(1, 2)
        # Use the `embed_dim` from the config (stored in the class) rather than `hidden_state` because `attn_output` can be
        # partitioned aross GPUs when using tensor-parallelism.
        attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
        attn_output = self.out_proj(attn_output)
        return attn_output, attn_weights_reshaped, past_key_value




class ClsLayer_Y(BertLayer):
    """
    Modified from BertLayer to add support for output_hidden_states.
    """

    def __init__(self, config):
        super(ClsLayer_Y, self).__init__(config)
        self.ensemble = nn.Linear(config.hidden_size * 2, 1)
        self.cross_attention = cross_attention_Y(config.hidden_size, 8, dropout=0.3, is_decoder=True)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(self, self_chunk_align, cls, word_mask=None, prior_score=None, cls_2=None):
        CLS_ensem_1 = self.cross_attention(cls.unsqueeze(1), self_chunk_align, tau=1.0, neg_type=False, prior_score=prior_score)
        #CLS_ensem_2 = self.cross_attention_abs(cls_2.unsqueeze(1), self_chunk_align, tau=1.0, neg_type=False, prior_score=prior_score)
        #CLS_ensem_F = self.cross_attention(cls.unsqueeze(1), self_chunk_align, tau=0.5, neg_type=True)
        #CLS_ensem_T_F = self.ensemble(torch.cat([CLS_ensem_T[0].squeeze(1), CLS_ensem_F[0].squeeze(1)], dim=-1))
        #cls_sig_1 = torch.sigmoid(self.ensemble(torch.cat([CLS_ensem_1[0].squeeze(1), cls], dim=-1)))
        #cls_sig_2 = torch.sigmoid(self.ensemble(torch.cat([CLS_ensem_2[0].squeeze(1), cls], dim=-1)))
        #cls_attn_output = self.dropout(cls_attn_output)
        #cls_with_align = self.LayerNorm(cls_attn_output + cls)
        #cls_with_align = cls_sig_1 * CLS_ensem_1[0].squeeze(1) + cls_sig_2 * CLS_ensem_2[0].squeeze(1) + cls
        cls_with_align = self.dropout(CLS_ensem_1[0].squeeze(1))
        cls_with_align = self.LayerNorm(cls_with_align + cls)
        intermediate_output = self.intermediate(cls_with_align)
        layer_output = self.output(intermediate_output, cls_with_align)
        return layer_output


class cls_layer_both(nn.Module):
    def __init__(self,  input_dim, output_dim):
        super(cls_layer_both, self).__init__()
        self.proj_norm = self.LayerNorm = nn.LayerNorm(input_dim)
        self.proj = nn.Linear(input_dim, output_dim)
        
    def forward(self, lang_feat, img_feat):
        proj_feat = lang_feat + img_feat
        proj_feat = self.proj_norm(proj_feat)
        proj_feat = self.proj(proj_feat)

        return proj_feat


class MTCCMBertForMMTokenClassificationCRF(torch.nn.Module):
    def __init__(self, config, embedding, last_encoder, layer_num1=1, layer_num2=1, layer_num3=1,  num_labels=2):
        super().__init__()
        self.num_labels = num_labels
        self.last_encoder = last_encoder
        self.bert = embedding
        self.hidden_size = config.hidden_size
        self.self_attention = BertSelfEncoder(config)
        self.self_attention_v2 = BertSelfEncoder(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.vismap2text = nn.Linear(2048, config.hidden_size)
        # clip输出的维度可能是512，需要转换成hidden_size
        self.vismapping = nn.Linear(512, config.hidden_size)
        self.txt2img_attention = BertCrossEncoder(config, layer_num1)
        self.cls_layer_Y = nn.ModuleList([BertCrossEncoder(config, layer_num1) for _ in range(2)])
        # self.text_coattention = nn.ModuleList([BertCrossEncoder(config, layer_num1) for _ in range(2)])
        self.embedding_layer = nn.Embedding(config.vocab_size, config.hidden_size)
        
        self.lstm = nn.LSTM(input_size=config.hidden_size,
                            hidden_size=config.hidden_size,
                            batch_first=True,
                            bidirectional=True)
        # self.dropout_2 = nn.Dropout(p=0.3)
        self.classifier = torch.nn.Linear(config.hidden_size * 2, num_labels)
        self.crf = CRF(num_tags=num_labels,
                        batch_first=True)
        self.prompt_len = 5
        self.mapping_network_alignment = nn.Sequential(
            nn.Dropout(p=0.3),
            torch.nn.Linear(config.hidden_size, 756 * self.prompt_len, bias=True),
            torch.nn.Tanh(),
            nn.Dropout(p=0.3),
            torch.nn.Linear(756 * self.prompt_len, config.hidden_size * self.prompt_len, bias=True)
        )

        self.mapping_network_vision = nn.Sequential(
            nn.Dropout(p=0.3),
            torch.nn.Linear(2048, 756 * self.prompt_len, bias=True),
            torch.nn.Tanh(),
            nn.Dropout(p=0.3),
            torch.nn.Linear(756 * self.prompt_len, config.hidden_size * self.prompt_len, bias=True)
        )

        self.lastproj = torch.nn.Linear(config.hidden_size, 1024)


        self.cls_layer = cls_layer_both(config.hidden_size,config.hidden_size)
        self.aux_head = nn.Linear(config.hidden_size, 1)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        # for name, parameter in self.bert.named_parameters():
        #     parameter.requires_grad = False


    def forward(self, input_ids, segment_ids, input_mask, ori_input_ids, ori_input_mask, ori_segment_ids, added_attention_mask, 
                clip_features, visual_embeds_mean, visual_embeds_att, offsets, output_mask,rela_score,
                temp=None, temp_lamb=None,lamb=None, labels=None, negative_rate=None, mode = None):
        
        # sequence_output, sequence_output_pooler = self.bert(ori_input_ids, token_type_ids=ori_segment_ids, attention_mask=ori_input_mask,
        #                                output_all_encoded_layers=False)  # batch_size * seq_len * hidden_size
        
        offset = offsets.tolist()[0]
        sequence_output= self.bert(ori_input_ids, token_type_ids=ori_segment_ids, 
                                        attention_mask=ori_input_mask)[0].float()
        # sequence_output = self.embedding_layer(ori_input_ids)

        sequence_output = self.dropout(sequence_output)
        clip_features = self.vismapping(clip_features.to(dtype=next(self.parameters()).dtype).squeeze(1))

        vis_embed_map = visual_embeds_att.view(-1, 2048, 49).permute(0, 2, 1)  # self.batch_size, 49, 2048

        converted_vis_embed_map = self.vismap2text(vis_embed_map)  # self.batch_size, 49, hidden_dim

        # '''
        # apply txt2img attention mechanism to obtain image-based text representations
        img_mask = added_attention_mask[:, :49]  # batch_size * 49
        extended_img_mask = img_mask.unsqueeze(1).unsqueeze(2)  # batch_size * 1 * 1 * 49
        extended_img_mask = extended_img_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_img_mask = (1.0 - extended_img_mask) * -10000.0


        cross_encoder = self.txt2img_attention(sequence_output, converted_vis_embed_map, extended_img_mask)
        cross_output_layer = cross_encoder[-1]  #  batch_size * text_len * hidden_dim
        # kd_loss = 0.0
        # for student_rep, teacher_rep in zip(cross_output_layer[:,0,:], clip_features):
        #     tmp_loss = F.mse_loss(F.normalize(student_rep, p=2, dim=-1), F.normalize(teacher_rep, p=2, dim=-1))
        #     kd_loss += tmp_loss

        # extended_text_mask = ori_input_mask[:, :1].unsqueeze(1).unsqueeze(2)  # batch_size * 1 * 1 * 1
        extended_text_mask = ori_input_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_text_mask = (1.0 - extended_text_mask) * -10000.0
        
        
        # ==using BertCrossEncoder
        clip_features = clip_features.unsqueeze(1)
        extended_text_mask = extended_text_mask.unsqueeze(1).unsqueeze(2)
        # ===using BertCrossEncoder
        for i, layer_module in enumerate(self.cls_layer_Y):

            # clip_features = layer_module(cross_output_layer, clip_features, extended_text_mask, None, None)

            # using BertCrossEncoder
            clip_features = layer_module(clip_features, cross_output_layer, extended_text_mask)[-1]
            # using BertCrossEncoder

        
        # 去掉第三个
        # clip_features = cross_output_layer[:,0,:].unsqueeze(1)
        Alignment_prompt = self.mapping_network_alignment(clip_features).unsqueeze(1).view(ori_input_ids.size(0), self.prompt_len, -1)
        # 去掉第三个
        
        prefix_vision = self.mapping_network_vision(visual_embeds_mean)
        prefix_vision = prefix_vision.reshape(input_ids.size(0), self.prompt_len, -1)


        prefix_emb = torch.cat([prefix_vision, Alignment_prompt], dim=1)
        if prefix_emb.size(2) != 1024:
            prefix_emb = self.lastproj(prefix_emb)


        align_mask = input_mask[:, :1].repeat(1, self.prompt_len)
        vision_mask = input_mask[:, :1].repeat(1, self.prompt_len)
        prompt_mask = torch.cat([vision_mask, align_mask], dim=1)
        roberta_encoder_outputs = self.last_encoder(input_ids=input_ids,token_type_ids=segment_ids,
                                              attention_mask=input_mask, prompt_embeddings=prefix_emb, 
                                              input_mask=prompt_mask, offset=offset)
        roberta_encoder_output = roberta_encoder_outputs[0] 
        # roberta_encoder_output = [batch_size, input_ids.size(1) - 2 + prefix_emb.size(1), 1024 ]
        # last_encoder是roberta_large,它的输出总是1024
        



        # offset: 总输入中，代表ori_input的最开头token的相应下标，- 2 + prefix_emb.size(1)将Mask换成prefix_embedding的长度
        # 也是要根据prompt_text被分词后的结果动态调整
        offset = offset - 2 + prefix_emb.size(1)
 
        token_embedding = roberta_encoder_output[:,offset: offset+128,:]
        
        
        # result = token_embedding * 0.5 + cross_output_layer * 0.5

        cross_output_layer_global = cross_output_layer[:,0,:]
        token_embedding_global = token_embedding[:,0,:]
        

        related_feat = self.cls_layer(cross_output_layer_global, token_embedding_global)
        logits = self.aux_head(related_feat)
        gate_signal = torch.sigmoid(logits).view(token_embedding.size(0), 1, 1)
        result = gate_signal*token_embedding + (1 - gate_signal)* cross_output_layer

        # fixed_relation_score 
        # rela_score = rela_score.view(token_embedding.size(0), 1, 1)
        # result = rela_score*token_embedding + (1 - rela_score)* cross_output_layer

        x, _ = self.lstm(result)
        emissions = self.classifier(x)

        output_mask = (output_mask != 0)
        if mode == 'train':
            loss = - self.crf(emissions, tags=labels, mask=output_mask,
                          reduction='token_mean')
            return loss
        elif mode == 'dev':
            pred_tags = self.crf.decode(emissions, mask=output_mask)
            loss = - self.crf(emissions, tags=labels, mask=output_mask,
                                    reduction='token_mean')
            return pred_tags,loss
        elif mode == 'test':
            pred_tags = self.crf.decode(emissions, mask=output_mask)
            return pred_tags
        


class MTCCMBertForMMTokenClassificationCRF_bert(torch.nn.Module):
    def __init__(self, config, embedding, last_encoder, layer_num1=1, layer_num2=1, layer_num3=1,  num_labels=2):
        super().__init__()
        self.num_labels = num_labels
        self.last_encoder = last_encoder
        self.bert = embedding
        self.hidden_size = config.hidden_size
        self.self_attention = BertSelfEncoder(config)
        self.self_attention_v2 = BertSelfEncoder(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.vismap2text = nn.Linear(2048, config.hidden_size)
        # clip输出的维度可能是512，需要转换成hidden_size
        self.vismapping = nn.Linear(512, config.hidden_size)
        self.txt2img_attention = BertCrossEncoder(config, layer_num1)
        self.cls_layer_Y = nn.ModuleList([BertCrossEncoder(config, layer_num1) for _ in range(5)])
        # self.text_coattention = nn.ModuleList([BertCrossEncoder(config, layer_num1) for _ in range(2)])
        self.embedding_layer = nn.Embedding(config.vocab_size, config.hidden_size)
        
        self.lstm = nn.LSTM(input_size=config.hidden_size,
                            hidden_size=config.hidden_size,
                            batch_first=True,
                            bidirectional=True)
        # self.dropout_2 = nn.Dropout(p=0.3)
        self.classifier = torch.nn.Linear(config.hidden_size * 2, num_labels)
        self.crf = CRF(num_tags=num_labels,
                        batch_first=True)
        self.prompt_len = 5
        self.mapping_network_alignment = nn.Sequential(
            nn.Dropout(p=0.3),
            torch.nn.Linear(config.hidden_size, 756 * self.prompt_len, bias=True),
            torch.nn.Tanh(),
            nn.Dropout(p=0.3),
            torch.nn.Linear(756 * self.prompt_len, config.hidden_size * self.prompt_len, bias=True)
        )

        self.mapping_network_vision = nn.Sequential(
            nn.Dropout(p=0.3),
            torch.nn.Linear(2048, 756 * self.prompt_len, bias=True),
            torch.nn.Tanh(),
            nn.Dropout(p=0.3),
            torch.nn.Linear(756 * self.prompt_len, config.hidden_size * self.prompt_len, bias=True)
        )

        self.lastproj = torch.nn.Linear(config.hidden_size, 1024)


        self.cls_layer = cls_layer_both(config.hidden_size,config.hidden_size)
        self.aux_head = nn.Linear(config.hidden_size, 1)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        # for name, parameter in self.bert.named_parameters():
        #     parameter.requires_grad = False


    def forward(self, input_ids, segment_ids, input_mask, ori_input_ids, ori_input_mask, ori_segment_ids, added_attention_mask, 
                clip_features, visual_embeds_mean, visual_embeds_att, offsets, output_mask,rela_score,
                temp=None, temp_lamb=None,lamb=None, labels=None, negative_rate=None, mode = None):
        
        # sequence_output, sequence_output_pooler = self.bert(ori_input_ids, token_type_ids=ori_segment_ids, attention_mask=ori_input_mask,
        #                                output_all_encoded_layers=False)  # batch_size * seq_len * hidden_size
        
        offset = offsets.tolist()[0]
        sequence_output= self.bert(ori_input_ids, token_type_ids=ori_segment_ids, 
                                        attention_mask=ori_input_mask)[0].float()
        # sequence_output = self.embedding_layer(ori_input_ids)

        sequence_output = self.dropout(sequence_output)
        clip_features = self.vismapping(clip_features.to(dtype=next(self.parameters()).dtype).squeeze(1))

        vis_embed_map = visual_embeds_att.view(-1, 2048, 49).permute(0, 2, 1)  # self.batch_size, 49, 2048

        converted_vis_embed_map = self.vismap2text(vis_embed_map)  # self.batch_size, 49, hidden_dim

        # '''
        # apply txt2img attention mechanism to obtain image-based text representations
        img_mask = added_attention_mask[:, :49]  # batch_size * 49
        extended_img_mask = img_mask.unsqueeze(1).unsqueeze(2)  # batch_size * 1 * 1 * 49
        extended_img_mask = extended_img_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_img_mask = (1.0 - extended_img_mask) * -10000.0


        cross_encoder = self.txt2img_attention(sequence_output, converted_vis_embed_map, extended_img_mask)
        cross_output_layer = cross_encoder[-1]  #  batch_size * text_len * hidden_dim
        # kd_loss = 0.0
        # for student_rep, teacher_rep in zip(cross_output_layer[:,0,:], clip_features):
        #     tmp_loss = F.mse_loss(F.normalize(student_rep, p=2, dim=-1), F.normalize(teacher_rep, p=2, dim=-1))
        #     kd_loss += tmp_loss

        # extended_text_mask = ori_input_mask[:, :1].unsqueeze(1).unsqueeze(2)  # batch_size * 1 * 1 * 1
        extended_text_mask = ori_input_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_text_mask = (1.0 - extended_text_mask) * -10000.0
        
        
        # ==using BertCrossEncoder
        clip_features = clip_features.unsqueeze(1)
        extended_text_mask = extended_text_mask.unsqueeze(1).unsqueeze(2)
        # ===using BertCrossEncoder
        for i, layer_module in enumerate(self.cls_layer_Y):

            # clip_features = layer_module(cross_output_layer, clip_features, extended_text_mask, None, None)

            # using BertCrossEncoder
            clip_features = layer_module(clip_features, cross_output_layer, extended_text_mask)[-1]
            # using BertCrossEncoder

        
        # 去掉第三个
        # clip_features = cross_output_layer[:,0,:].unsqueeze(1)
        Alignment_prompt = self.mapping_network_alignment(clip_features).unsqueeze(1).view(ori_input_ids.size(0), self.prompt_len, -1)
        # 去掉第三个
        
        prefix_vision = self.mapping_network_vision(visual_embeds_mean)
        prefix_vision = prefix_vision.reshape(input_ids.size(0), self.prompt_len, -1)


        prefix_emb = torch.cat([prefix_vision, Alignment_prompt], dim=1)
        # if prefix_emb.size(2) != 1024:
        #     prefix_emb = self.lastproj(prefix_emb)


        align_mask = input_mask[:, :1].repeat(1, self.prompt_len)
        vision_mask = input_mask[:, :1].repeat(1, self.prompt_len)
        prompt_mask = torch.cat([vision_mask, align_mask], dim=1)
        roberta_encoder_outputs = self.last_encoder(input_ids=input_ids,token_type_ids=segment_ids,
                                              attention_mask=input_mask, prompt_embeddings=prefix_emb, 
                                              input_mask=prompt_mask, offset=offset)
        roberta_encoder_output = roberta_encoder_outputs[0] 
        # roberta_encoder_output = [batch_size, input_ids.size(1) - 2 + prefix_emb.size(1), 1024 ]
        # last_encoder是roberta_large,它的输出总是1024
        



        # offset: 总输入中，代表ori_input的最开头token的相应下标，- 2 + prefix_emb.size(1)将Mask换成prefix_embedding的长度
        # 也是要根据prompt_text被分词后的结果动态调整
        offset = offset - 2 + prefix_emb.size(1)
 
        token_embedding = roberta_encoder_output[:,offset: offset+128,:]
        
        
        # result = token_embedding * 0.5 + cross_output_layer * 0.5

        cross_output_layer_global = cross_output_layer[:,0,:]
        token_embedding_global = token_embedding[:,0,:]
        

        related_feat = self.cls_layer(cross_output_layer_global, token_embedding_global)
        logits = self.aux_head(related_feat)
        gate_signal = torch.sigmoid(logits).view(token_embedding.size(0), 1, 1)
        result = gate_signal*token_embedding + (1 - gate_signal)* cross_output_layer

        # fixed_relation_score 
        # rela_score = rela_score.view(token_embedding.size(0), 1, 1)
        # result = rela_score*token_embedding + (1 - rela_score)* cross_output_layer

        x, _ = self.lstm(result)
        emissions = self.classifier(x)

        output_mask = (output_mask != 0)
        if mode == 'train':
            loss = - self.crf(emissions, tags=labels, mask=output_mask,
                          reduction='token_mean')
            return loss
        elif mode == 'dev':
            pred_tags = self.crf.decode(emissions, mask=output_mask)
            loss = - self.crf(emissions, tags=labels, mask=output_mask,
                                    reduction='token_mean')
            return pred_tags,loss
        elif mode == 'test':
            pred_tags = self.crf.decode(emissions, mask=output_mask)
            return pred_tags



# 去掉跨模态注意力机制：在第对齐模块中，文本向量采用编码器输出的**文本向量**。
#       同时门控由token_embedding_global决定，并且由文本向量进行计算（而不是跨模态后的文本向量）
class MTCCMBertForMMTokenClassificationCRF_woCrossAtt_1(torch.nn.Module):
    def __init__(self, config, embedding, last_encoder, layer_num1=1, layer_num2=1, layer_num3=1,  num_labels=2):
        super().__init__()
        self.num_labels = num_labels
        self.last_encoder = last_encoder
        self.bert = embedding
        self.hidden_size = config.hidden_size
        self.self_attention = BertSelfEncoder(config)
        self.self_attention_v2 = BertSelfEncoder(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.vismap2text = nn.Linear(2048, config.hidden_size)
        # clip输出的维度可能是512，需要转换成hidden_size
        self.vismapping = nn.Linear(512, config.hidden_size)
        self.txt2img_attention = BertCrossEncoder(config, layer_num1)
        self.cls_layer_Y = nn.ModuleList([BertCrossEncoder(config, layer_num1) for _ in range(2)])
        # self.text_coattention = nn.ModuleList([BertCrossEncoder(config, layer_num1) for _ in range(2)])
        self.embedding_layer = nn.Embedding(config.vocab_size, config.hidden_size)
        
        self.lstm = nn.LSTM(input_size=config.hidden_size,
                            hidden_size=config.hidden_size,
                            batch_first=True,
                            bidirectional=True)
        # self.dropout_2 = nn.Dropout(p=0.3)
        self.classifier = torch.nn.Linear(config.hidden_size * 2, num_labels)
        self.crf = CRF(num_tags=num_labels,
                        batch_first=True)
        self.prompt_len = 5
        self.mapping_network_alignment = nn.Sequential(
            nn.Dropout(p=0.3),
            torch.nn.Linear(config.hidden_size, 756 * self.prompt_len, bias=True),
            torch.nn.Tanh(),
            nn.Dropout(p=0.3),
            torch.nn.Linear(756 * self.prompt_len, config.hidden_size * self.prompt_len, bias=True)
        )

        self.mapping_network_vision = nn.Sequential(
            nn.Dropout(p=0.3),
            torch.nn.Linear(2048, 756 * self.prompt_len, bias=True),
            torch.nn.Tanh(),
            nn.Dropout(p=0.3),
            torch.nn.Linear(756 * self.prompt_len, config.hidden_size * self.prompt_len, bias=True)
        )

        self.lastproj = torch.nn.Linear(config.hidden_size, 1024)


        self.cls_layer = cls_layer_both(config.hidden_size,config.hidden_size)
        self.aux_head = nn.Linear(config.hidden_size, 1)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        # for name, parameter in self.bert.named_parameters():
        #     parameter.requires_grad = False


    def forward(self, input_ids, segment_ids, input_mask, ori_input_ids, ori_input_mask, ori_segment_ids, added_attention_mask, 
                clip_features, visual_embeds_mean, visual_embeds_att, offsets, output_mask,rela_score,
                temp=None, temp_lamb=None,lamb=None, labels=None, negative_rate=None, mode = None):
        
        # sequence_output, sequence_output_pooler = self.bert(ori_input_ids, token_type_ids=ori_segment_ids, attention_mask=ori_input_mask,
        #                                output_all_encoded_layers=False)  # batch_size * seq_len * hidden_size
        
        offset = offsets.tolist()[0]
        sequence_output= self.bert(ori_input_ids, token_type_ids=ori_segment_ids, 
                                        attention_mask=ori_input_mask)[0].float()
        # sequence_output = self.embedding_layer(ori_input_ids)

        sequence_output = self.dropout(sequence_output)
        clip_features = self.vismapping(clip_features.to(dtype=next(self.parameters()).dtype).squeeze(1))

        vis_embed_map = visual_embeds_att.view(-1, 2048, 49).permute(0, 2, 1)  # self.batch_size, 49, 2048

        converted_vis_embed_map = self.vismap2text(vis_embed_map)  # self.batch_size, 49, hidden_dim

        # '''
        # apply txt2img attention mechanism to obtain image-based text representations
        img_mask = added_attention_mask[:, :49]  # batch_size * 49
        extended_img_mask = img_mask.unsqueeze(1).unsqueeze(2)  # batch_size * 1 * 1 * 49
        extended_img_mask = extended_img_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_img_mask = (1.0 - extended_img_mask) * -10000.0


        # kd_loss = 0.0
        # for student_rep, teacher_rep in zip(cross_output_layer[:,0,:], clip_features):
        #     tmp_loss = F.mse_loss(F.normalize(student_rep, p=2, dim=-1), F.normalize(teacher_rep, p=2, dim=-1))
        #     kd_loss += tmp_loss

        # extended_text_mask = ori_input_mask[:, :1].unsqueeze(1).unsqueeze(2)  # batch_size * 1 * 1 * 1
        extended_text_mask = ori_input_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_text_mask = (1.0 - extended_text_mask) * -10000.0
        
        
        # ==using BertCrossEncoder
        clip_features = clip_features.unsqueeze(1)
        extended_text_mask = extended_text_mask.unsqueeze(1).unsqueeze(2)
        # ===using BertCrossEncoder
        for i, layer_module in enumerate(self.cls_layer_Y):

            # clip_features = layer_module(cross_output_layer, clip_features, extended_text_mask, None, None)

            # using BertCrossEncoder
            clip_features = layer_module(clip_features, sequence_output, extended_text_mask)[-1]
            # using BertCrossEncoder

        
        # 去掉第三个
        # clip_features = cross_output_layer[:,0,:].unsqueeze(1)
        Alignment_prompt = self.mapping_network_alignment(clip_features).unsqueeze(1).view(ori_input_ids.size(0), self.prompt_len, -1)
        # 去掉第三个
        
        prefix_vision = self.mapping_network_vision(visual_embeds_mean)
        prefix_vision = prefix_vision.reshape(input_ids.size(0), self.prompt_len, -1)


        prefix_emb = torch.cat([prefix_vision, Alignment_prompt], dim=1)
        if prefix_emb.size(2) != 1024:
            prefix_emb = self.lastproj(prefix_emb)


        align_mask = input_mask[:, :1].repeat(1, self.prompt_len)
        vision_mask = input_mask[:, :1].repeat(1, self.prompt_len)
        prompt_mask = torch.cat([vision_mask, align_mask], dim=1)
        roberta_encoder_outputs = self.last_encoder(input_ids=input_ids,token_type_ids=segment_ids,
                                              attention_mask=input_mask, prompt_embeddings=prefix_emb, 
                                              input_mask=prompt_mask, offset=offset)
        roberta_encoder_output = roberta_encoder_outputs[0] 
        # roberta_encoder_output = [batch_size, input_ids.size(1) - 2 + prefix_emb.size(1), 1024 ]
        # last_encoder是roberta_large,它的输出总是1024
        



        # offset: 总输入中，代表ori_input的最开头token的相应下标，- 2 + prefix_emb.size(1)将Mask换成prefix_embedding的长度
        # 也是要根据prompt_text被分词后的结果动态调整
        offset = offset - 2 + prefix_emb.size(1)
 
        token_embedding = roberta_encoder_output[:,offset: offset+128,:]
        
        
        # result = token_embedding * 0.5 + cross_output_layer * 0.5

        cross_output_layer_global = sequence_output[:,0,:]
        token_embedding_global = token_embedding[:,0,:]
        

        related_feat = self.cls_layer(cross_output_layer_global, token_embedding_global)
        logits = self.aux_head(related_feat)
        gate_signal = torch.sigmoid(logits).view(token_embedding.size(0), 1, 1)
        result = gate_signal*token_embedding + (1 - gate_signal)* sequence_output

        # fixed_relation_score 
        # rela_score = rela_score.view(token_embedding.size(0), 1, 1)
        # result = rela_score*token_embedding + (1 - rela_score)* cross_output_layer

        x, _ = self.lstm(result)
        emissions = self.classifier(x)

        output_mask = (output_mask != 0)
        if mode == 'train':
            loss = - self.crf(emissions, tags=labels, mask=output_mask,
                          reduction='token_mean')
            return loss
        elif mode == 'dev':
            pred_tags = self.crf.decode(emissions, mask=output_mask)
            loss = - self.crf(emissions, tags=labels, mask=output_mask,
                                    reduction='token_mean')
            return pred_tags,loss
        elif mode == 'test':
            pred_tags = self.crf.decode(emissions, mask=output_mask)
            return pred_tags
        


# 去掉跨模态注意力机制_2：由于该处的去掉也会同时影响后续对齐模块，所以应该做到对CrossAtt进行削弱的同时保留一部分语义。
#       ————用Mask来去掉一部分Att值
#       门控依旧由CrossAtt的全局进行计算，同时也保留了CrossAtt的计算权重
class MTCCMBertForMMTokenClassificationCRF_woCrossAtt_2(torch.nn.Module):
    def __init__(self, config, embedding, last_encoder, layer_num1=1, layer_num2=1, layer_num3=1,  num_labels=2):
        super().__init__()
        self.num_labels = num_labels
        self.last_encoder = last_encoder
        self.bert = embedding
        self.hidden_size = config.hidden_size
        self.self_attention = BertSelfEncoder(config)
        self.self_attention_v2 = BertSelfEncoder(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.vismap2text = nn.Linear(2048, config.hidden_size)
        # clip输出的维度可能是512，需要转换成hidden_size
        self.vismapping = nn.Linear(512, config.hidden_size)
        self.txt2img_attention = BertCrossEncoder(config, layer_num1)
        self.cls_layer_Y = nn.ModuleList([BertCrossEncoder(config, layer_num1) for _ in range(2)])
        # self.text_coattention = nn.ModuleList([BertCrossEncoder(config, layer_num1) for _ in range(2)])
        self.embedding_layer = nn.Embedding(config.vocab_size, config.hidden_size)
        
        self.lstm = nn.LSTM(input_size=config.hidden_size,
                            hidden_size=config.hidden_size,
                            batch_first=True,
                            bidirectional=True)
        # self.dropout_2 = nn.Dropout(p=0.3)
        self.classifier = torch.nn.Linear(config.hidden_size * 2, num_labels)
        self.crf = CRF(num_tags=num_labels,
                        batch_first=True)
        self.prompt_len = 5
        self.mapping_network_alignment = nn.Sequential(
            nn.Dropout(p=0.3),
            torch.nn.Linear(config.hidden_size, 756 * self.prompt_len, bias=True),
            torch.nn.Tanh(),
            nn.Dropout(p=0.3),
            torch.nn.Linear(756 * self.prompt_len, config.hidden_size * self.prompt_len, bias=True)
        )

        self.mapping_network_vision = nn.Sequential(
            nn.Dropout(p=0.3),
            torch.nn.Linear(2048, 756 * self.prompt_len, bias=True),
            torch.nn.Tanh(),
            nn.Dropout(p=0.3),
            torch.nn.Linear(756 * self.prompt_len, config.hidden_size * self.prompt_len, bias=True)
        )

        self.lastproj = torch.nn.Linear(config.hidden_size, 1024)


        self.cls_layer = cls_layer_both(config.hidden_size,config.hidden_size)
        self.aux_head = nn.Linear(config.hidden_size, 1)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        # for name, parameter in self.bert.named_parameters():
        #     parameter.requires_grad = False


    def forward(self, input_ids, segment_ids, input_mask, ori_input_ids, ori_input_mask, ori_segment_ids, added_attention_mask, 
                clip_features, visual_embeds_mean, visual_embeds_att, offsets, output_mask,rela_score,
                temp=None, temp_lamb=None,lamb=None, labels=None, negative_rate=None, random_mask=None, mode = None):
        
        # sequence_output, sequence_output_pooler = self.bert(ori_input_ids, token_type_ids=ori_segment_ids, attention_mask=ori_input_mask,
        #                                output_all_encoded_layers=False)  # batch_size * seq_len * hidden_size
        
        offset = offsets.tolist()[0]
        sequence_output= self.bert(ori_input_ids, token_type_ids=ori_segment_ids, 
                                        attention_mask=ori_input_mask)[0].float()
        # sequence_output = self.embedding_layer(ori_input_ids)

        sequence_output = self.dropout(sequence_output)
        clip_features = self.vismapping(clip_features.to(dtype=next(self.parameters()).dtype).squeeze(1))

        vis_embed_map = visual_embeds_att.view(-1, 2048, 49).permute(0, 2, 1)  # self.batch_size, 49, 2048

        converted_vis_embed_map = self.vismap2text(vis_embed_map)  # self.batch_size, 49, hidden_dim

        # 随机mask掉0.3的token
        img_mask = added_attention_mask[:, :49]  # batch_size * 49
        if random_mask != None:
            img_mask = img_mask * random_mask
        extended_img_mask = img_mask.unsqueeze(1).unsqueeze(2)  # batch_size * 1 * 1 * 49
        extended_img_mask = extended_img_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_img_mask = (1.0 - extended_img_mask) * -10000.0


        cross_encoder = self.txt2img_attention(sequence_output, converted_vis_embed_map, extended_img_mask)
        cross_output_layer = cross_encoder[-1]  #  batch_size * text_len * hidden_dim

        # extended_text_mask = ori_input_mask[:, :1].unsqueeze(1).unsqueeze(2)  # batch_size * 1 * 1 * 1
        extended_text_mask = ori_input_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_text_mask = (1.0 - extended_text_mask) * -10000.0
        
        
        # ==using BertCrossEncoder
        clip_features = clip_features.unsqueeze(1)
        extended_text_mask = extended_text_mask.unsqueeze(1).unsqueeze(2)
        # ===using BertCrossEncoder
        for i, layer_module in enumerate(self.cls_layer_Y):

            # clip_features = layer_module(cross_output_layer, clip_features, extended_text_mask, None, None)

            # using BertCrossEncoder
            clip_features = layer_module(clip_features, cross_output_layer, extended_text_mask)[-1]
            # using BertCrossEncoder

        
        # 去掉第三个
        # clip_features = cross_output_layer[:,0,:].unsqueeze(1)
        Alignment_prompt = self.mapping_network_alignment(clip_features).unsqueeze(1).view(ori_input_ids.size(0), self.prompt_len, -1)
        # 去掉第三个


        prefix_vision = self.mapping_network_vision(visual_embeds_mean)
        prefix_vision = prefix_vision.reshape(input_ids.size(0), self.prompt_len, -1)


        prefix_emb = torch.cat([prefix_vision, Alignment_prompt], dim=1)
        if prefix_emb.size(2) != 1024:
            prefix_emb = self.lastproj(prefix_emb)


        align_mask = input_mask[:, :1].repeat(1, self.prompt_len)
        vision_mask = input_mask[:, :1].repeat(1, self.prompt_len)
        prompt_mask = torch.cat([vision_mask, align_mask], dim=1)
        roberta_encoder_outputs = self.last_encoder(input_ids=input_ids,token_type_ids=segment_ids,
                                              attention_mask=input_mask, prompt_embeddings=prefix_emb, 
                                              input_mask=prompt_mask, offset=offset)
        roberta_encoder_output = roberta_encoder_outputs[0] 
        # roberta_encoder_output = [batch_size, input_ids.size(1) - 2 + prefix_emb.size(1), 1024 ]
        # last_encoder是roberta_large,它的输出总是1024
        



        # offset: 总输入中，代表ori_input的最开头token的相应下标，- 2 + prefix_emb.size(1)将Mask换成prefix_embedding的长度
        # 也是要根据prompt_text被分词后的结果动态调整
        offset = offset - 2 + prefix_emb.size(1)
 
        token_embedding = roberta_encoder_output[:,offset: offset+128,:]
        
        
        # result = token_embedding * 0.5 + cross_output_layer * 0.5

        cross_output_layer_global = cross_output_layer[:,0,:]
        token_embedding_global = token_embedding[:,0,:]
        

        related_feat = self.cls_layer(cross_output_layer_global, token_embedding_global)
        logits = self.aux_head(related_feat)
        gate_signal = torch.sigmoid(logits).view(token_embedding.size(0), 1, 1)
        result = gate_signal*token_embedding + (1 - gate_signal)* cross_output_layer

        # fixed_relation_score 
        # rela_score = rela_score.view(token_embedding.size(0), 1, 1)
        # result = rela_score*token_embedding + (1 - rela_score)* cross_output_layer

        x, _ = self.lstm(result)
        emissions = self.classifier(x)

        output_mask = (output_mask != 0)
        if mode == 'train':
            loss = - self.crf(emissions, tags=labels, mask=output_mask,
                          reduction='token_mean')
            return loss
        elif mode == 'dev':
            pred_tags = self.crf.decode(emissions, mask=output_mask)
            loss = - self.crf(emissions, tags=labels, mask=output_mask,
                                    reduction='token_mean')
            return pred_tags,loss
        elif mode == 'test':
            pred_tags = self.crf.decode(emissions, mask=output_mask)
            return pred_tags



# 关于对齐模块的消融实验1： 直接把 跨模态注意力的第一个全局token作为 image的桥梁。证明了对齐模块对整体效果提升的必要性
#       门控依旧由CrossAtt的全局进行计算，同时也保留了CrossAtt的计算权重
class MTCCMBertForMMTokenClassificationCRF_woPart2_1(torch.nn.Module):

    def __init__(self, config, embedding, last_encoder, layer_num1=1, layer_num2=1, layer_num3=1,  num_labels=2):
        super().__init__()
        self.num_labels = num_labels
        self.last_encoder = last_encoder
        self.bert = embedding
        self.hidden_size = config.hidden_size
        self.self_attention = BertSelfEncoder(config)
        self.self_attention_v2 = BertSelfEncoder(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.vismap2text = nn.Linear(2048, config.hidden_size)
        # clip输出的维度可能是512，需要转换成hidden_size
        self.vismapping = nn.Linear(512, config.hidden_size)
        self.txt2img_attention = BertCrossEncoder(config, layer_num1)
        self.cls_layer_Y = nn.ModuleList([BertCrossEncoder(config, layer_num1) for _ in range(2)])
        # self.text_coattention = nn.ModuleList([BertCrossEncoder(config, layer_num1) for _ in range(2)])
        self.embedding_layer = nn.Embedding(config.vocab_size, config.hidden_size)
        
        self.lstm = nn.LSTM(input_size=config.hidden_size,
                            hidden_size=config.hidden_size,
                            batch_first=True,
                            bidirectional=True)
        # self.dropout_2 = nn.Dropout(p=0.3)
        self.classifier = torch.nn.Linear(config.hidden_size * 2, num_labels)
        self.crf = CRF(num_tags=num_labels,
                        batch_first=True)
        self.prompt_len = 5
        self.mapping_network_alignment = nn.Sequential(
            nn.Dropout(p=0.3),
            torch.nn.Linear(config.hidden_size, 756 * self.prompt_len, bias=True),
            torch.nn.Tanh(),
            nn.Dropout(p=0.3),
            torch.nn.Linear(756 * self.prompt_len, config.hidden_size * self.prompt_len, bias=True)
        )

        self.mapping_network_vision = nn.Sequential(
            nn.Dropout(p=0.3),
            torch.nn.Linear(2048, 756 * self.prompt_len, bias=True),
            torch.nn.Tanh(),
            nn.Dropout(p=0.3),
            torch.nn.Linear(756 * self.prompt_len, config.hidden_size * self.prompt_len, bias=True)
        )

        self.lastproj = torch.nn.Linear(config.hidden_size, 1024)


        self.cls_layer = cls_layer_both(config.hidden_size,config.hidden_size)
        self.aux_head = nn.Linear(config.hidden_size, 1)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        # for name, parameter in self.bert.named_parameters():
        #     parameter.requires_grad = False


    def forward(self, input_ids, segment_ids, input_mask, ori_input_ids, ori_input_mask, ori_segment_ids, added_attention_mask, 
                clip_features, visual_embeds_mean, visual_embeds_att, offsets, output_mask,rela_score,
                temp=None, temp_lamb=None,lamb=None, labels=None, negative_rate=None, mode = None):
        
        # sequence_output, sequence_output_pooler = self.bert(ori_input_ids, token_type_ids=ori_segment_ids, attention_mask=ori_input_mask,
        #                                output_all_encoded_layers=False)  # batch_size * seq_len * hidden_size
        
        offset = offsets.tolist()[0]
        sequence_output= self.bert(ori_input_ids, token_type_ids=ori_segment_ids, 
                                        attention_mask=ori_input_mask)[0].float()
        # sequence_output = self.embedding_layer(ori_input_ids)

        sequence_output = self.dropout(sequence_output)
        clip_features = self.vismapping(clip_features.to(dtype=next(self.parameters()).dtype).squeeze(1))

        vis_embed_map = visual_embeds_att.view(-1, 2048, 49).permute(0, 2, 1)  # self.batch_size, 49, 2048

        converted_vis_embed_map = self.vismap2text(vis_embed_map)  # self.batch_size, 49, hidden_dim

        # '''
        # apply txt2img attention mechanism to obtain image-based text representations
        img_mask = added_attention_mask[:, :49]  # batch_size * 49
        extended_img_mask = img_mask.unsqueeze(1).unsqueeze(2)  # batch_size * 1 * 1 * 49
        extended_img_mask = extended_img_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_img_mask = (1.0 - extended_img_mask) * -10000.0


        cross_encoder = self.txt2img_attention(sequence_output, converted_vis_embed_map, extended_img_mask)
        cross_output_layer = cross_encoder[-1]  #  batch_size * text_len * hidden_dim
        
        extended_text_mask = ori_input_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_text_mask = (1.0 - extended_text_mask) * -10000.0
        
        
        # 去掉第三个
        clip_features = cross_output_layer[:,0,:].unsqueeze(1)
        Alignment_prompt = self.mapping_network_alignment(clip_features).unsqueeze(1).view(ori_input_ids.size(0), self.prompt_len, -1)
        # 去掉第三个
        
        prefix_vision = self.mapping_network_vision(visual_embeds_mean)
        prefix_vision = prefix_vision.reshape(input_ids.size(0), self.prompt_len, -1)


        prefix_emb = torch.cat([prefix_vision, Alignment_prompt], dim=1)
        if prefix_emb.size(2) != 1024:
            prefix_emb = self.lastproj(prefix_emb)


        align_mask = input_mask[:, :1].repeat(1, self.prompt_len)
        vision_mask = input_mask[:, :1].repeat(1, self.prompt_len)
        prompt_mask = torch.cat([vision_mask, align_mask], dim=1)
        roberta_encoder_outputs = self.last_encoder(input_ids=input_ids,token_type_ids=segment_ids,
                                              attention_mask=input_mask, prompt_embeddings=prefix_emb, 
                                              input_mask=prompt_mask, offset=offset)
        roberta_encoder_output = roberta_encoder_outputs[0] 
        # roberta_encoder_output = [batch_size, input_ids.size(1) - 2 + prefix_emb.size(1), 1024 ]
        # last_encoder是roberta_large,它的输出总是1024
        



        # offset: 总输入中，代表ori_input的最开头token的相应下标，- 2 + prefix_emb.size(1)将Mask换成prefix_embedding的长度
        # 也是要根据prompt_text被分词后的结果动态调整
        offset = offset - 2 + prefix_emb.size(1)
 
        token_embedding = roberta_encoder_output[:,offset: offset+128,:]
        
        
        # result = token_embedding * 0.5 + cross_output_layer * 0.5

        cross_output_layer_global = cross_output_layer[:,0,:]
        token_embedding_global = token_embedding[:,0,:]
        

        related_feat = self.cls_layer(cross_output_layer_global, token_embedding_global)
        logits = self.aux_head(related_feat)
        gate_signal = torch.sigmoid(logits).view(token_embedding.size(0), 1, 1)
        result = gate_signal*token_embedding + (1 - gate_signal)* cross_output_layer

        # fixed_relation_score 
        # rela_score = rela_score.view(token_embedding.size(0), 1, 1)
        # result = rela_score*token_embedding + (1 - rela_score)* cross_output_layer

        x, _ = self.lstm(result)
        emissions = self.classifier(x)

        output_mask = (output_mask != 0)
        if mode == 'train':
            loss = - self.crf(emissions, tags=labels, mask=output_mask,
                          reduction='token_mean')
            return loss
        elif mode == 'dev':
            pred_tags = self.crf.decode(emissions, mask=output_mask)
            loss = - self.crf(emissions, tags=labels, mask=output_mask,
                                    reduction='token_mean')
            return pred_tags,loss
        elif mode == 'test':
            pred_tags = self.crf.decode(emissions, mask=output_mask)
            return pred_tags


# 关于对齐模块的消融实验2： 将对齐模块中文本序列部分直接采用文字输出的向量
#       这里门控依旧由CrossAtt的全局进行计算，同时也保留了CrossAtt的计算权重
class MTCCMBertForMMTokenClassificationCRF_woPart2_2(torch.nn.Module):
    def __init__(self, config, embedding, last_encoder, layer_num1=1, layer_num2=1, layer_num3=1,  num_labels=2):
        super().__init__()
        self.num_labels = num_labels
        self.last_encoder = last_encoder
        self.bert = embedding
        self.hidden_size = config.hidden_size
        self.self_attention = BertSelfEncoder(config)
        self.self_attention_v2 = BertSelfEncoder(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.vismap2text = nn.Linear(2048, config.hidden_size)
        # clip输出的维度可能是512，需要转换成hidden_size
        self.vismapping = nn.Linear(512, config.hidden_size)
        self.txt2img_attention = BertCrossEncoder(config, layer_num1)
        self.cls_layer_Y = nn.ModuleList([BertCrossEncoder(config, layer_num1) for _ in range(2)])
        # self.text_coattention = nn.ModuleList([BertCrossEncoder(config, layer_num1) for _ in range(2)])
        self.embedding_layer = nn.Embedding(config.vocab_size, config.hidden_size)
        
        self.lstm = nn.LSTM(input_size=config.hidden_size,
                            hidden_size=config.hidden_size,
                            batch_first=True,
                            bidirectional=True)
        # self.dropout_2 = nn.Dropout(p=0.3)
        self.classifier = torch.nn.Linear(config.hidden_size * 2, num_labels)
        self.crf = CRF(num_tags=num_labels,
                        batch_first=True)
        self.prompt_len = 5
        self.mapping_network_alignment = nn.Sequential(
            nn.Dropout(p=0.3),
            torch.nn.Linear(config.hidden_size, 756 * self.prompt_len, bias=True),
            torch.nn.Tanh(),
            nn.Dropout(p=0.3),
            torch.nn.Linear(756 * self.prompt_len, config.hidden_size * self.prompt_len, bias=True)
        )

        self.mapping_network_vision = nn.Sequential(
            nn.Dropout(p=0.3),
            torch.nn.Linear(2048, 756 * self.prompt_len, bias=True),
            torch.nn.Tanh(),
            nn.Dropout(p=0.3),
            torch.nn.Linear(756 * self.prompt_len, config.hidden_size * self.prompt_len, bias=True)
        )

        self.lastproj = torch.nn.Linear(config.hidden_size, 1024)


        self.cls_layer = cls_layer_both(config.hidden_size,config.hidden_size)
        self.aux_head = nn.Linear(config.hidden_size, 1)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        # for name, parameter in self.bert.named_parameters():
        #     parameter.requires_grad = False


    def forward(self, input_ids, segment_ids, input_mask, ori_input_ids, ori_input_mask, ori_segment_ids, added_attention_mask, 
                clip_features, visual_embeds_mean, visual_embeds_att, offsets, output_mask,rela_score,
                temp=None, temp_lamb=None,lamb=None, labels=None, negative_rate=None, mode = None):
        
        # sequence_output, sequence_output_pooler = self.bert(ori_input_ids, token_type_ids=ori_segment_ids, attention_mask=ori_input_mask,
        #                                output_all_encoded_layers=False)  # batch_size * seq_len * hidden_size
        
        offset = offsets.tolist()[0]
        sequence_output= self.bert(ori_input_ids, token_type_ids=ori_segment_ids, 
                                        attention_mask=ori_input_mask)[0].float()
        # sequence_output = self.embedding_layer(ori_input_ids)

        sequence_output = self.dropout(sequence_output)
        clip_features = self.vismapping(clip_features.to(dtype=next(self.parameters()).dtype).squeeze(1))

        vis_embed_map = visual_embeds_att.view(-1, 2048, 49).permute(0, 2, 1)  # self.batch_size, 49, 2048

        converted_vis_embed_map = self.vismap2text(vis_embed_map)  # self.batch_size, 49, hidden_dim

        # '''
        # apply txt2img attention mechanism to obtain image-based text representations
        img_mask = added_attention_mask[:, :49]  # batch_size * 49
        extended_img_mask = img_mask.unsqueeze(1).unsqueeze(2)  # batch_size * 1 * 1 * 49
        extended_img_mask = extended_img_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_img_mask = (1.0 - extended_img_mask) * -10000.0


        cross_encoder = self.txt2img_attention(sequence_output, converted_vis_embed_map, extended_img_mask)
        cross_output_layer = cross_encoder[-1]  #  batch_size * text_len * hidden_dim

        extended_text_mask = ori_input_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_text_mask = (1.0 - extended_text_mask) * -10000.0
        
        
        # ==using BertCrossEncoder
        clip_features = clip_features.unsqueeze(1)
        extended_text_mask = extended_text_mask.unsqueeze(1).unsqueeze(2)
        # ===using BertCrossEncoder
        for i, layer_module in enumerate(self.cls_layer_Y):

            # clip_features = layer_module(cross_output_layer, clip_features, extended_text_mask, None, None)

            # using BertCrossEncoder
            clip_features = layer_module(clip_features, sequence_output, extended_text_mask)[-1]
            # using BertCrossEncoder


        Alignment_prompt = self.mapping_network_alignment(clip_features).unsqueeze(1).view(ori_input_ids.size(0), self.prompt_len, -1)
        
        prefix_vision = self.mapping_network_vision(visual_embeds_mean)
        prefix_vision = prefix_vision.reshape(input_ids.size(0), self.prompt_len, -1)


        prefix_emb = torch.cat([prefix_vision, Alignment_prompt], dim=1)
        if prefix_emb.size(2) != 1024:
            prefix_emb = self.lastproj(prefix_emb)


        align_mask = input_mask[:, :1].repeat(1, self.prompt_len)
        vision_mask = input_mask[:, :1].repeat(1, self.prompt_len)
        prompt_mask = torch.cat([vision_mask, align_mask], dim=1)
        roberta_encoder_outputs = self.last_encoder(input_ids=input_ids,token_type_ids=segment_ids,
                                              attention_mask=input_mask, prompt_embeddings=prefix_emb, 
                                              input_mask=prompt_mask, offset=offset)
        roberta_encoder_output = roberta_encoder_outputs[0] 
        # roberta_encoder_output = [batch_size, input_ids.size(1) - 2 + prefix_emb.size(1), 1024 ]
        # last_encoder是roberta_large,它的输出总是1024
        



        # offset: 总输入中，代表ori_input的最开头token的相应下标，- 2 + prefix_emb.size(1)将Mask换成prefix_embedding的长度
        # 也是要根据prompt_text被分词后的结果动态调整
        offset = offset - 2 + prefix_emb.size(1)
 
        token_embedding = roberta_encoder_output[:,offset: offset+128,:]
        
        
        # result = token_embedding * 0.5 + cross_output_layer * 0.5

        cross_output_layer_global = cross_output_layer[:,0,:]
        token_embedding_global = token_embedding[:,0,:]
        

        related_feat = self.cls_layer(cross_output_layer_global, token_embedding_global)
        logits = self.aux_head(related_feat)
        gate_signal = torch.sigmoid(logits).view(token_embedding.size(0), 1, 1)
        result = gate_signal*token_embedding + (1 - gate_signal)* cross_output_layer

        # fixed_relation_score 
        # rela_score = rela_score.view(token_embedding.size(0), 1, 1)
        # result = rela_score*token_embedding + (1 - rela_score)* cross_output_layer

        x, _ = self.lstm(result)
        emissions = self.classifier(x)

        output_mask = (output_mask != 0)
        if mode == 'train':
            loss = - self.crf(emissions, tags=labels, mask=output_mask,
                          reduction='token_mean')
            return loss
        elif mode == 'dev':
            pred_tags = self.crf.decode(emissions, mask=output_mask)
            loss = - self.crf(emissions, tags=labels, mask=output_mask,
                                    reduction='token_mean')
            return pred_tags,loss
        elif mode == 'test':
            pred_tags = self.crf.decode(emissions, mask=output_mask)
            return pred_tags
        


# 关于对齐模块的消融实验3： 直接把 Clip token作为 image的桥梁。证明了对齐模块对整体效果提升的必要性
#       这里门控依旧由CrossAtt的全局进行计算，同时也保留了CrossAtt的计算权重
class MTCCMBertForMMTokenClassificationCRF_woPart2_3(torch.nn.Module):
    def __init__(self, config, embedding, last_encoder, layer_num1=1, layer_num2=1, layer_num3=1,  num_labels=2):
        super().__init__()
        self.num_labels = num_labels
        self.last_encoder = last_encoder
        self.bert = embedding
        self.hidden_size = config.hidden_size
        self.self_attention = BertSelfEncoder(config)
        self.self_attention_v2 = BertSelfEncoder(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.vismap2text = nn.Linear(2048, config.hidden_size)
        # clip输出的维度可能是512，需要转换成hidden_size
        self.vismapping = nn.Linear(512, config.hidden_size)
        self.txt2img_attention = BertCrossEncoder(config, layer_num1)
        self.cls_layer_Y = nn.ModuleList([BertCrossEncoder(config, layer_num1) for _ in range(2)])
        # self.text_coattention = nn.ModuleList([BertCrossEncoder(config, layer_num1) for _ in range(2)])
        self.embedding_layer = nn.Embedding(config.vocab_size, config.hidden_size)
        
        self.lstm = nn.LSTM(input_size=config.hidden_size,
                            hidden_size=config.hidden_size,
                            batch_first=True,
                            bidirectional=True)
        # self.dropout_2 = nn.Dropout(p=0.3)
        self.classifier = torch.nn.Linear(config.hidden_size * 2, num_labels)
        self.crf = CRF(num_tags=num_labels,
                        batch_first=True)
        self.prompt_len = 5
        self.mapping_network_alignment = nn.Sequential(
            nn.Dropout(p=0.3),
            torch.nn.Linear(config.hidden_size, 756 * self.prompt_len, bias=True),
            torch.nn.Tanh(),
            nn.Dropout(p=0.3),
            torch.nn.Linear(756 * self.prompt_len, config.hidden_size * self.prompt_len, bias=True)
        )

        self.mapping_network_vision = nn.Sequential(
            nn.Dropout(p=0.3),
            torch.nn.Linear(2048, 756 * self.prompt_len, bias=True),
            torch.nn.Tanh(),
            nn.Dropout(p=0.3),
            torch.nn.Linear(756 * self.prompt_len, config.hidden_size * self.prompt_len, bias=True)
        )

        self.lastproj = torch.nn.Linear(config.hidden_size, 1024)


        self.cls_layer = cls_layer_both(config.hidden_size,config.hidden_size)
        self.aux_head = nn.Linear(config.hidden_size, 1)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        # for name, parameter in self.bert.named_parameters():
        #     parameter.requires_grad = False


    def forward(self, input_ids, segment_ids, input_mask, ori_input_ids, ori_input_mask, ori_segment_ids, added_attention_mask, 
                clip_features, visual_embeds_mean, visual_embeds_att, offsets, output_mask,rela_score,
                temp=None, temp_lamb=None,lamb=None, labels=None, negative_rate=None, mode = None):
        
        # sequence_output, sequence_output_pooler = self.bert(ori_input_ids, token_type_ids=ori_segment_ids, attention_mask=ori_input_mask,
        #                                output_all_encoded_layers=False)  # batch_size * seq_len * hidden_size
        
        offset = offsets.tolist()[0]
        sequence_output= self.bert(ori_input_ids, token_type_ids=ori_segment_ids, 
                                        attention_mask=ori_input_mask)[0].float()
        # sequence_output = self.embedding_layer(ori_input_ids)

        sequence_output = self.dropout(sequence_output)
        clip_features = self.vismapping(clip_features.to(dtype=next(self.parameters()).dtype).squeeze(1))

        vis_embed_map = visual_embeds_att.view(-1, 2048, 49).permute(0, 2, 1)  # self.batch_size, 49, 2048

        converted_vis_embed_map = self.vismap2text(vis_embed_map)  # self.batch_size, 49, hidden_dim

        # '''
        # apply txt2img attention mechanism to obtain image-based text representations
        img_mask = added_attention_mask[:, :49]  # batch_size * 49
        extended_img_mask = img_mask.unsqueeze(1).unsqueeze(2)  # batch_size * 1 * 1 * 49
        extended_img_mask = extended_img_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_img_mask = (1.0 - extended_img_mask) * -10000.0


        cross_encoder = self.txt2img_attention(sequence_output, converted_vis_embed_map, extended_img_mask)
        cross_output_layer = cross_encoder[-1]  #  batch_size * text_len * hidden_dim

        extended_text_mask = ori_input_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_text_mask = (1.0 - extended_text_mask) * -10000.0
        
        
        # ==using BertCrossEncoder
        clip_features = clip_features.unsqueeze(1)
        extended_text_mask = extended_text_mask.unsqueeze(1).unsqueeze(2)
        # ===using BertCrossEncoder
        # for i, layer_module in enumerate(self.cls_layer_Y):

        #     # clip_features = layer_module(cross_output_layer, clip_features, extended_text_mask, None, None)

        #     # using BertCrossEncoder
        #     clip_features = layer_module(clip_features, sequence_output, extended_text_mask)[-1]
        #     # using BertCrossEncoder


        Alignment_prompt = self.mapping_network_alignment(clip_features).unsqueeze(1).view(ori_input_ids.size(0), self.prompt_len, -1)
        
        prefix_vision = self.mapping_network_vision(visual_embeds_mean)
        prefix_vision = prefix_vision.reshape(input_ids.size(0), self.prompt_len, -1)


        prefix_emb = torch.cat([prefix_vision, Alignment_prompt], dim=1)
        if prefix_emb.size(2) != 1024:
            prefix_emb = self.lastproj(prefix_emb)


        align_mask = input_mask[:, :1].repeat(1, self.prompt_len)
        vision_mask = input_mask[:, :1].repeat(1, self.prompt_len)
        prompt_mask = torch.cat([vision_mask, align_mask], dim=1)
        roberta_encoder_outputs = self.last_encoder(input_ids=input_ids,token_type_ids=segment_ids,
                                              attention_mask=input_mask, prompt_embeddings=prefix_emb, 
                                              input_mask=prompt_mask, offset=offset)
        roberta_encoder_output = roberta_encoder_outputs[0] 
        # roberta_encoder_output = [batch_size, input_ids.size(1) - 2 + prefix_emb.size(1), 1024 ]
        # last_encoder是roberta_large,它的输出总是1024
        



        # offset: 总输入中，代表ori_input的最开头token的相应下标，- 2 + prefix_emb.size(1)将Mask换成prefix_embedding的长度
        # 也是要根据prompt_text被分词后的结果动态调整
        offset = offset - 2 + prefix_emb.size(1)
 
        token_embedding = roberta_encoder_output[:,offset: offset+128,:]
        
        
        # result = token_embedding * 0.5 + cross_output_layer * 0.5

        cross_output_layer_global = cross_output_layer[:,0,:]
        token_embedding_global = token_embedding[:,0,:]
        

        related_feat = self.cls_layer(cross_output_layer_global, token_embedding_global)
        logits = self.aux_head(related_feat)
        gate_signal = torch.sigmoid(logits).view(token_embedding.size(0), 1, 1)
        result = gate_signal*token_embedding + (1 - gate_signal)* cross_output_layer

        # fixed_relation_score 
        # rela_score = rela_score.view(token_embedding.size(0), 1, 1)
        # result = rela_score*token_embedding + (1 - rela_score)* cross_output_layer

        x, _ = self.lstm(result)
        emissions = self.classifier(x)

        output_mask = (output_mask != 0)
        if mode == 'train':
            loss = - self.crf(emissions, tags=labels, mask=output_mask,
                          reduction='token_mean')
            return loss
        elif mode == 'dev':
            pred_tags = self.crf.decode(emissions, mask=output_mask)
            loss = - self.crf(emissions, tags=labels, mask=output_mask,
                                    reduction='token_mean')
            return pred_tags,loss
        elif mode == 'test':
            pred_tags = self.crf.decode(emissions, mask=output_mask)
            return pred_tags


# 关于prompt的修改：bos + <mask> <mask> eos + text is + .....
class MTCCMBertForMMTokenClassificationCRF_prompt_1(torch.nn.Module):
    def __init__(self, config, embedding, last_encoder, layer_num1=1, layer_num2=1, layer_num3=1,  num_labels=2):
        super().__init__()
        self.num_labels = num_labels
        self.last_encoder = last_encoder
        self.bert = embedding
        self.hidden_size = config.hidden_size
        self.self_attention = BertSelfEncoder(config)
        self.self_attention_v2 = BertSelfEncoder(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.vismap2text = nn.Linear(2048, config.hidden_size)
        # clip输出的维度可能是512，需要转换成hidden_size
        self.vismapping = nn.Linear(512, config.hidden_size)
        self.txt2img_attention = BertCrossEncoder(config, layer_num1)
        self.cls_layer_Y = nn.ModuleList([BertCrossEncoder(config, layer_num1) for _ in range(2)])
        # self.text_coattention = nn.ModuleList([BertCrossEncoder(config, layer_num1) for _ in range(2)])
        self.embedding_layer = nn.Embedding(config.vocab_size, config.hidden_size)
        
        self.lstm = nn.LSTM(input_size=config.hidden_size,
                            hidden_size=config.hidden_size,
                            batch_first=True,
                            bidirectional=True)
        # self.dropout_2 = nn.Dropout(p=0.3)
        self.classifier = torch.nn.Linear(config.hidden_size * 2, num_labels)
        self.crf = CRF(num_tags=num_labels,
                        batch_first=True)
        self.prompt_len = 5
        self.mapping_network_alignment = nn.Sequential(
            nn.Dropout(p=0.3),
            torch.nn.Linear(config.hidden_size, 756 * self.prompt_len, bias=True),
            torch.nn.Tanh(),
            nn.Dropout(p=0.3),
            torch.nn.Linear(756 * self.prompt_len, config.hidden_size * self.prompt_len, bias=True)
        )

        self.mapping_network_vision = nn.Sequential(
            nn.Dropout(p=0.3),
            torch.nn.Linear(2048, 756 * self.prompt_len, bias=True),
            torch.nn.Tanh(),
            nn.Dropout(p=0.3),
            torch.nn.Linear(756 * self.prompt_len, config.hidden_size * self.prompt_len, bias=True)
        )

        self.lastproj = torch.nn.Linear(config.hidden_size, 1024)


        self.cls_layer = cls_layer_both(config.hidden_size,config.hidden_size)
        self.aux_head = nn.Linear(config.hidden_size, 1)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        # for name, parameter in self.bert.named_parameters():
        #     parameter.requires_grad = False


    def forward(self, input_ids, segment_ids, input_mask, ori_input_ids, ori_input_mask, ori_segment_ids, added_attention_mask, 
                clip_features, visual_embeds_mean, visual_embeds_att, offsets, output_mask,rela_score,
                temp=None, temp_lamb=None,lamb=None, labels=None, negative_rate=None, mode = None):
        
        # sequence_output, sequence_output_pooler = self.bert(ori_input_ids, token_type_ids=ori_segment_ids, attention_mask=ori_input_mask,
        #                                output_all_encoded_layers=False)  # batch_size * seq_len * hidden_size
        
        offset = offsets.tolist()[0]
        sequence_output= self.bert(ori_input_ids, token_type_ids=ori_segment_ids, 
                                        attention_mask=ori_input_mask)[0].float()
        # sequence_output = self.embedding_layer(ori_input_ids)

        sequence_output = self.dropout(sequence_output)
        clip_features = self.vismapping(clip_features.to(dtype=next(self.parameters()).dtype).squeeze(1))

        vis_embed_map = visual_embeds_att.view(-1, 2048, 49).permute(0, 2, 1)  # self.batch_size, 49, 2048

        converted_vis_embed_map = self.vismap2text(vis_embed_map)  # self.batch_size, 49, hidden_dim

        # '''
        # apply txt2img attention mechanism to obtain image-based text representations
        img_mask = added_attention_mask[:, :49]  # batch_size * 49
        extended_img_mask = img_mask.unsqueeze(1).unsqueeze(2)  # batch_size * 1 * 1 * 49
        extended_img_mask = extended_img_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_img_mask = (1.0 - extended_img_mask) * -10000.0


        cross_encoder = self.txt2img_attention(sequence_output, converted_vis_embed_map, extended_img_mask)
        cross_output_layer = cross_encoder[-1]  #  batch_size * text_len * hidden_dim
        # kd_loss = 0.0
        # for student_rep, teacher_rep in zip(cross_output_layer[:,0,:], clip_features):
        #     tmp_loss = F.mse_loss(F.normalize(student_rep, p=2, dim=-1), F.normalize(teacher_rep, p=2, dim=-1))
        #     kd_loss += tmp_loss

        # extended_text_mask = ori_input_mask[:, :1].unsqueeze(1).unsqueeze(2)  # batch_size * 1 * 1 * 1
        extended_text_mask = ori_input_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_text_mask = (1.0 - extended_text_mask) * -10000.0
        
        
        # ==using BertCrossEncoder
        clip_features = clip_features.unsqueeze(1)
        extended_text_mask = extended_text_mask.unsqueeze(1).unsqueeze(2)
        # ===using BertCrossEncoder
        for i, layer_module in enumerate(self.cls_layer_Y):

            # clip_features = layer_module(cross_output_layer, clip_features, extended_text_mask, None, None)

            # using BertCrossEncoder
            clip_features = layer_module(clip_features, cross_output_layer, extended_text_mask)[-1]
            # using BertCrossEncoder

        
        # 去掉第三个
        # clip_features = cross_output_layer[:,0,:].unsqueeze(1)
        Alignment_prompt = self.mapping_network_alignment(clip_features).unsqueeze(1).view(ori_input_ids.size(0), self.prompt_len, -1)
        # 去掉第三个
        
        prefix_vision = self.mapping_network_vision(visual_embeds_mean)
        prefix_vision = prefix_vision.reshape(input_ids.size(0), self.prompt_len, -1)


        prefix_emb = torch.cat([prefix_vision, Alignment_prompt], dim=1)
        if prefix_emb.size(2) != 1024:
            prefix_emb = self.lastproj(prefix_emb)


        align_mask = input_mask[:, :1].repeat(1, self.prompt_len)
        vision_mask = input_mask[:, :1].repeat(1, self.prompt_len)
        prompt_mask = torch.cat([vision_mask, align_mask], dim=1)
        roberta_encoder_outputs = self.last_encoder(input_ids=input_ids,token_type_ids=segment_ids,
                                              attention_mask=input_mask, prompt_embeddings=prefix_emb, 
                                              input_mask=prompt_mask, offset=offset)
        roberta_encoder_output = roberta_encoder_outputs[0] 
        # roberta_encoder_output = [batch_size, input_ids.size(1) - 2 + prefix_emb.size(1), 1024 ]
        # last_encoder是roberta_large,它的输出总是1024
        



        # offset: 总输入中，代表ori_input的最开头token的相应下标，- 2 + prefix_emb.size(1)将Mask换成prefix_embedding的长度
        # 也是要根据prompt_text被分词后的结果动态调整
        offset = offset - 2 + prefix_emb.size(1)
 
        token_embedding = roberta_encoder_output[:,offset: offset+128,:]
        
        
        # result = token_embedding * 0.5 + cross_output_layer * 0.5

        cross_output_layer_global = cross_output_layer[:,0,:]
        token_embedding_global = token_embedding[:,0,:]
        

        related_feat = self.cls_layer(cross_output_layer_global, token_embedding_global)
        logits = self.aux_head(related_feat)
        gate_signal = torch.sigmoid(logits).view(token_embedding.size(0), 1, 1)
        result = gate_signal*token_embedding + (1 - gate_signal)* cross_output_layer

        # fixed_relation_score 
        # rela_score = rela_score.view(token_embedding.size(0), 1, 1)
        # result = rela_score*token_embedding + (1 - rela_score)* cross_output_layer

        x, _ = self.lstm(result)
        emissions = self.classifier(x)

        output_mask = (output_mask != 0)
        if mode == 'train':
            loss = - self.crf(emissions, tags=labels, mask=output_mask,
                          reduction='token_mean')
            return loss
        elif mode == 'dev':
            pred_tags = self.crf.decode(emissions, mask=output_mask)
            loss = - self.crf(emissions, tags=labels, mask=output_mask,
                                    reduction='token_mean')
            return pred_tags,loss
        elif mode == 'test':
            pred_tags = self.crf.decode(emissions, mask=output_mask)
            return pred_tags
        


# 关于prompt的修改：bos + Image is <mask> eos + text is + .....
class MTCCMBertForMMTokenClassificationCRF_prompt_2(torch.nn.Module):
    def __init__(self, config, embedding, last_encoder, layer_num1=1, layer_num2=1, layer_num3=1,  num_labels=2):
        super().__init__()
        self.num_labels = num_labels
        self.last_encoder = last_encoder
        self.bert = embedding
        self.hidden_size = config.hidden_size
        self.self_attention = BertSelfEncoder(config)
        self.self_attention_v2 = BertSelfEncoder(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.vismap2text = nn.Linear(2048, config.hidden_size)
        # clip输出的维度可能是512，需要转换成hidden_size
        self.vismapping = nn.Linear(512, config.hidden_size)
        self.txt2img_attention = BertCrossEncoder(config, layer_num1)
        self.cls_layer_Y = nn.ModuleList([BertCrossEncoder(config, layer_num1) for _ in range(2)])
        # self.text_coattention = nn.ModuleList([BertCrossEncoder(config, layer_num1) for _ in range(2)])
        self.embedding_layer = nn.Embedding(config.vocab_size, config.hidden_size)
        
        self.lstm = nn.LSTM(input_size=config.hidden_size,
                            hidden_size=config.hidden_size,
                            batch_first=True,
                            bidirectional=True)
        # self.dropout_2 = nn.Dropout(p=0.3)
        self.classifier = torch.nn.Linear(config.hidden_size * 2, num_labels)
        self.crf = CRF(num_tags=num_labels,
                        batch_first=True)
        self.prompt_len = 5
        self.mapping_network_alignment = nn.Sequential(
            nn.Dropout(p=0.3),
            torch.nn.Linear(config.hidden_size, 756 * self.prompt_len, bias=True),
            torch.nn.Tanh(),
            nn.Dropout(p=0.3),
            torch.nn.Linear(756 * self.prompt_len, config.hidden_size * self.prompt_len, bias=True)
        )

        self.mapping_network_vision = nn.Sequential(
            nn.Dropout(p=0.3),
            torch.nn.Linear(2048, 756 * self.prompt_len, bias=True),
            torch.nn.Tanh(),
            nn.Dropout(p=0.3),
            torch.nn.Linear(756 * self.prompt_len, config.hidden_size * self.prompt_len, bias=True)
        )

        self.lastproj = torch.nn.Linear(config.hidden_size, 1024)


        self.cls_layer = cls_layer_both(config.hidden_size,config.hidden_size)
        self.aux_head = nn.Linear(config.hidden_size, 1)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        # for name, parameter in self.bert.named_parameters():
        #     parameter.requires_grad = False


    def forward(self, input_ids, segment_ids, input_mask, ori_input_ids, ori_input_mask, ori_segment_ids, added_attention_mask, 
                clip_features, visual_embeds_mean, visual_embeds_att, offsets, output_mask,rela_score,
                temp=None, temp_lamb=None,lamb=None, labels=None, negative_rate=None, mode = None):
        
        # sequence_output, sequence_output_pooler = self.bert(ori_input_ids, token_type_ids=ori_segment_ids, attention_mask=ori_input_mask,
        #                                output_all_encoded_layers=False)  # batch_size * seq_len * hidden_size
        
        offset = offsets.tolist()[0]
        sequence_output= self.bert(ori_input_ids, token_type_ids=ori_segment_ids, 
                                        attention_mask=ori_input_mask)[0].float()
        # sequence_output = self.embedding_layer(ori_input_ids)

        sequence_output = self.dropout(sequence_output)
        
        prefix_vision = self.mapping_network_vision(visual_embeds_mean)
        prefix_vision = prefix_vision.reshape(input_ids.size(0), self.prompt_len, -1)


        prefix_emb = prefix_vision
        if prefix_emb.size(2) != 1024:
            prefix_emb = self.lastproj(prefix_emb)


        vision_mask = input_mask[:, :1].repeat(1, self.prompt_len)
        prompt_mask = vision_mask
        roberta_encoder_outputs = self.last_encoder(input_ids=input_ids,token_type_ids=segment_ids,
                                              attention_mask=input_mask, prompt_embeddings=prefix_emb, 
                                              input_mask=prompt_mask, offset=offset)
        roberta_encoder_output = roberta_encoder_outputs[0] 
        # roberta_encoder_output = [batch_size, input_ids.size(1) - 2 + prefix_emb.size(1), 1024 ]
        # last_encoder是roberta_large,它的输出总是1024
        



        # offset: 总输入中，代表ori_input的最开头token的相应下标，- 2 + prefix_emb.size(1)将Mask换成prefix_embedding的长度
        # 也是要根据prompt_text被分词后的结果动态调整
        offset = offset - 2 + prefix_emb.size(1)
 
        token_embedding = roberta_encoder_output[:,offset: offset+128,:]
        
        
        # result = token_embedding * 0.5 + cross_output_layer * 0.5

        cross_output_layer_global = sequence_output[:,0,:]
        token_embedding_global = token_embedding[:,0,:]
        

        related_feat = self.cls_layer(cross_output_layer_global, token_embedding_global)
        logits = self.aux_head(related_feat)
        gate_signal = torch.sigmoid(logits).view(token_embedding.size(0), 1, 1)
        result = gate_signal*token_embedding + (1 - gate_signal)* sequence_output

        # fixed_relation_score 
        # rela_score = rela_score.view(token_embedding.size(0), 1, 1)
        # result = rela_score*token_embedding + (1 - rela_score)* cross_output_layer

        x, _ = self.lstm(result)
        emissions = self.classifier(x)

        output_mask = (output_mask != 0)
        if mode == 'train':
            loss = - self.crf(emissions, tags=labels, mask=output_mask,
                          reduction='token_mean')
            return loss
        elif mode == 'dev':
            pred_tags = self.crf.decode(emissions, mask=output_mask)
            loss = - self.crf(emissions, tags=labels, mask=output_mask,
                                    reduction='token_mean')
            return pred_tags,loss
        elif mode == 'test':
            pred_tags = self.crf.decode(emissions, mask=output_mask)
            return pred_tags
        

# 关于门控的消融实验1： 直接把 跨模态注意力作为标准
class MTCCMBertForMMTokenClassificationCRF_gate_1(torch.nn.Module):
    
    def __init__(self, config, embedding, last_encoder, layer_num1=1, layer_num2=1, layer_num3=1,  num_labels=2):
        super().__init__()
        self.num_labels = num_labels
        self.last_encoder = last_encoder
        self.bert = embedding
        self.hidden_size = config.hidden_size
        self.self_attention = BertSelfEncoder(config)
        self.self_attention_v2 = BertSelfEncoder(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.vismap2text = nn.Linear(2048, config.hidden_size)
        # clip输出的维度可能是512，需要转换成hidden_size
        self.vismapping = nn.Linear(512, config.hidden_size)
        self.txt2img_attention = BertCrossEncoder(config, layer_num1)
        self.cls_layer_Y = nn.ModuleList([BertCrossEncoder(config, layer_num1) for _ in range(2)])
        # self.text_coattention = nn.ModuleList([BertCrossEncoder(config, layer_num1) for _ in range(2)])
        self.embedding_layer = nn.Embedding(config.vocab_size, config.hidden_size)
        
        self.lstm = nn.LSTM(input_size=config.hidden_size,
                            hidden_size=config.hidden_size,
                            batch_first=True,
                            bidirectional=True)
        # self.dropout_2 = nn.Dropout(p=0.3)
        self.classifier = torch.nn.Linear(config.hidden_size * 2, num_labels)
        self.crf = CRF(num_tags=num_labels,
                        batch_first=True)
        self.prompt_len = 5
        self.mapping_network_alignment = nn.Sequential(
            nn.Dropout(p=0.3),
            torch.nn.Linear(config.hidden_size, 756 * self.prompt_len, bias=True),
            torch.nn.Tanh(),
            nn.Dropout(p=0.3),
            torch.nn.Linear(756 * self.prompt_len, config.hidden_size * self.prompt_len, bias=True)
        )

        self.mapping_network_vision = nn.Sequential(
            nn.Dropout(p=0.3),
            torch.nn.Linear(2048, 756 * self.prompt_len, bias=True),
            torch.nn.Tanh(),
            nn.Dropout(p=0.3),
            torch.nn.Linear(756 * self.prompt_len, config.hidden_size * self.prompt_len, bias=True)
        )

        self.lastproj = torch.nn.Linear(config.hidden_size, 1024)


        self.cls_layer = cls_layer_both(config.hidden_size,config.hidden_size)
        self.aux_head = nn.Linear(config.hidden_size, 1)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        # for name, parameter in self.bert.named_parameters():
        #     parameter.requires_grad = False


    def forward(self, input_ids, segment_ids, input_mask, ori_input_ids, ori_input_mask, ori_segment_ids, added_attention_mask, 
                clip_features, visual_embeds_mean, visual_embeds_att, offsets, output_mask,rela_score,
                temp=None, temp_lamb=None,lamb=None, labels=None, negative_rate=None, mode = None):
        
        # sequence_output, sequence_output_pooler = self.bert(ori_input_ids, token_type_ids=ori_segment_ids, attention_mask=ori_input_mask,
        #                                output_all_encoded_layers=False)  # batch_size * seq_len * hidden_size
        
        offset = offsets.tolist()[0]
        sequence_output= self.bert(ori_input_ids, token_type_ids=ori_segment_ids, 
                                        attention_mask=ori_input_mask)[0].float()
        # sequence_output = self.embedding_layer(ori_input_ids)

        sequence_output = self.dropout(sequence_output)
        clip_features = self.vismapping(clip_features.to(dtype=next(self.parameters()).dtype).squeeze(1))

        vis_embed_map = visual_embeds_att.view(-1, 2048, 49).permute(0, 2, 1)  # self.batch_size, 49, 2048

        converted_vis_embed_map = self.vismap2text(vis_embed_map)  # self.batch_size, 49, hidden_dim

        # '''
        # apply txt2img attention mechanism to obtain image-based text representations
        img_mask = added_attention_mask[:, :49]  # batch_size * 49
        extended_img_mask = img_mask.unsqueeze(1).unsqueeze(2)  # batch_size * 1 * 1 * 49
        extended_img_mask = extended_img_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_img_mask = (1.0 - extended_img_mask) * -10000.0


        cross_encoder = self.txt2img_attention(sequence_output, converted_vis_embed_map, extended_img_mask)
        cross_output_layer = cross_encoder[-1]  #  batch_size * text_len * hidden_dim

        x, _ = self.lstm(cross_output_layer)
        emissions = self.classifier(x)

        output_mask = (output_mask != 0)
        if mode == 'train':
            loss = - self.crf(emissions, tags=labels, mask=output_mask,
                          reduction='token_mean')
            return loss
        elif mode == 'dev':
            pred_tags = self.crf.decode(emissions, mask=output_mask)
            loss = - self.crf(emissions, tags=labels, mask=output_mask,
                                    reduction='token_mean')
            return pred_tags,loss
        elif mode == 'test':
            pred_tags = self.crf.decode(emissions, mask=output_mask)
            return pred_tags
        
        