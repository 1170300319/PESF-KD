import transformers
from transformers.models.bert.modeling_bert import BertPreTrainedModel, BertEmbeddings, BertLayer, BertPooler, BertModel, \
    BertEncoder
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss, MSELoss
from utils_glue import Adapter
from transformers.configuration_utils import PretrainedConfig


class BertForSequenceClassification(BertPreTrainedModel):
    r"""
        **labels**: (`optional`) ``torch.LongTensor`` of shape ``(batch_size,)``:
            Labels for computing the sequence classification/regression loss.
            Indices should be in ``[0, ..., config.num_labels - 1]``.
            If ``config.num_labels == 1`` a regression loss is computed (Mean-Square loss),
            If ``config.num_labels > 1`` a classification loss is computed (Cross-Entropy).

    Outputs: `Tuple` comprising various elements depending on the configuration (config) and inputs:
        **loss**: (`optional`, returned when ``labels`` is provided) ``torch.FloatTensor`` of shape ``(1,)``:
            Classification (or regression if config.num_labels==1) loss.
        **logits**: ``torch.FloatTensor`` of shape ``(batch_size, config.num_labels)``
            Classification (or regression if config.num_labels==1) scores (before SoftMax).
        **hidden_states**: (`optional`, returned when ``config.output_hidden_states=True``)
            list of ``torch.FloatTensor`` (one for the output of each layer + the output of the embeddings)
            of shape ``(batch_size, sequence_length, hidden_size)``:
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        **attentions**: (`optional`, returned when ``config.output_attentions=True``)
            list of ``torch.FloatTensor`` (one for each layer) of shape ``(batch_size, num_heads, sequence_length, sequence_length)``:
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.

    Examples::

        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertForSequenceClassification.from_pretrained('bert-base-uncased')
        input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute")).unsqueeze(0)  # Batch size 1
        labels = torch.tensor([1]).unsqueeze(0)  # Batch size 1
        outputs = model(input_ids, labels=labels)
        loss, logits = outputs[:2]

    """

    def __init__(self, config, use_adapter=False, adapter_size=64):
        super(BertForSequenceClassification, self).__init__(config)

        
        self.num_labels = config.num_labels

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, self.config.num_labels)
        #self.classifier2 = nn.Linear(config.hidden_size, self.config.num_labels)
        if use_adapter:
            self.adapter = Adapter(768, adapter_size, config.num_labels)
        else:
            self.adapter = None


        self.init_weights()

    def forward(self, input_ids, attention_mask=None, token_type_ids=None,
                position_ids=None, head_mask=None, labels=None):

        outputs = self.bert(input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            position_ids=position_ids,
                            head_mask=head_mask)

        pooled_output = outputs[1]
        # print(pooled_output.shape)

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        if self.adapter is not None:
            logits += self.adapter(pooled_output)

        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here

        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions)


def get_prune_paramerts(model):
    prune_paramerts = {}
    for name, param in model.named_parameters():
        if 'embeddings' in name:
            prune_paramerts[name] = param
        elif name.startswith('encoder.layer.0.'):
            prune_paramerts[name] = param
        elif name.startswith('encoder.layer.2.'):
            pro_name = name.split('encoder.layer.2.')
            prune_paramerts['encoder.layer.1.' + pro_name[1]] = param
        elif name.startswith('encoder.layer.4.'):
            pro_name = name.split('encoder.layer.4.')
            prune_paramerts['encoder.layer.2.' + pro_name[1]] = param
        elif name.startswith('encoder.layer.6.'):
            pro_name = name.split('encoder.layer.6.')
            prune_paramerts['encoder.layer.3.' + pro_name[1]] = param
        elif name.startswith('encoder.layer.8.'):
            pro_name = name.split('encoder.layer.8.')
            prune_paramerts['encoder.layer.4.' + pro_name[1]] = param
        elif name.startswith('encoder.layer.10.'):
            pro_name = name.split('encoder.layer.10.')
            prune_paramerts['encoder.layer.5.' + pro_name[1]] = param
        elif 'pooler' in name:
            prune_paramerts[name] = param
    return prune_paramerts


def get_prune_config(config, n):
    prune_config = config
    prune_config['num_hidden_layers'] = n
    return prune_config


def get_prune_model(model, prune_parameters):
    prune_model = model.state_dict()
    for name in list(prune_model.keys()):
        if 'embeddings.position_ids' == name:
            continue
        if 'embeddings' in name:
            prune_model[name] = prune_parameters[name]
        elif name.startswith('encoder.layer.0.'):
            prune_model[name] = prune_parameters[name]
        elif name.startswith('encoder.layer.1.'):
            prune_model[name] = prune_parameters[name]
        elif name.startswith('encoder.layer.2.'):
            prune_model[name] = prune_parameters[name]
        elif name.startswith('encoder.layer.3.'):
            prune_model[name] = prune_parameters[name]
        elif name.startswith('encoder.layer.4.'):
            prune_model[name] = prune_parameters[name]
        elif name.startswith('encoder.layer.5.'):
            prune_model[name] = prune_parameters[name]
        elif 'pooler' in name:
            prune_model[name] = prune_parameters[name]
        else:
            del prune_model[name]
    return prune_model


def prune_main(model, config):
    prune_parameters = get_prune_paramerts(model)
    prune_config = get_prune_config(config)
    prune_model = get_prune_model(model, prune_parameters)

    return prune_model, prune_config

