# coding=utf-8
# Copyright 2021 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" PyTorch VLE model."""


from typing import Optional, Tuple, Union

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

from transformers.modeling_utils import PreTrainedModel
from transformers.utils import add_start_docstrings, add_start_docstrings_to_model_forward, logging, replace_return_docstrings, ModelOutput
from transformers.models.auto.configuration_auto import AutoConfig
from transformers.models.auto.modeling_auto import AutoModel

from transformers.models.bert.modeling_bert import BertAttention, BertIntermediate, BertOutput, apply_chunking_to_forward
from transformers.models.clip.modeling_clip import CLIPOutput, CLIPVisionConfig, CLIPVisionModel
from transformers.models.deberta_v2.modeling_deberta_v2 import DebertaV2OnlyMLMHead
from .configuration_vle import VLEConfig
from dataclasses import dataclass

logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "VLEConfig"


@dataclass
class VLEModelOutput(ModelOutput):

    pooler_output: torch.FloatTensor = None
    text_embeds: torch.FloatTensor = None
    image_embeds: torch.FloatTensor = None


@dataclass
class VLEForITMOutput(ModelOutput):

    loss: torch.FloatTensor = None
    logits: torch.FloatTensor = None

@dataclass
class VLEForPBCOutput(ModelOutput):

    loss: torch.FloatTensor = None
    logits: torch.FloatTensor = None

@dataclass
class VLEForMLMOutput(ModelOutput):

    loss: torch.FloatTensor = None
    logits: torch.FloatTensor = None

@dataclass
class VLEForVQAOutput(ModelOutput):

    loss : torch.FloatTensor = None
    logits: torch.FloatTensor = None

@dataclass
class VLEForVCRQ2AOutput(ModelOutput):

    loss : torch.FloatTensor = None
    logits: torch.FloatTensor = None

@dataclass
class VLEForVCRQA2ROutput(ModelOutput):

    loss : torch.FloatTensor = None
    logits: torch.FloatTensor = None

class ITMHead(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.fc = nn.Linear(hidden_size, 2)

    def forward(self, x):
        x = self.fc(x)
        return x


def extend_position_embedding(state_dict, patch_size, after):
    """
    modify state_dict in-place for longer position embeddings
    """
    keys = {}
    for k,v in state_dict.items():
        if k.endswith('vision_model.embeddings.position_embedding.weight'):
            assert k not in keys
            keys['pe'] = (k,v)
        if k.endswith('vision_model.embeddings.position_ids'):
            assert k not in keys
            keys['pi'] = (k,v)

    pe_weight = keys['pe'][1]
    position_length_before = pe_weight.shape[0]
    embed_dim = pe_weight.shape[1]
    grid_before = int((position_length_before - 1)**(1/2))
    position_length_after = (after // patch_size) ** 2 + 1 
    grid_after = int((position_length_after - 1)**(1/2))

    new_pe_weight = pe_weight[1:].reshape((grid_before,grid_before,-1))
    new_pe_weight =  torch.nn.functional.interpolate(
        new_pe_weight.permute(2,0,1).unsqueeze(0),
        size = (grid_after,grid_after), mode = 'bicubic')
    new_pe_weight = new_pe_weight.squeeze(0).permute(1,2,0).reshape(grid_after*grid_after, -1)
    new_pe_weight = torch.cat((pe_weight[0:1],new_pe_weight), dim=0)
    assert new_pe_weight.shape == (grid_after*grid_after + 1, embed_dim)
    
    state_dict[keys['pe'][0]] = new_pe_weight
    state_dict[keys['pi'][0]] = torch.arange(grid_after*grid_after + 1).unsqueeze(0)
    return state_dict


class Pooler(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class BertCrossLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.attention = BertAttention(config)
        self.is_decoder = config.is_decoder
        self.add_cross_attention = config.add_cross_attention
        self.crossattention = BertAttention(config)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(
        self,
        hidden_states,
        encoder_hidden_states,
        attention_mask=None,
        encoder_attention_mask=None,
        output_attentions=False,
    ):
        # decoder uni-directional self-attention cached key/values tuple is at positions 1,2
        self_attn_past_key_value = None #past_key_value[:2] if past_key_value is not None else None
        self_attention_outputs = self.attention(
            hidden_states,
            attention_mask,
            head_mask=None,
            output_attentions=output_attentions,
            past_key_value=None,
        )
        attention_output = self_attention_outputs[0]

        # if decoder, the last output is tuple of self-attn cache
        outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights

        cross_attn_present_key_value = None
        cross_attention_outputs = self.crossattention(
            attention_output,
            attention_mask,
            None,
            encoder_hidden_states,
            encoder_attention_mask,
            None,
            output_attentions,
        )
        attention_output = cross_attention_outputs[0]
        outputs = outputs + cross_attention_outputs[1:]  # add cross attentions if we output attention weights

        layer_output = apply_chunking_to_forward(
            self.feed_forward_chunk, self.chunk_size_feed_forward, self.seq_len_dim, attention_output
        )
        outputs = (layer_output,) + outputs

        return outputs

    def feed_forward_chunk(self, attention_output):
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output


class VLEPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization.
    """

    config_class = VLEConfig
    base_model_prefix = "vle"
    supports_gradient_checkpointing = False
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
    # no supported
    # def _set_gradient_checkpointing(self, module, value=False):
    #     if isinstance(module, BertEncoder):
    #         module.gradient_checkpointing = value

class VLEModel(VLEPreTrainedModel):
    def __init__(
        self,
        config: Optional[VLEConfig] = None,
        vision_model: Optional[PreTrainedModel] = None,
        text_model: Optional[PreTrainedModel] = None,
    ):

        if config is None and (vision_model is None or text_model is None):
            raise ValueError("Either a configuration or an vision and a text model has to be provided")

        if config is None:
            config = VLEConfig(text_config=text_model.config, vision_config=vision_model.config)
        else:
            if not isinstance(config, self.config_class):
                raise ValueError(f"config: {config} has to be of type {self.config_class}")

        # initialize with config
        super().__init__(config)

        if vision_model is None:
            if isinstance(config.vision_config, CLIPVisionConfig):
                vision_model = CLIPVisionModel(config.vision_config)
            else:
                vision_model = AutoModel.from_config(config.vision_config)

        if text_model is None:
            text_model = AutoModel.from_config(config.text_config)

        self.vision_model = vision_model
        self.text_model = text_model

        # make sure that the individual model's config refers to the shared config
        # so that the updates to the config will be synced
        self.vision_model.config = self.config.vision_config
        self.text_model.config = self.config.text_config

        self.vision_embed_dim = config.vision_config.hidden_size
        self.text_embed_dim = config.text_config.hidden_size
        self.coattention_dim = config.hidden_size

        # add projection layers
        self.text_projection_layer = nn.Linear(self.text_embed_dim, self.coattention_dim)
        self.image_projection_layer = nn.Linear(self.vision_embed_dim, self.coattention_dim)

        #self.logit_scale = nn.Parameter(torch.ones([]) * self.config.logit_scale_init_value)
        self.token_type_embeddings = nn.Embedding(config.num_token_types, config.hidden_size)

        self.cross_modal_image_layers = nn.ModuleList([BertCrossLayer(config) for _ in range(config.num_hidden_layers)])
        self.cross_modal_text_layers = nn.ModuleList([BertCrossLayer(config) for _ in range(config.num_hidden_layers)])
        self.cross_modal_image_pooler = Pooler(config.hidden_size)
        self.cross_modal_text_pooler = Pooler(config.hidden_size)

        # Initialize weights and apply final processing
        self.token_type_embeddings.apply(self._init_weights)
        self.cross_modal_image_layers.apply(self._init_weights)
        self.cross_modal_text_layers.apply(self._init_weights)
        self.cross_modal_image_pooler.apply(self._init_weights)
        self.cross_modal_text_pooler.apply(self._init_weights)
        if hasattr(self,"text_projection_layer"):
            self.text_projection_layer.apply(self._init_weights)
        if hasattr(self,"image_projection_layer"):
            self.image_projection_layer.apply(self._init_weights)


    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        patch_ids = None,
        extend_token_type_ids = None,
        return_loss: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], VLEModelOutput]:

        return_dict = return_dict if return_dict is not None else self.config.return_dict

        vision_outputs = self.vision_model(
            pixel_values=pixel_values,
            return_dict=return_dict,
        )

        text_outputs = self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            return_dict=return_dict,
        )

        image_embeds = self.vision_model.vision_model.post_layernorm(vision_outputs[0])  # last_hidden_state
        image_embeds = self.image_projection_layer(image_embeds)

        text_embeds = text_outputs[0]  # last_hidden_state
        text_embeds = self.text_projection_layer(text_embeds)

        if patch_ids is not None:
            # add box image embeddings (mean)
            # image_embeds : batch_size * (num_patch+1) * dims
            image_embeds_size_1 = image_embeds.size(1)
            new_image_embeds = []
            for item_image_embeds, item_patch_ids  in zip(image_embeds, patch_ids):
                add_item_image_embeds = []
                for i_, box_patch_ids in enumerate(item_patch_ids):
                    # skip cls embedding
                    box_image_embeds = item_image_embeds[torch.as_tensor(box_patch_ids) + 1]
                    box_image_embeds = torch.mean(box_image_embeds, dim=0, keepdim=True)
                    add_item_image_embeds.append(box_image_embeds)
                new_image_embeds.append(torch.cat([item_image_embeds] + add_item_image_embeds))
            image_embeds = pad_sequence(new_image_embeds, batch_first=True)

            len_of_ones = torch.as_tensor([len(box_p_ids) for box_p_ids in patch_ids], dtype=torch.int64, device=image_embeds.device) + image_embeds_size_1
            image_mask_ones = [torch.ones(l, dtype=torch.long, device=image_embeds.device) for l in len_of_ones]
            image_mask_zeros = [torch.zeros(image_embeds.size(1) - l, dtype=torch.long, device=image_embeds.device) for l in len_of_ones]
            image_masks = torch.cat([torch.cat([one, zero]).unsqueeze(0) for one, zero in zip(image_mask_ones, image_mask_zeros)], dim=0)

            text_token_type_ids = pad_sequence(list(map(lambda x: torch.as_tensor(x, device=image_embeds.device, dtype=torch.long),extend_token_type_ids[1])),batch_first=True)
            text_token_type_ids = torch.cat([text_token_type_ids, torch.zeros(text_embeds.size(0), text_embeds.size(1) - text_token_type_ids.size(1), device=image_embeds.device, dtype=torch.long)], dim=1)
            image_added_token_type_ids = pad_sequence(list(map(lambda x: torch.as_tensor(x, device=image_embeds.device, dtype=torch.long),extend_token_type_ids[0])),batch_first=True)
            image_token_type_ids = torch.cat([torch.ones(image_embeds.size(0), image_embeds_size_1, dtype=torch.long, device=image_embeds.device), image_added_token_type_ids], dim=1)
        else:
            image_masks = torch.ones((image_embeds.size(0), image_embeds.size(1)), dtype=torch.long, device=image_embeds.device)
        extend_image_masks = self.text_model.get_extended_attention_mask(image_masks, image_masks.size())
        extend_text_masks = self.text_model.get_extended_attention_mask(attention_mask, attention_mask.size())
        
        if patch_ids is not None and extend_token_type_ids is not None:
            text_embeds = text_embeds + self.token_type_embeddings(text_token_type_ids)
            image_embeds = image_embeds + self.token_type_embeddings(image_token_type_ids)
        else:
            image_embeds = image_embeds + self.token_type_embeddings(torch.full_like(image_masks, 1))
            text_embeds = text_embeds  + self.token_type_embeddings(torch.zeros_like(attention_mask))

        x, y = text_embeds, image_embeds
        for text_layer, image_layer in zip(self.cross_modal_text_layers, self.cross_modal_image_layers):
            x1 = text_layer(x, y, extend_text_masks, extend_image_masks)
            y1 = image_layer(y, x, extend_image_masks, extend_text_masks)
            x, y = x1[0], y1[0]

        text_embeds, image_embeds = x, y
        text_pooler_output = self.cross_modal_text_pooler(x)
        image_pooler_output =  self.cross_modal_image_pooler(y)
        pooler_output = torch.cat([text_pooler_output, image_pooler_output], dim=-1)

        if not return_dict:
            output = (pooler_output, text_embeds, image_embeds)
            return output
        return VLEModelOutput(
            pooler_output = pooler_output,
            text_embeds = text_embeds,
            image_embeds = image_embeds
        )


    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        # At the moment fast initialization is not supported
        # for composite models
        kwargs["_fast_init"] = False
        return super().from_pretrained(*args, **kwargs)

    @classmethod
    def from_vision_text_pretrained(
        cls,
        vision_model_name_or_path: str = None,
        text_model_name_or_path: str = None,
        *model_args,
        **kwargs,
    ) -> PreTrainedModel:

        kwargs_vision = {
            argument[len("vision_") :]: value for argument, value in kwargs.items() if argument.startswith("vision_")
        }

        kwargs_text = {
            argument[len("text_") :]: value for argument, value in kwargs.items() if argument.startswith("text_")
        }

        # remove vision, text kwargs from kwargs
        for key in kwargs_vision.keys():
            del kwargs["vision_" + key]
        for key in kwargs_text.keys():
            del kwargs["text_" + key]

        # Load and initialize the vision and text model
        vision_model = kwargs_vision.pop("model", None)
        if vision_model is None:
            if vision_model_name_or_path is None:
                raise ValueError(
                    "If `vision_model` is not defined as an argument, a `vision_model_name_or_path` has to be defined"
                )

            if "config" not in kwargs_vision:
                vision_config = AutoConfig.from_pretrained(vision_model_name_or_path)

            if vision_config.model_type == "clip":
                kwargs_vision["config"] = vision_config.vision_config
                vision_model = CLIPVisionModel.from_pretrained(vision_model_name_or_path, *model_args, **kwargs_vision)
            else:
                kwargs_vision["config"] = vision_config
                vision_model = AutoModel.from_pretrained(vision_model_name_or_path, *model_args, **kwargs_vision)

        text_model = kwargs_text.pop("model", None)
        if text_model is None:
            if text_model_name_or_path is None:
                raise ValueError(
                    "If `text_model` is not defined as an argument, a `text_model_name_or_path` has to be defined"
                )

            if "config" not in kwargs_text:
                text_config = AutoConfig.from_pretrained(text_model_name_or_path)
                kwargs_text["config"] = text_config

            text_model = AutoModel.from_pretrained(text_model_name_or_path, *model_args, **kwargs_text)

        # instantiate config with corresponding kwargs
        config = VLEConfig(text_config=text_model.config, vision_config=vision_model.config, **kwargs)

        # init model
        model = cls(config=config, vision_model=vision_model, text_model=text_model)

        # the projection layers are always newly initialized when loading the model
        # using pre-trained vision and text model.
        logger.warning(
            "The coattention layers and projection layers are newly initialized. You should probably TRAIN this model on a down-stream task to be"
            " able to use it for predictions and inference."
        )
        return model


    def get_text_features(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        token_type_ids=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        text_outputs = self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            #output_attentions=output_attentions,
            #output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        return text_outputs[0] # last_hidden_state

    def get_image_features(
        self,
        pixel_values=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        Returns:
            image_features (`torch.FloatTensor` of shape `(batch_size, output_dim`): The image embeddings obtained by
            applying the projection layer to the pooled output of [`CLIPVisionModel`].

        Examples:

        ```python
        >>> from PIL import Image
        >>> import requests
        >>> from transformers import VLEModel, AutoImageProcessor

        >>> model = VLEModel.from_pretrained("clip-italian/clip-italian")
        >>> image_processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224")

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> inputs = image_processor(images=image, return_tensors="pt")

        >>> image_features = model.get_image_features(**inputs)
        ```"""
        vision_outputs = self.vision_model(
            pixel_values=pixel_values,
            #output_attentions=output_attentions,
            #output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        last_hidden_state = self.vision_model.vision_model.post_layernorm(vision_outputs[0])
        return last_hidden_state
    def get_input_embeddings(self):
        return self.text_model.embeddings.word_embeddings

    def set_input_embeddings(self, new_embeddings):
        self.text_model.embeddings.word_embeddings = new_embeddings

class VLEForVQA(VLEPreTrainedModel):
    def __init__(
        self,
        config: Optional[VLEConfig] = None,
        vision_model: Optional[PreTrainedModel] = None,
        text_model: Optional[PreTrainedModel] = None,
    ):
        super().__init__(config)
        self.vle = VLEModel(config, vision_model, text_model)

        hidden_size = config.hidden_size
        self.num_vqa_labels = len(self.config.id2label)
        self.vqa_classifier = nn.Sequential(
                                    nn.Linear(hidden_size * 2, hidden_size * 2),
                                    nn.LayerNorm(hidden_size * 2),
                                    nn.GELU(),
                                    nn.Linear(hidden_size * 2, self.num_vqa_labels),
        )
        self.vqa_classifier.apply(self._init_weights)
    
    def forward(self,
                input_ids: Optional[torch.LongTensor],
                pixel_values: Optional[torch.FloatTensor],
                attention_mask: Optional[torch.Tensor] = None,
                position_ids: Optional[torch.LongTensor] = None,
                token_type_ids: Optional[torch.LongTensor] = None,
                patch_ids = None,
                vqa_labels = None,
                vqa_scores = None,
                return_loss: Optional[bool] = None,
                return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], VLEForVQAOutput]:

        return_dict = return_dict if return_dict is not None else self.config.return_dict

        vle_output = self.vle(
            input_ids = input_ids,
            pixel_values = pixel_values,
            attention_mask = attention_mask,
            position_ids = position_ids,
            token_type_ids = token_type_ids,
            patch_ids = patch_ids,)
        pooler_output = vle_output[0]
        vqa_logits = self.vqa_classifier(pooler_output)


        vqa_loss = None
        if return_loss and vqa_labels is not None and vqa_scores is not None:
            vqa_targets = torch.zeros(len(vqa_logits), self.num_vqa_labels,device=vqa_logits.device)
            for i, (_label, _score) in enumerate(zip(vqa_labels, vqa_scores)):
                for l, s in zip(_label, _score):
                    vqa_targets[i, l] = s
            vqa_loss = F.binary_cross_entropy_with_logits(vqa_logits, vqa_targets) * vqa_targets.shape[1]
            # https://github.com/jnhwkim/ban-vqa/blob/master/train.py#L19

        if not return_dict:
            output = (vqa_logits,)
            return ((vqa_loss,) + output) if vqa_loss is not None else output
        return VLEForVQAOutput(
            loss = vqa_loss,
            logits = vqa_logits
        )


class VLEForITM(VLEPreTrainedModel):
    def __init__(
        self,
        config: Optional[VLEConfig] = None,
        vision_model: Optional[PreTrainedModel] = None,
        text_model: Optional[PreTrainedModel] = None,
    ):
        super().__init__(config)
        self.vle = VLEModel(config, vision_model, text_model)

        hidden_size = config.hidden_size
        self.itm_score = ITMHead(hidden_size*2)
        self.itm_score.apply(self._init_weights)

    def forward(self,
                input_ids: Optional[torch.LongTensor],
                pixel_values: Optional[torch.FloatTensor],
                attention_mask: Optional[torch.Tensor] = None,
                position_ids: Optional[torch.LongTensor] = None,
                token_type_ids: Optional[torch.LongTensor] = None,
                patch_ids = None,
                itm_labels = None,
                return_loss: Optional[bool] = None,
                return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], VLEForITMOutput]:

        return_dict = return_dict if return_dict is not None else self.config.return_dict

        vle_output = self.vle(
            input_ids = input_ids,
            pixel_values = pixel_values,
            attention_mask = attention_mask,
            position_ids = position_ids,
            token_type_ids = token_type_ids,
            patch_ids = patch_ids,)
        pooler_output = vle_output[0]

        itm_logits = self.itm_score(pooler_output)
        itm_loss = None
        if return_loss and itm_labels is not None:
            itm_loss = nn.functional.cross_entropy(itm_logits, torch.tensor(itm_labels).long().to(itm_logits.device))
        if not return_dict:
            output = (itm_logits,)
            return ((itm_loss,) + output) if itm_loss is not None else output
        return VLEForITMOutput(loss = itm_loss, logits = itm_logits)


class VLEForPBC(VLEPreTrainedModel):
    def __init__(
        self,
        config: Optional[VLEConfig] = None,
        vision_model: Optional[PreTrainedModel] = None,
        text_model: Optional[PreTrainedModel] = None,
    ):
        super().__init__(config)
        self.vle = VLEModel(config, vision_model, text_model)

        hidden_size = config.hidden_size
        self.pbc_classifier = nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.LayerNorm(hidden_size),
                nn.GELU(),
                nn.Linear(hidden_size, 2),
            )
        self.pbc_classifier.apply(self._init_weights)
    
    def forward(self,
                input_ids: Optional[torch.LongTensor],
                pixel_values: Optional[torch.FloatTensor],
                attention_mask: Optional[torch.Tensor] = None,
                position_ids: Optional[torch.LongTensor] = None,
                token_type_ids: Optional[torch.LongTensor] = None,
                patch_ids = None,
                pbc_labels = None,
                return_loss: Optional[bool] = None,
                return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], VLEForPBCOutput]:

        return_dict = return_dict if return_dict is not None else self.config.return_dict

        vle_output = self.vle(
            input_ids = input_ids,
            pixel_values = pixel_values,
            attention_mask = attention_mask,
            position_ids = position_ids,
            token_type_ids = token_type_ids,
            patch_ids = patch_ids,)
        image_embeds = vle_output['image_embeds']
        pbc_logits = self.pbc_classifier(image_embeds[:,1:,:])

        pbc_loss = None
        if return_loss and pbc_labels is not None:
            pbc_loss = F.cross_entropy(pbc_logits, torch.tensor(pbc_labels).long().to(pbc_logits.device))

        if not return_dict:
            output = (pbc_logits,)
            return ((pbc_loss,) + output) if pbc_loss is not None else output
        return VLEForPBCOutput(loss = pbc_loss, logits = pbc_logits)


class VLEForMLM(VLEPreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"mlm_score.1.predictions.decoder.weight",r"mlm_score.1.predictions.decoder.bias"]
    def __init__(
        self,
        config: Optional[VLEConfig] = None,
        vision_model: Optional[PreTrainedModel] = None,
        text_model: Optional[PreTrainedModel] = None,
    ):
        super().__init__(config)
        self.vle = VLEModel(config, vision_model, text_model)

        hidden_size = config.hidden_size
        mlm_head = DebertaV2OnlyMLMHead(self.config.text_config)
        mlm_transform = nn.Linear(hidden_size, self.config.text_config.hidden_size)
        self.mlm_score = nn.Sequential(
                        mlm_transform,
                        mlm_head,
                    )

    def forward(self,
                input_ids: Optional[torch.LongTensor],
                pixel_values: Optional[torch.FloatTensor],
                attention_mask: Optional[torch.Tensor] = None,
                position_ids: Optional[torch.LongTensor] = None,
                token_type_ids: Optional[torch.LongTensor] = None,
                patch_ids = None,
                mlm_labels = None,
                return_loss: Optional[bool] = None,
                return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], VLEForMLMOutput]:

        return_dict = return_dict if return_dict is not None else self.config.return_dict

        vle_output = self.vle(
            input_ids = input_ids,
            pixel_values = pixel_values,
            attention_mask = attention_mask,
            position_ids = position_ids,
            token_type_ids = token_type_ids,
            patch_ids = patch_ids,)
        text_feats = vle_output.text_embeds

        mlm_logits = self.mlm_score(text_feats)
        mlm_loss = None
        if return_loss and mlm_labels is not None:
            mlm_loss = F.cross_entropy(
                mlm_logits.view(-1, self.config.text_config.vocab_size),
                mlm_labels.view(-1),
                ignore_index=-100,
            )
        if not return_dict:
            output = (mlm_logits,)
            return ((mlm_loss,) + output) if mlm_loss is not None else output
        return VLEForMLMOutput(loss = mlm_loss, logits = mlm_logits)


    def get_output_embeddings(self):
        return self.mlm_score[1].predictions.decoder

    def set_output_embeddings(self, new_embeddings):
        self.mlm_score[1].predictions.decoder = new_embeddings

        
class VLEForVCRQ2A(VLEPreTrainedModel):
    def __init__(
        self,
        config: Optional[VLEConfig] = None,
        vision_model: Optional[PreTrainedModel] = None,
        text_model: Optional[PreTrainedModel] = None,
    ):
        super().__init__(config)
        self.vle = VLEModel(config, vision_model, text_model)

        hidden_size = config.hidden_size
        self.vcr_q2a_logit = nn.Sequential(
                                    nn.Linear(hidden_size * 2, hidden_size * 2),
                                    nn.LayerNorm(hidden_size * 2),
                                    nn.GELU(),
                                    nn.Linear(hidden_size * 2, 1),
        )
        self.vcr_q2a_logit.apply(self._init_weights)
    
    def forward(self,
                input_ids: Optional[torch.LongTensor],
                pixel_values: Optional[torch.FloatTensor],
                attention_mask: Optional[torch.Tensor] = None,
                position_ids: Optional[torch.LongTensor] = None,
                token_type_ids: Optional[torch.LongTensor] = None,
                patch_ids = None,
                vcr_labels = None,
                extend_token_type_ids = None,
                return_loss: Optional[bool] = None,
                return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], VLEForVQAOutput]:

        return_dict = return_dict if return_dict is not None else self.config.return_dict

        infers = []
        for i in range(4):
            vle_output = self.vle(
                input_ids = input_ids[i],
                pixel_values = pixel_values,
                attention_mask = attention_mask[i],
                position_ids = position_ids,
                token_type_ids = token_type_ids[i],
                patch_ids = patch_ids[i],
                extend_token_type_ids = extend_token_type_ids[i])
            pooler_output = vle_output[0]
            logits = self.vcr_q2a_logit(pooler_output)
            infers.append(logits)

        vcr_logits = torch.cat(infers, dim=-1)
        vcr_loss = None
        if return_loss and vcr_labels is not None:
            vcr_targets = torch.zeros(len(vcr_logits), dtype=torch.long).to(self.device)
            for i, _label in enumerate(vcr_labels):
                vcr_targets[i] = _label
            vcr_loss = F.cross_entropy(vcr_logits, vcr_targets.view(-1))

        if not return_dict:
            output = (vcr_logits,)
            return ((vcr_loss,) + output) if vcr_loss is not None else output
        return VLEForVCRQ2AOutput(
            loss = vcr_loss,
            logits = vcr_logits
        )


class VLEForVCRQA2R(VLEPreTrainedModel):
    def __init__(
        self,
        config: Optional[VLEConfig] = None,
        vision_model: Optional[PreTrainedModel] = None,
        text_model: Optional[PreTrainedModel] = None,
    ):
        super().__init__(config)
        self.vle = VLEModel(config, vision_model, text_model)

        hidden_size = config.hidden_size
        self.vcr_qa2r_logit = nn.Sequential(
                                    nn.Linear(hidden_size * 2, hidden_size * 2),
                                    nn.LayerNorm(hidden_size * 2),
                                    nn.GELU(),
                                    nn.Linear(hidden_size * 2, 1),
        )
        self.vcr_qa2r_logit.apply(self._init_weights)
    
    def forward(self,
                input_ids: Optional[torch.LongTensor],
                pixel_values: Optional[torch.FloatTensor],
                attention_mask: Optional[torch.Tensor] = None,
                position_ids: Optional[torch.LongTensor] = None,
                token_type_ids: Optional[torch.LongTensor] = None,
                patch_ids = None,
                vcr_labels = None,
                extend_token_type_ids = None,
                return_loss: Optional[bool] = None,
                return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], VLEForVQAOutput]:

        return_dict = return_dict if return_dict is not None else self.config.return_dict

        infers = []
        for i in range(4):
            vle_output = self.vle(
                input_ids = input_ids[i],
                pixel_values = pixel_values,
                attention_mask = attention_mask[i],
                position_ids = position_ids,
                token_type_ids = token_type_ids[i],
                patch_ids = patch_ids[i],
                extend_token_type_ids = extend_token_type_ids[i])
            pooler_output = vle_output[0]
            logits = self.vcr_qa2r_logit(pooler_output)
            infers.append(logits)

        vcr_logits = torch.cat(infers, dim=-1)
        vcr_loss = None
        if return_loss and vcr_labels is not None:
            vcr_targets = torch.zeros(len(vcr_logits), dtype=torch.long).to(self.device)
            for i, _label in enumerate(vcr_labels):
                vcr_targets[i] = _label
            vcr_loss = F.cross_entropy(vcr_logits, vcr_targets.view(-1))

        if not return_dict:
            output = (vcr_logits,)
            return ((vcr_loss,) + output) if vcr_loss is not None else output
        return VLEForVCRQA2ROutput(
            loss = vcr_loss,
            logits = vcr_logits
        )
