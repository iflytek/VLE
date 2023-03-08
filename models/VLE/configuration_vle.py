# coding=utf-8
# Copyright The HuggingFace Inc. team. All rights reserved.
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
""" VLE model configuration"""

import copy

from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging
from transformers.models.auto.configuration_auto import AutoConfig
from transformers.models.clip.configuration_clip import CLIPVisionConfig
from typing import Union, Dict

logger = logging.get_logger(__name__)


class VLEConfig(PretrainedConfig):

    model_type = "vle"
    is_composition = True

    def __init__(
        self, 
        text_config: Union[PretrainedConfig, Dict],
        vision_config: Union[PretrainedConfig, Dict],
        num_token_types=2,
        hidden_size=768,
        num_hidden_layers=6,
        num_attention_heads=12,
        intermediate_size=3072,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        initializer_range=0.02,
        layer_norm_eps=1e-12,
        classifier_dropout=None,
        **kwargs):
        super().__init__(**kwargs)

        if not isinstance(text_config,PretrainedConfig):
            text_model_type = text_config.pop('model_type')
            text_config = AutoConfig.for_model(text_model_type, **text_config)
        self.text_config = text_config

        if not isinstance(vision_config, PretrainedConfig):
            vision_model_type = vision_config.pop('model_type')
            if vision_model_type == "clip":
                vision_config = AutoConfig.for_model(vision_model_type, **vision_config).vision_config
            elif vision_model_type == "clip_vision_model":
                vision_config = CLIPVisionConfig(**vision_config)
            else:
                vision_config = AutoConfig.for_model(vision_model_type, **vision_config)
            self.vision_config = vision_config
        else:
            vision_model_type = vision_config.model_type
            if vision_model_type== "clip":
                vision_config = vision_config.vision_config
            self.vision_config = vision_config



        # co-attention
        self.num_token_types=num_token_types
        self.hidden_size=hidden_size
        self.num_hidden_layers=num_hidden_layers
        self.num_attention_heads=num_attention_heads
        self.intermediate_size=intermediate_size
        self.hidden_act=hidden_act
        self.hidden_dropout_prob=hidden_dropout_prob
        self.attention_probs_dropout_prob=attention_probs_dropout_prob
        self.initializer_range=initializer_range
        self.layer_norm_eps=layer_norm_eps
        self.classifier_dropout=classifier_dropout


    def to_dict(self):
        """
        Serializes this instance to a Python dictionary. Override the default [`~PretrainedConfig.to_dict`].

        Returns:
            `Dict[str, any]`: Dictionary of all the attributes that make up this configuration instance,
        """
        output = copy.deepcopy(self.__dict__)
        output["vision_config"] = self.vision_config.to_dict()
        output["text_config"] = self.text_config.to_dict()
        output["model_type"] = self.__class__.model_type
        return output
