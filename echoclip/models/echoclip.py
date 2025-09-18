import os.path as osp
from open_clip import tokenize, create_model_and_transforms
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision.transforms import Resize
import sys
import transformers
from echoclip.utils import get_logger
from .builder import MODELS
import random
from .prompt_learners import PROMPT_LEARNERS
from .prompt_learners.plain_prompt_learner import PlainPromptLearner

from utils import zero_shot_prompts, compute_regression_metric,compute_regression_metric_mean_std,compute_regression_prediction, LVEF_fine_prompts,LVEF_fine_prompts_new, LVEF_fine_prompt0,LVEF_fine_prompt1,LVEF_fine_prompt2,LVEF_fine_prompt3,LVEF_fine_prompt4
from s2wrapper import forward as multiscale_forward

logger = get_logger(__name__)


@MODELS.register_module()
class EchoCLIP(nn.Module):
    def __init__(
        self,
        **kwargs,
    ) -> None:
        super().__init__()

        if kwargs:
            logger.info(f"irrelevant kwargs: {kwargs}")

        clip_model, _, preprocess_val = create_model_and_transforms(
            "hf-hub:mkaichristensen/echo-clip", precision="bf16", device="cuda"
        )

        clip_model.float()


        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)

        self.embed_dim = 512
        self.logit_scale = clip_model.logit_scale

        self.abmil = VideoFeatureAggregator(feature_dim=512, hidden_dim=64)
        self.layernorm2d = LayerNormConv2d((2048,),eps=1e-06)


# ------------------------------the deeper one-----------------------------
        self.image_clip_fea_encoder = nn.Sequential(*[
                clip_model.visual.trunk.stem,
                clip_model.visual.trunk.stages[0],
                clip_model.visual.trunk.stages[1],
                clip_model.visual.trunk.stages[2],
                clip_model.visual.trunk.stages[3],
                clip_model.visual.trunk.norm_pre,
                nn.Flatten(start_dim=2,end_dim=-1)
            ])
    
        self.head = nn.Sequential(*[
                clip_model.visual.trunk.head.global_pool,
                self.layernorm2d,
                clip_model.visual.trunk.head.flatten,
                clip_model.visual.trunk.head.pre_logits,
                clip_model.visual.trunk.head.drop,
                clip_model.visual.trunk.head.fc,
                clip_model.visual.head.drop,
                nn.Linear(in_features=2048,out_features=512,bias=False)
            ])



        self.last_project = nn.Sequential(
            nn.Linear(512, 512, bias=False),
            nn.ReLU(),
            nn.Linear(512, 128, bias=False),
            nn.ReLU(),
            nn.Linear(128, 1, bias=True))



    def forward(self, video):


        ejection_fraction_prompts = LVEF_fine_prompts_new
        ejection_fraction_prompts = tokenize(ejection_fraction_prompts).cuda()


        text_features = self.text_encoder(ejection_fraction_prompts)
        text_features = F.normalize(text_features, dim=-1)



        x = video.cuda()
        bs,c,T,h,w = x.shape
        x = x.reshape(-1,c,h,w)
        up = x

        fea_multiscale = multiscale_forward(self.image_clip_fea_encoder,up,scales=[1,2],max_split_size=112)
        dim = bs * T

        fea_multiscale = fea_multiscale.reshape(dim,2048,3,3)
        fea_after = self.head(fea_multiscale)


        video_features = fea_after.reshape(bs,T,-1)
        fusion_feature = self.abmil(video_features)


        fusion_feature = F.normalize(fusion_feature,dim=-1)
        logit_scale = self.logit_scale.exp()
        logits = logit_scale * fusion_feature @ text_features.t()

        
        ejection_fraction_predictions = compute_regression_metric(video_embeddings=fusion_feature,
        prompt_embeddings=text_features,logits=logits)

        return ejection_fraction_predictions, logits



    def encode_image(self, x):

        image_features = self.image_encoder(x)
        return image_features


    def forward_text_only(self):
        text_features = self.last_project[2].weight

        return text_features

    def create_psudo_sentence_tokens(self, num_tokens_per_rank, num_context_tokens, num_ranks):
        psudo_sentence_tokens = torch.zeros(num_ranks, 77, dtype=torch.long)

        sentence_length = 1 + num_context_tokens + num_tokens_per_rank + 1 + 1
        psudo_sentence_tokens[:, :sentence_length] = torch.arange(0, sentence_length, dtype=torch.long)
        
        return psudo_sentence_tokens



class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.token_embedding = clip_model.token_embedding
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection


    def forward(self, prompts):
        
        x = self.token_embedding(prompts).type(self.dtype)
        x = x + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        x = x[torch.arange(x.shape[0]), prompts.argmax(dim=-1)] @ self.text_projection

        return x
    
    @property
    def dtype(self):
        return self.transformer.resblocks[0].mlp.c_fc.weight.dtype





class LayerNormConv2d(nn.Module):
    """
    Layer norm the just works on the channel axis for a Conv2d

    Ref:
    - code modified from https://github.com/Scitator/Run-Skeleton-Run/blob/master/common/modules/LayerNorm.py
    - paper: https://arxiv.org/abs/1607.06450

    Usage:
        ln = LayerNormConv(3)
        x = Variable(torch.rand((1,3,4,2)))
        ln(x).size()
    """

    def __init__(self, features, eps=1e-6):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(features).cuda()).unsqueeze(-1).unsqueeze(-1)
        self.beta = nn.Parameter(torch.zeros(features).cuda()).unsqueeze(-1).unsqueeze(-1)
        self.eps = eps
        self.features = features

    def _check_input_dim(self, input):
        if input.size(1) != self.gamma.nelement():
            raise ValueError('got {}-feature tensor, expected {}'
                             .format(input.size(1), self.features))

    def forward(self, x):
        self._check_input_dim(x)
        x_flat = x.transpose(1,-1).contiguous().view((-1, x.size(1)))
        mean = x_flat.mean(0).unsqueeze(-1).unsqueeze(-1).expand_as(x)
        std = x_flat.std(0).unsqueeze(-1).unsqueeze(-1).expand_as(x)
        return self.gamma.expand_as(x) * (x - mean) / (std + self.eps) + self.beta.expand_as(x)





class VideoFeatureAggregator(nn.Module):
    def __init__(self, feature_dim, hidden_dim):
        super(VideoFeatureAggregator, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )


        self.feature_aggregation = nn.Sequential(
            nn.Linear(feature_dim, feature_dim)
        )

    def forward(self, features):
 
        batch_size, b, d = features.shape

        flat_features = features.view(-1, d)  # [batch_size * b, d]
        attention_weights = self.attention(flat_features)  # [batch_size * b, 1]
        attention_weights = attention_weights.view(batch_size, b, 1)  # [batch_size, b, 1]
        attention_weights = F.softmax(attention_weights, dim=1)  # [batch_size, b, 1]


        weighted_features = attention_weights * features  # [batch_size, b, d]
        aggregated_feature = torch.sum(weighted_features, dim=1)  # [batch_size, d]
        video_feature = self.feature_aggregation(aggregated_feature)  # [batch_size, d]

        return video_feature