import torch
from torch import nn
# import torch.nn.functional as F
from torchvision import transforms
from timm.models import create_model

from allennlp.common import Params
from kb.include_all import ModelArchiveFromParams
from kb.knowbert_utils import KnowBertBatchifier


class Net(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        self.vision_net = create_model(
                'vit_base_patch16_224',
                pretrained=True,
                num_classes=self.cfg.NUM_CLASSES,
        )
        self.vision_net.head = nn.Identity()

        # archive_file = 'https://allennlp.s3-us-west-2.amazonaws.com/knowbert/models/knowbert_wiki_wordnet_model.tar.gz'
        archive_file = 'datasets/knowbert_wiki_wordnet_model.tar.gz'
        params = Params({'archive_file': archive_file})
        self.language_net = ModelArchiveFromParams.from_params(params = params)
        self.language_batcher = KnowBertBatchifier(archive_file)

        for name, child in self.vision_net.named_children():
            print(name, 'is frozen.')
            for parameter in child.parameters():
                parameter.requires_grad = False

        for name, child in self.language_net.named_children():
            print(name, 'is frozen.')
            for parameter in child.parameters():
                parameter.requires_grad = False

        self.resize = transforms.Resize(224)

        dim = 768
        self.linear_global = nn.Linear(dim, dim)
        self.linear_local = nn.Linear(dim, dim)
        self.linear_text = nn.Linear(dim, dim)
        self.norm_global = nn.LayerNorm(dim)
        self.norm_local = nn.LayerNorm(dim)
        self.norm_text = nn.LayerNorm(dim)
        self.activation = nn.LeakyReLU(inplace=True)

        self.global_e = nn.Parameter(torch.zeros(1, 1, dim), requires_grad=True)
        self.local_e = nn.Parameter(torch.zeros(1, 1, dim), requires_grad=True)
        self.text_e = nn.Parameter(torch.zeros(1, 1, dim), requires_grad=True)

        nhead = 8
        self.transformer_1 = nn.TransformerEncoderLayer(d_model=dim, nhead=nhead, batch_first=True)
        self.transformer_2 = nn.TransformerEncoderLayer(d_model=dim, nhead=nhead, batch_first=True)
        self.head = nn.Linear(dim, self.cfg.NUM_CLASSES)

        self.criterion = nn.CrossEntropyLoss()

    @staticmethod
    def convert_kb_input_to_device(inputs, device):
        inputs['tokens']['tokens'] = inputs['tokens']['tokens'].to(device)
        inputs['segment_ids'] = inputs['segment_ids'].to(device)

        inputs['candidates']['wiki']['candidate_entity_priors'] = inputs['candidates']['wiki']['candidate_entity_priors'].to(device)
        inputs['candidates']['wiki']['candidate_entities']['ids'] = inputs['candidates']['wiki']['candidate_entities']['ids'].to(device)
        inputs['candidates']['wiki']['candidate_spans'] = inputs['candidates']['wiki']['candidate_spans'].to(device)
        inputs['candidates']['wiki']['candidate_segment_ids'] = inputs['candidates']['wiki']['candidate_segment_ids'].to(device)

        inputs['candidates']['wordnet']['candidate_entity_priors'] = inputs['candidates']['wordnet']['candidate_entity_priors'].to(device)
        inputs['candidates']['wordnet']['candidate_entities']['ids'] = inputs['candidates']['wordnet']['candidate_entities']['ids'].to(device)
        inputs['candidates']['wordnet']['candidate_spans'] = inputs['candidates']['wordnet']['candidate_spans'].to(device)
        inputs['candidates']['wordnet']['candidate_segment_ids'] = inputs['candidates']['wordnet']['candidate_segment_ids'].to(device)

    def forward_global(self, image):
        global_f = torch.unsqueeze(self.vision_net(image), dim=1)
        return global_f

    def forward_local(self, image):
        local_f1 = torch.unsqueeze(self.vision_net(self.resize(image[:, :,   0:112,   0:112])), dim=1)
        local_f2 = torch.unsqueeze(self.vision_net(self.resize(image[:, :,   0:112, 112:224])), dim=1)
        local_f3 = torch.unsqueeze(self.vision_net(self.resize(image[:, :, 112:224,   0:112])), dim=1)
        local_f4 = torch.unsqueeze(self.vision_net(self.resize(image[:, :, 112:224, 112:224])), dim=1)
        local_f5 = torch.unsqueeze(self.vision_net(self.resize(image[:, :,  56:168,  56:168])), dim=1)
        local_f = torch.cat((local_f1, local_f2, local_f3, local_f4, local_f5), dim=1)
        return local_f

    def forward_text(self, image, text):
        for kb_input in self.language_batcher.iter_batches(text, verbose=False):
            self.convert_kb_input_to_device(kb_input, image.device)
            kb_output = self.language_net(**kb_input)
        text_f = kb_output['contextual_embeddings']
        return text_f

    def forward(self, image, text, targets=None):
        global_f = self.forward_global(image)
        local_f = self.forward_local(image)
        text_f = self.forward_text(image, text)

        global_f = self.activation(self.linear_global(self.norm_global(global_f)))
        local_f = self.activation(self.linear_local(self.norm_local(local_f)))
        text_f = self.activation(self.linear_text(self.norm_text(text_f)))

        global_f = global_f + self.global_e
        local_f = local_f + self.local_e
        text_f = text_f + self.text_e

        general_f = torch.cat((global_f, local_f, text_f), dim=1)
        general_f = self.transformer_1(general_f)
        general_f = self.transformer_2(general_f)

        general_f = torch.mean(general_f, dim=1)
        logits = self.head(general_f)
        # logits = F.dropout(logits, p=0.3, training=self.training)

        if targets is None:
            return logits
        else:
            return self.criterion(logits, targets)
