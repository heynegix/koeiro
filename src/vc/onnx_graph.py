"""Fixed-batch functional LLVC export, adapted from MIT KoeAI model.py.

Same weights/equations. Return context by concatenation rather than repeated
ScatterND of the entire encoder history. Static 13/26/52 ms graphs are checked
against the original implementation for consecutive non-zero chunks.
"""
import torch
from torch import nn
import torch.nn.functional as F


class FunctionalLLVC(nn.Module):
    def __init__(self, model, factor):
        super().__init__()
        self.model, self.factor = model, factor
        with torch.no_grad():
            self.register_buffer('label', model.label_embedding(torch.zeros(1, 1)).unsqueeze(-1))

    def project(self, module, audio):
        conv = module[0]
        weight, bias = conv.weight[:, :, 0], conv.bias.reshape(1, -1, 1)
        if conv.in_channels//conv.groups == 2:
            result = audio[:, 0::2]*weight[:, 0].reshape(1,-1,1) + audio[:, 1::2]*weight[:, 1].reshape(1,-1,1)
        else:
            result = (audio.unsqueeze(2)*weight.reshape(1,conv.groups,2,1)).reshape(1,conv.out_channels,-1)
        return F.relu(result+bias)

    def decoder(self, target, memory, context):
        decoder = self.model.mask_gen.decoder
        mem = torch.cat((context[:,0], memory.transpose(1,2)),dim=1)
        tgt = torch.cat((context[:,1], target.transpose(1,2)),dim=1)
        next_context = torch.stack((mem[:,-13:],tgt[:,-13:]),dim=1)
        mem = torch.cat([mem[:,i*13:i*13+26] for i in range(self.factor)],dim=0)
        tgt = torch.cat([tgt[:,i*13:i*13+26] for i in range(self.factor)],dim=0)
        position = decoder.pos_enc.pe[:,:26]
        mem, tgt = mem+position, tgt+position
        layer = decoder.tf_dec_layers[0]
        query = tgt[:,-13:]
        attended = layer.self_attn(query,tgt,tgt,need_weights=False)[0]
        query = layer.norm1(query+attended)
        attended = layer.multihead_attn(query,mem,mem,need_weights=False)[0]
        query = layer.norm2(query+attended)
        query = layer.norm3(query+layer.linear2(F.relu(layer.linear1(query))))
        return query.reshape(1,13*self.factor,256).transpose(1,2),next_context

    def forward(self, audio, encoder, decoder, output, prenet):
        # Single-channel causal prenet; no full-state in-place ScatterND.
        x = audio
        next_prenet = []
        pre = self.model.convnet_pre
        for i,block in enumerate(pre.down_convs):
            start,length = pre.buf_indices[i],pre.buf_lengths[i]
            joined = torch.cat((prenet[:,:,start:start+length],x),dim=-1)
            next_prenet.append(joined[:,:,-length:])
            x = joined[:,:,block.output_crop:] + torch.tanh(block.filter(joined))*torch.sigmoid(block.gate(joined))
        audio = audio+x
        encoded = self.model.in_conv(audio)
        x = encoded
        next_encoder = []
        layers = self.model.mask_gen.encoder
        frames = 13*self.factor
        for i,block in enumerate(layers.dcc_layers):
            start,length = layers.buf_indices[i],layers.buf_lengths[i]
            joined = torch.cat((encoder[:,:,start:start+length],x),dim=-1)
            next_encoder.append(joined[:,:,-length:])
            depth = block.layers[0]
            dilation = 2**i
            value = depth.bias.reshape(1,-1,1)
            for k in range(3):
                value = value+joined[:,:,k*dilation:k*dilation+frames]*depth.weight[:,0,k].reshape(1,-1,1)
            value = F.relu(block.layers[1](value))
            point = block.layers[3]
            value = F.linear(value.transpose(1,2),point.weight[:,:,0],point.bias).transpose(1,2)
            x = x+F.relu(block.layers[4](value))
        labelled = x*self.label
        mask, next_decoder = self.decoder(self.project(self.model.mask_gen.proj_e2d_l,labelled),
                                          self.project(self.model.mask_gen.proj_e2d_e,x),decoder)
        mask = labelled+self.project(self.model.mask_gen.proj_d2e,mask)
        x = encoded*mask
        x = torch.cat((output,x),dim=-1)
        result = self.model.out_conv(x)
        return result,torch.cat(next_encoder,dim=-1),next_decoder,x[:,:,-4:],torch.cat(next_prenet,dim=-1)
