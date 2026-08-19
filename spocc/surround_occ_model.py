import torch



class SurroundOccModel(torch.nn.Module):


    def __init__(self, model, enable_hook=False):
        super(SurroundOccModel, self).__init__()
        self.model = model
        self.enable_hook = enable_hook
        self.init_temp()

        occ_head = self.model.pts_bbox_head
        self.transformers = occ_head.transformer
        self.volume_h, self.volume_w, self.volume_z  = occ_head.volume_h, occ_head.volume_w, occ_head.volume_z



    def init_temp(self):
        self.sampling_locations_list  = []
        self.attention_weights_logit_list = []
        self.volume_size_list = []
        self.indexes_list = []
        self.volume_mask_list = []
        self.att_layer_value = []
        self.att_block_value = {'cross_att': [], 'norm1': [], 'ffn': [], 'norm2':[], 'conv': []}

    def to(self, arg):
        self.model = self.model.to(arg)
        return self

    def eval(self):
        self.model.eval()
        return self

    def get_df_att_forward_hook(self, idx_vit, idx_layer):
        def forward_hook(module, input, output_org):
            volume_h, volume_w, volume_z = self.volume_h[idx_vit], self.volume_w[idx_vit], self.volume_z[idx_vit]
            output, sampling_locations, attention_weights_logit = output_org[0], output_org[1], output_org[2]
            indexes = [idx.detach() for idx in output_org[3]]
            volume_mask = output_org[4].detach()
            volum_size = [volume_h, volume_w, volume_z]

            self.sampling_locations_list.append(sampling_locations)
            self.attention_weights_logit_list.append(attention_weights_logit)
            self.volume_size_list.append(volum_size)
            self.indexes_list.append(indexes)
            self.volume_mask_list.append(volume_mask)




        return forward_hook

    def get_block_forwar_hook(self, idx_vit, idx_layer, name):

        def forward_block_hook(module, input, output):
            if isinstance(output, tuple) or isinstance(output, list):
                output = output[0]
            self.att_block_value[name].append(output)



        return forward_block_hook


    def get_layer_forward_hook(self, idx_vit, idx_layer):

        def forward_block_hook(module, input, output):
            if isinstance(output, tuple) or isinstance(output, list):
                output = output[0]
            self.att_layer_value.append(output)



        return forward_block_hook


    def forward(self, img, img_metas):
        self.init_temp()
        hook_list = []
        for i, transformer_i in enumerate(self.transformers):
            encoder_i = transformer_i.encoder
            occ_layers_i = encoder_i.layers
            for j, layers_j in enumerate(occ_layers_i):
                cross_att = layers_j.attentions[0]
                deformable_attention = cross_att.deformable_attention
                hook_fn = self.get_df_att_forward_hook(i,j)
                handle = cross_att.register_forward_hook(hook_fn)
                hook_list.append(handle)

                if self.enable_hook:
                    norm_1 = layers_j.norms[0]
                    ffn = layers_j.ffns[0]
                    norm2 = layers_j.norms[1]
                    conv = layers_j.deblock[len(layers_j.deblock)-1]
                    hook0 = cross_att.register_forward_hook(self.get_block_forwar_hook(i, j, "cross_att"))
                    hook1 = norm_1.register_forward_hook(self.get_block_forwar_hook(i, j, "norm1"))
                    hook2 = ffn.register_forward_hook(self.get_block_forwar_hook(i, j, "ffn"))
                    hook3 = norm2.register_forward_hook(self.get_block_forwar_hook(i, j, "norm2"))
                    hook4 = conv.register_forward_hook(self.get_block_forwar_hook(i, j, "conv"))
                    hook_list.extend([hook0,hook1,hook2,hook3, hook4])

                    hook_layer = layers_j.register_forward_hook(self.get_layer_forward_hook(i, j))
                    hook_list.append(hook_layer)

        output = self.model.simple_test(img_metas=img_metas, img=img, rescale=True)
        for handle in hook_list:
            handle.remove()
        pred_occ = output['occ_preds'] 
        if type(pred_occ) == list:
            pred_occ = pred_occ[-1]
        out_dic = {'pred_occ': pred_occ, 'fpn_3d_outputs': output['fpn_3d_outputs'], 'vit_outputs': output['vit_outputs'], 'sampling_locations_list': self.sampling_locations_list,'attention_weights_logit_list': self.attention_weights_logit_list, 
                  'volume_size_list':self.volume_size_list, 'indexes_list': self.indexes_list, 'volume_mask_list': self.volume_mask_list}
        if self.enable_hook:
            out_dic['att_layer_value'] = self.att_layer_value
            out_dic['att_block_value'] = self.att_block_value
        return out_dic

    def __call__(self, img, img_metas):
        return self.forward(img, img_metas)




