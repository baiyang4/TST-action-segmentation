import torch
import numpy as np
import configs.asrf_config as asrf_cfg
import pdb
import sys
import torch.nn.functional as F
sys.path.append('./backbones/asrf')
from libs.postprocess import PostProcessor
    
    
def predict_refiner(model, main_backbone_name, backbones, split_dict, model_dir, result_dir, features_path, vid_list_file, epoch, actions_dict, device, sample_rate):
    model.eval()
    with torch.no_grad():
        model.to(device)

        # model.load_state_dict(torch.load(model_dir + "/epoch-" + str(epoch) + ".model"))
        file_ptr = open(vid_list_file, 'r')
        list_of_vids = file_ptr.read().split('\n')[:-1]
        file_ptr.close()
        for vid in list_of_vids:
            features = np.load(features_path + vid.split('.')[0] + '.npy')
            features = features[:, ::sample_rate]
            input_x = torch.tensor(features, dtype=torch.float)
            input_x.unsqueeze_(0)
            input_x = input_x.to(device)

            #####################################################################
            # >>>>>>>>>> step3: our model <<<<<<<<<<<<<<<<
            #####################################################################
            # segment_cls: [num_decoder, bs, num_seg, num_class+1]  # segment_mask: [num_decoder, bs, num_seg, T] # action_idx: [T]
            segment_cls, segment_mask, action_idx, _ = model(None, input_x) # [bs, num_seg]

            #####################################################################
            # >>>>>>>>>> action idx <<<<<<<<<<<<<<<<
            #####################################################################
            
            # T = action_idx.shape[0]
            # targets_l = action_idx.tolist()
            # prev_action_class = targets_l[0]

            # targets_segment = []
            # targets_segment_cls = []
            # location_segment = []
            # start_id = 0
            # for idx, action_class in enumerate(targets_l):
            #     if action_class != prev_action_class:
            #         end_id = idx
            #         tmp = torch.zeros(T)
            #         tmp[start_id:end_id] = 1
            #         targets_segment.append(tmp)
            #         targets_segment_cls.append(prev_action_class)
            #         location_segment.append([start_id, end_id])
            #         prev_action_class = action_class
            #         start_id = idx
            # tmp = torch.zeros(T)
            # tmp[start_id:T] = 1
            # targets_segment.append(tmp)
            # targets_segment_cls.append(action_class) 
            # location_segment.append([start_id, T])

            # targets_segment = torch.stack(targets_segment, dim = 0)   # [num_video_seg, T]

            # mask_cls = F.softmax(segment_cls[-1], dim=-1) # [num_pred_seg, num_cls]
            # # mask_pred = segment_mask[-1].sigmoid() # [bs, num_pred_seg, T]
            # predicted = torch.bmm(mask_cls.permute(0, 2, 1), targets_segment.unsqueeze(0).to(mask_cls.device)) # [num_cls, T]
            # _, predicted = torch.max(predicted.data, 1)

            # #####################################################################
            # # >>>>>>>>>> step4: inference <<<<<<<<<<<<<<<<
            # #####################################################################
            # mask_cls = F.softmax(segment_cls[-1], dim=-1)[..., :-1] # [bs, num_pred_seg, num_cls]
            # mask_pred = segment_mask[-1].sigmoid() # [bs, num_pred_seg, T]
            # predicted = torch.bmm(mask_cls.permute(0, 2, 1), mask_pred) # [num_cls, T]
            # _, predicted = torch.max(predicted.data, 1) # [bs, T]

            # # #####################################################################
            # # # >>>>>>>>>> step4: roolout inference <<<<<<<<<<<<<<<<
            # # #####################################################################
            mask_cls = F.softmax(segment_cls[-1], dim=-1)[..., :-1][0] # [num_pred_seg, num_cls]
            pred_cls = torch.max(mask_cls, 1)[1] # [num_pred_seg]

            T = action_idx.shape[0]
            predicted = torch.zeros(T)
            action_idx = action_idx.tolist() # [T]
            prev_action_class = action_idx[0]
            start_idx = 0
            cls_idx = 0
            
            for idx, action_class in enumerate(action_idx):
                if action_class != prev_action_class:
                    end_idx = idx
                    predicted[start_idx:end_idx] = pred_cls[cls_idx]
                    cls_idx += 1
                    prev_action_class = action_class
                    start_idx = idx
            predicted[start_idx:T] = pred_cls[-1]

            predicted = predicted.squeeze()
            recognition = []
            for i in range(len(predicted)):
                recognition = np.concatenate((recognition, [list(actions_dict.keys())[list(actions_dict.values()).index(predicted[i].item())]]*sample_rate))
            f_name = vid.split('/')[-1].split('.')[0]
            f_ptr = open(result_dir + "/" + f_name, "w")
            f_ptr.write("### Frame level recognition: ###\n")
            f_ptr.write(' '.join(recognition))
            f_ptr.close()
            

def predict_refiner_best(model, main_backbone_name, backbones, split_dict, best_fp, result_dir, features_path, vid_list_file_tst, actions_dict, device, sample_rate):
    model.eval()
    with torch.no_grad():
        model.to(device)
        model.load_state_dict(torch.load(best_fp))
        file_ptr = open(vid_list_file_tst, 'r')
        list_of_vids = file_ptr.read().split('\n')[:-1]
        file_ptr.close()
        for vid in list_of_vids:
            features = np.load(features_path + vid.split('.')[0] + '.npy')
            features = features[:, ::sample_rate]
            input_x = torch.tensor(features, dtype=torch.float)
            input_x.unsqueeze_(0)
            input_x = input_x.to(device)
            
            split_idx = 0
            for i in range(len(split_dict.keys())):
                if vid.split('.')[0] in split_dict[i+1]:
                    split_idx = i+1
                    break
            
            curr_backbone = backbones[split_idx]
            curr_backbone.eval()
            
            if main_backbone_name != 'asrf':
                if main_backbone_name == 'mstcn':
                    mask = torch.ones(input_x.size(), device=device)
                    action_pred = curr_backbone(input_x, mask)[-1]
                elif main_backbone_name == 'mgru':
                    action_pred = curr_backbone(input_x)
                elif main_backbone_name == 'sstda':
                    mask = torch.ones(input_x.size(), device=device)
                    action_pred, _, _, _, _, _, _, _, _, _, _, _, _, _ = curr_backbone(input_x, input_x, mask, mask, [0, 0], reverse=False)
                    action_pred = action_pred[:, -1, :, :]

                action_idx = torch.argmax(action_pred, dim=1).squeeze().detach()
                
            else:
                out_cls, out_bound = curr_backbone(input_x)
                postprocessor = PostProcessor("refinement_with_boundary", asrf_cfg.boundary_th)
                refined_output_cls = postprocessor(out_cls.cpu().data.numpy(), boundaries=out_bound.cpu().data.numpy(), 
                                                   masks=torch.ones(1, 1, input_x.shape[-1]).bool().data.numpy())
                action_idx = torch.Tensor(refined_output_cls).squeeze().detach()

            _, predictions, _ = model(action_idx.to(device), input_x)
            _, predicted = torch.max(predictions.data, 1)
                
            predicted = predicted.squeeze()
            recognition = []
            for i in range(len(predicted)):
                recognition = np.concatenate((recognition, [list(actions_dict.keys())[list(actions_dict.values()).index(predicted[i].item())]]*sample_rate))
            f_name = vid.split('/')[-1].split('.')[0]
            f_ptr = open(result_dir + "/" + f_name, "w")
            f_ptr.write("### Frame level recognition: ###\n")
            f_ptr.write(' '.join(recognition))
            f_ptr.close()
            

def predict_backbone(name, model, model_dir, result_dir, features_path, vid_list_file, epoch, actions_dict, device, sample_rate):
    model.eval()
    with torch.no_grad():
        model.to(device)
        # model.load_state_dict(torch.load(model_dir + "/epoch-" + str(epoch) + ".model"))
        file_ptr = open(vid_list_file, 'r')
        list_of_vids = file_ptr.read().split('\n')[:-1]
        file_ptr.close()
        for vid in list_of_vids:
            features = np.load(features_path + vid.split('.')[0] + '.npy')
            features = features[:, ::sample_rate]
            input_x = torch.tensor(features, dtype=torch.float)
            input_x.unsqueeze_(0)
            input_x = input_x.to(device)
            if name == 'asrf':
                out_cls, out_bound, _, _ = model(input_x)
                if isinstance(out_cls, list):
                    out_cls = out_cls[-1]
                else:
                    out_cls = out_cls
                if isinstance(out_bound, list):
                    out_bound = out_bound[-1]
                else:
                    out_bound = out_bound
                # # 只推理上面
                # predicted = torch.max(out_cls, 1)[1] # [bs, T]

                # 论文的推理
                postprocessor = PostProcessor("refinement_with_boundary", asrf_cfg.boundary_th)
                refined_output_cls = postprocessor(out_cls.cpu().data.numpy(), boundaries=out_bound.cpu().data.numpy(), 
                                                   masks=torch.ones(1, 1, input_x.shape[-1]).bool().data.numpy())
                predicted = refined_output_cls # [bs, T]
                
            elif name == 'mstcn':
                predictions = model(input_x, torch.ones(input_x.size(), device=device))
                _, predicted = torch.max(predictions[-1].data, 1)
                
            elif name == 'sstda':
                mask = torch.ones(input_x.size(), device=device)
                predictions, _, _, _, _, _, _, _, _, _, _, _, _, _ = model(input_x, input_x, mask, mask, [0, 0], reverse=False)
                _, predicted = torch.max(predictions[:, -1, :, :].data, 1)
                
            elif name == 'mgru':
                predictions = model(input_x)
                _, predicted = torch.max(predictions.data, 1)
                
                
            predicted = predicted.squeeze() # [T]
            recognition = []

            for i in range(len(predicted)):
                recognition = np.concatenate((recognition, [list(actions_dict.keys())[list(actions_dict.values()).index(predicted[i].item())]]*sample_rate))
       
            f_name = vid.split('/')[-1].split('.')[0]
            f_ptr = open(result_dir + "/" + f_name, "w")
            f_ptr.write("### Frame level recognition: ###\n")
            f_ptr.write(' '.join(recognition))
            f_ptr.close()
            


def predict_backbone_new(name, model, model_dir, result_dir, features_path, vid_list_file, epoch, actions_dict, device, sample_rate):
    model.eval()
    with torch.no_grad():
        model.to(device)
        # model.load_state_dict(torch.load(model_dir + "/epoch-" + str(epoch) + ".model"))
        file_ptr = open(vid_list_file, 'r')
        list_of_vids = file_ptr.read().split('\n')[:-1]
        file_ptr.close()
        for vid in list_of_vids:
            features = np.load(features_path + vid.split('.')[0] + '.npy')
            features = features[:, ::sample_rate]
            input_x = torch.tensor(features, dtype=torch.float)
            input_x.unsqueeze_(0)
            input_x = input_x.to(device)
            if name == 'asrf':

                # output_cls: [bs, num_class, T],  output_bound: List-4. [0]-[bs, 1, T]
                # segment_cls: [bs, num_seg, num_class],  segment_mask: [bs, num_seg, T]
                _, out_bound, segment_cls, segment_mask, _, _, _ = model(input_x)

                # [bs, num_class, T]
                segment_cls = F.softmax(segment_cls[0], dim=1) # [num_seg, num_class]
                segment_mask = segment_mask[0].sigmoid() # [num_seg, T]
                out_cls = torch.matmul(segment_cls.permute(1, 0) , segment_mask).unsqueeze(0) # [bs, num_class, T]

                # 只测上面的branch
                # predicted = torch.max(out_cls, 1)[1] # [bs, T]

                # 作者的方法
                if isinstance(out_bound, list):
                    out_bound = out_bound[-1]
                else:
                    out_bound = out_bound
                postprocessor = PostProcessor("refinement_with_boundary", asrf_cfg.boundary_th)
                refined_output_cls = postprocessor(out_cls.cpu().data.numpy(), boundaries=out_bound.cpu().data.numpy(), 
                                                   masks=torch.ones(1, 1, input_x.shape[-1]).bool().data.numpy())
                predicted = refined_output_cls # [bs, T]
                
            elif name == 'mstcn':
                predictions = model(input_x, torch.ones(input_x.size(), device=device))
                _, predicted = torch.max(predictions[-1].data, 1)
                
            elif name == 'sstda':
                mask = torch.ones(input_x.size(), device=device)
                predictions, _, _, _, _, _, _, _, _, _, _, _, _, _ = model(input_x, input_x, mask, mask, [0, 0], reverse=False)
                _, predicted = torch.max(predictions[:, -1, :, :].data, 1)
                
            elif name == 'mgru':
                predictions = model(input_x)
                _, predicted = torch.max(predictions.data, 1)
                
                
            predicted = predicted.squeeze() # [T]
            recognition = []

            for i in range(len(predicted)):
                recognition = np.concatenate((recognition, [list(actions_dict.keys())[list(actions_dict.values()).index(predicted[i].item())]]*sample_rate))
       
            f_name = vid.split('/')[-1].split('.')[0]
            f_ptr = open(result_dir + "/" + f_name, "w")
            f_ptr.write("### Frame level recognition: ###\n")
            f_ptr.write(' '.join(recognition))
            f_ptr.close()



def predict_backbone_gtea(name, model, model_dir, result_dir, features_path, vid_list_file, epoch, actions_dict, device, sample_rate):
    model.eval()
    with torch.no_grad():
        model.to(device)
        # model.load_state_dict(torch.load(model_dir + "/epoch-" + str(epoch) + ".model"))
        file_ptr = open(vid_list_file, 'r')
        list_of_vids = file_ptr.read().split('\n')[:-1]
        file_ptr.close()
        for vid in list_of_vids:
            features = np.load(features_path + vid.split('.')[0] + '.npy')
            features = features[:, ::sample_rate]
            input_x = torch.tensor(features, dtype=torch.float)
            input_x.unsqueeze_(0)
            input_x = input_x.to(device)
            if name == 'asrf':
                out_cls,  _ = model(input_x)
                out_cls = out_cls[-1] # [bs, num_class, T]

                predicted = torch.max(out_cls, 1)[1] # [bs, T]

                # # 论文的推理
                # postprocessor = PostProcessor("refinement_with_boundary", asrf_cfg.boundary_th)
                # refined_output_cls = postprocessor(out_cls.cpu().data.numpy(), boundaries=out_bound.cpu().data.numpy(), 
                #                                    masks=torch.ones(1, 1, input_x.shape[-1]).bool().data.numpy())
                # predicted = refined_output_cls # [bs, T]
                
            elif name == 'mstcn':
                predictions = model(input_x, torch.ones(input_x.size(), device=device))
                _, predicted = torch.max(predictions[-1].data, 1)
                
            elif name == 'sstda':
                mask = torch.ones(input_x.size(), device=device)
                predictions, _, _, _, _, _, _, _, _, _, _, _, _, _ = model(input_x, input_x, mask, mask, [0, 0], reverse=False)
                _, predicted = torch.max(predictions[:, -1, :, :].data, 1)
                
            elif name == 'mgru':
                predictions = model(input_x)
                _, predicted = torch.max(predictions.data, 1)
                
                
            predicted = predicted.squeeze() # [T]
            recognition = []

            for i in range(len(predicted)):
                recognition = np.concatenate((recognition, [list(actions_dict.keys())[list(actions_dict.values()).index(predicted[i].item())]]*sample_rate))
       
            f_name = vid.split('/')[-1].split('.')[0]
            f_ptr = open(result_dir + "/" + f_name, "w")
            f_ptr.write("### Frame level recognition: ###\n")
            f_ptr.write(' '.join(recognition))
            f_ptr.close()


def predict_backbone_gtea2(name, model, model_dir, result_dir, features_path, vid_list_file, epoch, actions_dict, device, sample_rate):
    model.eval()
    with torch.no_grad():
        model.to(device)
        # model.load_state_dict(torch.load(model_dir + "/epoch-" + str(epoch) + ".model"))
        file_ptr = open(vid_list_file, 'r')
        list_of_vids = file_ptr.read().split('\n')[:-1]
        file_ptr.close()
        for vid in list_of_vids:
            features = np.load(features_path + vid.split('.')[0] + '.npy')
            features = features[:, ::sample_rate]
            input_x = torch.tensor(features, dtype=torch.float)
            input_x.unsqueeze_(0)
            input_x = input_x.to(device)
            if name == 'asrf':

                # output_cls: [bs, num_class, T],  output_bound: List-4. [0]-[bs, 1, T]
                # segment_cls: [bs, num_seg, num_class],  segment_mask: [bs, num_seg, T]
                _, _, segment_cls, segment_mask, _, _, _ = model(input_x)

                # [bs, num_class, T]
                segment_cls = F.softmax(segment_cls[0], dim=1) # [num_seg, num_class]
                segment_mask = segment_mask[0].sigmoid() # [num_seg, T]
                out_cls = torch.matmul(segment_cls.permute(1, 0) , segment_mask).unsqueeze(0) # [bs, num_class, T]

                # 只测上面的branch
                predicted = torch.max(out_cls, 1)[1] # [bs, T]

                # # 作者的方法
                # if isinstance(out_bound, list):
                #     out_bound = out_bound[-1]
                # else:
                #     out_bound = out_bound
                # postprocessor = PostProcessor("refinement_with_boundary", asrf_cfg.boundary_th)
                # refined_output_cls = postprocessor(out_cls.cpu().data.numpy(), boundaries=out_bound.cpu().data.numpy(), 
                #                                    masks=torch.ones(1, 1, input_x.shape[-1]).bool().data.numpy())
                # predicted = refined_output_cls # [bs, T]
                
            elif name == 'mstcn':
                predictions = model(input_x, torch.ones(input_x.size(), device=device))
                _, predicted = torch.max(predictions[-1].data, 1)
                
            elif name == 'sstda':
                mask = torch.ones(input_x.size(), device=device)
                predictions, _, _, _, _, _, _, _, _, _, _, _, _, _ = model(input_x, input_x, mask, mask, [0, 0], reverse=False)
                _, predicted = torch.max(predictions[:, -1, :, :].data, 1)
                
            elif name == 'mgru':
                predictions = model(input_x)
                _, predicted = torch.max(predictions.data, 1)
                
                
            predicted = predicted.squeeze() # [T]
            recognition = []

            for i in range(len(predicted)):
                recognition = np.concatenate((recognition, [list(actions_dict.keys())[list(actions_dict.values()).index(predicted[i].item())]]*sample_rate))
       
            f_name = vid.split('/')[-1].split('.')[0]
            f_ptr = open(result_dir + "/" + f_name, "w")
            f_ptr.write("### Frame level recognition: ###\n")
            f_ptr.write(' '.join(recognition))
            f_ptr.close()


def predict_backbone_gtea3(name, model, model_dir, result_dir, features_path, vid_list_file, epoch, actions_dict, device, sample_rate):
    model.eval()
    with torch.no_grad():
        model.to(device)
        # model.load_state_dict(torch.load(model_dir + "/epoch-" + str(epoch) + ".model"))
        file_ptr = open(vid_list_file, 'r')
        list_of_vids = file_ptr.read().split('\n')[:-1]
        file_ptr.close()
        for vid in list_of_vids:
            features = np.load(features_path + vid.split('.')[0] + '.npy')
            features = features[:, ::sample_rate]
            input_x = torch.tensor(features, dtype=torch.float)
            input_x.unsqueeze_(0)
            input_x = input_x.to(device)
            if name == 'asrf':

                # output_cls: [bs, num_class, T],  output_bound: List-4. [0]-[bs, 1, T]
                # segment_cls: [bs, num_seg, num_class],  segment_mask: [bs, num_seg, T]
                _, _, segment_cls, segment_mask, _, _, _ = model(input_x)

                # [bs, num_class, T]
                segment_cls = F.softmax(segment_cls[-1][0], dim=1) # [num_seg, num_class]
                segment_mask = segment_mask[-1][0].sigmoid() # [num_seg, T]
                out_cls = torch.matmul(segment_cls.permute(1, 0) , segment_mask).unsqueeze(0) # [bs, num_class, T]

                # 只测上面的branch
                predicted = torch.max(out_cls, 1)[1] # [bs, T]

                
            elif name == 'mstcn':
                predictions = model(input_x, torch.ones(input_x.size(), device=device))
                _, predicted = torch.max(predictions[-1].data, 1)
                
            elif name == 'sstda':
                mask = torch.ones(input_x.size(), device=device)
                predictions, _, _, _, _, _, _, _, _, _, _, _, _, _ = model(input_x, input_x, mask, mask, [0, 0], reverse=False)
                _, predicted = torch.max(predictions[:, -1, :, :].data, 1)
                
            elif name == 'mgru':
                predictions = model(input_x)
                _, predicted = torch.max(predictions.data, 1)
                
                
            predicted = predicted.squeeze() # [T]
            recognition = []

            for i in range(len(predicted)):
                recognition = np.concatenate((recognition, [list(actions_dict.keys())[list(actions_dict.values()).index(predicted[i].item())]]*sample_rate))
       
            f_name = vid.split('/')[-1].split('.')[0]
            f_ptr = open(result_dir + "/" + f_name, "w")
            f_ptr.write("### Frame level recognition: ###\n")
            f_ptr.write(' '.join(recognition))
            f_ptr.close()



def predict_backbone_wosd(name, model, model_dir, result_dir, features_path, vid_list_file, epoch, actions_dict, device, sample_rate):
    model.eval()
    with torch.no_grad():
        model.to(device)
        # model.load_state_dict(torch.load(model_dir + "/epoch-" + str(epoch) + ".model"))
        file_ptr = open(vid_list_file, 'r')
        list_of_vids = file_ptr.read().split('\n')[:-1]
        file_ptr.close()
        for vid in list_of_vids:
            features = np.load(features_path + vid.split('.')[0] + '.npy')
            features = features[:, ::sample_rate]
            input_x = torch.tensor(features, dtype=torch.float)
            input_x.unsqueeze_(0)
            input_x = input_x.to(device)
            if name == 'asrf':
                out_cls = model(input_x)[1]

                predicted = torch.max(out_cls, 1)[1] # [bs, T]

                # # 论文的推理
                # postprocessor = PostProcessor("refinement_with_boundary", asrf_cfg.boundary_th)
                # refined_output_cls = postprocessor(out_cls.cpu().data.numpy(), boundaries=out_bound.cpu().data.numpy(), 
                #                                    masks=torch.ones(1, 1, input_x.shape[-1]).bool().data.numpy())
                # predicted = refined_output_cls # [bs, T]
                
            elif name == 'mstcn':
                predictions = model(input_x, torch.ones(input_x.size(), device=device))
                _, predicted = torch.max(predictions[-1].data, 1)
                
            elif name == 'sstda':
                mask = torch.ones(input_x.size(), device=device)
                predictions, _, _, _, _, _, _, _, _, _, _, _, _, _ = model(input_x, input_x, mask, mask, [0, 0], reverse=False)
                _, predicted = torch.max(predictions[:, -1, :, :].data, 1)
                
            elif name == 'mgru':
                predictions = model(input_x)
                _, predicted = torch.max(predictions.data, 1)
                
                
            predicted = predicted.squeeze() # [T]
            recognition = []

            for i in range(len(predicted)):
                recognition = np.concatenate((recognition, [list(actions_dict.keys())[list(actions_dict.values()).index(predicted[i].item())]]*sample_rate))
       
            f_name = vid.split('/')[-1].split('.')[0]
            f_ptr = open(result_dir + "/" + f_name, "w")
            f_ptr.write("### Frame level recognition: ###\n")
            f_ptr.write(' '.join(recognition))
            f_ptr.close()


def predict_backbone_wopd(name, model, model_dir, result_dir, features_path, vid_list_file, epoch, actions_dict, device, sample_rate):
    model.eval()
    with torch.no_grad():
        model.to(device)
        # model.load_state_dict(torch.load(model_dir + "/epoch-" + str(epoch) + ".model"))
        file_ptr = open(vid_list_file, 'r')
        list_of_vids = file_ptr.read().split('\n')[:-1]
        file_ptr.close()
        for vid in list_of_vids:
            features = np.load(features_path + vid.split('.')[0] + '.npy')
            features = features[:, ::sample_rate]
            input_x = torch.tensor(features, dtype=torch.float)
            input_x.unsqueeze_(0)
            input_x = input_x.to(device)
            if name == 'asrf':
                out_cls = model(input_x)[1] 
                out_cls = out_cls.permute(1, 0, 2) # [bs, num_seg, num_cls]
                pred = torch.max(out_cls[0], -1)[1] # [num_seg]

                # repeat
                action_idx = model(input_x)[2].tolist() # [T]
                num_frame = len(action_idx)
                predicted = torch.full((1, num_frame), -1).squeeze(0).to(device)

                segment_idx = [0]
                prev_seg = action_idx[0]
                for ii, idx in enumerate(action_idx):
                    if idx != prev_seg:
                        segment_idx.append(ii)
                    prev_seg = idx
                segment_idx.append(len(action_idx))
                
                for s_i in range(len(segment_idx)-1):
                    prev_idx = segment_idx[s_i]
                    curr_idx = segment_idx[s_i+1]
                    predicted[prev_idx:curr_idx] = pred[s_i]

                predicted = predicted.unsqueeze(0)
                # predicted = torch.max(out_cls, 1)[1] # [bs, T]

                # # 论文的推理
                # postprocessor = PostProcessor("refinement_with_boundary", asrf_cfg.boundary_th)
                # refined_output_cls = postprocessor(out_cls.cpu().data.numpy(), boundaries=out_bound.cpu().data.numpy(), 
                #                                    masks=torch.ones(1, 1, input_x.shape[-1]).bool().data.numpy())
                # predicted = refined_output_cls # [bs, T]
                
            elif name == 'mstcn':
                predictions = model(input_x, torch.ones(input_x.size(), device=device))
                _, predicted = torch.max(predictions[-1].data, 1)
                
            elif name == 'sstda':
                mask = torch.ones(input_x.size(), device=device)
                predictions, _, _, _, _, _, _, _, _, _, _, _, _, _ = model(input_x, input_x, mask, mask, [0, 0], reverse=False)
                _, predicted = torch.max(predictions[:, -1, :, :].data, 1)
                
            elif name == 'mgru':
                predictions = model(input_x)
                _, predicted = torch.max(predictions.data, 1)
                
                
            predicted = predicted.squeeze() # [T]
            recognition = []

            for i in range(len(predicted)):
                recognition = np.concatenate((recognition, [list(actions_dict.keys())[list(actions_dict.values()).index(predicted[i].item())]]*sample_rate))
       
            f_name = vid.split('/')[-1].split('.')[0]
            f_ptr = open(result_dir + "/" + f_name, "w")
            f_ptr.write("### Frame level recognition: ###\n")
            f_ptr.write(' '.join(recognition))
            f_ptr.close()