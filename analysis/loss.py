from optimizer.sam import disable_running_stats, enable_running_stats
import torch
from tqdm import tqdm
import torch.nn.functional as F
import sys
import numpy as np
from utilities import dict_to_

@torch.no_grad
def compute_loss(graphs, model, loss_name, criterion, criterion_summed, device, num_classes, loader_abridged, test_loader, opt_params, compute_acc=False, compute_model_output=False):
    disable_running_stats(model)
    loss_sum = 0
    accuracy_sum = 0
    accuracy = 0
    
    model_output = []
    for batch_idx, input in enumerate(loader_abridged, start=1):
        if opt_params["wild_data"]:
            data, target, metadata = input
            data, target = data.to(device), target.to(device)
            out = model(data)
            loss = criterion_summed(out, target)
        elif opt_params["cub_data"]:
            data, target, group = input
            data, target, group = data.to(device), target.to(device), group.to(device)
            out = model(data)
            loss = criterion_summed(out, target, group, False)
        elif not opt_params["hf_model"]:
            data, target = input
            data, target = data.to(device), target.to(device)
            out = model(data)
            loss = criterion_summed(out, target)
        else:
            dict_to_(input, device)
            target = input["labels"]
            output = model(**input)
            loss, out = output.loss * output.logits.shape[0], output.logits

        if compute_acc:
            if out.dim() > 1:
                if out.shape != target.shape:
                    accuracy = torch.sum((torch.argmax(out,dim=1)==target).float()).item()
                else:
                    accuracy = torch.sum((torch.argmax(out,dim=1)==torch.argmax(target,dim=1)).float()).item()
            else:
                accuracy = torch.sum((out*target > 0).float()).item()

        if compute_model_output:
            model_output.append(((out-target)*target).detach().cpu().numpy())
        loss_sum += loss.item()
        accuracy_sum += accuracy
    graphs.loss.append(loss_sum / len(loader_abridged.dataset))
    graphs.accuracy.append(accuracy_sum / len(loader_abridged.dataset))
    
    if compute_model_output:
        graphs.model_output.append(np.concatenate(model_output))
    
    model.eval()
    pbar = tqdm(total=len(test_loader), position=0, leave=True)
    loss_sum = 0
    accuracy_sum = 0
    for batch_idx, input in enumerate(test_loader, start=1):
        if opt_params["wild_data"]:
            data, target, metadata = input
            data, target = data.to(device), target.to(device)
            out = model(data)
            loss = criterion_summed(out, target)
            physical_batch_size = data.shape[0]
        elif opt_params["cub_data"]:
            data, target, group = input
            data, target, group = data.to(device), target.to(device), group.to(device)
            out = model(data)
            loss = criterion_summed(out, target, group, False)
            physical_batch_size = data.shape[0]
        elif not opt_params["hf_model"]:
            data, target = input
            data, target = data.to(device), target.to(device)
            out = model(data)
            loss = criterion_summed(out, target)
            physical_batch_size = data.shape[0]
        else:
            dict_to_(input, device)
            target = input["labels"]
            output = model(**input)
            physical_batch_size = output.logits.shape[0]
            loss, out = output.loss * physical_batch_size, output.logits
            

        if compute_acc:
            if out.dim() > 1:
                if out.shape != target.shape:
                    accuracy = torch.sum((torch.argmax(out,dim=1)==target).float()).item()
                else:
                    accuracy = torch.sum((torch.argmax(out,dim=1)==torch.argmax(target,dim=1)).float()).item()
            else:
                accuracy = torch.sum((out*target > 0).float()).item()
        pbar.update(1)
        pbar.set_description(
            'Test  [{}/{} ({:.0f}%)] '
            'Batch Loss: {:.6f} '
            'Batch Accuracy: {:.6f}'.format(
                batch_idx,
                len(test_loader),
                100. * batch_idx / len(test_loader),
                (loss / physical_batch_size).item(),
                accuracy / physical_batch_size))
        loss_sum += loss.item()
        accuracy_sum += accuracy

    #print(loss_sum / len(test_loader.dataset))
    #print(out[:10])
    #print(target[:10])
    
    pbar.close()
    graphs.test_loss.append(loss_sum / len(test_loader.dataset))
    graphs.test_accuracy.append(accuracy_sum / len(test_loader.dataset))
    
    print("Mean Train Loss: {} \t Accuarcy: {}".format(graphs.loss[-1], graphs.accuracy[-1]))
    print("Mean Test Loss: {} \t Accuarcy: {}".format(graphs.test_loss[-1], graphs.test_accuracy[-1]))

    enable_running_stats(model)

from transformers.trainer_pt_utils import LabelSmoother

def compute_loss_hf(graphs, model, criterion_summed, loader_abridged, test_loader):
    disable_running_stats(model)
    loss_sum = 0
    accuracy_sum = 0

    for batch_idx, inputs in enumerate(loader_abridged, start=1):
        #inputs keys ['labels', 'input_ids', 'token_type_ids', 'attention_mask']
        outputs = model(**inputs)
        loss = outputs['loss']
        metrics = criterion_summed(torch.argmax(outputs['logits'],dim=1), inputs['labels']) #keys accuracy f1

        loss_sum += loss.item()
        accuracy_sum += metrics['accuracy']

    graphs.loss.append(loss_sum / len(loader_abridged.dataset))
    graphs.accuracy.append(accuracy_sum / len(loader_abridged.dataset))

    model.eval()
    pbar = tqdm(total=len(test_loader), position=0, leave=True)
    loss_sum = 0
    accuracy_sum = 0
    for batch_idx, inputs in enumerate(test_loader, start=1):
        outputs = model(**inputs)
        loss = outputs['loss']
        metrics = criterion_summed(torch.argmax(outputs['logits'],dim=1), inputs['labels']) #keys accuracy f1
        accuracy = metrics['accuracy']

        pbar.update(1)
        pbar.set_description(
            'Test\t\t [{}/{} ({:.0f}%)] \t'
            'Batch Loss: {:.6f} \t'
            'Batch Accuracy: {:.6f}'.format(
                batch_idx,
                len(test_loader),
                100. * batch_idx / len(test_loader),
                (loss / inputs['labels'].shape[0]).item(),
                accuracy / inputs['labels'].shape[0]))
        loss_sum += loss.item()
        accuracy_sum += accuracy
    pbar.close()
    graphs.test_loss.append(loss_sum / len(test_loader.dataset))
    graphs.test_accuracy.append(accuracy_sum / len(test_loader.dataset))
    print("Mean Test Loss: {} \t Accuarcy: {}".format(graphs.test_loss[-1], graphs.test_accuracy[-1]))
    enable_running_stats(model)
