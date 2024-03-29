from optimizer.sam import disable_running_stats, enable_running_stats
import torch
from tqdm import tqdm
import torch.nn.functional as F
import sys
import numpy as np

def compute_loss(graphs, model, loss_name, criterion, criterion_summed, device, num_classes, loader_abridged, test_loader, compute_acc=False, compute_model_output=False):
    disable_running_stats(model)
    loss_sum = 0
    accuracy_sum = 0
    accuracy = 0

    model_output = []
    for batch_idx, (data, target) in enumerate(loader_abridged, start=1):
        data, target = data.to(device), target.to(device)
        out = model(data)
        loss = criterion_summed(out, target)
        """
        if loss_name == 'BCELoss':
            accuracy = torch.sum(out*target > 0)
        elif out.dim() > 1:
            accuracy = torch.sum((torch.argmax(out,dim=1)==target).float()).item()
        """
        if compute_acc:
            if out.dim() > 1:
                accuracy = torch.sum((torch.argmax(out,dim=1)==target).float()).item()
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
    for batch_idx, (data, target) in enumerate(test_loader, start=1):
        data, target = data.to(device), target.to(device)
        out = model(data)
        loss = criterion_summed(out, target)
        """
        if loss_name == 'BCELoss':
            accuracy = torch.sum(out*target > 0).item()
        elif out.dim() > 1:
            accuracy = torch.sum((torch.argmax(out,dim=1)==target).float()).item()
            #wrong_index = torch.where((torch.argmax(out,dim=1)==target).float() == 0)[0]
        """
        if compute_acc:
            if out.dim() > 1:
                accuracy = torch.sum((torch.argmax(out,dim=1)==target).float()).item()
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
                (loss / data.shape[0]).item(),
                accuracy / data.shape[0]))
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
