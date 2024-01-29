from optimizer.sam import disable_running_stats, enable_running_stats
import torch
from tqdm import tqdm
import torch.nn.functional as F

def compute_loss(graphs, model, loss_name, criterion, criterion_summed, device, num_classes, loader_abridged, test_loader):
    disable_running_stats(model)
    loss_sum = 0
    accuracy_sum = 0
    accuracy = 0
    for batch_idx, (data, target) in enumerate(loader_abridged, start=1):
        data, target = data.to(device), target.to(device)
        out = model(data)
        if str(loss_name) == 'CrossEntropyLoss':
            loss = criterion(out, target)
        elif str(loss_name) == 'MSELoss':
            #print(out[0], target[0])
            #loss = criterion(out, F.one_hot(target, num_classes=num_classes).float()) * num_classes
            #if transform_to_one_hot:
            #    loss = criterion_summed(out, F.one_hot(target, num_classes=num_classes).float()).float()
            #else:
            loss = criterion_summed(out, target)
        if out.dim() > 1:
            accuracy = torch.sum((torch.argmax(out,dim=1)==target).float()).item()
        loss_sum += loss.item()
        accuracy_sum += accuracy
    graphs.loss.append(loss_sum / len(loader_abridged.dataset))
    graphs.accuracy.append(accuracy_sum / len(loader_abridged.dataset))

    model.eval()
    pbar = tqdm(total=len(test_loader), position=0, leave=True)
    loss_sum = 0
    accuracy_sum = 0
    for batch_idx, (data, target) in enumerate(test_loader, start=1):
        data, target = data.to(device), target.to(device)
        out = model(data)
        if str(criterion) == 'CrossEntropyLoss':
            loss = criterion(out, target)
        elif str(criterion) == 'MSELoss':
            #print(out[0], target[0])
            #loss = criterion(out, F.one_hot(target, num_classes=num_classes).float()) * num_classes
            #if transform_to_one_hot:
            #    loss = criterion_summed(out, F.one_hot(target, num_classes=num_classes).float()).float()
            #else:
            loss = criterion_summed(out, target)
        if out.dim() > 1:
            accuracy = torch.sum((torch.argmax(out,dim=1)==target).float()).item()

        pbar.update(1)
        pbar.set_description(
            'Test\t\t [{}/{} ({:.0f}%)] \t'
            'Batch Loss: {:.6f} \t'
            'Batch Accuracy: {:.6f}'.format(
                batch_idx,
                len(test_loader),
                100. * batch_idx / len(test_loader),
                (loss / data.shape[0]).item(),
                accuracy / data.shape[0]))
        loss_sum += loss.item()
        accuracy_sum += accuracy
    pbar.close()
    graphs.test_loss.append(loss_sum / len(test_loader.dataset))
    graphs.test_accuracy.append(accuracy_sum / len(test_loader.dataset))
    print("Mean Test Loss: {} \t Accuarcy: {}".format(graphs.test_loss[-1], graphs.test_accuracy[-1]))

    enable_running_stats(model)
