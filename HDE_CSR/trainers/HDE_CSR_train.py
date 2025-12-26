import torch
from torch.nn import NLLLoss


def default_train(train_loader, model, optimizer, writer, iter_counter, args):

    way = model.way
    query_shot = model.shots[-1]
    target = torch.LongTensor([i // query_shot for i in range(query_shot * way)]).cuda()
    criterion = NLLLoss().cuda()

    lr = optimizer.param_groups[0]['lr']
    writer.add_scalar('lr', lr, iter_counter)


    avg_loss = 0
    avg_acc = 0

    for i, (inp, _) in enumerate(train_loader):
        iter_counter += 1

        inp = inp.cuda()
        log_prediction_h, log_prediction_d = model(inp)

        alpha = args.alpha

        loss_h = criterion(log_prediction_h, target)
        loss_d = criterion(log_prediction_d, target)

        loss_total = ((loss_h + loss_d)/2)*2

        optimizer.zero_grad()
        loss_total.backward()
        optimizer.step()

        loss_value = loss_total.item()

        log_prediction = 0.5*log_prediction_h + 0.5*log_prediction_d
        _, max_index = torch.max(log_prediction, 1)
        acc = 100 * torch.sum(torch.eq(max_index, target)).item() / query_shot / way

        avg_acc += acc
        avg_loss += loss_value

    avg_acc = avg_acc / (i + 1)
    avg_loss = avg_loss / (i + 1)

    writer.add_scalar('HDE_CSR_loss', avg_loss, iter_counter)
    writer.add_scalar('train_acc', avg_acc, iter_counter)

    return iter_counter, avg_acc, avg_loss
