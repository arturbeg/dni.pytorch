from model import *
from plot import *
import torch
import logging
from datetime import datetime

class classifier():

    def __init__(self, args, data):
        self.train_loader = data.train_loader
        self.test_loader = data.test_loader
        self.batch_size = args.batch_size
        self.num_train = data.num_train
        self.num_classes = data.num_classes
        assert args.model_type == 'mlp' or args.model_type == 'cnn'
        if args.model_type == 'mlp':
            self.net = mlp(args.conditioned, data.input_dims, data.num_classes, hidden_size=256)

        self.classificationCriterion = nn.CrossEntropyLoss()
        self.syntheticCriterion = nn.MSELoss()
        self.plot = args.plot
        self.num_epochs = args.num_epochs
        self.model_name = args.model_name
        self.conditioned = args.conditioned
        self.best_perf = 0.
        self.stats = dict(grad_loss=[], classify_loss=[])
        print("[%] model name will be", self.model_name)

        logging.basicConfig(filename='./log_files/'+datetime.today().strftime('%Y_%m_%d_%H_%M_%S')+'.log', level=logging.DEBUG)
        logging.info(args)

    def optimizer_module(self, optimizer, forward, out, label_onehot=None):
        optimizer.zero_grad()
        out, grad = forward(out, label_onehot)
        out.backward(grad.detach().data)
        optimizer.step()
        out = out.detach()
        return out

    def save_grad(self, name):
        def hook(grad):
            self.backprop_grads[name] = grad
            #self.backprop_grads[name].volatile = False
        return hook

    def optimizer_dni_module(self, images, labels, label_onehot, grad_optimizer, optimizer, forward):
        # synthetic model
        # Forward + Backward + Optimize
        grad_optimizer.zero_grad()
        optimizer.zero_grad()
        outs, grads = forward(images, label_onehot)
        self.backprop_grads = {}
        handles = {}
        keys = []
        for i, (out, grad) in enumerate(zip(outs, grads)):
            handles[str(i)] = out.register_hook(self.save_grad(str(i)))
            keys.append(str(i))
        outputs = outs[-1]
        loss = self.classificationCriterion(outputs, labels)
        loss.backward(retain_graph=True)
        for (k, v) in handles.items():
            v.remove()
        grad_loss = 0.
        for k in keys:
            grad_loss += self.syntheticCriterion(grads[int(k)], self.backprop_grads[k].detach())

        grad_loss.backward()
        grad_optimizer.step()
        self.stats['grad_loss'].append(grad_loss.item())
        self.stats['classify_loss'].append(loss.item())
        return loss, grad_loss

    def train_model(self):
        for epoch in range(self.num_epochs):
            for i, (images, labels) in enumerate(self.train_loader):
                # Convert torch tensor to Variable
                labels_onehot = torch.zeros([labels.size(0), self.num_classes])
                labels_onehot.scatter_(1, labels.unsqueeze(1), 1)
                out = images
                # Forward + Backward + Optimize
                for (optimizer, forward) in zip(self.net.optimizers, self.net.forwards):
                    if self.conditioned:
                        out = self.optimizer_module(optimizer, forward, out, labels_onehot)
                    else:
                        out = self.optimizer_module(optimizer, forward, out)
                # synthetic model
                # Forward + Backward + Optimize
                loss, grad_loss = self.optimizer_dni_module(images, labels, labels_onehot,
                                          self.net.grad_optimizer, self.net.optimizer, self.net)

                if (i+1) % 100 == 0:
                    print ('Epoch [%d/%d], Step [%d/%d], Loss: %.4f, Grad Loss: %.4f'
                         %(epoch+1, self.num_epochs, i+1, self.num_train//self.batch_size, loss.item(), grad_loss.item()))

                    logging.info('Epoch [%d/%d], Step [%d/%d], Loss: %.4f, Grad Loss: %.4f'
                         %(epoch+1, self.num_epochs, i+1, self.num_train//self.batch_size, loss.item(), grad_loss.item()))

            if (epoch+1) % 10 == 0:
                perf = self.test_model(epoch+1)
                if perf > self.best_perf:
                    torch.save(self.net.state_dict(), self.model_name+'_model_best.pkl')
                    self.net.train()

        # Save the Model ans Stats
        pkl.dump(self.stats, open(self.model_name+'_stats.pkl', 'wb'))
        torch.save(self.net.state_dict(), self.model_name+'_model.pkl')
        if self.plot:
            plot(self.stats, name=self.model_name)

    def test_model(self, epoch):
        # Test the Model
        self.net.eval()
        correct = 0
        total = 0
        for images, labels in self.test_loader:
            outputs = self.net(images)
            outputs = outputs[-1]
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted.cpu() == labels).sum()
        perf = 100 * correct / total
        print('Epoch %d: Accuracy of the network on the 10000 test images: %d %%' % (epoch, perf))
        logging.info('Epoch %d: Accuracy of the network on the 10000 test images: %d %%' % (epoch, perf))
        return perf