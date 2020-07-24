# from model import *
# from plot import *
# import torch
# import logging
# from datetime import datetime
# from data import get_mnist_np, sample_minibatch_deterministically, preprocess, shuffle
#
# class classifier():
#
#     def __init__(self, args, data):
#         self.train_loader = data.train_loader
#         self.test_loader = data.test_loader
#         self.batch_size = args.batch_size
#         self.num_train = data.num_train
#         self.num_classes = data.num_classes
#         self.args = args
#         assert args.model_type == 'mlp' or args.model_type == 'cnn'
#         if args.model_type == 'mlp':
#             self.net = mlp(args.conditioned, data.input_dims, data.num_classes, hidden_size=256)
#
#         self.classificationCriterion = nn.CrossEntropyLoss()
#         self.syntheticCriterion = nn.MSELoss()
#         self.plot = args.plot
#         self.num_epochs = args.num_epochs
#         self.model_name = args.model_name
#         self.conditioned = args.conditioned
#         self.best_perf = 0.
#         self.stats = dict(grad_loss=[], classify_loss=[])
#         print("[%] model name will be", self.model_name)
#
#         logging.basicConfig(filename='./log_files/'+datetime.today().strftime('%Y_%m_%d_%H_%M_%S')+'.log', level=logging.DEBUG)
#         logging.info(args)
#
#     def optimizer_module(self, optimizer, forward, out, label_onehot=None, weight=0.0):
#         optimizer.zero_grad()
#         out, grad = forward(out, label_onehot)
#         out.backward(grad.detach().data*100000.0) # TODO: after the backward pass the fc1.weight stays zero
#         optimizer.step()
#         out = out.detach()
#         return out
#
#     def save_grad(self, name):
#         def hook(grad):
#             self.backprop_grads[name] = grad
#             #self.backprop_grads[name].volatile = False
#         return hook
#
#     def optimize_grad_and_net(self, images, labels, label_onehot, grad_optimizer, optimizer, forward):
#         # synthetic model
#         # Forward + Backward + Optimize
#         grad_optimizer.zero_grad()
#         optimizer.zero_grad()
#         outs, grads = forward(images, label_onehot)
#         self.backprop_grads = {}
#         handles = {}
#         keys = []
#         for i, (out, grad) in enumerate(zip(outs, grads)):
#             handles[str(i)] = out.register_hook(self.save_grad(str(i)))
#             keys.append(str(i))
#         outputs = outs[-1]
#         loss = self.classificationCriterion(outputs, labels)
#         loss.backward(retain_graph=True)
#         for (k, v) in handles.items():
#             v.remove()
#         grad_loss = 0.
#         for k in keys:
#             grad_loss += self.syntheticCriterion(grads[int(k)], self.backprop_grads[k].detach())
#
#         grad_loss.backward()
#         grad_optimizer.step()
#         optimizer.step()
#         self.stats['grad_loss'].append(grad_loss.item())
#         self.stats['classify_loss'].append(loss.item())
#         return loss, grad_loss
#
#
#     def unlabelled_weight_schedule(self, iteration, T1=72000, T2=250000, af=1.0):
#         # enough iterations for the syn grad to reach very high training accuracy
#         if iteration <= T1:
#             return 0.0
#         elif T1 < iteration < T2:
#             return ((iteration-T1) / (T2-T1)) * af
#         else:
#             return af
#
#     def train_model(self):
#         proportion_labeled = 0.1
#         assert proportion_labeled == 0.1
#
#         train_data_np, train_labels_np, test_data_np, test_labels_np = get_mnist_np(root='./data', download=True)
#         x_labeled, x_unlabelled, x_test, y_labeled, _, y_unlabelled, y_test = preprocess(train_data_np=train_data_np,
#                                                                                          train_labels_np=train_labels_np,
#                                                                                          test_data_np=test_data_np,
#                                                                                          test_labels_np=test_labels_np,
#                                                                                          proportion_labeled=proportion_labeled)
#         self.x_test = x_test
#         self.y_test = y_test
#
#
#         for i in range(self.args.num_iterations):
#             self.train_model_unsupervised(x=x_unlabelled, y=y_unlabelled, weight=self.unlabelled_weight_schedule(i))
#             loss, grad_loss = self.train_model_supervised(x=x_labeled, y=y_labeled)
#             # self.train_model_unsupervised(x=x_unlabelled, y=y_unlabelled, weight=self.unlabelled_weight_schedule(i))
#             if (i + 1) % 100 == 0:
#                 print('Iteration [%d/%d], Loss: %.6f, Grad Loss: %.8f'
#                       % (i + 1, self.args.num_iterations, loss.item(), grad_loss.item()))
#
#                 logging.info('Iteration [%d/%d], Loss: %.6f, Grad Loss: %.8f'
#                       % (i + 1, self.args.num_iterations, loss.item(), grad_loss.item()))
#
#                 if self.unlabelled_weight_schedule(i) != 0.0:
#                     print('Current synthetic gradient weigth is: %.4f' % (self.unlabelled_weight_schedule(i)))
#                     logging.info('Current synthetic gradient weigth is: %.4f' % (self.unlabelled_weight_schedule(i)))
#                 perf = self.test_model(i + 1)
#
#
#     def train_model_supervised(self, x, y):
#         # can replace the two lines below with sample_minibatch
#         x, y = shuffle(x=x, y=y)
#         x_l, y_l, _ = sample_minibatch_deterministically(x, y, batch_i=1, batch_size=self.args.batch_size)
#
#         labels_onehot = torch.zeros([y_l.size(0), self.num_classes])
#         labels_onehot.scatter_(1, y_l.long().unsqueeze(1), 1)
#         # Forward + Backward + Optimize
#         loss, grad_loss = self.optimize_grad_and_net(x_l, y_l.long(), labels_onehot,
#                                   self.net.grad_optimizer, self.net.optimizer, self.net)
#         return loss, grad_loss
#
#     def train_model_unsupervised(self, x, y, weight=0.0):
#         x, y = shuffle(x=x, y=y)
#         x_l, y_l, _ = sample_minibatch_deterministically(x, y, batch_i=1, batch_size=self.args.batch_size)
#
#         labels_onehot = torch.zeros([y_l.size(0), self.num_classes])
#         labels_onehot.scatter_(1, y_l.unsqueeze(1).long(), 1)
#         out = x_l
#         # Forward + Backward + Optimize
#         for (optimizer, forward) in zip(self.net.optimizers, self.net.forwards):
#             if self.conditioned:
#                 out = self.optimizer_module(optimizer, forward, out, labels_onehot, weight=weight)
#             else:
#                 out = self.optimizer_module(optimizer, forward, out, weight=weight)
#
#     def test_model(self, epoch):
#         # Test the Model
#         self.net.eval()
#         correct = 0
#         total = 0
#         with torch.no_grad():
#             for i in range(int(len(self.x_test) / self.args.batch_size)):
#                 images, labels, _ = sample_minibatch_deterministically(x=self.x_test, y=self.y_test, batch_i=i, batch_size=self.args.batch_size)
#                 outputs = self.net(images)
#                 outputs = outputs[-1]
#                 _, predicted = torch.max(outputs.data, 1)
#                 total += labels.size(0)
#                 correct += (predicted.cpu() == labels).sum()
#             perf = 100 * correct / total
#         print('Iteration %d: Accuracy of the network on the 10000 test images: %d %%' % (epoch, perf))
#         logging.info('Iteration %d: Accuracy of the network on the 10000 test images: %d %%' % (epoch, perf))
#         return perf