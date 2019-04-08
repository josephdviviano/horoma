from time import time

import numpy as np
from torch.utils.data import DataLoader

from utils.dataset import SplitDataset
from trainer.basetrainer import BaseTrainer
from time import time

import numpy as np
from sklearn.cluster import KMeans
import torch
from torch.nn import functional

from trainer.trainer import Trainer


class IIC_unsupervised(BaseTrainer):
      
    def __init__(self, model, optimizer, resume, config, unlabelled, labelled, helios_run, 
                  experiment_folder=None, n_clusters=20, kmeans_interval=0, kmeans_headstart=0, kmeans_weight=1):

        super(IIC_unsupervised, self).__init__(self, model, optimizer, resume, config, unlabelled, labelled,
                 helios_run, experiment_folder=None):

        self.kmeans = KMeans(n_clusters=n_clusters)
        self.kmeans_interval = kmeans_interval
        self.kmeans_headstart = kmeans_headstart
        self.kmeans_weight = kmeans_weight

        ############################################
        #    Splitting into training/validation    #
        ############################################

        # Splitting 9:1 by default
        split = config['data']['dataloader'].get('split', .9)

        splitter = SplitDataset(split)

        train_set, valid_set = splitter(unlabelled)

        ############################################
        #  Creating the corresponding dataloaders  #
        ############################################

        train_loader = DataLoader(
            dataset=train_set,
            **config['data']['dataloader']['train'],
            pin_memory=True
        )

        valid_loader = DataLoader(
            dataset=valid_set,
            **config['data']['dataloader']['valid'],
            pin_memory=True
        )

        print(
            '>> Total batch number for training: {}'.format(len(train_loader)))
        print('>> Total batch number for validation: {}'.format(
            len(valid_loader)))
        print()

        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.log_step = int(np.sqrt(len(train_loader)))

        self.kmeans = KMeans(n_clusters=n_clusters)
        self.kmeans_interval = kmeans_interval
        self.kmeans_headstart = kmeans_headstart
        self.kmeans_weight = kmeans_weight

    def _fit_kmeans(self):
        """
        Train the kmeans.

        :return:
        """
        embeddings = []

        self.model.eval()

        with torch.no_grad():
            for batch_idx, (data) in enumerate(self.train_loader):
                data = data.to(self.device)

                z = self.model.encode(data).cpu()

                embeddings.append(z.detach().numpy())

            embeddings = np.concatenate(embeddings)

            self.kmeans.fit(embeddings)

    def _train_epoch_kmeans(self, epoch):
        """
        Add the distance to the centroid in the loss.

        :param epoch: Current training epoch.
        :return: the loss for this epoch
        """

        self._fit_kmeans()

        self.model.train()

        total_loss = 0
        total_kmeans_loss = 0
        total_model_loss = 0

        self.logger.info('K-Means Train Epoch: {}'.format(epoch))

        for batch_idx, (data) in enumerate(self.train_loader):
            start_it = time()
            data = data.to(self.device)

            self.optimizer.zero_grad()

            output = self.model(data)
            z = self.model.encode(data)

            if isinstance(output, tuple):
                model_loss = self.model.loss(data, *output)
            else:
                model_loss = self.model.loss(data, output)

            centroids = torch.tensor(self.kmeans.cluster_centers_).to(
                self.device)
            clusters = torch.tensor(self.kmeans.predict(z.cpu().detach()),
                                    dtype=torch.long).to(self.device)

            closest_centroids = torch.index_select(centroids, 0, clusters)

            kmeans_loss = functional.mse_loss(z, closest_centroids)

            loss = model_loss + kmeans_loss * self.kmeans_weight

            loss.backward()
            self.optimizer.step()

            step = epoch * len(self.train_loader) + batch_idx
            self.tb_writer.add_scalar('train/loss', loss.item(), step)

            total_loss += loss.item()
            total_kmeans_loss += kmeans_loss.item()
            total_model_loss += model_loss.item()

            end_it = time()
            time_it = end_it - start_it
            if batch_idx % self.log_step == 0:
                self.logger.info(
                    '   > [{}/{} ({:.0f}%), {:.2f}s] '
                    'Loss: {:.6f} ({:.3f} + {:.3f} x {:.1f})'.format(
                        batch_idx * self.train_loader.batch_size + data.size(
                            0),
                        len(self.train_loader.dataset),
                        100.0 * batch_idx / len(self.train_loader),
                        time_it * (len(self.train_loader) - batch_idx),
                        loss.item(),
                        model_loss.item(),
                        kmeans_loss.item(),
                        self.kmeans_weight
                    ))
                # grid = make_grid(data.cpu(), nrow=8, normalize=True)
                # self.tb_writer.add_image('input', grid, step)

        self.logger.info(
            '   > Total loss: {:.6f} ({:.3f} + {:.3f} x {:.1f})'.format(
                total_loss / len(self.train_loader),
                total_model_loss / len(self.train_loader),
                total_kmeans_loss / len(self.train_loader),
                self.kmeans_weight
            ))

        # We return the model loss for coherence
        return total_model_loss / len(self.train_loader)

    def train(self):
        """
        Full training logic for the kmeanstrainer
        """

        t0 = time()

        for epoch in range(self.start_epoch, self.epochs):

            if epoch % (
                    self.kmeans_interval + 1) == 0 \
                    and epoch >= self.kmeans_headstart:
                train_loss = self._train_epoch_kmeans(epoch)
            else:
                train_loss = self._train_epoch(epoch)

            valid_loss = self._valid_epoch(epoch)

            self.tb_writer.add_scalar("train/epoch_loss", train_loss,
                                      epoch)
            self.tb_writer.add_scalar("valid/epoch_loss", valid_loss,
                                      epoch)

            self._save_checkpoint(epoch, train_loss, valid_loss)

            time_elapsed = time() - t0

            # Break the loop if there is no more time left
            if time_elapsed * (1 + 1 / (
                    epoch - self.start_epoch + 1)) > .95 \
                    * self.wall_time * 3600:
                break

        # Save the checkpoint if it's not already done.
        if not epoch % self.save_period == 0:
            self._save_checkpoint(epoch, train_loss, valid_loss)

        
dataloaders_head_A, dataloaders_head_B, mapping_assignment_dataloader, \
mapping_test_dataloader = cluster_twohead_create_dataloaders(config)

net = archs.__dict__[config.arch](config)
if config.restart:
  model_path = os.path.join(config.out_dir, net_name)
  net.load_state_dict(
    torch.load(model_path, map_location=lambda storage, loc: storage))

net.cuda()
net = torch.nn.DataParallel(net)
net.train()

optimiser = get_opt(config.opt)(net.module.parameters(), lr=config.lr)
if config.restart:
  opt_path = os.path.join(config.out_dir, opt_name)
  optimiser.load_state_dict(torch.load(opt_path))

heads = ["B", "A"]
if config.head_A_first:
  heads = ["A", "B"]

head_epochs = {}
head_epochs["A"] = config.head_A_epochs
head_epochs["B"] = config.head_B_epochs

# Results ----------------------------------------------------------------------

if config.restart:
  if not config.restart_from_best:
    next_epoch = config.last_epoch + 1  # corresponds to last saved model
  else:
    # sanity check
    next_epoch = np.argmax(np.array(config.epoch_acc)) + 1
    assert (next_epoch == config.last_epoch + 1)
  print("starting from epoch %d" % next_epoch)

  config.epoch_acc = config.epoch_acc[:next_epoch]  # in case we overshot
  config.epoch_avg_subhead_acc = config.epoch_avg_subhead_acc[:next_epoch]
  config.epoch_stats = config.epoch_stats[:next_epoch]

  if config.double_eval:
    config.double_eval_acc = config.double_eval_acc[:next_epoch]
    config.double_eval_avg_subhead_acc = config.double_eval_avg_subhead_acc[
                                         :next_epoch]
    config.double_eval_stats = config.double_eval_stats[:next_epoch]

  config.epoch_loss_head_A = config.epoch_loss_head_A[:(next_epoch - 1)]
  config.epoch_loss_no_lamb_head_A = config.epoch_loss_no_lamb_head_A[
                                     :(next_epoch - 1)]
  config.epoch_loss_head_B = config.epoch_loss_head_B[:(next_epoch - 1)]
  config.epoch_loss_no_lamb_head_B = config.epoch_loss_no_lamb_head_B[
                                     :(next_epoch - 1)]
else:
  config.epoch_acc = []
  config.epoch_avg_subhead_acc = []
  config.epoch_stats = []

  if config.double_eval:
    config.double_eval_acc = []
    config.double_eval_avg_subhead_acc = []
    config.double_eval_stats = []

  config.epoch_loss_head_A = []
  config.epoch_loss_no_lamb_head_A = []

  config.epoch_loss_head_B = []
  config.epoch_loss_no_lamb_head_B = []

  _ = cluster_eval(config, net,
                   mapping_assignment_dataloader=mapping_assignment_dataloader,
                   mapping_test_dataloader=mapping_test_dataloader,
                   sobel=True)

  print("Pre: time %s: \n %s" % (datetime.now(), nice(config.epoch_stats[-1])))
  if config.double_eval:
    print("double eval: \n %s" % (nice(config.double_eval_stats[-1])))
  sys.stdout.flush()
  next_epoch = 1

fig, axarr = plt.subplots(6 + 2 * int(config.double_eval), sharex=False,
                          figsize=(20, 20))

# Train ------------------------------------------------------------------------

for e_i in range(next_epoch, config.num_epochs):
  print("Starting e_i: %d" % (e_i))

  if e_i in config.lr_schedule:
    optimiser = update_lr(optimiser, lr_mult=config.lr_mult)

  for head_i in range(2):
    head = heads[head_i]
    if head == "A":
      dataloaders = dataloaders_head_A
      epoch_loss = config.epoch_loss_head_A
      epoch_loss_no_lamb = config.epoch_loss_no_lamb_head_A
    elif head == "B":
      dataloaders = dataloaders_head_B
      epoch_loss = config.epoch_loss_head_B
      epoch_loss_no_lamb = config.epoch_loss_no_lamb_head_B

    avg_loss = 0.  # over heads and head_epochs (and sub_heads)
    avg_loss_no_lamb = 0.
    avg_loss_count = 0

    for head_i_epoch in range(head_epochs[head]):
      sys.stdout.flush()

      iterators = (d for d in dataloaders)

      b_i = 0
      for tup in zip(*iterators):
        net.module.zero_grad()

        # one less because this is before sobel
        all_imgs = torch.zeros(config.batch_sz, config.in_channels - 1,
                               config.input_sz,
                               config.input_sz).cuda()
        all_imgs_tf = torch.zeros(config.batch_sz, config.in_channels - 1,
                                  config.input_sz,
                                  config.input_sz).cuda()
        
        if isistance(tup[0],list): # always the first
          imgs_curr = tup[0][0] 
        else:
          imgs_curr = tup[0]

        curr_batch_sz = imgs_curr.size(0)
        for d_i in range(config.num_dataloaders):
          if isinstance(tup[1 + d_i],list):# from 2nd to last
            imgs_tf_curr = tup[1 + d_i][0]
          else:
            imgs_tf_curr = tup[1 + d_i]
          assert (curr_batch_sz == imgs_tf_curr.size(0))

          actual_batch_start = d_i * curr_batch_sz
          actual_batch_end = actual_batch_start + curr_batch_sz
          all_imgs[actual_batch_start:actual_batch_end, :, :, :] = \
            imgs_curr.cuda()
          all_imgs_tf[actual_batch_start:actual_batch_end, :, :, :] = \
            imgs_tf_curr.cuda()

        if not (curr_batch_sz == config.dataloader_batch_sz):
          print("last batch sz %d" % curr_batch_sz)

        curr_total_batch_sz = curr_batch_sz * config.num_dataloaders
        all_imgs = all_imgs[:curr_total_batch_sz, :, :, :]
        all_imgs_tf = all_imgs_tf[:curr_total_batch_sz, :, :, :]

        all_imgs = sobel_process(all_imgs, config.include_rgb)
        all_imgs_tf = sobel_process(all_imgs_tf, config.include_rgb)

        x_outs = net(all_imgs, head=head)
        x_tf_outs = net(all_imgs_tf, head=head)

        avg_loss_batch = None  # avg over the sub_heads
        avg_loss_no_lamb_batch = None
        for i in range(config.num_sub_heads):
          loss, loss_no_lamb = IID_loss(x_outs[i], x_tf_outs[i],
                                        lamb=config.lamb)
          if avg_loss_batch is None:
            avg_loss_batch = loss
            avg_loss_no_lamb_batch = loss_no_lamb
          else:
            avg_loss_batch += loss
            avg_loss_no_lamb_batch += loss_no_lamb

        avg_loss_batch /= config.num_sub_heads
        avg_loss_no_lamb_batch /= config.num_sub_heads

        if ((b_i % 100) == 0) or (e_i == next_epoch and b_i < 10):
          print("Model ind %d epoch %d head %s head_i_epoch %d batch %d: avg "
                "loss %f avg loss no lamb %f time %s" % \
                (config.model_ind, e_i, head, head_i_epoch, b_i,
                 avg_loss_batch.item(), avg_loss_no_lamb_batch.item(),
                 datetime.now()))
          sys.stdout.flush()

        if not np.isfinite(avg_loss_batch.item()):
          print("Loss is not finite... %s:" % str(avg_loss_batch))
          exit(1)

        avg_loss += avg_loss_batch.item()
        avg_loss_no_lamb += avg_loss_no_lamb_batch.item()
        avg_loss_count += 1

        avg_loss_batch.backward()
        optimiser.step()

        b_i += 1
        if b_i == 2 and config.test_code:
          break

    avg_loss = float(avg_loss / avg_loss_count)
    avg_loss_no_lamb = float(avg_loss_no_lamb / avg_loss_count)

    epoch_loss.append(avg_loss)
    epoch_loss_no_lamb.append(avg_loss_no_lamb)

  # Eval -----------------------------------------------------------------------

  is_best = cluster_eval(config, net,
                         mapping_assignment_dataloader=mapping_assignment_dataloader,
                         mapping_test_dataloader=mapping_test_dataloader,
                         sobel=True)

  print("Pre: time %s: \n %s" % (datetime.now(), nice(config.epoch_stats[-1])))
  if config.double_eval:
    print("     double eval: \n %s" % (nice(config.double_eval_stats[-1])))
  sys.stdout.flush()

  axarr[0].clear()
  axarr[0].plot(config.epoch_acc)
  axarr[0].set_title("acc (best), top: %f" % max(config.epoch_acc))

  axarr[1].clear()
  axarr[1].plot(config.epoch_avg_subhead_acc)
  axarr[1].set_title("acc (avg), top: %f" % max(config.epoch_avg_subhead_acc))

  axarr[2].clear()
  axarr[2].plot(config.epoch_loss_head_A)
  axarr[2].set_title("Loss head A")

  axarr[3].clear()
  axarr[3].plot(config.epoch_loss_no_lamb_head_A)
  axarr[3].set_title("Loss no lamb head A")

  axarr[4].clear()
  axarr[4].plot(config.epoch_loss_head_B)
  axarr[4].set_title("Loss head B")

  axarr[5].clear()
  axarr[5].plot(config.epoch_loss_no_lamb_head_B)
  axarr[5].set_title("Loss no lamb head B")

  if config.double_eval:
    axarr[6].clear()
    axarr[6].plot(config.double_eval_acc)
    axarr[6].set_title("double eval acc (best), top: %f" %
                       max(config.double_eval_acc))

    axarr[7].clear()
    axarr[7].plot(config.double_eval_avg_subhead_acc)
    axarr[7].set_title("double eval acc (avg), top: %f" %
                       max(config.double_eval_avg_subhead_acc))

  fig.tight_layout()
  fig.canvas.draw_idle()
  fig.savefig(os.path.join(config.out_dir, "plots.png"))

  if is_best or (e_i % config.save_freq == 0):
    net.module.cpu()

    if e_i % config.save_freq == 0:
      torch.save(net.module.state_dict(),
                 os.path.join(config.out_dir, "latest_net.pytorch"))
      torch.save(optimiser.state_dict(),
                 os.path.join(config.out_dir, "latest_optimiser.pytorch"))
      config.last_epoch = e_i  # for last saved version

    if is_best:
      torch.save(net.module.state_dict(),
                 os.path.join(config.out_dir, "best_net.pytorch"))
      torch.save(optimiser.state_dict(),
                 os.path.join(config.out_dir, "best_optimiser.pytorch"))
      with open(os.path.join(config.out_dir, "best_config.pickle"),
                'wb') as outfile:
        pickle.dump(config, outfile)

      with open(os.path.join(config.out_dir, "best_config.txt"),
                "w") as text_file:
        text_file.write("%s" % config)

    net.module.cuda()

  with open(os.path.join(config.out_dir, "config.pickle"),
            'wb') as outfile:
    pickle.dump(config, outfile)

  with open(os.path.join(config.out_dir, "config.txt"),
            "w") as text_file:
    text_file.write("%s" % config)

  if config.test_code:
    exit(0)

    def _valid_epoch(self, epoch):
        """
        Validation logic for an epoch

        :param epoch: Current training epoch.
        :return: the loss for this epoch
        """
        self.model.eval()
        total_loss = 0

        self.logger.info('Valid Epoch: {}'.format(epoch))

        for batch_idx, (data) in enumerate(self.valid_loader):
            start_it = time()
            data = data.to(self.device)

            output = self.model(data)
            if isinstance(output, tuple):
                loss = self.model.loss(data, *output)
            else:
                loss = self.model.loss(data, output)

            step = epoch * len(self.valid_loader) + batch_idx
            self.tb_writer.add_scalar('valid/loss', loss.item(), step)

            total_loss += loss.item()

            end_it = time()
            time_it = end_it - start_it
            if batch_idx % self.log_step == 0:
                self.logger.info(
                    '   > [{}/{} ({:.0f}%), {:.2f}s] Loss: {:.6f} '.format(
                        batch_idx * self.valid_loader.batch_size + data.size(
                            0),
                        len(self.valid_loader.dataset),
                        100.0 * batch_idx / len(self.valid_loader),
                        time_it * (len(self.valid_loader) - batch_idx),
                        loss.item()))
                # grid = make_grid(data.cpu(), nrow=8, normalize=True)
                # self.tb_writer.add_image('input', grid, step)

        self.logger.info('   > Total loss: {:.6f}'.format(
            total_loss / len(self.valid_loader)
        ))

        return total_loss / len(self.valid_loader)
