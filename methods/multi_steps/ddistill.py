import collections
import copy
import logging
import os
import pickle

import numpy as np
import torch
from scipy.spatial.distance import cdist
from torch import nn
from scipy.special import softmax
from torch.nn import functional as F
from tqdm import tqdm

from inclearn.lib import factory, herding, losses, network, schedulers, utils
# from inclearn.lib.classifiers import Classifier

from inclearn.lib.network import hook
from inclearn.lib import metrics, results_utils, utils
from inclearn.models.base import IncrementalLearner
import sklearn.metrics as sk
from scipy.special import softmax

EPSILON = 1e-8

logger = logging.getLogger(__name__)

# cifar100

# # expert_loss_part2_weight = 0.5
# expert_loss_part2_weight = 0.0
# expert_scheduling = "cosine"
# # Misc
# expert_epochs = 250
# expert_lr = 0.1
# expert_lr_decay = 0.1
# expert_optimizer = "sgd"
# expert_proxy_per_class = 1
# expert_weight_decay = 0.0005


# skin

# expert_loss_part2_weight = 0.5
expert_loss_part2_weight = 0.0
expert_scheduling = "cosine"
# Misc
expert_epochs = 250

expert_lr = 0.1
expert_lr_decay = 0.1
expert_optimizer = "sgd"
expert_proxy_per_class = 1
expert_weight_decay = 0.0005


old_features_weight = 0.0
expert_features_weight = 0.0
denominator = 1.0


class DDistill(IncrementalLearner):
    """Implementation of dual distill
    """

    def __init__(self, args):
        super().__init__()

        print("create ddistill here !!")

        nowdate = utils.get_date()
        folder = results_utils.get_save_folder(args["model"], nowdate, args["label"])
        filename = "detail.log"
        handler = logging.FileHandler(filename=os.path.join(folder, filename), mode='w')

        formatter = logging.Formatter(fmt='%(asctime)s [%(filename)s]: %(message)s', datefmt='%Y-%m-%d:%H:%M:%S')
        handler.setFormatter(formatter)

        logger.addHandler(handler)
        self.args = args
        self._disable_progressbar = args.get("no_progressbar", False)

        self._device = args["device"][0]
        self._multiple_devices = args["device"]

        self._opt_name = args["optimizer"]
        self._lr = args["lr"]
        self._weight_decay = args["weight_decay"]
        self._n_epochs = args["epochs"]

        self._scheduling = args["scheduling"]
        self._lr_decay = args["lr_decay"]

        self._warmup_config = args.get("warmup", {})
        if self._warmup_config and self._warmup_config["total_epoch"] > 0:
            self._lr /= self._warmup_config["multiplier"]

        self._eval_every_x_epochs = args.get("eval_every_x_epochs")
        self._early_stopping = args.get("early_stopping", {})

        self._memory_size = args["memory_size"]
        self._fixed_memory = args["fixed_memory"]
        self._herding_selection = args.get("herding_selection", {"type": "icarl"})
        self._n_classes = 0
        self._last_results = None
        self._validation_percent = args["validation"]

        self._rotations_config = args.get("rotations_config", {})
        self._random_noise_config = args.get("random_noise_config", {})

        self._network = network.BasicNet(
            args["convnet"],
            convnet_kwargs=args.get("convnet_config", {}),
            classifier_kwargs=args.get("classifier_config", {
                "type": "fc",
                "use_bias": True
            }),
            device=self._device,
            extract_no_act=True,
            classifier_no_act=False,
            rotations_predictor=bool(self._rotations_config)
        )

        # debug
        # self._expert_network = network.BasicNet(
        #     args["convnet"],
        #     convnet_kwargs=args.get("convnet_config", {}),
        #     classifier_kwargs=args.get("classifier_config", {
        #         "type": "fc",
        #         "use_bias": True
        #     }),
        #     device=self._device,
        #     extract_no_act=True,
        #     classifier_no_act=False,
        #     rotations_predictor=bool(self._rotations_config)
        # )

        self._temperature = args["temperature"]
        self.lambda1 = args["lambda1"]
        self.lambda2 = args["lambda2"]

        self._expert_network = None

        self._examplars = {}
        self._means = None
        self._herding_indexes = []
        self._data_memory, self._targets_memory = None, None

        self._old_model = None

        # self._clf_loss = F.binary_cross_entropy_with_logits
        # self._distil_loss = F.binary_cross_entropy_with_logits

        self._epoch_metrics = collections.defaultdict(list)

        self._meta_transfer = args.get("meta_transfer", {})

    def set_meta_transfer(self):
        if self._meta_transfer["type"] not in ("repeat", "once", "none"):
            raise ValueError(f"Invalid value for meta-transfer {self._meta_transfer}.")

        if self._task == 0:
            self._network.convnet.apply_mtl(False)
        elif self._task == 1:
            if self._meta_transfer["type"] != "none":
                self._network.convnet.apply_mtl(True)

            if self._meta_transfer.get("mtl_bias"):
                self._network.convnet.apply_mtl_bias(True)
            elif self._meta_transfer.get("bias_on_weight"):
                self._network.convnet.apply_bias_on_weights(True)

            if self._meta_transfer["freeze_convnet"]:
                self._network.convnet.freeze_convnet(
                    True,
                    bn_weights=self._meta_transfer.get("freeze_bn_weights"),
                    bn_stats=self._meta_transfer.get("freeze_bn_stats")
                )
        elif self._meta_transfer["type"] != "none":
            if self._meta_transfer["type"] == "repeat" or (
                    self._task == 2 and self._meta_transfer["type"] == "once"
            ):
                self._network.convnet.fuse_mtl_weights()
                self._network.convnet.reset_mtl_parameters()

                if self._meta_transfer["freeze_convnet"]:
                    self._network.convnet.freeze_convnet(
                        True,
                        bn_weights=self._meta_transfer.get("freeze_bn_weights"),
                        bn_stats=self._meta_transfer.get("freeze_bn_stats")
                    )

    def save_metadata(self, directory, run_id):
        path = os.path.join(directory, f"meta_{run_id}_task_{self._task}.pkl")

        logger.info("Saving metadata at {}.".format(path))
        with open(path, "wb+") as f:
            pickle.dump(
                [self._data_memory, self._targets_memory, self._herding_indexes, self._class_means],
                f
            )

    def load_metadata(self, directory, run_id):
        path = os.path.join(directory, f"meta_{run_id}_task_{self._task}.pkl")
        if not os.path.exists(path):
            return

        logger.info("Loading metadata at {}.".format(path))
        with open(path, "rb") as f:
            self._data_memory, self._targets_memory, self._herding_indexes, self._class_means = pickle.load(
                f
            )

    @property
    def epoch_metrics(self):
        return dict(self._epoch_metrics)

    # ----------
    # Public API
    # ----------

    def _before_task(self, train_loader, val_loader):
        self._n_classes += self._task_size
        self._network.add_classes(self._task_size)

        logger.info("Now {} examplars per class.".format(self._memory_per_class))

        # gen classifier for expert
        if self._task == 1:
            # self._expert_network = network.BasicNet(
            #     self.args["convnet"],
            #     convnet_kwargs=self.args.get("convnet_config", {}),
            #     classifier_kwargs=self.args.get("classifier_config", {
            #         "type": "fc",
            #         "use_bias": True
            #     }),
            #     device=self._device,
            #     extract_no_act=True,
            #     classifier_no_act=False,
            #     rotations_predictor=bool(self._rotations_config)
            # )
            # self._expert_network.add_classes(self._task_size)
            self._expert_network = self._old_model.copy().to(self._device)


            self._expert_network.classifier = network.Classifier(self._expert_network.convnet.out_dim, device=self._device, type="fc", use_bias=True)
            self._expert_network.add_classes(self._task_size)
            logger.info(f"self._task_size {self._task_size}")


        ##############
        if self._task > 0:
            self.last_task_network = self._old_model.copy().to(self._device)
        ##############

        self._optimizer = factory.get_optimizer(
            self._network.parameters(), self._opt_name, self._lr, self._weight_decay
        )

        base_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            self._optimizer, self._scheduling, gamma=self._lr_decay
        )

        if self._task > 0:
            self._expert_optimizer = factory.get_optimizer(
                self._expert_network.parameters(), expert_optimizer, expert_lr, expert_weight_decay
            )

            expert_base_scheduler = factory.get_lr_scheduler(
                expert_scheduling,
                self._expert_optimizer,
                nb_epochs=expert_epochs,
                lr_decay=expert_lr_decay,
                # task=self._task # do not warmup
            )
            self._expert_scheduler = expert_base_scheduler

        if self._warmup_config:
            if self._warmup_config.get("only_first_step", True) and self._task != 0:
                pass
            else:
                logger.info("Using WarmUp")
                self._scheduler = schedulers.GradualWarmupScheduler(
                    optimizer=self._optimizer,
                    after_scheduler=base_scheduler,
                    **self._warmup_config
                )
        else:
            self._scheduler = base_scheduler

    def _train_task(self, train_loader, val_loader):
        logger.debug("nb {}.".format(len(train_loader.dataset)))

        if self._task > 0:
            logger.info(" start training expert CNN ")
            self._training_expert_step(train_loader, val_loader, 0, expert_epochs)


        logger.info(" start training main CNN after training CNN")
        self._training_step(train_loader, val_loader, 0, self._n_epochs)

    def _training_step(
            self, train_loader, val_loader, initial_epoch, nb_epochs, record_bn=True, clipper=None
    ):
        best_epoch, best_acc = -1, -1.
        wait = 0

        grad, act = None, None
        if len(self._multiple_devices) > 1:
            logger.info("Duplicating model on {} gpus.".format(len(self._multiple_devices)))
            training_network = nn.DataParallel(self._network, self._multiple_devices)
            if self._network.gradcam_hook:
                grad, act, back_hook, for_hook = hook.get_gradcam_hook(training_network)
                training_network.module.convnet.last_conv.register_backward_hook(back_hook)
                training_network.module.convnet.last_conv.register_forward_hook(for_hook)
        else:
            training_network = self._network

        for epoch in range(initial_epoch, nb_epochs):
            self._metrics = collections.defaultdict(float)

            self._epoch_percent = epoch / (nb_epochs - initial_epoch)

            if epoch == nb_epochs - 1 and record_bn and len(self._multiple_devices) == 1 and \
                    hasattr(training_network.convnet, "record_mode"):
                logger.info("Recording BN means & vars for MCBN...")
                training_network.convnet.clear_records()
                training_network.convnet.record_mode()

            prog_bar = tqdm(
                train_loader,
                disable=self._disable_progressbar,
                ascii=True,
                bar_format="{desc}: {percentage:3.0f}% | {n_fmt}/{total_fmt} | {rate_fmt}{postfix}"
            )
            for i, input_dict in enumerate(prog_bar, start=1):
                inputs, targets = input_dict["inputs"], input_dict["targets"]
                memory_flags = input_dict["memory_flags"]

                if grad is not None:
                    _clean_list(grad)
                    _clean_list(act)

                self._optimizer.zero_grad()
                loss = self._forward_loss(
                    training_network,
                    inputs,
                    targets,
                    memory_flags,
                    gradcam_grad=grad,
                    gradcam_act=act
                )
                loss.backward()
                self._optimizer.step()

                if clipper:
                    training_network.apply(clipper)

                self._print_metrics(prog_bar, epoch, nb_epochs, i)

            if self._scheduler:
                self._scheduler.step(epoch)

            if self._eval_every_x_epochs and epoch != 0 and epoch % self._eval_every_x_epochs == 0:
                self._network.eval()
                self._data_memory, self._targets_memory, self._herding_indexes, self._class_means = self.build_examplars(
                    self.inc_dataset, self._herding_indexes
                )
                yraw, ytrue = self._eval_task(val_loader)
                # import pdb
                # pdb.set_trace()
                ypred = np.argmax(yraw, axis=1)

                acc = 100 * round((ypred == ytrue).sum() / len(ytrue), 3)
                logger.info("Val accuracy: {}".format(acc))
                self._network.train()

                if acc > best_acc:
                    best_epoch = epoch
                    best_acc = acc
                    wait = 0
                else:
                    wait += 1

                if self._early_stopping and self._early_stopping["patience"] > wait:
                    logger.warning("Early stopping!")
                    break

        if self._eval_every_x_epochs:
            logger.info("Best accuracy reached at epoch {} with {}%.".format(best_epoch, best_acc))

        if len(self._multiple_devices) == 1 and hasattr(training_network.convnet, "record_mode"):
            training_network.convnet.normal_mode()

    def _print_metrics(self, prog_bar, epoch, nb_epochs, nb_batches):
        pretty_metrics = ", ".join(
            "{}: {}".format(metric_name, round(metric_value / nb_batches, 3))
            for metric_name, metric_value in self._metrics.items()
        )

        prog_bar.set_description(
            "T{}/{}, E{}/{} => {}".format(
                self._task + 1, self._n_tasks, epoch + 1, nb_epochs, pretty_metrics
            )
        )

    def _forward_loss(
            self,
            training_network,
            inputs,
            targets,
            memory_flags,
            gradcam_grad=None,
            gradcam_act=None,
            **kwargs
    ):
        inputs, targets = inputs.to(self._device), targets.to(self._device)
        onehot_targets = utils.to_onehot(targets, self._n_classes).to(self._device)

        outputs = training_network(inputs)
        if gradcam_act is not None:
            outputs["gradcam_gradients"] = gradcam_grad
            outputs["gradcam_activations"] = gradcam_act

        # import pdb
        # pdb.set_trace()

        loss, loss_ce, loss_o, loss_n = self._compute_loss(inputs, outputs, targets, onehot_targets, memory_flags)

        if not utils.check_loss(loss):
            raise ValueError("A loss is NaN: {}".format(self._metrics))

        self._metrics["loss"] += loss.item()
        self._metrics["loss_ce"] += loss_ce.item()
        self._metrics["loss_o"] += loss_o.item()
        self._metrics["loss_n"] += loss_n.item()

        return loss

    def _after_task_intensive(self, inc_dataset):
        if self._herding_selection["type"] == "confusion":
            self._compute_confusion_matrix()

        self._data_memory, self._targets_memory, self._herding_indexes, self._class_means = self.build_examplars(
            inc_dataset, self._herding_indexes
        )

    def _after_task(self, inc_dataset):
        self._old_model = self._network.copy().freeze().to(self._device)
        self._network.on_task_end()
        # self.plot_tsne()

    def _compute_confusion_matrix(self):
        use_validation = self._validation_percent > 0.
        _, loader = self.inc_dataset.get_custom_loader(
            list(range(self._n_classes - self._task_size, self._n_classes)),
            memory=self.get_val_memory() if use_validation else self.get_memory(),
            mode="test",
            data_source="val" if use_validation else "train"
        )
        ypreds, ytrue = self._eval_task(loader)
        self._last_results = (ypreds, ytrue)

    def plot_tsne(self):
        if self.folder_result:
            loader = self.inc_dataset.get_custom_loader([], memory=self.get_memory())[1]
            embeddings, targets = utils.extract_features(self._network, loader)
            utils.plot_tsne(
                os.path.join(self.folder_result, "tsne_{}".format(self._task)), embeddings, targets
            )

    def _eval_task(self, data_loader):

        # Original icarl NME method
        ori_nme = True
        if ori_nme:
            ypred, ytrue = self.compute_accuracy(self._network, data_loader, self._class_means)

        # ours NME
        ours_nme = False
        if ours_nme:
            ypred, ytrue = [], []

            for input_dict in data_loader:

                inputs = input_dict["inputs"].to(self._device)
                features = self._network.extract(inputs)
                features = features.detach()

                if self._task > 0:

                    self.last_task_network.eval()

                    old_features = self.last_task_network.extract(inputs)
                    expert_features = self._expert_network.extract(inputs)

                    features = features.cpu().numpy()
                    old_features = old_features.detach().cpu().numpy()
                    expert_features = expert_features.detach().cpu().numpy()

                    # add->dis
                    features += (1.0) * old_features
                    features += (1.0) * expert_features

                    features = (features.T / (np.linalg.norm(features.T, axis=0) + EPSILON)).T
                    sqd = cdist(self._class_means, features, 'sqeuclidean')
                    preds = (-sqd).T
                    # preds = softmax(preds, axis=1)

                    # # dis->add
                    # now dis
                    # features = (features.T / (np.linalg.norm(features.T, axis=0) + EPSILON)).T
                    # sqd = cdist(self._class_means, features, 'sqeuclidean')
                    # preds = (-sqd).T
                    # preds = softmax(preds, axis=1)
                    #
                    # # old dis
                    # old_features = (old_features.T / (np.linalg.norm(old_features.T, axis=0) + EPSILON)).T
                    # sqd = cdist(self._class_means, old_features, 'sqeuclidean')
                    # old_preds = (-sqd).T
                    # old_preds = softmax(old_preds, axis=1)
                    #
                    #
                    # # expert dis
                    # expert_features = (expert_features.T / (np.linalg.norm(expert_features.T, axis=0) + EPSILON)).T
                    # sqd = cdist(self._class_means, expert_features, 'sqeuclidean')
                    # expert_preds = (-sqd).T
                    # expert_preds = softmax(expert_preds, axis=1)
                    #
                    #
                    # preds += (1)*old_preds
                    # preds += (1)*expert_preds

                else:

                    features = features.cpu().numpy()
                    features = (features.T / (np.linalg.norm(features.T, axis=0) + EPSILON)).T

                    # Compute score for iCaRL
                    sqd = cdist(self._class_means, features, 'sqeuclidean')
                    preds = (-sqd).T

                ytrue.append(input_dict["targets"].numpy())
                ypred.append(preds)

            ytrue = np.concatenate(ytrue)
            ypred = np.concatenate(ypred)

        # ours CNN
        ours_cnn =False
        if ours_cnn:
            ypred, ytrue = [], []

            for input_dict in data_loader:
                outputs = self._network(input_dict["inputs"].to(self._device))
                logits = outputs["logits"]

                logits = logits.detach()

                if self._task > 0:
                    inputs = input_dict["inputs"].to(self._device)
                    self.last_task_network.eval()

                    old_logits = self.last_task_network(inputs)["logits"].detach()
                    expert_logits = self._expert_network(inputs)["logits"].detach()

                    preds = F.softmax(logits, dim=-1)
                    old_preds = F.softmax(old_logits, dim=-1)
                    expert_preds = F.softmax(expert_logits, dim=-1)

                    # 0
                    if self._task == 1:
                        preds = self._n_classes * preds
                        preds[..., :-self._task_size] += (1.0) * (self._n_classes - self._task_size) * old_preds
                        # preds[..., :-self._task_size] = preds[..., :-self._task_size]/2

                        preds[..., -self._task_size:] += (1.0) * (self._task_size) * expert_preds
                        # preds[..., -self._task_size:] = preds[..., -self._task_size:]/2

                    else:
                        preds = self._n_classes * preds
                        preds[..., :-self._task_size] += (1.0) * (self._n_classes - self._task_size) * old_preds
                        # preds[..., :-self._task_size] = preds[..., :-self._task_size] / 2

                        preds[..., -self._task_size:] += (0.7) * (self._task_size) * expert_preds
                        # preds[..., -self._task_size:] = preds[..., -self._task_size:] / 2

                    preds /= 2

                else:
                    preds = F.softmax(logits, dim=-1)

                ytrue.append(input_dict["targets"].numpy())
                ypred.append(preds.cpu().numpy())

            ytrue = np.concatenate(ytrue)
            ypred = np.concatenate(ypred)

        return ypred, ytrue

    # -----------
    # Private API
    # -----------

    def _compute_loss(self, inputs, outputs, targets, onehot_targets, memory_flags):
        logits = outputs["logits"]
        loss_o = torch.tensor(0.0).to(self._device)
        loss_n = torch.tensor(0.0).to(self._device)

        loss_ce = F.cross_entropy(logits, targets)


        if self._old_model is not None:
            with torch.no_grad():
                old_logits = self._old_model(inputs)["logits"] / self._temperature
                self._expert_network.eval()
                new_logits = self._expert_network(inputs)["logits"] / self._temperature

                old_targets = F.softmax(old_logits, dim=1)
                new_targets = F.softmax(new_logits, dim=1)

            loss_o = F.cross_entropy(logits[..., :-self._task_size] / self._temperature,
                                     old_targets)
            loss_n = F.cross_entropy(logits[..., -self._task_size:] / self._temperature,
                                     new_targets)

        # loss += F.binary_cross_entropy_with_logits(
        #     logits[..., :-self._task_size] / self._temperature,
        #     torch.sigmoid(old_targets / self._temperature)
        # )
        # clf_loss = F.cross_entropy(logits, targets)

        loss = loss_ce + self.lambda1 * loss_o + self.lambda2 * loss_n

        return loss, loss_ce, loss_o, loss_n

    @property
    def _memory_per_class(self):
        """Returns the number of examplars per class."""
        if self._fixed_memory:
            return self._memory_size // self._total_n_classes
        return self._memory_size // self._n_classes

    # -----------------
    # Memory management
    # -----------------

    def build_examplars(
            self, inc_dataset, herding_indexes, memory_per_class=None, data_source="train"
    ):
        logger.info("Building & updating memory.")
        memory_per_class = memory_per_class or self._memory_per_class
        herding_indexes = copy.deepcopy(herding_indexes)

        data_memory, targets_memory = [], []
        class_means = np.zeros((self._n_classes, self._network.features_dim))


        for class_idx in range(self._n_classes):
            # We extract the features, both normal and flipped:
            inputs, loader = inc_dataset.get_custom_loader(
                class_idx, mode="test", data_source=data_source
            )

            features, targets = utils.extract_features(self._network, loader)
            features_flipped, _ = utils.extract_features(
                self._network,
                inc_dataset.get_custom_loader(class_idx, mode="flip", data_source=data_source)[1]
            )

            if self._task > 0:
                # old network features
                self.last_task_network.eval()
                old_features, targets = utils.extract_features(self.last_task_network, loader)
                old_features_flipped, _ = utils.extract_features(
                    self.last_task_network,
                    inc_dataset.get_custom_loader(class_idx, mode="flip", data_source=data_source)[1]
                )

                # expert network features
                self._expert_network.eval()
                expert_features, targets = utils.extract_features(self._expert_network, loader)
                expert_features_flipped, _ = utils.extract_features(
                    self._expert_network,
                    inc_dataset.get_custom_loader(class_idx, mode="flip", data_source=data_source)[1]
                )

                features = (features + old_features_weight * old_features + expert_features_weight * expert_features) / denominator
                features_flipped = (features_flipped + old_features_weight * old_features_flipped + expert_features_weight * expert_features_flipped) / denominator


            if class_idx >= self._n_classes - self._task_size:
                # New class, selecting the examplars:
                if self._herding_selection["type"] == "icarl":
                    selected_indexes = herding.icarl_selection(features, memory_per_class)
                elif self._herding_selection["type"] == "closest":
                    selected_indexes = herding.closest_to_mean(features, memory_per_class)
                elif self._herding_selection["type"] == "random":
                    selected_indexes = herding.random(features, memory_per_class)
                elif self._herding_selection["type"] == "first":
                    selected_indexes = np.arange(memory_per_class)
                elif self._herding_selection["type"] == "kmeans":
                    selected_indexes = herding.kmeans(
                        features, memory_per_class, k=self._herding_selection["k"]
                    )
                elif self._herding_selection["type"] == "confusion":
                    selected_indexes = herding.confusion(
                        *self._last_results,
                        memory_per_class,
                        class_id=class_idx,
                        minimize_confusion=self._herding_selection["minimize_confusion"]
                    )
                elif self._herding_selection["type"] == "var_ratio":
                    selected_indexes = herding.var_ratio(
                        memory_per_class, self._network, loader, **self._herding_selection
                    )
                elif self._herding_selection["type"] == "mcbn":
                    selected_indexes = herding.mcbn(
                        memory_per_class, self._network, loader, **self._herding_selection
                    )
                else:
                    raise ValueError(
                        "Unknown herding selection {}.".format(self._herding_selection)
                    )

                herding_indexes.append(selected_indexes)

            # Reducing examplars:
            try:
                selected_indexes = herding_indexes[class_idx][:memory_per_class]
                herding_indexes[class_idx] = selected_indexes
            except:
                import pdb
                pdb.set_trace()

            # Re-computing the examplar mean (which may have changed due to the training):
            examplar_mean = self.compute_examplar_mean(
                features, features_flipped, selected_indexes, memory_per_class
            )

            data_memory.append(inputs[selected_indexes])
            targets_memory.append(targets[selected_indexes])

            if self._task > 0 and class_idx < self._n_classes - self._task_size:
                examplar_mean = (examplar_mean + self._class_means[class_idx, :]) / 2

            class_means[class_idx, :] = examplar_mean

        data_memory = np.concatenate(data_memory)
        targets_memory = np.concatenate(targets_memory)

        return data_memory, targets_memory, herding_indexes, class_means

    def get_memory(self):
        return self._data_memory, self._targets_memory

    @staticmethod
    def compute_examplar_mean(feat_norm, feat_flip, indexes, nb_max):
        D = feat_norm.T
        D = D / (np.linalg.norm(D, axis=0) + EPSILON)

        D2 = feat_flip.T
        D2 = D2 / (np.linalg.norm(D2, axis=0) + EPSILON)

        selected_d = D[..., indexes]
        selected_d2 = D2[..., indexes]

        mean = (np.mean(selected_d, axis=1) + np.mean(selected_d2, axis=1)) / 2
        mean /= (np.linalg.norm(mean) + EPSILON)

        return mean

    @staticmethod
    def compute_accuracy(model, loader, class_means):
        features, targets_ = utils.extract_features(model, loader)

        features = (features.T / (np.linalg.norm(features.T, axis=0) + EPSILON)).T

        # Compute score for iCaRL
        sqd = cdist(class_means, features, 'sqeuclidean')
        score_icarl = (-sqd).T

        return score_icarl, targets_

    def compute_accuracy_threenet(self, loader, class_means):

        features, targets_ = utils.extract_features(self._network, loader)

        old_features, old_targets_ = utils.extract_features(self._old_model, loader)

        expert_features, expert_targets_ = utils.extract_features(self._expert_network, loader)

        if not (targets_ == old_targets_).all():
            print("target error")
            assert False
        if not (targets_ == expert_targets_).all():
            print("target error")
            assert False

        features += (1.0) * old_features
        features += (1.0) * expert_features

        features = (features.T / (np.linalg.norm(features.T, axis=0) + EPSILON)).T

        # Compute score for iCaRL
        sqd = cdist(class_means, features, 'sqeuclidean')
        score_icarl = (-sqd).T

        return score_icarl, targets_

    # -----------------
    # about expert network
    # -----------------

    def _training_expert_step(
            self, train_loader, val_loader, initial_epoch, nb_epochs, record_bn=True, clipper=None, finetune=False
    ):
        best_epoch, best_acc = -1, -1.
        wait = 0

        grad, act = None, None
        if len(self._multiple_devices) > 1:
            logger.info("Duplicating model on {} gpus.".format(len(self._multiple_devices)))
            training_network = nn.DataParallel(self._expert_network, self._multiple_devices)
            if self._expert_network.gradcam_hook:
                grad, act, back_hook, for_hook = hook.get_gradcam_hook(training_network)
                training_network.module.convnet.last_conv.register_backward_hook(back_hook)
                training_network.module.convnet.last_conv.register_forward_hook(for_hook)
        else:
            training_network = self._expert_network
            print(" training network is _expert_network ")

        # debug one stage train expert
        finetune = True

        if finetune:
            # eval before finetune
            logger.info("eval expert before finetune")
            self._expert_network.eval()

            yraw, ytrue, new_class_logits, old_class_logits = self._eval_expert_before_or_after_finetune(val_loader)
            ypred = np.argmax(yraw, axis=1)
            acc = 100 * round((ypred == ytrue).sum() / len(ytrue), 3)
            logger.info("expert Val accuracy: {}".format(acc))

            ## get auroc
            auroc_val = get_auroc(new_class_logits, old_class_logits)
            logger.info(f"AUROC : {auroc_val}")

            # watch class average logits
            avg_new_logits = np.average(new_class_logits, axis=0)
            avg_new_logits = (avg_new_logits + 1) / 2

            avg_old_logits = np.average(old_class_logits, axis=0)
            avg_old_logits = (avg_old_logits + 1) / 2

            logger.info("old class avg logits: \n\n {} \n\n".format(avg_old_logits))
            logger.info("new class avg logits: \n\n {} \n\n".format(avg_new_logits))

            self._expert_network.train()

        for epoch in range(initial_epoch, nb_epochs):
            self._metrics = collections.defaultdict(float)

            self._epoch_percent = epoch / (nb_epochs - initial_epoch)

            if epoch == nb_epochs - 1 and record_bn and len(self._multiple_devices) == 1 and \
                    hasattr(training_network.convnet, "record_mode"):
                logger.info("Recording BN means & vars for MCBN...")
                training_network.convnet.clear_records()
                training_network.convnet.record_mode()

            prog_bar = tqdm(
                train_loader,
                disable=self._disable_progressbar,
                ascii=True,
                bar_format="{desc}: {percentage:3.0f}% | {n_fmt}/{total_fmt} | {rate_fmt}{postfix}"
            )

            sum_loss_part1 = 0
            sum_loss_part2 = 0
            batch_num = 0

            for i, input_dict in enumerate(prog_bar, start=1):
                inputs, targets = input_dict["inputs"], input_dict["targets"]
                memory_flags = input_dict["memory_flags"]

                if grad is not None:
                    _clean_list(grad)
                    _clean_list(act)

                # change label to [0, self._task_size]
                targets = targets - (self._n_classes - self._task_size)

                self._expert_optimizer.zero_grad()
                loss, loss_part1, loss_part2 = self._forward_expert_loss(
                    training_network,
                    inputs,
                    targets,
                    memory_flags,
                    gradcam_grad=grad,
                    gradcam_act=act,
                    finetune=finetune
                )
                loss.backward()
                self._expert_optimizer.step()

                batch_num += 1
                sum_loss_part1 += loss_part1
                sum_loss_part2 += loss_part2

                if clipper:
                    training_network.apply(clipper)

                self._print_metrics(prog_bar, epoch, nb_epochs, i)

            if self._scheduler:
                self._expert_scheduler.step(epoch)

            if finetune:
                logger.info(
                    f'epoch {epoch} , avg loss_part1 {sum_loss_part1 / batch_num} , avg loss_part2 {sum_loss_part2 / batch_num}')

        if finetune:
            # eval after finetune
            logger.info("eval expert after finetune")
            self._expert_network.eval()

            yraw, ytrue, new_class_logits, old_class_logits = self._eval_expert_before_or_after_finetune(val_loader)
            ypred = np.argmax(yraw, axis=1)
            acc = 100 * round((ypred == ytrue).sum() / len(ytrue), 3)
            logger.info("expert Val accuracy: {}".format(acc))

            ## get auroc
            auroc_val = get_auroc(new_class_logits, old_class_logits)
            logger.info(f"AUROC : {auroc_val}")

            # watch class average logits
            avg_new_logits = np.average(new_class_logits, axis=0)
            avg_new_logits = (avg_new_logits + 1) / 2

            avg_old_logits = np.average(old_class_logits, axis=0)
            avg_old_logits = (avg_old_logits + 1) / 2

            logger.info("old class avg logits: \n\n {} \n\n".format(avg_old_logits))
            logger.info("new class avg logits: \n\n {} \n\n".format(avg_new_logits))

            self._expert_network.train()

        if len(self._multiple_devices) == 1 and hasattr(training_network.convnet, "record_mode"):
            training_network.convnet.normal_mode()

    def _forward_expert_loss(
            self,
            training_network,
            inputs,
            targets,
            memory_flags,
            gradcam_grad=None,
            gradcam_act=None,
            finetune=False,
            **kwargs
    ):

        # if self._n_classes == 20:
        #     import pdb
        #     pdb.set_trace()

        inputs, targets = inputs.to(self._device), targets.to(self._device)

        # do not need
        onehot_targets = None

        # get new data index
        # new_mask = torch.where(targets >= 0)
        #
        # inputs = inputs[new_mask]
        # targets = targets[new_mask]
        # if inputs.shape[0] == 0:
        #     loss = torch.tensor(0.0).to(self._device)
        #     loss_part1 = torch.tensor(0.0).to(self._device)
        #     loss_part2 = torch.tensor(0.0).to(self._device)
        #     return loss, loss_part1, loss_part2

        outputs = training_network(inputs)
        if gradcam_act is not None:
            outputs["gradcam_gradients"] = gradcam_grad
            outputs["gradcam_activations"] = gradcam_act
        # import pdb;pdb.set_trace()
        loss, loss_part1, loss_part2 = self._compute_expert_loss(inputs, outputs, targets, onehot_targets, memory_flags,
                                                                 finetune)

        if not utils.check_loss(loss):
            raise ValueError("A loss is NaN: {}".format(self._metrics))

        self._metrics["loss"] += loss.item()

        return loss, loss_part1, loss_part2

    def _compute_expert_loss(self, inputs, outputs, targets, onehot_targets, memory_flags, finetune):

        # get new data index
        new_mask = torch.where(targets >= 0)
        # get old data index
        old_mask = torch.where(targets < 0)

        features, logits, atts = outputs["raw_features"], outputs["logits"], outputs["attention"]

        # if self._post_processing_type is None:
        #     scaled_logits = self._expert_network.post_process(logits)
        # else:
        #     scaled_logits = logits * self._post_processing_type

        if logits[new_mask].shape[0] != 0:
            loss_part1 = F.cross_entropy(logits[new_mask], targets[new_mask])
        else:
            loss_part1 = 0.0

        if logits[old_mask].shape[0] != 0 and finetune:

            loss_part2 = expert_loss_part2_weight * -(
                    logits[old_mask].mean(1) - torch.logsumexp(logits[old_mask], dim=1)).mean()
        else:
            loss_part2 = 0.0

        loss = loss_part1 + loss_part2

        return loss, loss_part1, loss_part2

    def _eval_expert_before_or_after_finetune(self, test_loader):

        # bic use cnn
        self._evaluation_type = "cnn"

        if self._evaluation_type in ("icarl", "nme"):
            return super()._eval_task(test_loader)
        elif self._evaluation_type in ("softmax", "cnn"):
            ypred = []
            ytrue = []
            old_class_logits = []
            new_class_logits = []

            for input_dict in test_loader:
                targets = input_dict["targets"]
                inputs = input_dict["inputs"]

                # change label to [0, self._task_size]
                targets = targets - (self._n_classes - self._task_size)

                # select new example
                new_mask = torch.where(targets >= 0)
                # select old example
                old_mask = torch.where(targets < 0)

                inputs = inputs.to(self._device)
                logits = self._expert_network(inputs)["logits"].detach()

                preds = F.softmax(logits[new_mask], dim=-1)
                ypred.append(preds.cpu().numpy())
                ytrue.append(targets[new_mask].numpy())

                old_logits = logits[old_mask]
                old_class_logits.append(old_logits.cpu().numpy())
                new_logits = logits[new_mask]
                new_class_logits.append(new_logits.cpu().numpy())

            ypred = np.concatenate(ypred)
            ytrue = np.concatenate(ytrue)
            new_class_logits = np.concatenate(new_class_logits)
            old_class_logits = np.concatenate(old_class_logits)

            self._last_results = (ypred, ytrue)

            return ypred, ytrue, new_class_logits, old_class_logits
        else:
            raise ValueError(self._evaluation_type)

    def eval(self):
        self._network.eval()
        if self._expert_network is not None:
            self._expert_network.eval()
        # if self._task > 0:
        #     self.last_task_netwerk.eval()

    def train(self):
        self._network.train()
        if self._expert_network is not None:
            self._expert_network.train()
        # if self._task > 0:
        #     self.last_task_netwerk.train()

    def save_parameters(self, directory, run_id):
        path = os.path.join(directory, f"net_{run_id}_task_{self._task}.pth")
        logger.info(f"Saving model at {path}.")
        torch.save({'network': self.network.state_dict(),
                    'expert_network': self._expert_network.state_dict() if self._task > 0 else None,
                    'old_network': self.last_task_network.state_dict() if self._task > 0 else None}, path)

    def load_parameters(self, directory, run_id):
        path = os.path.join(directory, f"net_{run_id}_task_{self._task}.pth")
        if not os.path.exists(path):
            return

        logger.info(f"Loading model at {path}.")
        try:
            # import pdb; pdb.set_trace()
            checkpoint = torch.load(path, map_location=torch.device('cuda'))
            # import pdb; pdb.set_trace()
            self.network.load_state_dict(checkpoint['network'])
            if self._task > 0:
                self._expert_network.load_state_dict(checkpoint['expert_network'])
                self.last_task_network.load_state_dict(checkpoint['old_network'])
        except Exception:
            logger.warning("Old method to save weights, it's deprecated!")
            self._network = torch.load(path)


def _clean_list(l):
    for i in range(len(l)):
        l[i] = None


def get_measures(_pos, _neg, recall_level=0.95):
    pos = np.array(_pos[:]).reshape((-1, 1))
    neg = np.array(_neg[:]).reshape((-1, 1))
    examples = np.squeeze(np.vstack((pos, neg)))
    labels = np.zeros(len(examples), dtype=np.int32)
    labels[:len(pos)] += 1

    auroc = sk.roc_auc_score(labels, examples)

    return auroc


def get_auroc(pos_class_logits, neg_class_logits):
    # get auroc
    smax = softmax(pos_class_logits, axis=1)
    print(smax[0])
    pos_class_score = np.max(smax, axis=1)

    smax = softmax(neg_class_logits, axis=1)
    print(smax[0])
    neg_class_score = np.max(smax, axis=1)

    # import pdb
    # pdb.set_trace()
    auroc_val = get_measures(pos_class_score, neg_class_score)
    return auroc_val


