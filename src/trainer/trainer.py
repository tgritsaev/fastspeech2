import random
from tqdm import tqdm

import PIL
import torch
from torchvision.transforms import ToTensor

from src.trainer.base_trainer import BaseTrainer
from src.logger.utils import plot_spectrogram_to_buf
from src.utils import inf_loop, MetricTracker
from waveglow import get_wav, get_waveglow
from src.utils import DEFAULT_SR


class Trainer(BaseTrainer):
    """
    Trainer class
    """

    def __init__(
        self,
        model,
        criterion,
        metrics,
        optimizer,
        config,
        device,
        dataloaders,
        lr_scheduler=None,
        len_epoch=None,
        skip_oom=True,
    ):
        super().__init__(model, criterion, metrics, optimizer, lr_scheduler, config, device)
        self.skip_oom = skip_oom
        self.config = config
        self.train_dataloader = dataloaders["train"]
        if len_epoch is None:
            # epoch-based training
            self.len_epoch = len(self.train_dataloader)
        else:
            # iteration-based training
            self.train_dataloader = inf_loop(self.train_dataloader)
            self.len_epoch = len_epoch
        self.evaluation_dataloaders = {k: v for k, v in dataloaders.items() if k != "train"}

        self.loss_names = ["loss", "mel_loss", "duration_loss", "energy_loss", "pitch_loss"]
        self.train_metrics = MetricTracker(*self.loss_names, "grad_norm")
        self.batch_expand_size = config["trainer"].get("batch_expand_size", 1)
        self.iters_to_accumulate = config["trainer"].get("iters_to_accumulate", 1)
        self.waveglow = get_waveglow(self.config["trainer"]["waveglow_path"])

    @staticmethod
    def move_batch_to_device(batch, device: torch.device):
        """
        Move all necessary tensors to the GPU
        """
        for tensor_for_gpu in ["src_seq", "mel_target", "pitch_target", "energy_target", "src_pos", "mel_pos", "length_target"]:
            batch[tensor_for_gpu] = batch[tensor_for_gpu].to(device)
        return batch

    def _clip_grad_norm(self):
        if self.config["trainer"].get("grad_norm_clip", None) is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config["trainer"]["grad_norm_clip"])

    def _progress(self, batch_idx):
        base = "[{}/{} ({:.0f}%)]"
        if hasattr(self.train_dataloader, "n_samples"):
            current = batch_idx * self.train_dataloader.batch_size
            total = self.train_dataloader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)

    @torch.no_grad()
    def _log_predictions(self, mel_prediction, examples_to_log=4, **kwargs):
        mel_prediction = mel_prediction[:examples_to_log].transpose(1, 2).to("cuda")
        wavs = get_wav(mel_prediction, self.waveglow)

        for i, wav in enumerate(wavs):
            self.writer.add_audio(f"audio-{i}", wav, sample_rate=DEFAULT_SR)

    def _log_spectrogram(self, spectrogram_batch):
        spectrogram = random.choice(spectrogram_batch.cpu())
        image = PIL.Image.open(plot_spectrogram_to_buf(spectrogram))
        self.writer.add_image("spectrogram", ToTensor()(image))

    @torch.no_grad()
    def get_grad_norm(self, norm_type=2):
        parameters = self.model.parameters()
        if isinstance(parameters, torch.Tensor):
            parameters = [parameters]
        parameters = [p for p in parameters if p.grad is not None]

        total_norm = torch.norm(
            torch.stack([torch.norm(torch.nan_to_num(p.grad.detach(), nan=0), norm_type).cpu() for p in parameters]),
            norm_type,
        )
        return total_norm.item()

    def _log_scalars(self, metric_tracker: MetricTracker):
        if self.writer is None:
            return
        for metric_name in metric_tracker.keys():
            self.writer.add_scalar(f"{metric_name}", metric_tracker.avg(metric_name))

    def process_batch(self, batch, is_train: bool, batch_idx: int, metrics: MetricTracker):
        batch = self.move_batch_to_device(batch, self.device)
        outputs = self.model(**batch)
        batch.update(outputs)
        if is_train:
            loss = self.criterion(**batch)
            for key in loss.keys():
                loss[key] /= self.iters_to_accumulate
            batch.update(loss)
            batch["loss"].backward()

            if (batch_idx + 1) % self.iters_to_accumulate == 0 or (batch_idx + 1) == self.len_epoch:
                self._clip_grad_norm()
                self.optimizer.step()
                self.train_metrics.update("grad_norm", self.get_grad_norm())
                self.optimizer.zero_grad()

            if self.lr_scheduler is not None:
                self.lr_scheduler.step()
            for loss_name in self.loss_names:
                metrics.update(loss_name, batch[loss_name].item())

        for metric in self.metrics:
            metrics.update(metric.name, metric(**batch))
        return batch

    # def _evaluation_epoch(self, epoch, part, dataloader):
    #     """
    #     Validate after training an epoch

    #     :param epoch: Integer, current training epoch.
    #     :return: A log that contains information about validation
    #     """
    #     self.model.eval()
    #     self.evaluation_metrics.reset()
    #     with torch.no_grad():
    #         for batch_idx, batch in tqdm(enumerate(dataloader), desc=part, total=len(dataloader)):
    #             batch = self.process_batch(batch, False, 0, metrics=self.evaluation_metrics)
    #         self.writer.set_step(epoch * self.len_epoch, part)
    #         self._log_predictions(False, **batch)
    #         # self._log_spectrogram(batch["spectrogram"])
    #         self._log_scalars(self.evaluation_metrics)

    #     return self.evaluation_metrics.result()

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        self.model.train()
        self.train_metrics.reset()
        self.writer.add_scalar("epoch", epoch)
        bar = tqdm(range(self.len_epoch), desc="train")
        for list_batch_idx, batches in enumerate(self.train_dataloader):
            for cur_batch_idx, batch in enumerate(batches):
                real_batch_idx = cur_batch_idx + list_batch_idx * self.batch_expand_size
                try:
                    batch = self.process_batch(batch, True, real_batch_idx, metrics=self.train_metrics)
                except RuntimeError as e:
                    if "out of memory" in str(e) and self.skip_oom:
                        self.logger.warning("OOM on batch. Skipping batch.")
                        for p in self.model.parameters():
                            if p.grad is not None:
                                del p.grad  # free some memory
                        torch.cuda.empty_cache()
                        continue
                    else:
                        raise e
                if real_batch_idx % self.log_step == 0:
                    self.writer.set_step((epoch - 1) * self.len_epoch + real_batch_idx)
                    self.logger.debug("Train Epoch: {} {} Loss: {:.6f}".format(epoch, self._progress(real_batch_idx), batch["loss"].item()))
                    self.writer.add_scalar("learning rate", self.lr_scheduler.get_last_lr()[0])
                    self._log_scalars(self.train_metrics)
                    # we don't want to reset train metrics at the start of every epoch
                    # because we are interested in recent train metrics
                    last_train_metrics = self.train_metrics.result()
                    self.train_metrics.reset()
                    bar.update(self.log_step)

                if real_batch_idx + 1 >= self.len_epoch:
                    self._log_predictions(**batch)
                    log = last_train_metrics
                    return log

        # for part, dataloader in self.evaluation_dataloaders.items():
        #     val_log = self._evaluation_epoch(epoch, part, dataloader)
        #     log.update(**{f"{part}_{name}": value for name, value in val_log.items()})
