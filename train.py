"""
Run training as specified in a config file.
"""
import hydra
from omegaconf import DictConfig, OmegaConf
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
import json
from data.curriculum import new_curriculum
from module import ToyDataFlow, ImageFlow
from util import read, DummyDataloader

@hydra.main(version_base=None, config_path="config", config_name="config")
def run_training(hparams : DictConfig) -> None:
	print(OmegaConf.to_yaml(hparams))

	# select task
	if hparams.data.dataset in ["pinwheel", "simplex_stark", "simplex_toy", "gaussian_mixture", "coupled_binary"]:
		model = ToyDataFlow(hparams)
	elif hparams.data.dataset in ["mnist", "cityscapes"]:
		model = ImageFlow(hparams)
	else:
		raise NotImplementedError

	curriculum = new_curriculum(hparams.data)
	checkpointing = ModelCheckpoint(
		every_n_train_steps=read("check_interval_batches", hparams.logging, default=None),
		save_top_k=read("checkpoint_topk", hparams.logging, default=1),
		every_n_epochs=read("check_interval_epochs", hparams.logging, default=None)
	)
	if hparams.data.dataset == "simplex_stark":
		val_dl = DummyDataloader(hparams.data.num_val_batches)
	else:
		val_dl = DummyDataloader()
	trainer = L.Trainer(
		accelerator="auto",
		fast_dev_run=read("debug", hparams, default=False),
		max_epochs=hparams.training["epochs"],
		max_steps=read("steps", hparams.training, default=-1),
		check_val_every_n_epoch=read("eval_interval_epochs", hparams.logging, default=None),
		val_check_interval=read("eval_interval_batches", hparams.logging, default=1.0),
		callbacks=[checkpointing]
	)

	batch_size = hparams.training["batch_size"]
	train_dl = curriculum.dataloader(batch_size=batch_size)
	trainer.fit(model, train_dl, val_dataloaders=val_dl)

if __name__ == "__main__":
	run_training()
