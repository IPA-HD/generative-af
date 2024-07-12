"""
Run training as specified in a config file.
"""
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
import json
import argparse
from data.curriculum import new_curriculum
from module import ToyDataFlow, ImageFlow
from util import read, DummyDataloader

parser = argparse.ArgumentParser(
	prog='FlowMatchingAF',
	description='Train flow-matching assignment flows for discrete data as specified in JSON config file.')
parser.add_argument('--debug', action='store_true', help="Debug mode, short training and no logs.")
parser.add_argument("config", type=str, help="Filepath for the training configuration in JSON format.")
args = parser.parse_args()

with open(args.config, "r", encoding="utf-8") as f:
	hparams = json.load(f)

# select task
if hparams["data"]["dataset"] in ["pinwheel", "simplex_stark", "simplex_toy", "gaussian_mixture", "coupled_binary"]:
	model = ToyDataFlow(hparams)
elif hparams["data"]["dataset"] in ["mnist", "cityscapes"]:
	model = ImageFlow(hparams)
else:
	raise NotImplementedError

curriculum = new_curriculum(hparams["data"])
training_params = hparams["training"]
checkpointing = ModelCheckpoint(
	every_n_train_steps=read("check_interval_batches", hparams["logging"], default=None),
	save_top_k=read("checkpoint_topk", hparams["logging"], default=1),
	every_n_epochs=read("check_interval_epochs", hparams["logging"], default=None)
)
if hparams["data"]["dataset"] == "simplex_stark":
	val_dl = DummyDataloader(hparams["data"]["num_val_batches"])
else:
	val_dl = DummyDataloader()
trainer = L.Trainer(
	accelerator="auto",
	fast_dev_run=args.debug,
	max_epochs=training_params["epochs"],
	max_steps=read("steps", training_params, default=-1),
	check_val_every_n_epoch=read("eval_interval_epochs", hparams["logging"], default=None),
	val_check_interval=read("eval_interval_batches", hparams["logging"], default=1.0),
	callbacks=[checkpointing]
)

batch_size = training_params["batch_size"]
train_dl = curriculum.dataloader(batch_size=batch_size)
trainer.fit(model, train_dl, val_dataloaders=val_dl)
