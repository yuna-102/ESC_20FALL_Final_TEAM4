import os
import argparse
import datetime
import torch
import model
import train
import dataset
import pretrained_vectors


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('--batch-size', default=50, type=int)
	parser.add_argument('--dropout', default=0.5, type=float)
	parser.add_argument('--epoch', default=300, type=int)
	parser.add_argument('--learning-rate', default=0.1, type=float)
	parser.add_argument("--mode", default="non-static", help="available models: rand, static, non-static")
	parser.add_argument('--num-feature-maps', default=100, type=int) 
 	parser.add_argument("--pretrained-word-vectors", default="fasttext", help="available models: fasttext, Word2Vec")
 
  args = parser.parse_args()

  train_dataloader, val_dataloader = data_loader(train_inputs, val_inputs, train_labels, val_labels, batch_size = args.batch_size)
  
  if args.mode == 'rand':
    # CNN-rand: Word vectors are randomly initialized.
    train.set_seed(42)
    cnn_rand, optimizer = model.initilize_model(vocab_size=len(word2idx),
                                          embed_dim=300,
                                          learning_rate=args.learning_rate,
                                          dropout=args.dropout)
    train.train(cnn_rand, optimizer, train_dataloader, val_dataloader, epochs=args.epoch)

  elif args.mode == 'static':
    # CNN-static: fastText pretrained word vectors are used and freezed during training.
    train.set_seed(42)
    cnn_static, optimizer = initilize_model(pretrained_embedding=pretrained_vectors.get_vector(args.pretrained-word-vectors),
                                            freeze_embedding=True,
                                            learning_rate=args.learning_rate,
                                            dropout=args.dropout)
    train.train(cnn_static, optimizer, train_dataloader, val_dataloader, epochs=args.epoch)

  else:
    # CNN-non-static: fastText pretrained word vectors are fine-tuned during training.
    train.set_seed(42)
    cnn_non_static, optimizer = initilize_model(pretrained_embedding=pretrained_vectors.get_vector(args.pretrained-word-vectors),
                                                freeze_embedding=False,
                                                learning_rate=args.learning_rate,
                                                dropout=dropout)
    train.train(cnn_non_static, optimizer, train_dataloader, val_dataloader, epochs=20)