import torch
import random
import time
import os
import logging
import argparse
import sys
import numpy as np
import dill as pickle

from utils.dataset import Dataset
from utils.misc import save_config, validate_config, check_device
from utils.misc import get_memory_alloc, log_ckpts
from utils.misc import _convert_to_words_batchfirst, _convert_to_words
from utils.misc import _convert_to_tensor, _convert_to_tensor_pad
from utils.misc import plot_alignment, plot_attention, combine_weights
from utils.config import PAD, EOS
from modules.checkpoint import Checkpoint
from modules.loss import NLLLoss, BCELoss, CrossEntropyLoss, KLDivLoss, MSELoss
from torch.distributions import Categorical

logging.basicConfig(level=logging.INFO)


def load_arguments(parser):

	""" Seq2Seq eval """

	# paths
	parser.add_argument('--test_path_src', type=str, required=True, help='test src dir')
	parser.add_argument('--test_path_tgt', type=str, default='None', help='test tgt dir')
	parser.add_argument('--path_vocab_src', type=str, default='None', help='vocab src dir, no need')
	parser.add_argument('--path_vocab_tgt', type=str, default='None', help='vocab tgt dir, not needed')
	parser.add_argument('--use_type', type=str, default='char', help='use char | word level prediction')
	parser.add_argument('--acous_norm', type=str, default='False', help='input acoustic fbk normalisation')
	parser.add_argument('--acous_norm_path', type=str, default='None', help='acoustics norm')
	parser.add_argument('--test_acous_path', type=str, default='None', help='test set acoustics')

	parser.add_argument('--load', type=str, required=True, help='model load dir')
	parser.add_argument('--combine_path', type=str, default='None', help='combine multiple ckpts if given dir')
	parser.add_argument('--test_path_out', type=str, required=True, help='test out dir')

	# others
	parser.add_argument('--max_seq_len', type=int, default=32, help='maximum sequence length')
	parser.add_argument('--batch_size', type=int, default=64, help='batch size')
	parser.add_argument('--beam_width', type=int, default=0, help='beam width; set to 0 to disable beam search')
	parser.add_argument('--use_gpu', type=str, default='False', help='whether or not using GPU')
	parser.add_argument('--eval_mode', type=int, default=2, help='which evaluation mode to use')
	parser.add_argument('--gen_mode', type=str, default='ASR', help='AE|ASR|MT|ST')
	parser.add_argument('--lm_mode', type=str, default='null', help='null|s-4g|s-rnn|d')
	parser.add_argument('--seqrev', type=str, default=False, help='whether or not to reverse sequence')

	return parser


def translate(test_set, model, test_path_out, use_gpu,
	max_seq_len, beam_width, device, seqrev=False,
	gen_mode='ASR', lm_mode='null', history='HYP'):

	"""
		no reference tgt given - Run translation.
		Args:
			test_set: test dataset
				src, tgt using the same dir
			test_path_out: output dir
			load_dir: model dir
			use_gpu: on gpu/cpu
	"""

	modes = '_'.join([model.mode, gen_mode])
	# reset max_len
	if 'ASR' in modes or 'ST' in modes:
		model.las.decoder.max_seq_len = 150
	if 'MT' in modes:
		model.enc_src.expand_time(150)
	if 'ST' in modes or 'MT' in modes:
		model.dec_tgt.expand_time(max_seq_len)

	print('max seq len {}'.format(max_seq_len))
	sys.stdout.flush()

	# load lm
	mode = lm_mode.split('_')[0]
	if mode == 'null':
		lm_model = None
	elif mode == 's-4g':
		corpus = lm_mode.split('_')[1]
		LM_BASE = '/home/alta/BLTSpeaking/exp-ytl28/projects/lib/lms/pkl/'
		if corpus == 'ted':
			LM_PATH = os.path.join(LM_BASE, 'idlm-ted-train.pkl')
		elif corpus =='mustc':
			LM_PATH = os.path.join(LM_BASE, 'idlm-mustc-train.pkl')
		with open(LM_PATH, 'rb') as fin: lm_model = pickle.load(fin)
		print('LM {} - {} loaded'.format(lm_mode, LM_PATH))
	elif mode == 's-rnn':
		assert False, 'Not implemented'

	# load test
	test_set.construct_batches(is_train=False)
	evaliter = iter(test_set.iter_loader)

	print('num batches: {}'.format(len(evaliter)))
	with open(os.path.join(test_path_out, 'translate.txt'), 'w', encoding="utf8") as f:
		model.eval()
		with torch.no_grad():
			for idx in range(len(evaliter)):

				print(idx+1, len(evaliter))
				batch_items = evaliter.next()

				# load data
				src_ids = batch_items['srcid'][0]
				src_lengths = batch_items['srclen']
				tgt_ids = batch_items['tgtid'][0]
				tgt_lengths = batch_items['tgtlen']
				acous_feats = batch_items['acous_feat'][0]
				acous_lengths = batch_items['acouslen']

				src_len = max(src_lengths)
				tgt_len = max(tgt_lengths)
				acous_len = max(acous_lengths)
				src_ids = src_ids[:,:src_len].to(device=device)
				tgt_ids = tgt_ids.to(device=device)
				acous_feats = acous_feats.to(device=device)

				n_minibatch = int(tgt_len / 100 + tgt_len % 100 > 0)
				minibatch_size = int(src_ids.size(0) / n_minibatch)
				n_minibatch = int(src_ids.size(0) / minibatch_size) + \
					(src_ids.size(0) % minibatch_size > 0)

				for j in range(n_minibatch):

					st = j * minibatch_size
					ed = min((j+1) * minibatch_size, src_ids.size(0))
					src_ids_sub = src_ids[st:ed,:]
					tgt_ids_sub = tgt_ids[st:ed,:]
					acous_feats_sub = acous_feats[st:ed,:]
					acous_lengths_sub = acous_lengths[st:ed]
					print('minibatch: ', st, ed, src_ids.size(0))

					time1 = time.time()
					if history == 'HYP':
						preds = model.forward_translate(acous_feats=acous_feats_sub,
							acous_lens=acous_lengths_sub, src=src_ids_sub,
							beam_width=beam_width, use_gpu=use_gpu,
							max_seq_len=max_seq_len, mode=gen_mode,
							lm_mode=lm_mode, lm_model=lm_model)
					elif history == 'REF':
						preds = model.forward_translate_refen(acous_feats=acous_feats_sub,
							acous_lens=acous_lengths_sub, src=src_ids_sub,
							beam_width=beam_width, use_gpu=use_gpu,
							max_seq_len=max_seq_len, mode=gen_mode,
							lm_mode=lm_mode, lm_model=lm_model)
					time2 = time.time()
					print('comp time: ', time2-time1)

					# ------ debug ------
					# import pdb; pdb.set_trace()
					# out_dict = model.forward_eval(acous_feats=acous_feats_sub,
					# 	acous_lens=acous_lengths_sub, src=src_ids_sub,
					# 	use_gpu=use_gpu, mode=gen_mode)
					# -------------------

					# write to file
					if gen_mode == 'MT' or 'ST' in gen_mode:
						seqlist = preds[:,1:]
						seqwords =  _convert_to_words_batchfirst(seqlist, test_set.tgt_id2word)
						use_type = 'char'
					elif gen_mode == 'ASR':
						seqlist = preds
						seqwords =  _convert_to_words_batchfirst(seqlist, test_set.src_id2word)
						use_type = 'word'

					for i in range(len(seqwords)):
						words = []
						for word in seqwords[i]:
							if word == '<pad>':
								continue
							elif word == '<spc>':
								words.append(' ')
							elif word == '</s>':
								break
							else:
								words.append(word)
						if len(words) == 0:
							outline = ''
						else:
							if seqrev:
								words = words[::-1]
							if use_type == 'word':
								outline = ' '.join(words)
							elif use_type == 'char':
								outline = ''.join(words)
						f.write('{}\n'.format(outline))

						# import pdb; pdb.set_trace()
					sys.stdout.flush()


def plot_emb(test_set, model, test_path_out, use_gpu, max_seq_len, device):

	"""
		plot embedding spaces
	"""

	# load test
	test_set.construct_batches(is_train=False)
	evaliter = iter(test_set.iter_loader)
	print('num batches: {}'.format(len(evaliter)))
	path_out = os.path.join(test_path_out, 'embed.png')

	import torch.utils.tensorboard
	writer = torch.utils.tensorboard.writer.SummaryWriter(log_dir=test_path_out)

	model.eval()
	with torch.no_grad():
		for idx in range(len(evaliter)):

			print(idx+1, len(evaliter))
			batch_items = evaliter.next()

			# load data
			src_ids = batch_items['srcid'][0]
			src_lengths = batch_items['srclen']
			tgt_ids = batch_items['tgtid'][0]
			tgt_lengths = batch_items['tgtlen']
			acous_feats = batch_items['acous_feat'][0]
			acous_lengths = batch_items['acouslen']

			src_len = max(src_lengths)
			tgt_len = max(tgt_lengths)
			acous_len = max(acous_lengths)
			src_ids = src_ids[:,:src_len].to(device=device)
			tgt_ids = tgt_ids.to(device=device)
			acous_feats = acous_feats.to(device=device)

			n_minibatch = int(tgt_len / 100 + tgt_len % 100 > 0)
			minibatch_size = int(src_ids.size(0) / n_minibatch)
			n_minibatch = int(src_ids.size(0) / minibatch_size) + \
				(src_ids.size(0) % minibatch_size > 0)

			for j in range(n_minibatch):

				st = j * minibatch_size
				ed = min((j+1) * minibatch_size, src_ids.size(0))
				src_ids_sub = src_ids[st:ed,:]
				tgt_ids_sub = tgt_ids[st:ed,:]
				acous_feats_sub = acous_feats[st:ed,:]
				acous_lengths_sub = acous_lengths[st:ed]
				print('minibatch: ', st, ed, src_ids.size(0))

				# get dynamic
				dynamic_emb, logps, preds, lengths = model._encoder_acous(
					acous_feats_sub, acous_lengths_sub, device, use_gpu,
					is_training=False, teacher_forcing_ratio=0.0)

				# get static
				src = model._pre_proc_src(src_ids_sub, device)
				src_lengths = [elem - 1 for elem in src_lengths]
				_, static_emb, _ = model._get_src_emb(src, device)

				# prep plot
				commlen = min(dynamic_emb.size(1),static_emb.size(1))
				src_mask_input = (torch.arange(commlen).expand(len(src_lengths), commlen)
					< torch.LongTensor(src_lengths).unsqueeze(1)).to(device=device)
				dynamic_emb = dynamic_emb[:,:commlen][src_mask_input]
				static_emb = static_emb[:,:commlen][src_mask_input]
				hyp_ids = preds[:,:commlen][src_mask_input]
				ref_ids = src[:,:commlen][src_mask_input]
				hyp_words = [test_set.src_id2word[int(id)] for id in hyp_ids]
				ref_words = [test_set.src_id2word[int(id)] for id in ref_ids]

				# plot embeddings
				feats = torch.cat((dynamic_emb, static_emb), dim=0)
				hyp_words.extend(ref_words)
				meta = hyp_words
				color_dynamic = torch.Tensor([0,0,0]).repeat(dynamic_emb.size(0),1) #black
				color_static = torch.Tensor([1,0.5,0]).repeat(static_emb.size(0),1) #orange
				labels = torch.cat((color_dynamic, color_static),dim=0).view(-1,3,1,1)
				writer.add_embedding(feats,metadata=meta,label_img=labels)
				writer.close()

				import pdb; pdb.set_trace()


def gather_emb(test_set, model, test_path_out, use_gpu, max_seq_len, device):

	"""
		gather embedding statistics
	"""

	# load test
	test_set.construct_batches(is_train=False)
	evaliter = iter(test_set.iter_loader)
	print('num batches: {}'.format(len(evaliter)))

	# id = 0
	# size = 500
	# start = id * size
	# end = (id +1) * size
	# path_out = os.path.join(test_path_out, 'dyn_emb_ave_{}.npy'.format(id))

	path_out = os.path.join(test_path_out, 'dyn_emb_ave.npy')

	lis = []

	model.eval()
	with torch.no_grad():
		for idx in range(len(evaliter)):
		# for idx in range(0,10):
		# for idx in range(start,end):

			print(idx+1, len(evaliter))
			batch_items = evaliter.next()

			# load data
			src_ids = batch_items['srcid'][0]
			src_lengths = batch_items['srclen']
			tgt_ids = batch_items['tgtid'][0]
			tgt_lengths = batch_items['tgtlen']
			acous_feats = batch_items['acous_feat'][0]
			acous_lengths = batch_items['acouslen']

			src_len = max(src_lengths)
			tgt_len = max(tgt_lengths)
			acous_len = max(acous_lengths)
			src_ids = src_ids[:,:src_len].to(device=device)
			tgt_ids = tgt_ids.to(device=device)
			acous_feats = acous_feats.to(device=device)

			n_minibatch = int(tgt_len / 100 + tgt_len % 100 > 0)
			minibatch_size = int(src_ids.size(0) / n_minibatch)
			n_minibatch = int(src_ids.size(0) / minibatch_size) + \
				(src_ids.size(0) % minibatch_size > 0)

			for j in range(n_minibatch):

				st = j * minibatch_size
				ed = min((j+1) * minibatch_size, src_ids.size(0))
				src_ids_sub = src_ids[st:ed,:]
				tgt_ids_sub = tgt_ids[st:ed,:]
				acous_feats_sub = acous_feats[st:ed,:]
				acous_lengths_sub = acous_lengths[st:ed]
				print('minibatch: ', st, ed, src_ids.size(0))

				# get dynamic
				dynamic_emb, logps, preds, lengths = model._encoder_acous(
					acous_feats_sub, acous_lengths_sub, device, use_gpu,
					tgt=src_ids_sub, is_training=False, teacher_forcing_ratio=1.0)

				dynamic_emb_ave = torch.mean(dynamic_emb.view(-1,512), dim=0)
				lis.append(dynamic_emb_ave)

	# import pdb; pdb.set_trace()
	emb_ave = torch.mean(torch.stack(lis, dim=0), dim=0)
	np.save(path_out, emb_ave.cpu().numpy())
	print('saved to {}'.format(path_out))


def main():

	# load config
	parser = argparse.ArgumentParser(description='Evaluation')
	parser = load_arguments(parser)
	args = vars(parser.parse_args())
	config = validate_config(args)

	# load src-tgt pair
	test_path_src = config['test_path_src']
	test_path_tgt = config['test_path_tgt']
	if type(test_path_tgt) == type(None):
		test_path_tgt = test_path_src

	test_path_out = config['test_path_out']
	test_acous_path = config['test_acous_path']
	acous_norm_path = config['acous_norm_path']

	load_dir = config['load']
	max_seq_len = config['max_seq_len']
	batch_size = config['batch_size']
	beam_width = config['beam_width']
	use_gpu = config['use_gpu']
	seqrev = config['seqrev']
	use_type = config['use_type']

	# set test mode
	MODE = config['eval_mode']
	if MODE != 2:
		if not os.path.exists(test_path_out):
			os.makedirs(test_path_out)
		config_save_dir = os.path.join(test_path_out, 'eval.cfg')
		save_config(config, config_save_dir)

	# check device:
	device = check_device(use_gpu)
	print('device: {}'.format(device))

	# load model
	latest_checkpoint_path = load_dir
	resume_checkpoint = Checkpoint.load(latest_checkpoint_path)
	model = resume_checkpoint.model.to(device)
	vocab_src = resume_checkpoint.input_vocab
	vocab_tgt = resume_checkpoint.output_vocab
	print('Model dir: {}'.format(latest_checkpoint_path))
	print('Model laoded')

	# combine model
	if type(config['combine_path']) != type(None):
		model = combine_weights(config['combine_path'])
	# import pdb; pdb.set_trace()

	# load test_set
	test_set = Dataset(path_src=test_path_src, path_tgt=test_path_tgt,
						vocab_src_list=vocab_src, vocab_tgt_list=vocab_tgt,
						use_type=use_type,
						acous_path=test_acous_path,
						seqrev=seqrev,
						acous_norm=config['acous_norm'],
						acous_norm_path=config['acous_norm_path'],
						acous_max_len=6000,
						max_seq_len_src=900,
						max_seq_len_tgt=900,
						batch_size=batch_size,
						mode='ST',
						use_gpu=use_gpu)

	print('Test dir: {}'.format(test_path_src))
	print('Testset loaded')
	sys.stdout.flush()

	# '{AE|ASR|MT|ST}-{REF|HYP}'
	if len(config['gen_mode'].split('-')) == 2:
		gen_mode = config['gen_mode'].split('-')[0]
		history = config['gen_mode'].split('-')[1]
	elif len(config['gen_mode'].split('-')) == 1:
		gen_mode = config['gen_mode']
		history = 'HYP'

	# add external language model
	lm_mode = config['lm_mode']

	# run eval:
	if MODE == 1:
		translate(test_set, model, test_path_out, use_gpu,
			max_seq_len, beam_width, device, seqrev=seqrev,
			gen_mode=gen_mode, lm_mode=lm_mode, history=history)

	elif MODE == 2: # save combined model
		ckpt = Checkpoint(model=model,
				   optimizer=None, epoch=0, step=0,
				   input_vocab=test_set.vocab_src,
				   output_vocab=test_set.vocab_tgt)
		saved_path = ckpt.save_customise(
			os.path.join(config['combine_path'].strip('/')+'-combine','combine'))
		log_ckpts(config['combine_path'], config['combine_path'].strip('/')+'-combine')
		print('saving at {} ... '.format(saved_path))

	elif MODE == 3:
		gather_emb(test_set, model, test_path_out, use_gpu, max_seq_len, device)




if __name__ == '__main__':
	main()
