import argparse


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def ParseArgs():
	parser = argparse.ArgumentParser(description='Model Params')
	parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')	
	parser.add_argument('--batch', default=512, type=int, help='batch size')
	parser.add_argument('--tst_bat', default=256, type=int, help='number of users in a testing batch')	
	parser.add_argument('--epoch', default=200, type=int, help='number of epochs')
	parser.add_argument('--sim_epoch', default=5, type=int, help='number of simulative epochs')
	parser.add_argument('--mask_bat', default=128, type=int, help='batch size for masking')
	parser.add_argument('--niter', default=2, type=int, help='number of iteration in svd')
	parser.add_argument('--decay', default=1.0, type=float, help='weight decay rate')
	parser.add_argument('--save_path', default=None, help='file name to save model and training record')                                   
	parser.add_argument('--latdim', default=128, type=int, help='embedding size')
	parser.add_argument('--gnn_layer', default=3, type=int, help='number of gnn layers')
	parser.add_argument('--unlearn_layer', default=0, type=int, help='number of unlearning gnn layers')
	parser.add_argument('--early_stop', default=10, type=int, help='number of epochs for early stop')
	parser.add_argument('--load_model', default=None, help='model name to load')    
	parser.add_argument('--trained_model', default=None, help='model name to load')                        	     		
	parser.add_argument('--topk', default=20, type=int, help='K of top K')	
	parser.add_argument('--data', default='gowalla', type=str, help='name of dataset')     
	parser.add_argument('--tst_epoch', default=3, type=int, help='number of epoch to test while training')	
	parser.add_argument('--gpu', default='0', type=str, help='indicates which gpu to use')

	# Simgcl
	parser.add_argument('--ssl_reg', default=1e-2, type=float, help='weight decay regularizer')	
	parser.add_argument('--eps', default=0.2, type=float, help='epsilon in the model')
	parser.add_argument('--temp', default=0.1, type=float, help='temperature in ssl')     
	parser.add_argument('--graphSampleN', default=15000, type=int, help='number of nodes to sample each time')
	parser.add_argument('--noiseRate', default=-0.1, type=float, help='ratio of edges to add noise')
	parser.add_argument('--reg_version', default='v1', type=str, help='the version of reg loss')
     
	# SGL
	parser.add_argument('--sgl_ssl_reg', default=1e-2, type=float, help='weight decay regularizer')	
	parser.add_argument('--sgltemp', default=0.1, type=float, help='temperature in ssl')     
	parser.add_argument('--sglkeepRate', default=0.8, type=float, help='temperature in ssl')    
      

	# Model type
	parser.add_argument('--model', default='simgcl', type=str, help='the model type to unlearn')

	# Loss weights
	parser.add_argument('--reg', default=1e-7, type=float, help='weight decay regularizer')
	parser.add_argument('--bpr_wei', default=1., type=float, help='weight for BPR loss')
	parser.add_argument('--unlearn_wei', default=0.5, type=float, help='weight for BPR loss')
	parser.add_argument('--align_wei', default=0.02, type=float, help='weight for align loss')
	parser.add_argument('--align_temp', default=10., type=float, help='temperature for infoNCE loss')
	parser.add_argument('--align_type', default='v2', type=str, help='version for align function')	
	parser.add_argument('--unlearn_type', default='v1', type=str, help='version for unlearn function')	
	parser.add_argument('--perf_degrade', default=0.5, type=float, help='acceptable level of performance degradation')
     



	# Adversarial attack experiments
	parser.add_argument('--adversarial_attack', default=False, type=str2bool, help='whether to use the datasets with adversarial attack')
	parser.add_argument('--random_attack', default=False, type=str2bool, help='whether to use the datasets with random adversarial attack')
	parser.add_argument('--adv_method', default='lightgcn0.5', type=str, help='how to find the adversarial edges') 
	# parser.add_argument('--adv_method', default='lightgcn', type=str, help='how to find the adversarial edges') 

	# Random
	parser.add_argument('--seed', default=1234, type=int, help='random seed')
    
	# Unlearn
	parser.add_argument('--overall_withdraw_rate', default=0.1, type=int, help='overall withdraw rate')	
	parser.add_argument('--withdraw_rate_init', default=1, type=int, help='overall withdraw vector initialization')	
	parser.add_argument('--allgrad', default=True, type=str2bool, help='update all the parameters')	
	parser.add_argument('--pretrain_drop_rate', default=0.05, type=float, help='ratio of edges to drop when pretraining')     
	parser.add_argument('--test_drop_rate', default=0.003, type=float, help='ratio of edges to drop when testing, only take effect when adversarial_attack is False')     
	parser.add_argument('--hyper_temp', default=1., type=int, help='temperature for unlearn ssl')
	parser.add_argument('--unlearn_ssl', default=1e-3, type=int, help='weight for unlearn ssl')
	parser.add_argument('--keep_rate', default=1., type=float, help='ratio of edges to keep')
	parser.add_argument('--act', default='leaky', type=str, help='activation function for unlearnning fine-tune part')
	parser.add_argument('--layer_mlp', default=2, type=float, help='layer for mlp (unlearnning fine-tune part)')
	parser.add_argument('--leaky', default=0.99, type=float, help='slope for the negative part of leaky relu')
    
	parser.add_argument('--keepRate', default=1.0, type=float, help='ratio of edges to keep')
	parser.add_argument('--fineTune', default=False, type=str2bool, help='update all the parameters')	

	 


	return parser.parse_args()
args = ParseArgs()


