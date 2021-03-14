''' @author: Karen Stengel
'''
from PhIREGANs import *

data_type = 'st'
data_path = 'example_data/'
r = [2]

# data_type = 'wind'
# data_path = 'example_data/wind_MR-HR.tfrecord'
# model_path = 'models/wind_mr-hr/trained_gan/gan'
# r = [5]
# mu_sig=[[0.7684, -0.4575], [5.02455, 5.9017]] When mu_sig not specified during pre-train => G loss >1


if __name__ == '__main__':

    phiregans = PhIREGANs(data_type=data_type, mu_sig=mu_sig)
    
    model_dir = phiregans.pretrain(r=r,
                                   data_path=data_path,
                                   batch_size=1)

    model_dir = phiregans.train(r=r,
                                data_path=data_path,
                                batch_size=1)
    
    # phiregans.test(r=r,
    #                data_path=data_path,
    #                model_path=model_dir,
    #                batch_size=1)


