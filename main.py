from PhIREGANs import *

# data_train_path = 'data/bw_jan01_14_regions.tfrecord'
data_train_path = 'data/jan01_14data.tfrecord'
# pt_bw_model_path = 'models/st-20210316-111628/cnn/cnn'

# trained_model_path = 'models/st-20210314-132058/gan-all/gan'
r = [2]
data_test_path = 'data/test_01_15_25regions.tfrecord'
# mu_sig=[[37.27572889,  38.75493871, 150.22220039], [89.68371972, 90.81342058, 46.8983685]]

# data_type = 'wind'
# data_path = 'data/wind_MR-HR.tfrecord'
# model_path = 'models/wind_mr-hr/trained_gan/gan'
# r = [5]
# mu_sig=[[0.7684, -0.4575], [5.02455, 5.9017]]



if __name__ == '__main__':
    gan = PhIREGANs(data_type='st')
    
    pt_model = gan.pretrain(r=r,
                             data_path=data_train_path,
                             batch_size=20)
    trained_model_path = gan.train(r=r, data_path=data_train_path, model_path=pt_bw_model_path, batch_size=20)
    gan.test(r=r, data_path=data_test_path, model_path=trained_model_path, plot_data=True)