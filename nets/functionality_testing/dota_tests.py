from nets.dataset.classification.dota_dataset import DotaDataset

def dota_to_numpy():
    d = DotaDataset()
    d.load_all_data('dota2_train.csv', test_fname='dota2_test.csv')
    d.split_train_into_val()
    d.serialize_all_data('np_dota2_train.npz',
                         'np_dota2_val.npz',
                         'np_dota2_test.npz')

def check_dota_numpy():
    d = DotaDataset()
    d.load_all_data('np_dota2_train.npz','np_dota2_val.npz','np_dota2_test.npz')
    assert (len(d._train_x) == len(d._train_y))
    assert (len(d._val_x) == len(d._val_y))
    assert (len(d._test_x) == len(d._test_y))

if __name__ == '__main__':
    dota_to_numpy()
    check_dota_numpy()
