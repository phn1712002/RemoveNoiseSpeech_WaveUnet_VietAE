from Dataset import Dataset_CV, Dataset_Vivos
class DatasetWaveUnet():
    def __init__(self, path='./Dataset/'):
        self.path = path
        self._train_dataset = None
        self._dev_dataset = None
        self._test_dataset = None
        
    def __call__(self):
        self._dev_dataset = Dataset_CV.Dataset(path=self.path + 'raw/CommonVoice/', name_file='train').loadDataset()
        self._train_dataset = Dataset_Vivos.Dataset(path=self.path + 'raw/Vivos/', name_file='train').loadDataset() + Dataset_Vivos.Dataset(path=self.path + 'raw/Vivos/', name_file='test').loadDataset()
        return self._train_dataset, self._dev_dataset, self._test_dataset