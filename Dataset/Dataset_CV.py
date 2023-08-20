import pandas as pd


class Dataset:
    def __init__(self, path, name_file):
        self.path = path
        self.name_file = name_file

    def loadDataset(self, lang='vi'):
        # Đọc dữ liệu
        data = pd.read_csv(self.path + lang + '/' + self.name_file + '.tsv', sep='\t')
        data = data['path']

        # Chèn thêm đầu path
        data = self.path + lang + '/clips/' + data.astype(str)

        return list(data)
