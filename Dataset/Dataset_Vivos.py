import pandas as pd


class Dataset:
    def __init__(self, path, name_file):
        self.path = path
        self.name_file = name_file

    def loadDataset(self, lang='vi'):
        def split(string, pathFolder):
            index = string.find(" ")
            name = string[0:index]
            sentence = string[index + 1: len(string)]

            index = name.find("_")
            path = pathFolder + name[0:index] + '/' + name + '.wav'
            return path, sentence

        # Đọc dữ liệu
        data = pd.read_csv(self.path + lang + '/' + self.name_file + '/prompts.txt', header=None)
        path = self.path + lang + '/' + self.name_file + '/waves/'
        # Cắt path và sentence
        listPath = []
        for index in range(0, len(data) - 1):
            path_full, _ = split(data[0][index], path)
            listPath.append(path_full)

        data = pd.DataFrame({'path': listPath})

        return list(data['path'])
