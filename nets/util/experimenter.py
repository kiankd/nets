import csv
from collections import defaultdict

DELIMITER = '\t'
SUFFIX = '.tsv'
MODEL_START_KEY = 'MODEL_NAME'

class Experimenter(object):
    def __init__(self, save_name, dataset_headers, model_headers):
        assert(not '.' in save_name and MODEL_START_KEY in model_headers)
        self.fname = save_name
        self.res_dict = {}
        self.param_to_val = {}
        self.idx_to_param = {}

        for idx, key in enumerate(dataset_headers + model_headers):
            self.param_to_val[key] = []
            self.idx_to_param[idx] = key

    def add_result(self, dataset_params, model_params):
        self.check_res_formatting(dataset_params, model_params)
        for i, value in enumerate(dataset_params + model_params):
            param = self.idx_to_param[i]
            self.param_to_val[param].append(value)

    def check_res_formatting(self, dataset_params, model_params):
        assert(len(model_params) + len(dataset_params) == len(self.idx_to_param))

    def analyze_results(self):
        pass

    def serialize(self):
        """
        Save the results data into a nice TSV file.
        """
        with open(self.fname + SUFFIX, 'w', newline='') as tsvf:
            writer = csv.writer(tsvf, delimiter=DELIMITER)
            writer.writerow(list(self.param_to_val.keys()))
            rows = zip(*list(self.param_to_val.values()))
            for row in rows:
                writer.writerow(row)

    @staticmethod
    def load_results(fname):
        """
        :param fname: complete filename with path to a results TSV file.
        :return: Experimenter object with parameters as in the results file.
        """
        with open(fname,  newline='') as tsvf:
            reader = csv.reader(tsvf, delimiter=DELIMITER)
            data = list(reader)
        headers = data[0]
        idx = headers.index(MODEL_START_KEY)
        exp = Experimenter(fname.rstrip(SUFFIX), headers[:idx], headers[idx:])
        for row in data[1:]:
            exp.add_result(row[:idx], row[idx:])
        return exp

