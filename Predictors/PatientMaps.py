import pandas
from pandas import DataFrame
import numpy as np
import os

class PatientMap(DataFrame):

    def __init__(self, map_folder_path):
        dataframe = self._load_patient_fmri(map_folder_path)
        super(PatientMap, self).__init__(dataframe)

    def _load_patient_fmri(self, fmri_directory_path):
        ids =[]
        maps = []
        for root, dirs, timeseries in os.walk(self.fmri_directory_path):
            for t in timeseries:
                patient_id = t.split(".")
                ids += patient_id[0]
                patient_map = np.genfromtxt(os.path.join(root, t), delimiter=",")
                maps += patient_map

        patient_data = {"id": ids, "map": maps }
        return patient_data

    @property
    def _constructor(self):
        return PatientMap
