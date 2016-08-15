import pandas
import numpy as np
import os

SUBJECTID = 'Subject'

def load_patient_metadata(metadata_file, subject_measures=None, subjects=None):
    """

    :param metadata_file: A csv filename containing HCP metadata similar
    http://www.humanconnectome.org/documentation/Q1/behavioral-measures-details.html
    requirements to work with some of the other functions are:
    :param subject_measures: arraylike/ iterable of string - each string is column header
    for each subject measure
    :param subjects: list of subjectIDs to return the data for
    :returns a panda dataframe containing subject measures
    """
    metadata = pandas.read_csv(metadata_file)
    if subjects is not None:
        metadata = get_subjects_data(metadata, subjects)
    if subject_measures is not None:
        metadata = metadata[subject_measures]
    return metadata

def get_subjects_data(dataframe, subejct_list, subject_id=SUBJECTID):
    return dataframe[dataframe[subject_id].isin(subejct_list)]


