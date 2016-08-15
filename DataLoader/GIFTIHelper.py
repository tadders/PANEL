import numpy as np
from DataLoader.NiftiIntents import NiftiIntents as intents

def get_gifti_vertices(gifti):
    """

    :param gifti: A nibabel GIFTI class
    :return: n * 3 numpy array containing the coordinate of the vertices
    in the gifti surface
    """
    return gifti.getArraysFromIntent(intents.NIFTI_INTENT_POINTSET)[0].data

def get_gifti_faces(gifti):
    """

    :param gifti:
    :return: returns the triangular faces making up the gifti
    numpy array form f x 3 where f is the number of faces and the columns
    represent the index of the vertices making up the face
    """
    return np.flipud(gifti.getArraysFromIntent(intents.NIFTI_INTENT_TRIANGLE)[0].data)