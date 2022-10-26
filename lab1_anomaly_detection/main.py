import pandas as pd
import numpy as np
import re


# Convert states for one user from type 'str' to np.array of integers
def string_to_array(states: str) -> np.array:
    numbers = states.split(';')
    string_to_int = np.vectorize(lambda x: int(x))
    return np.array(string_to_int(numbers))


def preprocess_data(data: pd.DataFrame, data_true: pd.DataFrame, data_fake: pd.DataFrame) -> tuple:
    # Cast states of type 'str' to type 'np.array'
    for (i, j, k) in zip(range(len(data)), range(len(data_true)), range(len(data_fake))):
        data.values[i, 1] = string_to_array(data.values[i, 1])
        data_true.values[j, 1] = string_to_array(data_true.values[j, 1])
        data_fake.values[k, 1] = string_to_array(data_fake.values[k, 1])
    # Change values 'user' by removing word 'user' and cast number to 'int' for sorting
    data_fake["user"] = data_fake["user"].apply(lambda word: re.sub("user", '', word))  # remove 'user'
    data_fake["user"] = data_fake["user"].apply(lambda string: int(string))  # cast to 'int'
    data_fake = data_fake.sort_values(by=["user"])
    return data, data_true, data_fake


# Get transition matrix for one user
# @return matrix: zero row and column contains all unique states'
#                 element(i, j) at the cross is transition probability from state mat[i, 0] to state mat[0, j]
def get_transition_matrix(states: np.array) -> np.array:
    unique_states = np.unique(states)
    matrix = np.zeros(shape=(unique_states.size + 1, unique_states.size + 1))
    matrix[0, 1:] = unique_states  # first column - all unique states
    matrix[1:, 0] = unique_states  # first row - all unique states

    state_indices = {}  # dict with key - unique state; value - index of unique state
    for i in range(unique_states.size):
        state_indices.update({unique_states[i]: i + 1})

    # Get count of states
    for i in range(0, states.size - 1):  # start: 0 -> 1 state
        matrix[state_indices[states[i]],
               state_indices[states[i + 1]]] += 1

    # Convert to probability
    for row in matrix[1:, 1:]:  # skip values in zero row and column
        row_sum = np.sum(row)
        row[:] /= row_sum

    return matrix


# Return list of anomaly transitions for one user
def predict(transition_matrix: np.array, states: np.array) -> list:
    probability_threshold = 0.01
    unique_states = transition_matrix[0, 1:].astype(np.int32)

    state_indices = {}  # dict with key - unique state; value - index of unique state
    for i in range(unique_states.size):
        state_indices.update({unique_states[i]: i + 1})

    anomalies = []
    for i in range(0, states.size - 1):  # start: 0 -> 1 state
        try:
            if transition_matrix[state_indices[states[i]],
                                 state_indices[states[i + 1]]] < probability_threshold:
                anomalies.append([states[i], states[i + 1]])

        except KeyError:  # if this state is not in the transition matrix
            anomalies.append([states[i], states[i + 1]])

    return anomalies


def main():
    data = pd.read_csv("time_series_data/data.txt", sep=':')
    data_true = pd.read_csv("time_series_data/data_true.txt", sep=':')
    data_fake = pd.read_csv("time_series_data/data_fake.txt", sep=':')

    # Cast states to 'np.array', sort data_fake rows
    data, data_true, data_fake = preprocess_data(data, data_true, data_fake)

    # Get transition matrices for all users
    transition_matrices = []
    for i in range(len(data)):
        transition_matrices.append(get_transition_matrix(data.values[i, 1]))

    # Get list of anomaly transition for all users
    anomalies_true = []
    anomalies_fake = []
    for (i, j) in zip(range(len(data_true)), range(len(data_fake))):
        anomalies_true.append(predict(transition_matrices[i], data_true.values[i, 1]))
        anomalies_fake.append(predict(transition_matrices[j], data_fake.values[j, 1]))
    print(f"Number of anomalies in data_true: {sum(len(x) for x in anomalies_true)}")
    print(f"Number of anomalies in data_fake: {sum(len(x) for x in anomalies_fake)}")


main()
