import os
from numpy import load
from shutil import copyfile


def trajectory_path(session_path, reset=False, read_only=False):
    session_path = os.path.abspath(session_path)

    if not os.path.exists(session_path):
        raise TypeError(f"The path doesn't exists {session_path}")

    if not os.path.isdir(session_path):
        try:
            load(session_path, allow_pickle=True).item()
            return get_duplicated(session_path, reset=reset, read_only=read_only)
        except:
            raise TypeError(f"Not a valid trajectory file {session_path}")

    path_N_particles = os.path.join(
        session_path, "trajectories_wo_gaps", "trajectories_wo_gaps.npy"
    )
    path_1_particle = os.path.join(session_path, "trajectories", "trajectories.npy")

    if os.path.exists(path_N_particles):
        return get_duplicated(path_N_particles, reset=reset, read_only=read_only)
    elif os.path.exists(path_1_particle):
        return get_duplicated(path_1_particle, reset=reset, read_only=read_only)
    else:
        raise TypeError(f"No trajectory file found in session {session_path}")


def get_duplicated(path, read_only, reset=True):
    if path.endswith("_corrected.npy") or read_only:
        return path

    duplicated_path = path[:-4] + "_corrected.npy"
    if reset or not os.path.exists(duplicated_path):
        print(f"Duplicating {path} to {duplicated_path} ")
        copyfile(path, duplicated_path)
    return duplicated_path


def dir_path(string):
    if os.path.isdir(string):
        return string
    else:
        raise NotADirectoryError(string)


def file_path(string):
    if os.path.exists(string) and not os.path.isdir(string):
        return string
    else:
        raise ValueError(f"File {string} not found")
