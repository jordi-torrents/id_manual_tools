import os


def get_trajectory_path(session_path):
    session_path = os.path.abspath(session_path)
    path_N_particles = os.path.join(
        session_path, "trajectories_wo_gaps", "trajectories_wo_gaps.npy"
    )
    path_1_particle = os.path.join(session_path, "trajectories", "trajectories.npy")

    if os.path.exists(path_N_particles):
        return path_N_particles
    elif os.path.exists(path_1_particle):
        return path_1_particle
    else:
        raise FileNotFoundError(f"No trajectory file found in session {session_path}")


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
