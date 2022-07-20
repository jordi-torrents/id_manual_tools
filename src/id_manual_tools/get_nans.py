import numpy as np
import csv
from id_manual_tools.utils import trajectory_path


def find_nans_1D(data, id=None):
    """Returns a list of the nans location for a 1D array:

        (fish_id (optional), start, end, length of slice)

    In such a way that data[start] is the first nan and data[end-1] is the last nan
    """
    assert (
        data.ndim == 1
    ), f"Only one dimensional arrays, the given array has shape {data.shape}"

    dif = np.diff(np.concatenate(([-1], np.where(~np.isnan(data))[0], [len(data)])))
    end = np.cumsum(dif) - 1
    nan = dif - 1
    valid = nan > 0
    if id is not None:
        return [(id, e - n, e, n) for e, n in zip(end[valid], nan[valid])]
    else:
        return [(e - n, e, n) for e, n in zip(end[valid], nan[valid])]


def get_list_of_nans_from_traj(traj, sort_by="length"):
    """Returns a list of the nans location for a trajectory 2D/3D array:

        (fish_id (only for 3D arrays), start, end, length of slice)

    In such a way that traj[fish_id, start] is the first nan and
    traj[fish_id, end-1] is the last nan.
    Elements in the list are sorted by length of slice (reverse)
    """
    if traj.ndim == 3:
        traj = traj[..., 0]

    if traj.ndim == 1:
        nans = find_nans_1D(traj)
    else:
        nans = []
        for fish_id in range(traj.shape[1]):
            nans += find_nans_1D(traj[:, fish_id], fish_id)

    if sort_by == "length":
        nans.sort(key=lambda x: x[-1], reverse=True)
    elif sort_by == "end":
        nans.sort(key=lambda x: x[-2], reverse=True)
    elif sort_by == "start":
        nans.sort(key=lambda x: x[-3], reverse=True)
    elif sort_by == "id":
        nans.sort(key=lambda x: x[-4], reverse=True)
    return nans


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Nans observation in idtrackeri.ai trajectories."
    )
    parser.add_argument(
        "-i",
        type=str,
        help="idTracker.ai succesful session directory or trajectory file",
    )
    parser.add_argument(
        "-o", type=str, default=None, help="output file, default input[:4]+'_nans.csv'"
    )
    args = parser.parse_args()

    input_path = trajectory_path(args.i, read_only=True)

    output_path = args.o if args.o else input_path[:-4] + "_nans.csv"

    traj = np.load(input_path, allow_pickle=True).item()["trajectories"][..., 0]

    nans = get_list_of_nans_from_traj(traj)

    with open(output_path, "w", newline="") as csvfile:
        csvfile.write("fish_id,start,end,duration\n")
        writer = csv.writer(csvfile)
        writer.writerows(nans)
    print(f"File saved at {output_path}")


if __name__ == "__main__":
    main()
