import id_manual_tools.manual_tracking
from argparse import ArgumentParser
import os


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


def trajectory_correction():
    parser = ArgumentParser()
    parser.add_argument(
        "-s",
        metavar="session",
        type=dir_path,
        help="idTracker.ai succesful session directory",
        required=True,
    )
    parser.add_argument(
        "-video",
        type=file_path,
        help="Video file (only one file)",
        required=True,
    )
    parser.add_argument(
        "-o",
        metavar="output",
        type=str,
        help="Output file name, default is (input)_tracked.mp4",
        default=None,
    )
    parser.add_argument(
        "-t",
        metavar="time",
        type=float,
        help="Duration of the tracked video (in seconds), default is entire video",
        default=None,
    )
    parser.add_argument(
        "-check_jumps",
        action="store_true",
        default=False,
        help="Check for impossible long jumps on the trajectories",
    )

    parser.add_argument(
        "-reset",
        action="store_true",
        default=False,
        help="Ignores any previously edited file",
    )

    parser.add_argument(
        "-auto_validation",
        default=0,
        type=int,
        help="Max length of nan episode to apply auto-correction",
    )

    parser.add_argument(
        "-fps",
        default=0,
        type=int,
        help="Overwrite the frame rate of the session",
    )
    parser.add_argument(
        "-n",
        type=int,
        default=4,
        help="number of threads for parallel processing. Default is 4",
    )

    args = parser.parse_args()
    tracker = id_manual_tools.manual_tracking.manual_tracker(
        args.video,
        args.session,
        # "/home/jordi/session_0174/trajectories/trajectories.npy",
        # ignore_Existing_session=False,
        check_impossible_jumps=True,
        # automatic_check=3,
        setup_points="corners_out",
        fps=args.fps,
    )
