import id_manual_tools.manual_tracking
from id_manual_tools.utils import dir_path, file_path, get_trajectory_path
from argparse import ArgumentParser


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
        "-jumps_check_sigma",
        type=float,
        default=None,
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

    id_manual_tools.manual_tracking.manual_tracker(
        args.video,
        get_trajectory_path(args.s),
        ignore_Existing_session=args.reset,
        jumps_check_sigma=args.jumps_check_sigma,
        automatic_check=args.auto_validation,
        setup_points="corners_out",
        fps=args.fps,
    )
