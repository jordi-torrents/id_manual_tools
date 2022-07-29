import numpy as np
from statistics import mode, StatisticsError
import matplotlib.pyplot as plt
import os
from id_manual_tools import utils
from argparse import ArgumentParser
from rich import print

# from idmatcherai.idmatcherai import IdMatcherAi
# TODO implement idmatcher
# TODO add min distance check to match identities if idmatcher doesn't work (or to double-check)


def arg_main(
    output_path: str,
    session_names,
    sessions_dir: str,
    algorithm_indx=str,
):

    permutation_algorithms = (
        "mode",
        "max_P1",
        "max_freq",
        "greedy",
        "hungarian_P1",
        "hungarian_freq",
    )

    try:
        permutation_algorithm = permutation_algorithms[int(algorithm_indx)]
    except ValueError:
        if algorithm_indx in permutation_algorithms:
            permutation_algorithm = algorithm_indx
        else:
            raise ValueError(f"Invalid permutation algorithm: {algorithm_indx}")
    except IndexError:
        raise ValueError(f"Invalid permutation algorithm: {algorithm_indx}")

    print(f"Using [green]{permutation_algorithm}")

    assert len(session_names) > 1

    session_paths = [
        os.path.abspath(f"{sessions_dir}/session_{s}") for s in session_names
    ]
    assert all(os.path.exists(session_path) for session_path in session_paths)
    print("Sessions to concatenate:", session_paths)

    main_out = np.load(
        utils.trajectory_path(session_paths[0], read_only=True),
        allow_pickle=True,
    ).item()

    N = main_out["trajectories"].shape[1]
    main_out["video_path"] = [main_out["video_path"]]
    main_out["body_length"] = [main_out["body_length"]]
    main_out["stats"] = [main_out["stats"]]

    N_concatenations = len(session_names) - 1
    n_cols = int(np.sqrt(N_concatenations) + 0.5)
    n_rows = N_concatenations // n_cols + (N_concatenations % n_cols > 0)
    fig, ax = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3, n_rows * 3))

    if (n_cols * n_rows) == 1:
        ax = [ax]
    else:
        ax = ax.flatten()

    fpss = [main_out["frames_per_second"]]
    permutation = np.empty(N, int)

    for i in range(1, len(session_names)):

        id_matcher_file_path = (
            session_paths[i] + "/matching_results/" + session_names[0] + "-.npy"
        )

        if not os.path.exists(id_matcher_file_path):
            raise ValueError(f"session {session_names[i]} has not been idmatched")
            # print(f"Launching idmatcher for session {session_paths[i]}")
            # IdMatcherAi(session_paths[0], session_paths[i]).match_identities()
            # assert os.path.exists(id_matcher_file_path), id_matcher_file_path

        matcher = np.load(id_matcher_file_path, allow_pickle=True,).item()[
            "matching_results_B_A"
        ]["transfer_dicts"]

        traj_path = utils.trajectory_path(session_paths[i], read_only=True)
        data = np.load(traj_path, allow_pickle=True).item()

        for n in range(N):
            id = n + 1
            if permutation_algorithm == "mode":
                permutation[n] = mode(
                    (
                        matcher["max_P1"]["assignments"][id],
                        matcher["max_freq"]["assignments"][id],
                        matcher["greedy"]["assignments"][id],
                        matcher["hungarian_P1"]["assignments"][id],
                        matcher["hungarian_freq"]["assignments"][id],
                    )
                )
            else:
                permutation[n] = matcher[permutation_algorithm]["assignments"][id]
        permutation = permutation - 1

        l = len(main_out["trajectories"])

        main_out["trajectories"] = np.concatenate(
            (main_out["trajectories"], data["trajectories"][:, permutation])
        )
        main_out["areas"] = np.concatenate(
            (main_out["areas"], data["areas"][:, permutation])
        )
        main_out["id_probabilities"] = np.concatenate(
            (
                main_out["id_probabilities"],
                data["id_probabilities"][:, permutation],
            )
        )

        main_out["video_path"].append(data["video_path"])
        main_out["body_length"].append(data["body_length"])
        main_out["stats"].append(data["stats"])

        assert main_out["version"] == data["version"]
        fpss.append(data["frames_per_second"])
        # assert all(
        #     [i == j for i, j in zip(main_out["setup_points"], data["setup_points"])]
        # )
        assert all(
            [
                i == j
                for i, j in zip(
                    main_out["identities_groups"], data["identities_groups"]
                )
            ]
        )

        ax[i - 1].set(
            title=(session_names[i - 1] + " + \n" + session_names[i]),
            aspect=1,
            xticks=(),
            yticks=(),
        )
        ax[i - 1].plot(
            main_out["trajectories"][l - 4 : l + 5, :, 0],
            main_out["trajectories"][l - 4 : l + 5, :, 1],
            ".-",
            ms=2,
            lw=1,
        )

    if len(set(fpss)) > 1:
        print(f"Differences in FPS: {fpss}.")
        try:
            fps = mode(fpss)
            print(f"We use the mode ({fps})")
        except StatisticsError:
            fps = int(input("There's no defined mode, enter the desired value: "))
        fps = 50
        main_out["frames_per_second"] = fps

    main_out["body_length"] = np.mean(main_out["body_length"])
    plt.tight_layout(pad=0.3)
    plt.show()
    fig.savefig(output_path + ".png", dpi=300)
    np.save(output_path, main_out)
    print("Concatenated data saved at", os.path.abspath(output_path + ".npy"))


def main():
    parser = ArgumentParser(
        description="Concatenate various trajectory files using idmatcher"
    )

    parser.add_argument(
        "sessions",
        help="idTracker.ai successful sessions directories or trajectory files to concatenate (ordered)",
        type=str,
        action="store",
        nargs="*",
    )
    parser.add_argument(
        "-dir",
        help="dir where to find sessions folders, default is current dir",
        type=str,
        default="./",
    )
    parser.add_argument(
        "-o",
        help="output file for concatenated trajectories, default is './concatenated_trajectories'",
        type=str,
        default="concatenated_trajectories",
    )

    parser.add_argument(
        "-algorithm",
        help='permutation algorithm. One of {0: "mode", 1: "max_P1", 2: "max_freq", 3: "greedy", 4: "hungarian_P1", 5: "hungarian_freq"}',
        type=str,
        default="0",
    )

    args = parser.parse_args()
    arg_main(args.o, args.sessions, args.dir, args.algorithm)


if __name__ == "__main__":
    main()
