import numpy as np
from statistics import mode
import matplotlib.pyplot as plt


def main():
    video = "20220613_0146"
    sessions_name = "0102030405060146", "0708091011120146"

    sessions = ["/home/jordi/session_" + s + "/" for s in sessions_name]

    main_out = np.load(
        sessions[0] + "trajectories_wo_gaps/trajectories_wo_gaps_corrected.npy",
        allow_pickle=True,
    ).item()

    N = main_out["trajectories"].shape[1]
    main_out["video_path"] = [main_out["video_path"]]
    main_out["body_length"] = [main_out["body_length"]]
    main_out["stats"] = [main_out["stats"]]

    N_concatenations = len(sessions) - 1
    n_cols = int(np.sqrt(N_concatenations)) + 1
    n_rows = int(N_concatenations / n_cols) + 1
    fig, ax = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3, n_rows * 3))
    ax = ax.flatten()

    fpss = [main_out["frames_per_second"]]

    for ax_i, session in enumerate(sessions[1:]):
        matcher = np.load(
            session + "matching_results/" + sessions_name[0] + "-.npy",
            allow_pickle=True,
        ).item()["matching_results_B_A"]["transfer_dicts"]

        data = np.load(
            session + "trajectories_wo_gaps/trajectories_wo_gaps_corrected.npy",
            allow_pickle=True,
        ).item()

        permutation = np.empty(N, int)

        # for i in range(N):
        #     id = i + 1
        #     permutation[i] = (
        #         mode(
        #             [
        #                 matcher["max_P1"]["assignments"][id],
        #                 matcher["max_freq"]["assignments"][id],
        #                 matcher["greedy"]["assignments"][id],
        #                 matcher["hungarian_P1"]["assignments"][id],
        #                 matcher["hungarian_freq"]["assignments"][id],
        #             ]
        #         )
        #         - 1
        #     )

        for i in range(N):
            id = i + 1
            permutation[i] = (
                # matcher["max_P1"]["assignments"][id]
                # matcher["max_freq"]["assignments"][id]
                # matcher["greedy"]["assignments"][id]
                # matcher["hungarian_P1"]["assignments"][id]
                matcher["hungarian_freq"]["assignments"][id]
                - 1
            )

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
        assert all(
            [i == j for i, j in zip(main_out["setup_points"], data["setup_points"])]
        )
        assert all(
            [
                i == j
                for i, j in zip(
                    main_out["identities_groups"], data["identities_groups"]
                )
            ]
        )

        ax[ax_i].set(title="adding " + session, aspect=1, xticks=(), yticks=())
        ax[ax_i].plot(
            main_out["trajectories"][l - 4 : l + 5, :, 0],
            main_out["trajectories"][l - 4 : l + 5, :, 1],
            ".-",
            ms=2,
            lw=1,
        )

    if len(set(fpss)) > 1:
        fps = mode(fpss)
        main_out["frames_per_second"] = fps
        print(f"Differences in FPS: {fpss}. We use the mode ({fps})")

    main_out["body_length"] = np.mean(main_out["body_length"])
    plt.tight_layout(pad=0.3)

    fig.savefig(video + ".png", dpi=300)
    np.save(video, main_out)


if __name__ == "__main__":
    main()
