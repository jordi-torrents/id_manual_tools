import matplotlib.pyplot as plt
import numpy as np
import cv2
from rich import print
from matplotlib.cm import get_cmap
from functools import lru_cache
from matplotlib.collections import LineCollection
import os
from scipy.interpolate import interp1d
from multiprocessing import Process
from id_manual_tools.get_nans import get_list_of_nans_from_traj
from csv import writer as csv_writer
from rich.console import Console
from scipy.ndimage import center_of_mass
from id_manual_tools.set_corners import arg_main as set_corners
from id_manual_tools.matplotlib_gui import matplotlib_gui
from rich.table import Table
import shutil
from time import sleep
from id_manual_tools.utils import file_path, trajectory_path
from argparse import ArgumentParser
from scipy.spatial.distance import cdist

# from PyQt5.QtWidgets import QToolBar
console = Console()


class trajectory_corrector(matplotlib_gui):
    def __init__(
        self,
        video_path,
        traj_path,
        setup_points=None,
        jumps_check_sigma=None,
        automatic_check=-1,
        fps=None,
        n_cores=4,
    ):
        self.jumps_check_sigma = jumps_check_sigma
        console.rule("[green]Welcome to the id_manual_tools manual validator")
        self.automatic_check = automatic_check
        self.video_path = os.path.abspath(video_path)
        self.traj_path = os.path.abspath(traj_path)

        self.preloaded_frames_path = os.path.abspath("Preloaded_frames")
        video_path_file = os.path.join(self.preloaded_frames_path, "video_path.txt")

        os.makedirs(self.preloaded_frames_path, exist_ok=True)
        try:
            with open(video_path_file, "r") as file:
                if file.readline() != self.video_path:
                    raise FileNotFoundError
        except FileNotFoundError:
            print(
                f"Creating new preloaded frames directory: {self.preloaded_frames_path}"
            )
            shutil.rmtree(self.preloaded_frames_path)
            os.makedirs(self.preloaded_frames_path)
            with open(video_path_file, "w") as file:
                file.write(self.video_path)
        else:
            print(f"Reusing frames from {self.preloaded_frames_path}")

        # TODO ask for an exclusion area with set_corners and set all points inside as nans
        self.cap = cv2.VideoCapture(video_path)
        print(f"Loaded video {self.video_path}")

        self.data = np.load(self.traj_path, allow_pickle=True).item()
        print(f"Loaded data  {self.traj_path}")

        if setup_points is not None:
            try:
                exist_required_setup_points = setup_points in self.data["setup_points"]
            except (KeyError, TypeError):
                exist_required_setup_points = False
            while not exist_required_setup_points:
                print(
                    f"Setup_points setter is launched to define the user required setup_points {setup_points}"
                )
                set_corners(self.video_path, self.data)

                try:
                    exist_required_setup_points = (
                        setup_points in self.data["setup_points"]
                    )
                except (KeyError, TypeError):
                    exist_required_setup_points = False

            corners = self.data["setup_points"][setup_points]
            self.xmin = int(np.min(corners[:, 0]))
            self.xmax = int(np.max(corners[:, 0]))
            self.ymin = int(np.min(corners[:, 1]))
            self.ymax = int(np.max(corners[:, 1]))
        else:
            self.xmin = 0
            self.xmax = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            self.ymin = 0
            self.ymax = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

        if fps:
            if fps != self.data["frames_per_second"]:
                self.data["frames_per_second"] = fps
                print(f"Frames per second updated to {fps}")

        self.all_traj = self.data["trajectories"]
        self.total_frames, self.N = self.all_traj.shape[:2]
        assert self.total_frames == self.cap.get(cv2.CAP_PROP_FRAME_COUNT)

        self.N -= 1
        self.BL = self.data["body_length"]

        self.not_preloaded_frame = np.empty(self.total_frames, bool)
        for frame in range(self.total_frames):
            self.not_preloaded_frame[frame] = not os.path.exists(
                os.path.join(self.preloaded_frames_path, f"{frame}.npz")
            )

        copy_of_traj = np.zeros_like(self.all_traj)
        if jumps_check_sigma is not None:
            vel = np.linalg.norm(np.diff(self.all_traj, axis=0), axis=2)
            impossible_jumps = vel > (
                np.nanmean(vel) + jumps_check_sigma * np.nanstd(vel)
            )
            copy_of_traj[:-1][impossible_jumps] = np.nan
            print(f"Number of impossible jumps: {np.sum(impossible_jumps)}")
            self.list_of_jumps = get_list_of_nans_from_traj(
                copy_of_traj, sort_by="start"
            )
        else:
            self.list_of_jumps = []

        self.list_of_nans = get_list_of_nans_from_traj(self.all_traj, sort_by="start")

        self.write_lists_of_nans_and_jumps()

        self.limits = (self.xmin, self.xmax, self.ymin, self.ymax)
        print(f"xmin, xmax, ymin, ymax = {self.limits}")
        self.Lx = 0.5 * (self.xmax - self.xmin)
        self.Ly = 0.5 * (self.ymax - self.ymin)
        self.pad = 7
        self.pad_extra = 150
        self.actual_plotted_frame = -1
        self.G_pressed = False

        if self.list_of_nans or self.list_of_jumps:
            list_of_frames_to_preload = set()
            for id, start, end, duration in self.list_of_nans + self.list_of_jumps:
                if duration <= self.automatic_check:
                    list_of_frames_to_preload.add(max(0, start - 1))
                else:
                    pad = min(self.pad, duration)
                    for frame in range(
                        max(0, start - pad), min(self.total_frames, end + pad)
                    ):
                        list_of_frames_to_preload.add(frame)
            list_of_frames_to_preload = list(list_of_frames_to_preload)
            print(f"{len(list_of_frames_to_preload)} frames needed")
            list_of_frames_to_preload = [
                frame
                for frame in list_of_frames_to_preload
                if not os.path.exists(
                    os.path.join(self.preloaded_frames_path, f"{frame}.npz")
                )
            ]
            print(f"{len(list_of_frames_to_preload)} frames to preload")
            list_of_frames_to_preload.sort()
            if list_of_frames_to_preload:
                self.preload_frames_list(list_of_frames_to_preload, n_cores=n_cores)

            self.create_figure()

            self.Delta = 1
            if self.list_of_nans:
                self.next_episode(self.list_of_nans.pop(-1))
            elif self.list_of_jumps:
                self.next_episode(self.list_of_jumps.pop(-1))

            plt.show()
        else:
            if jumps_check_sigma is not None:
                print("[red]There's no nans nor impossible jumps to correct")
            else:
                print("[red]There's no nans to correct")

    def next_episode(self, params):
        self.id, self.start, self.end, _ = params

        console.rule(
            f"[bold red]Episode for fish {self.id} from {self.start} to {self.end}, {self.end-self.start} nans"
        )
        self.id_traj = self.all_traj[:, self.id, :]
        self.traj = np.delete(self.all_traj, self.id, axis=1)

        if self.N:
            temp = self.traj.reshape(-1, self.N, 1, 2)
            self.segments = np.concatenate([temp[:-1], temp[1:]], axis=2)

        self.frame = max(0, self.start - 1)

        self.zoom = 0.3

        if not np.isnan(self.id_traj[self.frame, 0]):
            self.x_center, self.y_center = self.id_traj[self.frame]
        else:
            self.x_center, self.y_center = np.nanmean(self.traj[self.frame], axis=0)
        self.set_ax_lims(do_not_draw=True)
        self.interpolation_range = np.arange(self.start, self.end)
        self.continuous_interpolation_range = np.arange(self.start - 1, self.end, 0.2)

        self.user_detection_history = []

        self.fit_interpolator_and_draw_frame()

        if (self.end - self.start) <= self.automatic_check:
            sleep(0.1)
            self.key_enter()

    def preload_frames_list(self, list_of_frames, n_cores=10):

        n_frames = len(list_of_frames)
        chunks = max(50, n_frames // n_cores)
        print(
            f"[red]Starting {len(range(0, n_frames, chunks))} processes of {chunks} frames each"
        )

        # self.process_frame_list_and_save(
        #     self.video_path,
        #     list_of_frames,
        #     self.process_image,
        #     self.limits,
        # )
        for s in range(0, n_frames, chunks):
            # self.not_preloaded_frame[start:end] = False
            Process(
                target=trajectory_corrector.process_frame_list_and_save,
                args=(
                    self.preloaded_frames_path,
                    self.video_path,
                    list_of_frames[s : s + chunks],
                    trajectory_corrector.process_image,
                    self.limits,
                ),
            ).start()

    def draw_frame(self):

        self.points.set_offsets(self.traj[self.frame])
        if self.frame in self.interpolation_range:
            self.id_point.set_offsets(self.interpolator(self.frame))
        else:
            self.id_point.set_offsets(self.id_traj[self.frame])
        self.interpolated_points.set_data(self.interpolator(self.interpolation_range))
        self.interpolated_line.set_data(
            self.interpolator(self.continuous_interpolation_range)
        )

        self.interpolated_train.set_data(*self.interpolator.y)

        if self.frame != self.actual_plotted_frame:
            self.im.set_data(self.get_frame(self.frame))

            # self.im._A = self.get_frame(self.frame)
            # self.im._imcache = None
            # self.im._rgbacache = None
            # self.im.stale = True

            self.text.set_text(f"Frame {self.frame}")
            self.actual_plotted_frame = self.frame

        origin = max(0, self.frame - 30)
        for fish in range(self.N):
            self.LineCollections[fish].set_segments(
                self.segments[origin : self.frame, fish]
            )
        self.draw_and_flush()

    def create_figure(self):
        super().__init__("Trajectory correction")

        (self.interpolated_line,) = self.ax.plot([], [], "w-", zorder=8)
        (self.interpolated_points,) = self.ax.plot([], [], "w.", zorder=8)
        (self.interpolated_train,) = self.ax.plot([], [], "r.", zorder=9)

        self.im = self.ax.imshow(
            [[]],
            cmap="gray",
            vmax=255,
            vmin=0,
            extent=(
                self.xmin,
                self.xmax,
                self.ymax,
                self.ymin,
            ),
            interpolation="none",
            animated=True,
            resample=False,
            snap=False,
        )

        cmap = get_cmap("gist_rainbow")
        self.points = self.ax.scatter(
            *np.zeros((2, self.N)),
            c=cmap(np.arange(self.N) / (self.N - 1)),
            s=10.0,
        )

        self.id_point = self.ax.scatter([], [], c="k", s=10.0, zorder=10)
        self.text = self.ax.text(
            0.1, 0.1, "", size=15, zorder=15, transform=self.ax.transAxes
        )

        line_lenght = 30
        self.LineCollections = []
        for i in range(self.N):
            color = np.tile(cmap(i / (max(1, self.N - 1))), (line_lenght, 1))
            color[:, -1] = np.linspace(0, 1, line_lenght)
            self.LineCollections.append(LineCollection([], linewidths=2, color=color))

        for linecollection in self.LineCollections:
            self.ax.add_collection(linecollection)

    def fit_interpolator_and_draw_frame(self):

        time_range = np.arange(
            max(0, self.start - (self.pad + self.pad_extra)),
            min(self.total_frames, self.end + (self.pad + self.pad_extra)),
        )

        time_range = time_range[~np.isnan(self.id_traj[time_range, 0])]

        self.interpolator = interp1d(
            time_range,
            self.id_traj[time_range].T,
            axis=1,
            kind="cubic",
            fill_value="extrapolate",
        )
        self.draw_frame()

    def key_a(self, draw=True):
        """Go back Delta timesteps"""
        self.frame = max(0, self.frame - self.Delta)
        if draw:
            self.draw_frame()

    def key_d(self, draw=True):
        """Advance Delta timesteps"""
        self.frame = min(self.total_frames - 1, self.frame + self.Delta)
        if draw:
            self.draw_frame()

    def key_left(self):
        """Go back Delta timesteps"""
        self.key_a()

    def key_right(self):
        """Advance Delta timesteps"""
        self.key_d()

    def key_P(self):
        """Toggle 1500 extra timesteps in the interpolator data"""
        if self.pad_extra == 1500:
            self.pad_extra = 0
        else:
            self.pad_extra = 1500

        self.fit_interpolator_and_draw_frame()

    def key_p(self):
        """Toggle 150 extra timesteps in the interpolator data"""
        if self.pad_extra == 150:
            self.pad_extra = 0
        else:
            self.pad_extra = 150

        self.fit_interpolator_and_draw_frame()

    def key_z(self):
        """Undo the last point defined by user in the interpolation range"""
        if self.user_detection_history:
            frame, position = self.user_detection_history.pop()
            self.id_traj[frame] = position

            self.fit_interpolator_and_draw_frame()

    def key_e(self):
        """Togle actual frame from end to start and recenter"""
        if self.frame == self.end:
            self.frame = max(0, self.start - 1)
        else:
            self.frame = self.end
        self.recenter()
        self.draw_frame()

    def key_n(self):
        """Sets the actual position to nan (only on the boundaries and inside of the interpolation range)"""
        if self.frame == (self.start - 1) or self.frame == self.end:

            if self.frame == (self.start - 1):
                self.id_traj[
                    max(0, self.frame - self.Delta + 1) : self.frame + 1
                ] = np.nan
                while np.isnan(self.id_traj[self.frame, 0]):
                    self.start -= 1
                    self.frame -= 1
                    if self.start == 0:
                        self.frame = 0
                        break
            elif self.frame == self.end:
                self.id_traj[
                    self.frame : min(self.total_frames, self.frame + self.Delta)
                ] = np.nan
                while np.isnan(self.id_traj[self.frame, 0]):
                    self.end += 1
                    self.frame += 1
                    if self.end == (self.total_frames - 1):
                        break

            self.interpolation_range = np.arange(self.start, self.end)
            self.continuous_interpolation_range = np.arange(
                self.start - 1, self.end + 0.1, 0.2
            )
            self.fit_interpolator_and_draw_frame()

        elif (self.end - self.frame) < self.Delta:

            self.id_traj[
                self.frame : min(self.total_frames, self.frame + self.Delta)
            ] = np.nan

            while np.isnan(self.id_traj[self.frame, 0]):
                self.frame += 1
                if self.frame == (self.total_frames - 1):
                    break
            self.end = self.frame

            self.fit_interpolator_and_draw_frame()
        elif self.frame in self.interpolation_range:
            self.id_traj[self.frame] = np.nan
            self.fit_interpolator_and_draw_frame()

        else:
            print(f"You are not on the boundaries, you are at frame {self.frame}")
            print(
                f"You only can set nan values on frames {self.start-1} and {self.end}"
            )

    def key_enter(self):
        """Accept the interpolation, write it to the trajectory array and move on (this doesn't write on disk)"""
        print(
            f"Writting interploation into the array from {self.start} to {self.end} for fish {self.id}"
        )
        self.id_traj[self.interpolation_range] = self.interpolator(
            self.interpolation_range
        ).T

        self.list_of_nans = get_list_of_nans_from_traj(self.all_traj, sort_by="start")

        if self.list_of_nans:
            self.next_episode(self.list_of_nans.pop(-1))
        elif self.list_of_jumps:
            self.next_episode(self.list_of_jumps.pop(-1))
        else:
            self.key_w()
            plt.close()

    def key_w(self):
        """Write on disk the actual state of the trajectory array"""
        print(f"Saving data to {self.traj_path}")
        np.save(self.traj_path, self.data)
        self.write_lists_of_nans_and_jumps()

    def write_lists_of_nans_and_jumps(self):
        with open("list_of_nans.csv", "w", newline="") as csvfile:
            csvfile.write("fish_id,start,end,duration\n")
            writer = csv_writer(csvfile)
            writer.writerows(self.list_of_nans)
        print(f"List of nans saved at {os.path.abspath('list_of_nans.csv')}")

        if self.jumps_check_sigma is not None:
            with open("list_of_jumps.csv", "w", newline="") as csvfile:
                csvfile.write("fish_id,start,end,duration\n")
                writer = csv_writer(csvfile)
                writer.writerows(self.list_of_jumps)
            print(f"List of jumps saved at {os.path.abspath('list_of_jumps.csv')}")

    def recenter(self):
        if self.frame in self.interpolation_range:
            self.x_center, self.y_center = self.interpolator(self.frame)
        else:
            self.x_center, self.y_center = self.id_traj[self.frame]
        self.set_ax_lims(do_not_draw=True)

    def key_g(self):
        """Apply key d and key x sequentially"""
        self.key_d(draw=False)
        self.key_x(recenter=True)

    def key_G(self):
        """Apply key g until the end of the episode is reached"""
        self.G_pressed = not self.G_pressed
        while self.G_pressed:
            self.key_g()
            if self.frame >= self.end:
                self.G_pressed = False
        self.draw_frame()

    def key_number(self, number):
        if number:
            self.Delta = 2 ** (number - 1)

    def key_h(self):
        """Shows Key Bindings table"""
        table = Table(title="Key Bindings")

        keys = [
            "d",
            "right",
            "a",
            "left",
            "p",
            "P",
            "n",
            "z",
            "x",
            "g",
            "enter",
            "w",
            "h",
        ]
        table.add_column("Key", justify="center", style="cyan", no_wrap=True)
        table.add_column("Description", justify="center", style="magenta")

        for key in keys:
            table.add_row(key, getattr(self, f"key_{key}").__doc__)

        table.add_row("1-9", "Set Delta to 1, 2, 4, 8, 16, 32, 64, 128, 256")
        table.add_row("s", "Save screenshot")
        table.add_row("f", "Toggle full screen")
        table.add_row("q", "Quit application")
        console.print(table)

    def key_x(self, recenter=False):
        """Set the actual position of the blob by finding the center of mass of the image drawned around the interpolated position"""
        if recenter:
            self.recenter()
        if self.frame in self.interpolation_range:
            self.find_blob(*self.interpolator(self.frame))

    @lru_cache(maxsize=1024)
    def get_frame(self, frame):
        path = os.path.join(self.preloaded_frames_path, f"{frame}.npz")
        if os.path.exists(path):
            return np.load(path)["arr_0"]

        print(f"[red]Had to load frame {frame}")
        if self.cap.get(cv2.CAP_PROP_POS_FRAMES) != frame:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
        ret, image = self.cap.read()
        assert ret

        image = self.process_image(image, *self.limits)
        np.save(path, image)
        return image
        # return np.ma.masked_invalid(
        #     self.process_image(image, *self.limits)
        # ).shrink_mask()

    @staticmethod
    def process_image(image, xmin, xmax, ymin, ymax):
        image = np.mean(image[ymin:ymax, xmin:xmax], axis=-1)
        image -= np.min(image)
        image *= 255 / np.max(image)
        image = np.uint8(image)
        return image

    def find_blob(self, x, y):
        if x <= self.xmin or x >= self.xmax or y >= self.ymax or y <= self.ymin:
            return

        canvas_x_min = max(0, int(x - self.BL - self.xmin))
        canvas_y_min = max(0, int(y - self.BL - self.ymin))

        canvas_center = (x - canvas_x_min - self.xmin, y - canvas_y_min - self.ymin)

        fish_im = self.get_frame(self.frame)[
            canvas_y_min : int(y + self.BL - self.ymin),
            canvas_x_min : int(x + self.BL - self.xmin),
        ]

        fish_im = cv2.GaussianBlur(fish_im, (0, 0), 2)

        _, mask = cv2.threshold(
            fish_im,
            0,
            1,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU,
        )

        contours, hierarchy = cv2.findContours(
            1 - mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
        )

        blobs_positions = []
        for c in contours:
            M = cv2.moments(c)
            blobs_positions.append((M["m10"] / M["m00"], M["m01"] / M["m00"]))

        closer_blob = cdist(
            [
                canvas_center,
            ],
            blobs_positions,
        ).argmin()

        x_c, y_c = blobs_positions[closer_blob]

        # fig, ax = plt.subplots()
        # ax.imshow(fish_im, cmap="gray")
        # ax.plot(*blobs_positions.T, "r.")
        # ax.plot(*canvas_center, "b.")
        # ax.plot(x_c, y_c, "g.")
        # fig.savefig("id_dev/fish", dpi=300)

        self.user_detection_history.append(
            (self.frame, tuple(self.id_traj[self.frame]))
        )
        self.id_traj[self.frame] = (
            x_c + canvas_x_min + self.xmin,
            y_c + canvas_y_min + self.ymin,
        )
        self.fit_interpolator_and_draw_frame()

    def button_3(self, event):
        self.find_blob(event.xdata, event.ydata)

    def button_1(self, event):
        self.user_detection_history.append(
            (self.frame, tuple(self.id_traj[self.frame]))
        )
        self.id_traj[self.frame] = event.xdata, event.ydata
        self.fit_interpolator_and_draw_frame()

    def on_key(self, event):
        try:
            int_key = int(event.key)
            if int_key in range(1, 10):
                self.Delta = 2 ** (int_key - 1)
        except ValueError:
            try:
                fun = getattr(self, f"key_{event.key}")
                fun()
            except AttributeError:
                pass

    @staticmethod
    def process_frame_list_and_save(
        save_dir, video_path, list_of_frames, process_fun, lims
    ):
        # cv2.setNumThreads(1)
        cap = cv2.VideoCapture(video_path)
        for frame in list_of_frames:
            if cap.get(cv2.CAP_PROP_POS_FRAMES) != frame:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
            ret, image = cap.read()
            assert ret
            np.savez_compressed(
                os.path.join(save_dir, f"{frame}"),
                process_fun(image, *lims),
            )
        print(
            f"Preloaded episode with frames {list_of_frames[0]} => {list_of_frames[-1]}"
        )


def main():
    parser = ArgumentParser()
    parser.add_argument(
        "-s",
        metavar="session",
        type=str,
        help="idTracker.ai succesful session directory or trajectory file",
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
        default=10,
        help="number of threads for parallel processing. Default is 4",
    )

    args = parser.parse_args()

    trajectory_corrector(
        args.video,
        trajectory_path(args.s, reset=args.reset),
        jumps_check_sigma=args.jumps_check_sigma,
        automatic_check=args.auto_validation,
        setup_points="corners_out",
        fps=args.fps,
        n_cores=args.n,
    )


if __name__ == "__main__":
    main()
