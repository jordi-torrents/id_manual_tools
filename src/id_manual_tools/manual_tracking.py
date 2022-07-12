# import matplotlib

# # matplotlib.use("QtAgg")
# # matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import numpy as np
import cv2
from rich import print

# from rich.progress import track
from matplotlib.cm import get_cmap

# from PyQt5.QtWidgets import QToolBar
from functools import lru_cache
from matplotlib.collections import LineCollection
import os
from scipy.interpolate import interp1d
from multiprocessing import Process
from get_nans import get_list_of_nans_from_traj
from csv import writer as csv_writer
from rich.console import Console
from cv2 import threshold
from scipy.ndimage import center_of_mass
from set_corners import main as set_corners

# import matplotlib.cbook as cbook
from time import sleep

console = Console()


class manual_tracker:
    def __init__(
        self,
        video_path,
        traj_path,
        setup_points=None,
        ignore_Existing_session=False,
        check_impossible_jumps=False,
        automatic_check=0,
        fps=None,
    ):
        self.automatic_check = automatic_check
        os.makedirs("tmp", exist_ok=True)
        self.video_path = os.path.abspath(video_path)
        self.traj_path = os.path.abspath(traj_path)
        self.cap = cv2.VideoCapture(video_path)
        print(f"Loaded {self.video_path}")

        modified_path = self.traj_path[:-4] + "_corrected.npy"
        if os.path.exists(modified_path) and not ignore_Existing_session:
            self.data = np.load(modified_path, allow_pickle=True).item()
            print(f"Loaded {modified_path}")
        else:
            self.data = np.load(self.traj_path, allow_pickle=True).item()
            print(f"Loaded {self.traj_path}")

        if setup_points is not None:
            if isinstance(self.data["setup_points"], dict):
                if not setup_points in self.data["setup_points"]:
                    set_corners(self.video_path, self.traj_path)
                    if os.path.exists(modified_path) and not ignore_Existing_session:
                        self.data = np.load(modified_path, allow_pickle=True).item()
                        print(f"Loaded {modified_path}")
                    else:
                        self.data = np.load(self.traj_path, allow_pickle=True).item()
                        print(f"Loaded {self.traj_path}")
            else:
                set_corners(self.video_path, self.traj_path)
                if os.path.exists(modified_path) and not ignore_Existing_session:
                    self.data = np.load(modified_path, allow_pickle=True).item()
                    print(f"Loaded {modified_path}")
                else:
                    self.data = np.load(self.traj_path, allow_pickle=True).item()
                    print(f"Loaded {self.traj_path}")

        else:
            raise NotImplementedError
        corners = self.data["setup_points"][setup_points]

        if fps:
            if fps != self.data["frames_per_second"]:
                self.data["frames_per_second"] = fps
                print(f"Frames per second updated to {fps}")
        self.all_traj = self.data["trajectories"]
        self.total_frames, self.N = self.all_traj.shape[:2]
        self.N -= 1
        self.BL = self.data["body_length"]

        self.not_preloaded_frame = np.empty(self.total_frames, bool)
        for frame in range(self.total_frames):
            self.not_preloaded_frame[frame] = not os.path.exists(f"tmp/{frame}.npz")

        if check_impossible_jumps:
            vel = np.linalg.norm(np.diff(self.all_traj, axis=0), axis=2)
            impossible_jumps = vel > (np.nanmean(vel) + 8 * np.nanstd(vel))
            self.all_traj[:-1][impossible_jumps] = np.nan
            print(f"Number of impossible jumps: {np.sum(impossible_jumps)}")

        self.list_of_nans = get_list_of_nans_from_traj(self.all_traj, sort_by="start")

        output = os.path.abspath("./list_of_nans.csv")
        with open(output, "w", newline="") as csvfile:
            csvfile.write("fish_id,start,end,duration\n")
            writer = csv_writer(csvfile)
            writer.writerows(self.list_of_nans)
        print(f"List of nans saved at {output}")

        self.xmin = int(np.min(corners[:, 0]))
        self.xmax = int(np.max(corners[:, 0]))
        self.ymin = int(np.min(corners[:, 1]))
        self.ymax = int(np.max(corners[:, 1]))

        self.limits = (self.xmin, self.xmax, self.ymin, self.ymax)
        print(self.limits)
        self.Lx = 0.5 * (self.xmax - self.xmin)
        self.Ly = 0.5 * (self.ymax - self.ymin)
        self.pad = 7
        self.pad_extra = 150
        self.pad_extra_applied = 0
        self.actual_plotted_frame = -1

        if self.list_of_nans:
            list_of_frames_to_preload = set()
            for id, start, end, duration in self.list_of_nans:
                pad = min(self.pad, end - start)
                for frame in range(
                    max(0, start - self.pad), min(self.total_frames, end + self.pad)
                ):
                    list_of_frames_to_preload.add(frame)
            list_of_frames_to_preload = list(list_of_frames_to_preload)
            print(f"{len(list_of_frames_to_preload)} frames needed")
            list_of_frames_to_preload = [
                frame
                for frame in list_of_frames_to_preload
                if not os.path.exists(f"tmp/{frame}.npz")
            ]
            print(f"{len(list_of_frames_to_preload)} frames to preload")
            list_of_frames_to_preload.sort()
            if list_of_frames_to_preload:
                self.preload_frames_list(list_of_frames_to_preload)

            self.create_figure()
            self.Delta = 1
            self.next_episode(self.list_of_nans.pop(-1))
            self.mouse_pressed = False

            plt.show()
        else:
            if check_impossible_jumps:
                print("[red]There's no nans nor impossible jumps to correct")
            else:
                print("[red]There's no nans to correct")

    def next_episode(self, params, hard=True):
        self.id, self.start, self.end, _ = params

        if hard:
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

        self.interpolation_range = np.arange(self.start, self.end)
        self.continuous_interpolation_range = np.arange(self.start - 1, self.end, 0.2)

        if hard:
            self.user_detection_history = []
            self.set_ax_lims(do_not_draw=True)

        self.fit_interpolator_and_draw_frame()
        if hard and (self.end - self.start) <= self.automatic_check:
            sleep(0.1)
            self.key_enter()

    def preload_frames_list(self, list_of_frames, n_cores=10):

        n_frames = len(list_of_frames)
        chunks = max(50, n_frames // n_cores)
        print(
            f"[red]Starting {len(range(0, n_frames, chunks))} processes of {chunks} frames"
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
                target=manual_tracker.process_frame_list_and_save,
                args=(
                    self.video_path,
                    list_of_frames[s : s + chunks],
                    manual_tracker.process_image,
                    self.limits,
                ),
            ).start()

    def preload_frames(self, start, end, pad=0):
        start = max(0, start - pad)
        end = min(self.total_frames, end + pad)

        for s in range(start, end, 50):
            self.not_preloaded_frame[start:end] = False
            Process(
                target=manual_tracker.process_frame_and_save,
                args=(
                    self.video_path,
                    s,
                    min(s + 50, end),
                    manual_tracker.process_image,
                    self.limits,
                ),
            ).start()

    def draw_frame(self):
        # line.set_data(
        #     [x - 60, x + 60, x + 60, x - 60, x - 60],
        #     [y + 60, y + 60, y - 60, y - 60, y + 60],
        # )
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
            self.lines[fish].set_segments(self.segments[origin : self.frame, fish])
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

        # frames_that_should_be_preloaded = np.arange(
        #     max(0, self.frame - 15), min(self.total_frames, self.frame + 15)
        # )
        # frame_to_preload = frames_that_should_be_preloaded[
        #     self.not_preloaded_frame[frames_that_should_be_preloaded]
        # ]

        # if len(frame_to_preload) > 16:
        #     print(
        #         f"[gray]Sending frames from {frame_to_preload[0]} to {frame_to_preload[-1]} to preload"
        #     )
        #     self.preload_frames(frame_to_preload[0], frame_to_preload[-1])

    def create_figure(self):

        Lx = 0.5 * (self.xmax - self.xmin)
        Ly = 0.5 * (self.ymax - self.ymin)
        self.fig = plt.figure(figsize=(Lx / 150, Ly / 150))

        self.ax = self.fig.add_axes(
            [0, 0, 1, 1],
            xticks=(),
            yticks=(),
            facecolor="gray",
        )

        self.canvas_size = self.fig.get_size_inches() * self.fig.dpi
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

        # self.fig.canvas.manager.window.findChild(QToolBar).setVisible(False)
        self.fig.canvas.manager.set_window_title("Manual Tracking")
        self.fig.canvas.mpl_connect("button_press_event", self.on_click)
        self.fig.canvas.mpl_connect("button_release_event", self.on_click_release)
        self.fig.canvas.mpl_connect("key_release_event", self.on_key)
        self.fig.canvas.mpl_connect("scroll_event", self.on_scroll)
        self.fig.canvas.mpl_connect("motion_notify_event", self.on_motion)
        self.fig.canvas.mpl_connect("resize_event", self.on_resize)

        cmap = get_cmap("gist_rainbow")

        (self.interpolated_line,) = self.ax.plot([], [], "w-", zorder=8)
        (self.interpolated_points,) = self.ax.plot([], [], "w.", zorder=8)
        (self.interpolated_train,) = self.ax.plot([], [], "r.", zorder=9)

        self.points = self.ax.scatter(
            *np.zeros((2, self.N)),
            c=cmap(np.arange(self.N) / (self.N - 1)),
            s=10.0,
        )

        self.id_point = self.ax.scatter([], [], c="k", s=10.0, zorder=10)
        self.text = self.ax.text(
            0.1, 0.1, "", size=15, transform=self.ax.transAxes, zorder=15
        )

        line_lenght = 30
        self.lines = []  # DON'T ASK...
        for i in range(self.N):
            lc = LineCollection([], linewidths=2)
            color = np.tile(cmap(i / (max(1, self.N - 1))), (line_lenght, 1))
            color[:, -1] = np.linspace(0, 1, line_lenght)
            lc.set_color(color)
            self.ax.add_collection(lc)
            self.lines.append(lc)

    def fit_interpolator_and_draw_frame(self):

        time_range = np.arange(
            max(0, self.start - (self.pad + self.pad_extra * self.pad_extra_applied)),
            min(
                self.total_frames,
                self.end + (self.pad + self.pad_extra * self.pad_extra_applied),
            ),
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

    def key_a(self):
        self.frame = max(0, self.frame - self.Delta)
        self.draw_frame()

    def key_d(self):
        self.frame = min(self.total_frames - 1, self.frame + self.Delta)
        self.draw_frame()

    def key_left(self):
        self.key_a()

    def key_right(self):
        self.key_d()

    def key_P(self):
        if self.pad_extra_applied == 10:
            self.pad_extra_applied = 0
        else:
            self.pad_extra_applied = 10

        self.fit_interpolator_and_draw_frame()

    def key_p(self):
        if self.pad_extra_applied == 1:
            self.pad_extra_applied = 0
        else:
            self.pad_extra_applied = 1

        self.fit_interpolator_and_draw_frame()

    def key_z(self):
        if self.user_detection_history:
            frame, position = self.user_detection_history.pop()
            self.id_traj[frame] = position

            self.fit_interpolator_and_draw_frame()

    def key_n(self):
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

            self.next_episode((self.id, self.start, self.end, None), hard=False)
        else:
            print(f"You are not on the boundaries, you are at frame {self.frame}")
            print(
                f"You only can set nan values on frames {self.start-1} and {self.end}"
            )

    def key_enter(self):

        print(
            f"Writting interploation into the array from {self.start} to {self.end} for fish {self.id}"
        )
        self.id_traj[self.interpolation_range] = self.interpolator(
            self.interpolation_range
        ).T
        self.list_of_nans = get_list_of_nans_from_traj(self.all_traj, sort_by="start")
        if self.list_of_nans:
            self.next_episode(self.list_of_nans.pop(-1))
        else:
            self.key_w()
            plt.close()

    def key_w(self):
        out_path = self.traj_path[:-4] + "_corrected.npy"
        print(f"Saving data to {out_path}")
        np.save(out_path, self.data)
        output = os.path.abspath("./list_of_nans.csv")
        with open(output, "w", newline="") as csvfile:
            csvfile.write("fish_id,start,end,duration\n")
            writer = csv_writer(csvfile)
            writer.writerows(self.list_of_nans)
        print(f"List of nans saved at {output}")

    def key_g(self):
        self.key_d()
        self.key_x()

    def key_x(self):
        if self.frame in self.interpolation_range:
            x, y = self.interpolator(self.frame)

            fish_im = (
                255
                - self.get_frame(self.frame)[
                    int(y - 0.7 * self.BL - self.ymin) : int(
                        y + 0.7 * self.BL - self.ymin
                    ),
                    int(x - 0.7 * self.BL - self.xmin) : int(
                        x + 0.7 * self.BL - self.xmin
                    ),
                ]
            )

            _, fish_im = threshold(fish_im, 150, 255, 3)  # THRESH_TOZERO
            y_c, x_c = center_of_mass(fish_im)

            self.user_detection_history.append(
                (self.frame, tuple(self.id_traj[self.frame]))
            )
            self.id_traj[self.frame] = x_c + x - 0.7 * self.BL, y_c + y - 0.7 * self.BL
            self.fit_interpolator_and_draw_frame()

    @lru_cache(maxsize=1024)
    def get_frame(self, frame):
        if os.path.exists(f"tmp/{frame}.npz"):
            return np.load(f"tmp/{frame}.npz")["arr_0"]

        print(f"[red]Had to load frame {frame}")
        if self.cap.get(cv2.CAP_PROP_POS_FRAMES) != frame:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
        ret, image = self.cap.read()
        assert ret

        image = self.process_image(image, *self.limits)
        np.save(f"tmp/{frame}", image)
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

    def on_click(self, event):
        if event.button == 1:
            self.has_moved = False
            self.mouse_pressed = True
            self.click_origin = (event.x, event.y)

    def on_click_release(self, event):
        self.mouse_pressed = False
        if event.button == 1:
            if not self.has_moved:
                self.user_detection_history.append(
                    (self.frame, tuple(self.id_traj[self.frame]))
                )
                self.id_traj[self.frame] = event.xdata, event.ydata
                self.fit_interpolator_and_draw_frame()

        if event.button == 3:
            x, y = event.xdata, event.ydata

            fish_im = (
                255
                - self.get_frame(self.frame)[
                    int(y - self.BL - self.ymin) : int(y + self.BL - self.ymin),
                    int(x - self.BL - self.xmin) : int(x + self.BL - self.xmin),
                ]
            )

            _, fish_im = threshold(fish_im, 127, 255, 3)  # THRESH_TOZERO
            # fig2, ax2 = plt.subplots()
            # ax2.imshow(fish_im)
            # fig2.savefig("res.png")
            y_c, x_c = center_of_mass(fish_im)

            self.user_detection_history.append(
                (self.frame, tuple(self.id_traj[self.frame]))
            )
            self.id_traj[self.frame] = x_c + x - self.BL, y_c + y - self.BL
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

    def on_scroll(self, event):
        self.zoom += 0.1 * self.zoom * event.step
        self.set_ax_lims()

    def on_motion(self, event):
        if self.mouse_pressed:
            self.has_moved = True
            self.x_center -= (
                2
                * self.zoom
                * self.Lx
                * (event.x - self.click_origin[0])
                / self.canvas_size[0]
            )
            self.y_center += (
                2
                * self.zoom
                * self.Ly
                * (event.y - self.click_origin[1])
                / self.canvas_size[1]
            )
            self.click_origin = (event.x, event.y)
            self.set_ax_lims()

    def on_resize(self, event):
        self.Ly = event.height * self.Ly / self.canvas_size[1]
        self.Lx = event.width * self.Lx / self.canvas_size[0]
        self.canvas_size = (event.width, event.height)
        self.set_ax_lims()

    def set_ax_lims(self, do_not_draw=False):
        self.ax.set(
            xlim=(
                self.x_center - self.zoom * self.Lx,
                self.x_center + self.zoom * self.Lx,
            ),
            ylim=(
                self.y_center + self.zoom * self.Ly,
                self.y_center - self.zoom * self.Ly,
            ),
        )
        if not do_not_draw:
            self.fig.canvas.draw()

    @staticmethod
    def process_frame_list_and_save(video_path, list_of_frames, process_fun, lims):
        # cv2.setNumThreads(1)
        cap = cv2.VideoCapture(video_path)
        for frame in list_of_frames:
            if cap.get(cv2.CAP_PROP_POS_FRAMES) != frame:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
            ret, image = cap.read()
            assert ret
            np.savez_compressed(
                f"tmp/{frame}",
                process_fun(image, *lims),
            )
        print(
            f"Preloaded episode with frames {list_of_frames[0]} => {list_of_frames[-1]}"
        )

    @staticmethod
    def process_frame_and_save(video_path, start, end, process_fun, lims):
        # cv2.setNumThreads(1)
        frames = [
            frame
            for frame in range(start, end)
            if not os.path.exists(f"tmp/{frame}.npz")
        ]

        if frames:
            cap = cv2.VideoCapture(video_path)
            for frame in frames:
                if cap.get(cv2.CAP_PROP_POS_FRAMES) != frame:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
                ret, image = cap.read()
                assert ret
                # np.savez_compressed(
                #     f"tmp/{frame}",
                #     np.ma.masked_invalid(process_fun(image, *lims)).shrink_mask(),
                # )
                np.savez_compressed(
                    f"tmp/{frame}",
                    process_fun(image, *lims),
                )
            print(f"Preloaded episode with frames {start} => {end}")
        else:
            print(f"Frames {start} => {end} were already preloaded")


def test():
    print("Hello world!")


if __name__ == "__main__":
    tracker = manual_tracker(
        "/home/jordi/0155.MP4",
        "/home/jordi/session_0155/trajectories_wo_gaps/trajectories_wo_gaps.npy",
        # "/home/jordi/session_0174/trajectories/trajectories.npy",
        # ignore_Existing_session=False,
        check_impossible_jumps=True,
        # automatic_check=3,
        setup_points="corners_out",
        fps=50,
    )
