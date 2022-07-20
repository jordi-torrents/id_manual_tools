import matplotlib.pyplot as plt
import numpy as np
import cv2
from rich import print
from functools import lru_cache
import os
from rich.console import Console
from id_manual_tools.matplotlib_gui import matplotlib_gui
from rich.table import Table
from id_manual_tools.utils import trajectory_path
from argparse import ArgumentParser
from rich.align import Align


console = Console()


class setup_points_setter(matplotlib_gui):
    def __init__(self, video_path, data, name):

        self.cap = cv2.VideoCapture(video_path)
        self.total_frames = self.cap.get(cv2.CAP_PROP_FRAME_COUNT)
        print(f"Loaded {os.path.abspath(video_path)}")

        self.data = data

        self.setup_points = {}
        for name_i, points in self.data["setup_points"].items():
            self.setup_points[name_i] = [list(points[:, 0]), list(points[:, 1])]

        self.name = name

        if not self.name in self.setup_points:
            self.setup_points[self.name] = [[], []]
            console.rule(f"[bold red]Creating setup points: {self.name}")
        else:
            console.rule(f"[bold red]Modifying setup points: {self.name}")

        self.xmax = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.ymax = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

        self.create_figure()
        self.actual_plotted_frame = -1
        self.frame = 0
        self.x_center = self.xmax / 2
        self.y_center = self.ymax / 2

        self.zoom = self.ymax / (2 * self.Ly)
        self.set_ax_lims(do_not_draw=True)

        self.draw_frame()
        plt.show()

    def draw_frame(self):

        if self.frame != self.actual_plotted_frame:
            self.im.set_data(self.get_frame(self.frame))

            self.text.set_text(f"Frame {self.frame}")
            self.actual_plotted_frame = self.frame

        self.draw_and_flush()

    def create_figure(self):

        super().__init__("Setup points editor")

        self.im = self.ax.imshow(
            [[[0, 0, 0]]],
            extent=(
                0,
                self.xmax,
                self.ymax,
                0,
            ),
            interpolation="none",
            animated=True,
        )

        self.lines = {
            name: self.ax.plot(*self.close_line(*xy), ":.", label=name)[0]
            for name, xy in self.setup_points.items()
        }
        self.ax.legend()

        self.text = self.ax.text(
            0.1, 0.1, "", size=15, transform=self.ax.transAxes, zorder=15
        )

    @staticmethod
    def close_line(x, y):
        if len(x) < 3:
            return x, y
        else:
            return x + [x[0]], y + [y[0]]

    def key_a(self):
        self.frame = max(0, self.frame - 1)
        self.draw_frame()

    def key_d(self):
        self.frame = min(self.total_frames - 1, self.frame + 1)
        self.draw_frame()

    def key_left(self):
        self.key_a()

    def key_right(self):
        self.key_d()

    def key_enter(self):
        out_dict = {}
        for name, points in self.setup_points.items():
            out_dict[name] = np.array(points).astype(int).T
        self.data["setup_points"] = out_dict
        print(f"[green]Data writed in trajectory file")
        plt.close()

    @lru_cache(maxsize=20)
    def get_frame(self, frame):

        if self.cap.get(cv2.CAP_PROP_POS_FRAMES) != frame:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
        ret, image = self.cap.read()
        assert ret

        return image

    @staticmethod
    def closest_point(x, y, xs, ys):
        distances = [(x - x_i) ** 2 + (y - y_i) ** 2 for x_i, y_i in zip(xs, ys)]
        return np.argmin(distances), np.min(distances)

    @staticmethod
    def sort_points(x, y):

        atan2 = np.arctan2(y - np.mean(y), x - np.mean(x))
        x = [
            (x_i[1], x_i[2])
            for x_i in sorted(zip(atan2, x, y), key=lambda pair: pair[0])
        ]
        return list(map(list, zip(*x)))

    def button_1(self, event):
        self.setup_points[self.name][0].append(event.xdata)
        self.setup_points[self.name][1].append(event.ydata)
        self.setup_points[self.name] = self.sort_points(*self.setup_points[self.name])

        self.lines[self.name].set_data(*self.close_line(*self.setup_points[self.name]))
        self.draw_frame()

    def button_3(self, event):

        point_id, dist = self.closest_point(
            event.xdata, event.ydata, *self.setup_points[self.name]
        )
        if dist < 1000:
            self.setup_points[self.name][0].pop(point_id)
            self.setup_points[self.name][1].pop(point_id)
            self.lines[self.name].set_data(
                *self.close_line(*self.setup_points[self.name])
            )
            self.draw_frame()


def rename_setup_point(setup_points):
    old_name = console.input(f"Enter the OLD name of the setup points: ")
    if old_name in setup_points:
        new_name = console.input(f'Enter the NEW name for setup points "{old_name}": ')
        setup_points[new_name] = setup_points.pop(old_name)
        console.rule(f'[green]Succesfully renamed points "{old_name}" to "{new_name}"')
    else:
        print(f'[red]No setup points named "{old_name}", aborting...')


def delete_setup_point(setup_points):
    name = console.input(f"Enter the name of the setup points to delete: ")
    if name in setup_points:
        setup_points.pop(name)
        console.rule(f'[green]Succesfully deleted "{name}"')
    else:
        print(f'[red]No setup points named "{name}", aborting...')


def setup_points_table(setup_points):

    setup_points = [(name, len(points)) for name, points in setup_points.items()]
    table = Table(title="setting_points")

    table.add_column("key", justify="center", style="cyan")
    table.add_column("# of points", justify="center", style="magenta")

    for key, length in setup_points:
        table.add_row(key, f"{length}")

    return table


def arg_main(video_path, data):
    console.rule("Welcome to the setup_points setter!")

    if isinstance(data, str):
        input_arg = "path"
        path = trajectory_path(data)
        data = np.load(path, allow_pickle=True).item()
    else:
        input_arg = "data"

    if not isinstance(data["setup_points"], dict):
        data["setup_points"] = {}
        print(f"[green]Created an empty dict for setup_points")

    while True:
        console.print(Align.center(setup_points_table(data["setup_points"])))
        what_to_do = console.input(
            "[blue]Name an existing (or not) setup_points to modify it or deletet it (or create it). You can also write 'rename' or 'delete' to rename or delete an existing setup_point or 'q' to save and exit the app: "
        )
        if what_to_do == "q":
            if input_arg == "path":
                np.save(path, data)
                console.print(f"Data saved to {path}")
            console.rule("bye!")
            return
        elif what_to_do == "rename":
            rename_setup_point(data["setup_points"])
        elif what_to_do == "delete":
            delete_setup_point(data["setup_points"])
        elif what_to_do:
            setup_points_setter(video_path, data, name=what_to_do)


def main():
    parser = ArgumentParser()
    parser.add_argument(
        "-s",
        metavar="session",
        type=trajectory_path,
        help="idTracker.ai succesful session directory or trajectory file",
        required=True,
    )
    parser.add_argument(
        "-video",
        type=str,
        help="Video file (only one file)",
        required=True,
    )
    args = parser.parse_args()
    arg_main(args.video, args.s)


if __name__ == "__main__":
    main()
