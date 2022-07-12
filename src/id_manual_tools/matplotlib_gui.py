import matplotlib.pyplot as plt

# import numpy as np
# import cv2
# from rich import print
# from matplotlib.cm import get_cmap
# from matplotlib.collections import LineCollection
# import os
# from scipy.interpolate import interp1d
# from multiprocessing import Process
# from id_manual_tools.get_nans import get_list_of_nans_from_traj
# from csv import writer as csv_writer
# from rich.console import Console
from cv2 import threshold
from scipy.ndimage import center_of_mass

# console = Console()


class matplotlib_gui:
    def draw_frame(self):
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def add_plot(self, *args, **kwargs):
        return self.ax.plot(*args, **kwargs)

    def add_scatter(self, *args, **kwargs):
        return self.ax.scatter(*args, **kwargs)

    def add_image(self, *args, **kwargs):
        return self.ax.imshow(*args, **kwargs)

    def add_collection(self, *args, **kwargs):
        return self.ax.add_collection(*args, **kwargs)

    def add_text(self, *args, **kwargs):
        if "transform" in kwargs:
            if kwargs["transform"] == True:
                kwargs["transform"] = self.ax.transAxes
        return self.ax.text(*args, **kwargs)

    def connect(self, event: str, func):
        self.fig.canvas.mpl_connect(event, func)

    def __init__(self, title=" "):

        self.zoom = 1
        self.Ly = 500  # px
        self.Lx = 500  # px
        self.x_center = None
        self.y_center = None
        self.mouse_pressed = False

        self.fig = plt.figure(figsize=(8, 8))
        self.ax = self.fig.add_axes(
            [0, 0, 1, 1],
            xticks=(),
            yticks=(),
            facecolor="gray",
        )

        self.canvas_size = self.fig.get_size_inches() * self.fig.dpi

        # self.fig.canvas.manager.window.findChild(QToolBar).setVisible(False)
        self.fig.canvas.manager.set_window_title(title)
        self.fig.canvas.mpl_connect("button_press_event", self.on_click)
        self.fig.canvas.mpl_connect("button_release_event", self.on_click_release)
        # self.fig.canvas.mpl_connect("key_release_event", self.on_key)
        self.fig.canvas.mpl_connect("scroll_event", self.on_scroll)
        self.fig.canvas.mpl_connect("motion_notify_event", self.on_motion)
        self.fig.canvas.mpl_connect("resize_event", self.on_resize)

        # for artist in artists:
        #     self.ax.add_artist(artist)

        # for collection in collections:
        #     self.ax.add_collection(collection)

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
