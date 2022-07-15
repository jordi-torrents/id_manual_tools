import matplotlib.pyplot as plt


class matplotlib_gui:
    def draw_and_flush(self):
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def __init__(self, title=" "):

        self.zoom = 1
        self.Ly = 500  # px
        self.Lx = 500  # px
        self.x_center = 0
        self.y_center = 0
        self.mouse_pressed = False
        self.has_moved = False

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
        self.fig.canvas.mpl_connect("key_release_event", self.on_key)
        self.fig.canvas.mpl_connect("scroll_event", self.on_scroll)
        self.fig.canvas.mpl_connect("motion_notify_event", self.on_motion)
        self.fig.canvas.mpl_connect("resize_event", self.on_resize)

    def on_click(self, event):
        self.has_moved = False
        self.mouse_pressed = True
        self.click_origin = (event.x, event.y)

    def on_click_release(self, event):
        self.mouse_pressed = False
        if not self.has_moved:
            if hasattr(self, f"button_{event.button}"):
                getattr(self, f"button_{event.button}")(event)

    def on_key(self, event):
        try:
            int_key = int(event.key)
        except ValueError:
            try:
                getattr(self, f"key_{event.key}")()
            except AttributeError:
                pass
        else:
            if hasattr(self, "key_number"):
                self.key_number(int_key)

    def on_scroll(self, event):
        self.x_center += (self.x_center - event.xdata) * 0.1 * event.step
        self.y_center += (self.y_center - event.ydata) * 0.1 * event.step
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
