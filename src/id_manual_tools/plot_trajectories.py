from matplotlib.pyplot import Figure
import numpy as np
import cv2
from matplotlib.animation import FuncAnimation
from scipy.signal import savgol_filter
from matplotlib.collections import LineCollection
from matplotlib.cm import get_cmap
from argparse import ArgumentParser
from tqdm import trange
from multiprocessing import Pool
from subprocess import call
from os import makedirs, remove
from os.path import splitext
from shutil import rmtree
from warnings import catch_warnings, simplefilter
from scipy.ndimage import gaussian_filter1d


def interpolate_nans(arr):
    isnan = np.isnan(arr)
    arr[isnan] = np.interp(np.where(isnan)[0], xp=np.where(~isnan)[0], fp=arr[~isnan])


cmap = get_cmap("gist_rainbow")


# INPUT ARGUMENTS (ARGPARSER)
parser = ArgumentParser()
parser.add_argument(
    "-i", metavar="video", type=str, help="Input raw video file", required=True
)
parser.add_argument(
    "-traj",
    type=str,
    help="Path to trajectories to load (.npy file)",
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
    "-z",
    action="store_true",
    default=False,
    help="Activates dynamic zoom in the video",
)

parser.add_argument(
    "-vmin",
    default=0,
    help="Minimum value for video colormap, default 0",
)

parser.add_argument(
    "-vmax",
    default=255,
    help="Maximum value for video colormap, default 255",
)
parser.add_argument("-n", type=int, default=4, help="number of threads. Default is 4")

args = parser.parse_args()
line_lenght = 20
tmp_folder = "/tmp/plot_trajectories_tmp/"


# LOADING DATA (POSITIONS AND VIDEO)
data = np.load(args.traj, allow_pickle=True).item()
pos = data["trajectories"]
n_frames, N, _ = pos.shape

cap = cv2.VideoCapture(args.i)
fps = cap.get(cv2.CAP_PROP_FPS)
nx = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
ny = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
cap.release
cap = None


corners = data["setup_points"]["corners_out"]
xmin = int(np.min(corners[:, 0]))
xmax = int(np.max(corners[:, 0]))
ymin = int(np.min(corners[:, 1]))
ymax = int(np.max(corners[:, 1]))

#  COMPUTING DYNAMIC ZOOM (CENTER OF MASS AND ZOOM RADI)
if args.z:
    with catch_warnings():
        simplefilter("ignore", category=RuntimeWarning)
        c_mass = np.nanmean(pos, axis=1)
        z_radi = np.sqrt(
            np.mean(np.nansum((pos - c_mass[:, None, :]) ** 2, axis=-1), axis=-1)
        )

    # interpolate_nans(c_mass[:, 0])
    # interpolate_nans(c_mass[:, 1])
    # interpolate_nans(z_radi)

    c_mass = savgol_filter(c_mass, 31, 3, axis=0, mode="nearest")
    z_radi = savgol_filter(z_radi, 51, 3, mode="nearest")

    zoom = 2

    x_min = np.clip(c_mass[:, 0] - zoom * z_radi, xmin, None)
    y_min = np.clip(c_mass[:, 1] - zoom * z_radi, ymin, None)
    x_max = np.clip(x_min + 2 * zoom * z_radi, None, xmax)
    y_max = np.clip(y_min + 2 * zoom * z_radi, None, ymax)

    x_min = gaussian_filter1d(x_min, sigma=10)
    y_min = gaussian_filter1d(y_min, sigma=10)
    x_max = gaussian_filter1d(x_max, sigma=10)
    y_max = gaussian_filter1d(y_max, sigma=10)

    c_mass = None
    z_radi = None


# PREPARING PLOT LINES "SEGMENTS"
temp = pos.reshape(-1, N, 1, 2)
segments = np.concatenate([temp[:-1], temp[1:]], axis=2)
temp = None


# DISTRIBUTE FRAMES TO PROCESS BETWEEN THREADS
output_frames = int(args.t * fps) if args.t else n_frames
frame_limits = np.linspace(0, output_frames, args.n + 1, dtype=int)
frame_limits = np.column_stack([frame_limits[:-1], frame_limits[1:]])


# INITIALIZING FIGURE
dpi = 100
if args.z:
    fig = Figure(figsize=(15, 15), dpi=dpi)
else:
    Lx = xmax - xmin
    Ly = ymax - ymin
    diag = np.sqrt(Lx * Lx + Ly * Ly) / np.sqrt(2)
    fig = Figure(figsize=(15 * Lx / diag, 15 * Ly / diag), dpi=dpi)

fig.patch.set_facecolor("black")
axs = fig.add_axes([0, 0, 1, 1], facecolor="k", xticks=(), yticks=())
axs.set(xticks=(), yticks=(), xlim=(xmin, xmax), ylim=(ymax, ymin))

# colors = cmap((pos[0, :, 0] > 2000).astype(float) * 0.5)
colors = cmap(np.arange(N) / N)

points = axs.scatter(*np.zeros((2, N)), c=colors, s=20.0)

im = axs.imshow(
    np.zeros((ny, nx)), origin="upper", cmap="gray", vmax=args.vmax, vmin=args.vmin
)

LineCollections = []  # DON'T ASK...
for i in range(N):
    color = np.tile(colors[i], (line_lenght, 1))
    color[:, -1] = np.linspace(0, 1, line_lenght)
    LineCollections.append(axs.add_collection(LineCollection([], color=color)))


def thread_run(thread):
    """
    This function runs in every thread generating a portion of the final video
    """
    start, finish = frame_limits[thread]
    n_frames = finish - start

    cap = cv2.VideoCapture(args.i)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)

    def animate(video_frame_in_this_thread):
        """
        This animates the FuncAnimation
        """
        global_frame = video_frame_in_this_thread + start

        if args.z:
            axs.set(
                xlim=(x_min[global_frame], x_max[global_frame]),
                ylim=(y_max[global_frame], y_min[global_frame]),
            )

        points.set_offsets(pos[global_frame])
        im.set_data(np.mean(cap.read()[1], axis=2))

        origin = max(0, global_frame - line_lenght)
        for fish in range(N):
            LineCollections[fish].set_segments(segments[origin:global_frame, fish])
        return

    ani = FuncAnimation(
        fig,
        animate,
        frames=trange(
            n_frames,
            total=n_frames - 1,
            position=thread,
            desc=f"Thread {thread}",
        ),
        init_func=lambda: 0,
    )

    filename = f"{tmp_folder}part{thread:02d}.mp4"
    ani.save(filename, fps=fps)
    return filename


rmtree(tmp_folder, ignore_errors=True)
makedirs(tmp_folder)


def main():
    with Pool(args.n) as p, open("files.txt", "w") as file:
        for filename in p.map(thread_run, range(args.n)):
            file.write("file " + filename + "\n")
    print("\n" * args.n)

    call(
        [
            "ffmpeg",
            "-safe",
            "0",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "concat",
            "-y",
            "-i",
            "files.txt",
            "-c",
            "copy",
            args.o if args.o else splitext(args.i)[0] + "_tracked.mp4",
        ]
    )

    rmtree(tmp_folder)
    remove("files.txt")


if __name__ == "__main__":
    main()
