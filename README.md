# id manual tools <!-- omit in toc -->
<p align="center">
  <img src="https://github.com/jordi-torrents/id_manual_tools/raw/master/images/fishes.gif" alt="id_manual_tools"/>
</p>

id_manual_tools is a set of tools to use after a video has been tracked with idTracker.ai. They work on Python and are capable of visually correcting trajectories of animals, concatenate trajectory files and render awesome videos like the one above.

---
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/id-manual-tools)
![PyPI](https://img.shields.io/pypi/v/id-manual-tools)

# Table of Contents <!-- omit in toc -->
- [Install](#install)
- [Steps to track a video with idTracker.ai](#steps-to-track-a-video-with-idtrackerai)
  - [1 Download the episodes from Google Drive](#1-download-the-episodes-from-google-drive)
  - [2 Prepare the videos](#2-prepare-the-videos)
  - [3 Input parameters](#3-input-parameters)
    - [3.1 local_settings.py](#31-local_settingspy)
    - [3.2 segmentation parameters](#32-segmentation-parameters)
  - [4 Running idTracker.ai from terminal](#4-running-idtrackerai-from-terminal)
    - [4.1 Commands to run idTracker.ai from terminal](#41-commands-to-run-idtrackerai-from-terminal)
  - [4.2 idTracker.ai logger](#42-idtrackerai-logger)
  - [5 id_manual_tools](#5-id_manual_tools)
    - [5.1 `id_manual_tools_get_nans`](#51-id_manual_tools_get_nans)
    - [5.2 `id_manual_tools_set_corners`](#52-id_manual_tools_set_corners)
    - [5.3 `id_manual_tools_correct_traj`](#53-id_manual_tools_correct_traj)
    - [5.4 `id_manual_tools_concatenate_traj`](#54-id_manual_tools_concatenate_traj)
    - [5.5 `id_manual_tools_plot_traj`](#55-id_manual_tools_plot_traj)
- [Contact](#contact)
---

# Install



In the [idTracker.ai](https://idtrackerai.readthedocs.io/en/latest/) environment: `pip install id-manual-tools`

# Steps to track a video with idTracker.ai

## 1 Download the episodes from Google Drive

When downloading more than one file (or folder) of some GB each from Google Drive, Google tries to compress it but it can't, a bunch of files are then downloaded separately in the folder and you need to organize them.

These files should be saved at the external drive. Each one is around 4GB and you don't want to run out of space in the OS drive.

Once done, the files contain some extra numbers in their names (the video `GX010128.MP4` is downloaded as `GX010128-012.MP4`, for example). You can rename each file individually or use the bash command `rename -v 's/-[0-9]{3}././' *.MP4` to rename all of them at once

## 2 Prepare the videos

You don't want to track every 5 minutes episode individually and join them afterwards. The episodes (from the same experiment) must be concatenated using `ffmpeg`.

The larger the number of fishes in the experiment, the sorter the videos have to be so that idtracker.ai can track them. For up to 8 fishes, 30 minutes of 50fps videos (~90k frames) are ok for our computer. But try to track 30 minutes of 39 fish and your RAM will die (at least with our RAM with 64GB).

To concatenate all videos ended with `0154` in the current directory and write the output in the home directory you can run

`ffmpeg -safe 0 -f concat -i <(find . -type f -name '*0154.MP4' -printf "file '$PWD/%p'\n" | sort) -c copy /home/jordi/0154.MP4`

If you don't want to concatenate all of the videos and you want to specify the files you have to write an ordered `file` like (for example)
```
file './GX010154.MP4'
file './GX020154.MP4'
file './GX030154.MP4'
```

and then run `ffmpeg -safe 0 -f concat -i file -c copy /home/jordi/0102030154.MP4`

I use this name encoding, the lasts 4 digits are the video name and the firsts pairs are the episodes. So

- `010203040187.MP4` are episodes 01 02 03 and 04 of video 0187
- `0187.MP4` are all episodes of the video 0187

## 3 Input parameters

idTracker.ai has 3 levels of parameters with increasing priority.
1. `constants.py` file in the idtrackerai internal directory (you don't want to modify those parameters).
2. `local_settings.py` file in the current working directory.
3. Segmentation parameters from the idtrackerai GUI that wi will send to idtracker as `.json` file.

### 3.1 local_settings.py

Find complete info [the official idtracker website](https://idtrackerai.readthedocs.io/en/latest/advanced_parameters.html).

Here, you want to define the next parameters:
- `NUMBER_OF_JOBS_FOR_SEGMENTATION = 20`
  - Currently, idtrackerai consume so much RAM in the segmentation process so you want to set the number of jobs somewhere around 20 (although our computer has 36 cores)
- `IDENTIFICATION_IMAGE_SIZE = 55`
  - If you want to match identities after the tracking process, you have to fix the image size. In out videos 55 is a good value
- `DATA_POLICY = 'idmatcher.ai'`
  - This will remove useless data in the session directory when the tracking ends (this will free you from GBs of trash data)

### 3.2 segmentation parameters

The segmentation parameters will be unique for every video. To get them you have to run the command `idtrackerai` to enter the idtrackerai GUI. [Here](https://idtrackerai.readthedocs.io/en/latest/GUI_explained.html) you will find extended info of that.

These are the steps to obtain a good segmentation .json file from the GUI:

1. Open a video file by clicking _Open_.
2. If a window appears asking if you want to track multiple videos at once, say NO. I don't recommend tracking various videos, at least by now.
3. Define a ROI (Region of Interest) with a _Polygon_.
4. Subtract background. This may take a while and the GUI will freeze until the background is computed
5. Define the number of animals
6. Click to the Blob information and pressing Alt+space put it _Always on top_.
7. Modify the upper intensity threshold (the lower must be at 0) to obtain a minimum blob area of around 400 px and no noise blobs.
8. Modify the lower area threshold to around 300 px (so that all fish appear as blobs but you limit some accidental small noise blob)
9. Name you session with the same name as the video file
10. Save the .json with the same name as the video file
11. Close the GUI

Side notes:
- If you are working with 30 min videos, it's useful to use the GUI with one of the 5 min episodes so that the background computation takes less time. If you do that remember to **rename the video path** to the full 30 min video and **modify the _range_** (of frames to track) to something like $[0, 9999999999]$ in the .json.
- If you want to create the .json files for very similar videos (videos of the same day and characteristics). You can create the first one and copy-paste it changing the session name and the video path.

## 4 Running idTracker.ai from terminal

### 4.1 Commands to run idTracker.ai from terminal

The command to run idtrackerai in the terminal is

`idtrackerai terminal_mode --load file.json --exec track_video`

This will print output in the terminal and will be shut down when you exit the terminal.

If you want to run idtracker.ai not worrying about accidental shut downs, then

`nohup idtrackerai terminal_mode --load file.json --exec track_video > file.log 2>&1 &`

will print the output to a file and will keep running when you exit the terminal

A mini bash script for various videos to track could be

```bash
#!/bin/bash
declare -a files=("0102030405060146" "0708091011120146" "0147" "0148" "0149")

for file in "${files[@]}"
do
    idtrackerai terminal_mode --load $file.json --exec track_video > $file.log 2>&1
done
```

And run it like `nohup ./script.sh &`

Keep track of the output file to check the status of the program

## 4.2 idTracker.ai logger

Even if you run idtracker actively on the terminal (so that the log is printed in screen) or on background (the log is written in a .log file), you have to check some important aspects.

- The firsts lines of the log will indicate the parameters received by the software, check they are correct.

- Look for the line `INFO     --> check_segmentation`. After this line will appear the frames that contain more blobs than animal (if there is any). You don't want to have frames like that. For a 30 min 50 fps video you could accept < 20 error frames. But always think abut adjusting the segmentation parameters to avoid those frames.

- idTracker.ai has 3 tracking protocols depending of the needs of each video. If you video enters on protocol 3 (look for the line `INFO     --------------------> Protocol 2 failed -> Start protocol 3`), kill the process. Protocol 3 is eternal, it is time itself. Try to readjust the segmentation parameters and pray to god this doesn't happen again.

- The last lines of a successful log will be `INFO     Estimated accuracy: 0.????` and `INFO     Success`. A useful tracking should have an accuracy > 0.99.

## 5 id_manual_tools

When tracked, you may use the `id_manual_tools` project

For now, id_manual_tools has 5 tools:

- 5.1 `id_manual_tools_get_nans`
- 5.2 `id_manual_tools_set_corners`
- 5.3 `id_manual_tools_correct_traj`
- 5.4 `id_manual_tools_concatenate_traj`
- 5.5 `id_manual_tools_plot_traj`

All of them are wrapped with Python's ArgParser and can be ran with `-h` flag to print some basic information about input/output.

### 5.1 `id_manual_tools_get_nans`

The first tool checks for nans in the trajectory file. The raw trajectories from idTracker.ai use to have some NaNs (less than 1% of the total data). It reads the file and print a .csv list of NaNs

### 5.2 `id_manual_tools_set_corners`

This tool opens the video to set the `setting_points`. A list of coordinates that we use to indicate the position of the tank corners.

### 5.3 `id_manual_tools_correct_traj`

That's the main tool here. The trajectory corrector will use `id_manual_tools_get_nans` to look fort NaNs and will display a full matplotlib GUI to correct them using cubic interpolations. Additionally, the user will be asked to write the corners of the tank using `id_manual_tools_set_corners` to crop the video and speed up the GUI.

This tool can also be used to correct suspicious high velocities (jumps) using a gaussian threshold.

`id_manual_tools_correct_trajectories -s session_0146/ -video 0146.MP4 -fps 50 -n 10 -jumps_check_sigma 6`

### 5.4 `id_manual_tools_concatenate_traj`

If your video has been tracked in chunks. You can concatenate them with this tool. It is a wrapper of [idmatcher](https://gitlab.com/polavieja_lab/idmatcherai)

### 5.5 `id_manual_tools_plot_traj`

This is used to make composed videos (the original video with the trajectories overlapped)

# Contact

GitHub actions are recommended (issues, PR,...). Also, author's email is [jordi.torrentsm@gmail.com](jordi.torrentsm@gmail.com)
