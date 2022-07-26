
![Alt Text](https://github.com/jordi-torrents/id_manual_tools/images/fishes.gif)

# Table of content
- [Table of content](#table-of-content)
- [Install](#install)
- [Steps to track a video with idTracker.ai](#steps-to-track-a-video-with-idtrackerai)
  - [1 Download the episodes from Google Drive](#1-download-the-episodes-from-google-drive)
  - [2 Prepare the videos](#2-prepare-the-videos)
  - [3 Input parameters](#3-input-parameters)
    - [3.1 local_settings.py](#31-local_settingspy)
    - [3.2 segmentation parameters](#32-segmentation-parameters)
  - [4 Running idTracker.ai from terminal](#4-running-idtrackerai-from-terminal)
  - [5 id_manual_tools](#5-id_manual_tools)
    - [5.1 `id_manual_tools_get_nans`](#51-id_manual_tools_get_nans)
    - [5.2 `id_manual_tools_set_corners`](#52-id_manual_tools_set_corners)
    - [5.3 `id_manual_tools_correct_traj`](#53-id_manual_tools_correct_traj)
    - [5.4 `id_manual_tools_concatenate_traj`](#54-id_manual_tools_concatenate_traj)
    - [5.5 `id_manual_tools_plot_traj`](#55-id_manual_tools_plot_traj)
- [Contact](#contact)

# Install



In the [idTracker.ai](https://idtrackerai.readthedocs.io/en/latest/) environment: `pip install id-manual-tools`

# Steps to track a video with idTracker.ai

## 1 Download the episodes from Google Drive

When downloading more than one file (or folder) of some GB each from Google Drive, Google tries to compres it but it can't, a bunch of files are then downloaded separately in the folder and you need to organize them.

These files should be saved at the external drive. Each one is around 4GB and you don't want to run out of space in the OS drive.

Once done, the files contain some extra numbers in their names (the video `GX010128.MP4` is downloaded as `GX010128-012.MP4`, for example). You can rename each file individually or use the bash command `rename -v 's/-[0-9]{3}././' *.MP4` to rename all of them at once

## 2 Prepare the videos

You don't want to track every 5 minuts episode individually and join them afterwards. The episodes (from the same experiment) must be concatenated using `ffmpeg`.

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
1. `constants.py` file in the idtrackerai internal directory (you don't want to modify those parameters)
2. `local_settings.py` file in the current working directory that idtracker reads at startup
3. Segmentation parameters from the idtrackerai GUI or the `.json` file that you pass explicitly to idtrackerai

### 3.1 local_settings.py

Find complete info [the official idtracker website](https://idtrackerai.readthedocs.io/en/latest/advanced_parameters.html).

You want to define the next parameters here:
- `NUMBER_OF_JOBS_FOR_BACKGROUND_SUBTRACTION = -2`
  - Currently, idtrackerai does not paralelize properly the background subtraction so this paramenter will have some effect in the future
- `NUMBER_OF_JOBS_FOR_SEGMENTATION = 20`
  - Currently, idtrackerai consume so much RAM in the segmentation process so you want to set the number of jobs somewhere arround 20 (although our computer has 36 cores)
- `IDENTIFICATION_IMAGE_SIZE = 55`
  - If you want to match identities after the tracking process, you have to fix the image size. In out videos 55 is a good value
- `DATA_POLICY = 'idmatcher.ai'`
  - This will remove useless data in the session directory whenm the tracking ends (this will free you from GBs of trash data)

### 3.2 segmentation parameters

The segmentation parameters will be unique for every video. To get them you have to run the command `idtrackerai` to enter the idtrackerai GUI. [Here](https://idtrackerai.readthedocs.io/en/latest/GUI_explained.html) you will find extended info of that.

As a side note, a minimum area of ~400 px is perfect for our trackings.

I recommend using the GUI to set the segmentation parameters and save them in a `.json` file. Then you will use this file in to run idtrackerai from terminal

## 4 Running idTracker.ai from terminal

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

## 5 id_manual_tools

When tracked, you may use the `id_manual_tools` project

For now, id_manual_tools has 5 tools:

- 5.1 `id_manual_tools_get_nans`
- 5.2 `id_manual_tools_set_corners`
- 5.3 `id_manual_tools_correct_traj`
- 5.4 `id_manual_tools_concatenate_traj`
- 5.5 `id_manual_tools_plot_traj`

All of them are wrapped with Python's ArgParser and can be runned with `-h` flag to print some basic information about input/output.

### 5.1 `id_manual_tools_get_nans`

The first tool checks for nans in the trajectory file. The raw trajectories from idTracker.ai use to have some NaNs (less than 1% of the total data). It reads the file and print a .csv list of NaNs

### 5.2 `id_manual_tools_set_corners`

This tool opens the video to set the `setting_points`. A list of coordinates that we use to indicate the position of the tanck corners.

### 5.3 `id_manual_tools_correct_traj`

That's the main tool here. The trajectory corrector will use `id_manual_tools_get_nans` to look fort NaNs and will display a full matplotlib GUI to correct them using cubic interpolations. Aditionally, the user will be asked to write the corners of the tank using `id_manual_tools_set_corners` to crop the video and speed up the GUI.

This tool can also be used to correct suspicious high velicities (jumps) using a gaussian threshold.

`id_manual_tools_correct_trajectories -s session_0146/ -video 0146.MP4 -fps 50 -n 10 -jumps_check_sigma 6`

### 5.4 `id_manual_tools_concatenate_traj`

If your video has been tracked in chunks. You can concatenate them with this tool. It is a wrapper of [idmatcher](https://gitlab.com/polavieja_lab/idmatcherai)

### 5.5 `id_manual_tools_plot_traj`

This is used to make composed videos (the original video with the trajectories overlapped)

# Contact

GitHub actions are recommended (issues, PR,...). Also, author's email is [jordi.torrentsm@gmail.com](jordi.torrentsm@gmail.com)
