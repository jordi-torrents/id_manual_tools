
# Steps to track a video with idTracker.ai

## Download the episodes from Google Drive

When downloading more than one file (or folder) of some GB each from Google Drive, Google tries to compres it but it can't, a bunch of files are then downloaded separately in the folder and you need to organize them.

These files should be saved at the external drive. Each one is around 4GB and you don't want to run out of space in the OS drive.

Once done, the files contain some extra numbers in their names (the video `GX010128.MP4` is downloaded as `GX010128-012.MP4`, for example). You can rename each file individually or use the bash command `rename -v 's/-[0-9]{3}././' *.MP4` to rename all of them at once

## Prepare the videos

You don't want to track every 5 minuts episode individually and join them afterwards. The episodes (from the same experiment) must be concatenated using `ffmpeg`.

The larger the number of fishes in the experiment, the sorter the videos have to be so that idtracker.ai can track them. For up to 8 fishes, 30 minutes of 50fps videos (~90k frames) are ok for our computer. But try to track 30 minutes of 39 fish and your RAM will die (at least with our RAM with 64GB).

`ffmpeg -safe 0 -f concat -i <(find . -type f -name '*154.MP4' -printf "file '$PWD/%p'\n" | sort) -c copy /home/jordi/0154.MP4`

## ... working on it ...
