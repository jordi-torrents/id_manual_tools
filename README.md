useful commands
`rename -v 's/-[0-9]{3}././' *.MP4`
`ffmpeg -safe 0 -f concat -i <(find . -type f -name '*154.MP4' -printf "file '$PWD/%p'\n" | sort) -c copy /home/jordi/0154.MP4`
