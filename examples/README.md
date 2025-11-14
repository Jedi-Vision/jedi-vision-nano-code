# Examples

## YOLO
You can run the YOLO tracking model via 1. webcam, 2. video file, or 3. phone camera (if continuity is established on a Mac with an iPhone).

```bash
usage: run_yolo.py [-h] [-d {cpu,mps,cuda}] [-v VIDEO] [-w] [-p] [-t]

options:
  -h, --help            show this help message and exit
  -d {cpu,mps,cuda}, --device {cpu,mps,cuda}
  -v VIDEO, --video VIDEO
  -w, --webcam          Use webcam instead of input video.
  -p, --phone           Use phone instead of input video.
  -t, --text            Add text labels for classes on seg-masks.
```

## SegFormer
You can run the SegFormer model via 1. webcam, 2. video file, or 3. phone camera (if continuity is established on a Mac with an iPhone).

```bash
usage: run_segformer.py [-h] [-d {cpu,mps,cuda}] [-v VIDEO] [-w] [-p] [-t]

options:
  -h, --help            show this help message and exit
  -d {cpu,mps,cuda}, --device {cpu,mps,cuda}
  -v VIDEO, --video VIDEO
  -w, --webcam          Use webcam instead of input video.
  -p, --phone           Use phone instead of input video.
  -t, --text            Add text labels for classes on seg-masks.
```

## Vosk
Vosk is a speech recognition toolkit that allows for super-fast offline speech to text recognition. The example script utilizes the `KaldiRecognizer` from KaldiASR for automatic-speech recognition.

To use, manually install the `vosk` package with ```pip install vosk```.

You might run into an issue with `sounddevice` failing to find `PortAudio`, see [here](https://stackoverflow.com/questions/49333582/portaudio-library-not-found-by-sounddevice) for a fix

```bash
usage: run_vosk.py [-h] [-l] [-f FILENAME] [-d DEVICE] [-r SAMPLERATE] [-m MODEL]

options:
  -h, --help            show this help message and exit
  -l, --list-devices    show list of audio devices and exit
  -f FILENAME, --filename FILENAME
                        audio file to store recording to
  -d DEVICE, --device DEVICE
                        input device (numeric ID or substring)
  -r SAMPLERATE, --samplerate SAMPLERATE
                        sampling rate
  -m MODEL, --model MODEL
                        language model; e.g. en-us, fr, nl; default is en-us
```