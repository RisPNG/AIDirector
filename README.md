# GLaDOS Personality Core

This is a project dedicated to building a real-life version of GLaDOS.

*That being a hardware and software project that will create an aware, interactive, and embodied GLaDOS.*

This will entail:
- [x] Train GLaDOS voice generator
- [x] Generate prompt that leads to a realistic "Personality Core"
- [ ] Generate a [MemGPT](https://memgpt.readthedocs.io/en/latest/) medium- and long-term memory for GLaDOS
- [ ] Give GLaDOS vision via [LLaVA](https://llava-vl.github.io/)
- [ ] Create 3D-printable parts
- [ ] Design the animatronics system
  


## Sofware Architecture
The initial goals are to develop a low-latency platform, where GLaDOS can respond to voice interations within 600ms.

To do this, the system contantly record data to a circular buffer, waiting for [voice to be detected](https://github.com/snakers4/silero-vad). When it's determined that the voice has stopped (including detection of normal pauses), it will be [transcribed quickly](https://github.com/huggingface/distil-whisper). This is then passed to a streaming [local Large Language Model](https://github.com/ggerganov/llama.cpp), where the streamed text is broken by sentence, and passed to a [text-to-speech system](https://github.com/rhasspy/piper). This means futher sentences can be generated while the current is playing, reducing latency substantially.

### Subgoals
 - The another aim of the project is to minimise dependencies, so this can run on contrained hardware. That means no PyTorch or other large packages.  
 - As I want to fully understand the system, I have removed large amount of redirection: that means extracting and rewriting code. i.e. as GLaDOS only speaks English, I have rewritten the wrapper around [espeak](https://espeak.sourceforge.net/) and the entire Text-to-Speech subsystem is about 500 LOC and has only 3 dependencies: numpy, onnxruntime, and sounddevice. 


## Installation Instruction
If you want to install the TTS Engine on your machine, please follow the steps
below.  This has only been tested on Linux, but I wthink it will work on Windows with small tweaks.

1. Install the [`espeak`](https://github.com/espeak-ng/espeak-ng) synthesizer
   according to the [installation
   instructions](https://github.com/espeak-ng/espeak-ng/blob/master/docs/guide.md)
   for your operating system.
2. Install the required Python packages, e.g., by running `pip install -r
   requirements.txt`
3. For voice recognition, install [Whisper.cpp](https://github.com/ggerganov/whisper.cpp), and after compiling, mode the "libwhisper.so" file to the "glados" folder or add it to your path.  For Windows, check out the discussion in my [whisper pull request](https://github.com/ggerganov/whisper.cpp/pull/1524).  Then download the [voice recognition model](https://huggingface.co/distil-whisper/distil-medium.en/resolve/main/ggml-medium-32-2.en.bin?download=true), and put it tin the "models" directory.

## Testing
You can test the systems by exploring the 'demo.ipynb'.
