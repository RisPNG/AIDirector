<a href="https://trendshift.io/repositories/9828" target="_blank"><img src="https://trendshift.io/api/badge/repositories/9828" alt="dnhkng%2FGlaDOS | Trendshift" style="width: 250px; height: 55px;" width="250" height="55"/></a>

for [GLaDOS](https://github.com/dnhkng/GLaDOS) of course, I could never.

# What is GLaDOS?

GLaDOS is an amazing project dedicated to building a real-life version of GLaDOS!

If you want to chat or join their community, [Join the discord!](https://discord.com/invite/ERTDKwpjNB) If you want to support, [sponsor the project here!](https://ko-fi.com/dnhkng)

https://github.com/user-attachments/assets/c22049e4-7fba-4e84-8667-2c6657a656a0

## GLaDOS Goals

*GLaDOS is a hardware and software project that will create an aware, interactive, and embodied GLaDOS.*

It will entail:

- [X] Train GLaDOS voice generator
- [X] Generate a prompt that leads to a realistic "Personality Core"
- [ ] Generate a medium- and long-term memory for GLaDOS (Probably a custom vector DB in a simpy Numpy array!)
- [ ] Give GLaDOS vision via a VLM (either a full VLM for everything, or a 'vision module' using a tiny VLM the GLaDOS can function call!)
- [ ] Create 3D-printable parts
- [ ] Design the animatronics system

## Software Architecture

The initial goals are to develop a low-latency platform, where GLaDOS can respond to voice interactions within 600ms.

To do this, the system constantly records data to a circular buffer, waiting for [voice to be detected](https://github.com/snakers4/silero-vad). When it's determined that the voice has stopped (including detection of normal pauses), it will be [transcribed quickly](https://github.com/huggingface/distil-whisper). This is then passed to streaming [local Large Language Model](https://github.com/ggerganov/llama.cpp), where the streamed text is broken by sentence, and passed to a [text-to-speech system](https://github.com/rhasspy/piper). This means further sentences can be generated while the current is playing, reducing latency substantially.

### Subgoals

- The other aim of the project is to minimize dependencies, so this can run on constrained hardware. That means no PyTorch or other large packages.
- As I want to fully understand the system, I have removed a large amount of redirection: which means extracting and rewriting code.

## Hardware System

This will be based on servo- and stepper-motors. 3D printable STL will be provided to create GlaDOS's body, and she will be given a set of animations to express herself. The vision system will allow her to track and turn toward people and things of interest.

# Installation Instruction

Try this simplified process, but be aware it's still in the experimental stage!  For all operating systems, you'll first need to install [Ollama](https://github.com/ollama/ollama) to run the LLM.

## If Applicable, Install Drivers

If you are an Nvidia GPU, make sure you install the necessary drivers and CUDA which you can find here: [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit)

If you are using another accelerator (ROCm, DirectML etc.), after following the instructions below for you platform, follow up with installing the [best onnxruntime version](https://onnxruntime.ai/docs/install/) for your system.

 ___If you don't install the appropriate drivers, this system will still work, but the latency will be much greater!___

## Set up a local LLM server:

1. Download and install [Ollama](https://github.com/ollama/ollama) for your operating system.
2. Once installed, download a small 3B model for testing in a terminal or command prompt by using `ollama pull aya-expanse:8b`

Note: You can use any OpenAI or Ollama compatible server, local or cloud based. Just edit the glados_config.yaml and update the completion_url, model and the api_key if necessary.

## Operating specific instruction

#### Windows Installation Process

1. Open the Microsoft Store, search for `python` and install Python 3.12

#### macOS Installation Process

This is still experimental. Any issues can be addressed in the Discord server. If you create an issue related to this, you will be referred to the Discord server.  Note: I was getting Segfaults!  Please leave feedback!

#### Linux Installation Process

Install the PortAudio library, if you don't yet have it installed:

    sudo apt update
    sudo apt install libportaudio2

## Installing GLaDOS

1. Download this repository, either:

   1. Download and unzip this repository somewhere in your home folder, or
   2. At a terminal, git clone this repository using:

      `git clone https://github.com/dnhkng/GLaDOS.git`
2. In a terminal, go to the repository folder and run these commands:

   Mac/Linux:

   `python scripts/install.py`

   Windows:

   ```bash
   python scripts\install.py
   uv pip install kokoro misaki soundfile huggingface-hub gradio pydub espeakng-loader phonemizer-fork wheel setuptools num2words spacy aiofiles annotated-types anyio attrs av babel blis catalogue certifi cffi charset-normalizer click cloudpathlib cn2an colorama confection contourpy csvw curated-tokenizers curated-transformers cutlet cycler cymem distro dlinfo docopt fastapi ffmpy filelock fonttools fsspec fugashi g2pk2 gradio_client greenlet h11 httpcore httpx idna inflect isodate jaconv jamo jieba Jinja2 jiter joblib jsonschema jsonschema-specifications kiwisolver langcodes language_data language-tags loguru marisa-trie markdown-it-py MarkupSafe matplotlib mdurl mojimoji more-itertools mpmath munch murmurhash mutagen networkx nltk numpy nvidia-cublas-cu12 nvidia-cuda-cupti-cu12 nvidia-cuda-nvrtc-cu12 nvidia-cuda-runtime-cu12 nvidia-cudnn-cu12 nvidia-cufft-cu12 nvidia-curand-cu12 nvidia-cusolver-cu12 nvidia-cusparse-cu12 nvidia-cusparselt-cu12 nvidia-nvjitlink-cu12 nvidia-nvtx-cu12 openai ordered-set orjson packaging pandas pillow pip preshed proces psutil pycparser pydantic pydantic_core pydantic-settings Pygments pyparsing pypinyin python-dateutil python-dotenv python-multipart pytz PyYAML rdflib referencing regex requests rfc3986 rich rpds-py ruff safehttpx safetensors scipy segments semantic-version shellingham six smart-open sniffio spacy-curated-transformers spacy-legacy spacy-loggers SQLAlchemy srsly starlette sympy thinc tiktoken tokenizers tomlkit tqdm transformers typeguard typer typing_extensions tzdata unidic-lite uritemplate urllib3 uvicorn wasabi weasel websockets wrapt git+https://github.com/m-bain/whisperX.git
   ```

   Yeah, I know, I will figure out only the necessary packages soon.

   This will install Glados and download the needed AI models
3. To start GLaDOS, run:

   `uv run glados`
   If you want something more fancy, try the Text UI (TUI), with:

   `uv run glados tui`

## Speech Generation

You can also get her to say something with:

    `uv run glados say "The cake is real"`

## Changing the LLM Model

To use other models, use the command:
``ollama pull {modelname}``
and then add it to glados_config.yaml as the model:

    model: "{modelname}"

where __{modelname}__ is a placeholder to be replaced with the model you want to use. You can find [more models here!](https://ollama.com/library)

## Changing the Voice Model

You can use voices from Kokoro too!
Select a voice from the following:

- ### Female
- **US**
  - af_alloy
  - af_aoede
  - af_jessica
  - af_kore
  - af_nicole
  - af_nova
  - af_river
  - af_saraha
  - af_sky
- **British**
  - bf_alice
  - bf_emma
  - bf_isabella
  - bf_lily
- ### Male
- **US**
  - am_adam
  - am_echo
  - am_eric
  - am_fenrir
  - am_liam
  - am_michael
  - am_onyx
  - am_puck
- **British**
  - bm_daniel
  - bm_fable
  - bm_george
  - bm_lewis

and then add it to glados_config.yaml as the voice, e.g.:

    voice: "af_bella"

## More Personalities or LLM's

Make a copy of the file 'configs/glados_config.yaml' and give it a new name, then edit the parameters:

    model:  # the LLM model you want to use, see "Changing the LLM Model"
    personality_preprompt:
    system:  # A description of who the character should be
        - user:  # An example of a question you might ask
        - assistant:  # An example of how the AI should respond

To use these new settings, use the command:

    uv run glados start --config configs/assistant_config.yaml

## Common Issues

1. If you find you are getting stuck in loops, as GLaDOS is hearing herself speak, you have two options:
   1. Solve this by upgrading your hardware. You need to you either headphone, so she can't physically hear herself speak, or a conference-style room microphone/speaker. These have hardware sound cancellation, and prevent these loops.
   2. Disable voice interruption. This means neither you nor GLaDOS can interrupt when GLaDOS is speaking. To accomplish this, edit the `glados_config.yaml`, and change `interruptible:` to  `false`.
2. If you want to the the Text UI, you should use the glados-ui.py file instead of glado.py

## Testing the submodules

Want to mess around with the AI models? You can test the systems by exploring the 'demo.ipynb'.

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=RisPNG/AIDirector&type=Date)](https://star-history.com/#dnhkng/GlaDOS&Date)
