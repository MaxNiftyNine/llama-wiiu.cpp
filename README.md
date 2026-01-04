# llama-wiiu.cpp

This is just a quick little vibe-coded project I made.
The model I tested with was 'qwen2.5-0.5b-instruct-q4_0.gguf'
Currently the prompt has to be hardcoded before building.
```
### Instruction
You are a helpful assistant. Reply conversationally and directly.
### Input:
Hi
### Response:
```
To edit the "Hi" and other settings here in wiiu/main.cpp
```
//CONFIG ######################
const int n_predict = 32;
const std::string textInput = "Hi!";
const bool use_buggy_keyboard = false; 
//#############################
```

## Running on hardware

- Copy a model (`.gguf`) to `sd:/model/` (Don't put multiple in there).
- Run `llama-wiiu.wuhb`.
- A log is also written to `sd:/llama-wiiu.log`.

## Build

Requirements:
- devkitPro with WUT

Build the RPX/WUHB:

```sh
make
```

Outputs:
- `build-wiiu/llama-wiiu.rpx`
- `build-wiiu/llama-wiiu.wuhb`
