# Dear Google, I love you! Please don't ban me from colab T_T

#@title # *a*Start** 🚀 
# All necessary imports goes here
from IPython.display import clear_output, display, HTML
import os
import time
from datetime import timedelta
from google.colab import drive
from IPython.utils import capture
from subprocess import getoutput
from urllib.parse import unquote
from google.colab.output import eval_js
%cd /content
try:
  start_colab
except:
  start_colab = int(time.time())-5
clear_output()
print("\033[96m") #Cyan text


# Check if gpu exist, stop if don't.
try:
  output 
except:
  print('⌚ Checking GPU...', end='')
  output = getoutput('nvidia-smi --query-gpu=gpu_name --format=csv')
  if "name" in output:
    gpu_name = output[5:]
    print('\r✅ Current GPU:', gpu_name, flush=True)
  else:
    print('\r\033[91m❎ ERROR: No GPU detected. Please do step below to enable.\n', flush=True)
    display(HTML("<img src='https://i.ibb.co/HC9KH17/NVIDIA-Share-23-01-02-173037.png' width='800px'/>"))
    print('\033[91m\nIf it says "Cannot connect to GPU backend", meaning you\'ve either reached free usage limit. OR there\'s no gpu available.\n\nDon\'t mind me... I\'m destroying your current session for your own good...')
    display(HTML("<img src='https://media.tenor.com/E9omRGF7x0AAAAAC/hitori-gotou-bocchi-rock.gif' width='500px'/>"))
    time.sleep(5)
    from google.colab import runtime
    runtime.unassign()

# [ALL PARAMS]-----------------------------------------------
#@markdown ### **Install Configurations &nbsp;&nbsp;&nbsp;[?](https://rentry.org/ncpt_what)** 
stable_build = False
latest_webui = False #@param{type:"boolean"}
latest_extensions = False #@param{type:"boolean"}

#@markdown ### <br> **Configurations**
output_to_drive = False #@param{type:"boolean"}
configs_in_drive = False #@param{type:"boolean"}
fast_start = False #@param{type:"boolean"}
auto_vae = False #@param{type:"boolean"}
no_custom_theme = False #@param {type:"boolean"} 
merge_in_vram = False #@param {type:"boolean"} 
colab_optimizations = False #@param {type:"boolean"} 
ram_patch_for_sd2 = False #@param{type:"boolean"}
dpmpp_v2 = False 
krita_paint_ext = False #@param {type:"boolean"} 
umiai_ext = False #@param {type:"boolean"} 
verbose_download = False #@param {type:"boolean"} 
commandline_arguments = "--lowvram --enable-insecure-extension-access --opt-sdp-attention --share --no-half-vae --disable-safe-unpickle --theme dark --no-hashing " #@param{type:"string"}
commit_hash = "" #@param{type:"string"}
ngrok_token  = "" #@param{type:"string"}
ngrok_region = "jp" #@param ["us", "eu", "au", "ap", "sa", "jp", "in"]
ngrok_auto_save_load = False #@param{type:"boolean"}
alternative_tunnels = True #@param{type:"boolean"}
with_bore = False #@param{type:"boolean"}

#@markdown ### <br> **Models, VAEs, Embeddings, Hypernetworks, Yaml, LoRA**
#@markdown *Check only what you need, colab storage is not unlimited.*
optional_huggingface_token="" #@param{type:"string"}
model_url = "" # waifu_diffusion_beta2
# waifu_diffusion_AES = False #@param{type:"boolean"} 
# waifu_diffusion = False #@param{type:"boolean"}
waifu_diffusion_radiance = False #@param{type:"boolean"}
waifu_diffusion_ink = False #@param{type:"boolean"}
waifu_diffusion_mofu = False #@param{type:"boolean"}
waifu_diffusion_illusion = False #@param{type:"boolean"}
# anything_v3 = False #@param{type:"boolean"}
anything_v4_5 = True #@param{type:"boolean"}
holokuki_v2_2 = False #@param{type:"boolean"}
something_v2_2 = True #@param{type:"boolean"}
anything_vae = True #@param{type:"boolean"}
blessed2_vae = False #@param{type:"boolean"}
wd_vae = False #@param{type:"boolean"}
sd_vae = False #@param{type:"boolean"}
controlnet = "none" #@param ["v1.1", "v1.0", "v1.0-diff", "t2i", "none"]
null_model = False #@param{type:"boolean"}
custom_urls = "https://civitai.com/api/download/models/30163," #@param {type:"string"}
#@markdown &nbsp; &nbsp; <font color=gray><i> Put any models, embeddings, configs, hypernetworks, loras links and separate it with comma. To use lycoris, use `lycoris:https://link.com` in custom url<br>
#@markdown &nbsp; &nbsp;<img src="https://cdn.discordapp.com/emojis/930580027135901777.webp?size=56" width="18"/> [How to use this custom url box?](https://rentry.org/custom_url_nocrypt) - [**Awesome Models List**](https://rentry.org/ncpt_fav_models) ✨

# CONFIG DIR (not recommended to change unless you know what you're doing)
destination_dir = "/content/.downloaded/"
config_dir="/content/sdw/config.json"
models_dir = "/content/sdw/models/Stable-diffusion/"
vaes_dir = "/content/sdw/models/VAE/"
hypernetworks_dir = "/content/sdw/models/hypernetworks/"
embeddings_dir = "/content/sdw/embeddings/"
loras_dir = "/content/sdw/models/Lora"
lycoris_dir = "/content/sdw/models/LyCORIS"
patches_dir = "/content/patches"
extensions_dir = "/content/sdw/extensions/"
control_dir = "/content/sdw/extensions/sd-webui-controlnet/models"
drive_config_dir = "/content/gdrive/MyDrive/WebUI/configs/"
# -----------------------------------------------

# Append models to model_url
model_url+=custom_urls+", " if custom_urls else ""
# if anything_v3:
#   model_url+="https://huggingface.co/NoCrypt/safetensor_models/resolve/main/Anything-V3.0-pruned-fp32.safetensors, "

if anything_v4_5:
  model_url+="https://huggingface.co/andite/anything-v4.0/resolve/main/anything-v4.5-pruned.safetensors, "
if holokuki_v2_2:
  model_url+="https://huggingface.co/Aotsuyu/Kukicha/resolve/main/HoloKukiv2.2-fp16.safetensors, "
if something_v2_2:
  model_url+="https://huggingface.co/NoCrypt/SomethingV2_2/resolve/main/SomethingV2_2.safetensors, "
if waifu_diffusion_radiance:
  model_url+="https://huggingface.co/waifu-diffusion/wd-1-5-beta3/resolve/main/wd-radiance-fp16.safetensors, "
if waifu_diffusion_ink:
  model_url+="https://huggingface.co/waifu-diffusion/wd-1-5-beta3/resolve/main/wd-ink-fp16.safetensors, "
if waifu_diffusion_mofu:
  model_url+="https://huggingface.co/waifu-diffusion/wd-1-5-beta3/resolve/main/wd-mofu-fp16.safetensors, "
if waifu_diffusion_illusion:
  model_url+="https://huggingface.co/waifu-diffusion/wd-1-5-beta3/resolve/main/wd-illusion-fp16.safetensors, "

if anything_vae or anything_v4_5:
  model_url+=" https://huggingface.co/NoCrypt/Anything-v3-0/resolve/main/anything.vae.pt, "
if blessed2_vae:
  model_url+="https://huggingface.co/NoCrypt/blessed_vae/resolve/main/blessed2.vae.pt, "
if wd_vae or waifu_diffusion_radiance or waifu_diffusion_ink or waifu_diffusion_mofu or waifu_diffusion_illusion:
  model_url+="https://huggingface.co/hakurei/waifu-diffusion-v1-4/resolve/main/vae/kl-f8-anime2.ckpt, "
if sd_vae:
  model_url+="https://huggingface.co/stabilityai/sd-vae-ft-mse-original/resolve/main/vae-ft-mse-840000-ema-pruned.safetensors, "
if controlnet == "v1.0":
  model_url+="https://huggingface.co/webui/ControlNet-modules-safetensors/resolve/main/control_canny-fp16.safetensors, https://huggingface.co/webui/ControlNet-modules-safetensors/resolve/main/control_depth-fp16.safetensors, https://huggingface.co/webui/ControlNet-modules-safetensors/resolve/main/control_hed-fp16.safetensors, https://huggingface.co/webui/ControlNet-modules-safetensors/resolve/main/control_mlsd-fp16.safetensors, https://huggingface.co/webui/ControlNet-modules-safetensors/resolve/main/control_normal-fp16.safetensors, https://huggingface.co/webui/ControlNet-modules-safetensors/resolve/main/control_openpose-fp16.safetensors, https://huggingface.co/webui/ControlNet-modules-safetensors/resolve/main/control_scribble-fp16.safetensors, https://huggingface.co/webui/ControlNet-modules-safetensors/resolve/main/control_seg-fp16.safetensors, "
if controlnet == "v1.0-diff":
  model_url+="https://huggingface.co/kohya-ss/ControlNet-diff-modules/resolve/main/diff_control_sd15_canny_fp16.safetensors, https://huggingface.co/kohya-ss/ControlNet-diff-modules/resolve/main/diff_control_sd15_depth_fp16.safetensors, https://huggingface.co/kohya-ss/ControlNet-diff-modules/resolve/main/diff_control_sd15_hed_fp16.safetensors, https://huggingface.co/kohya-ss/ControlNet-diff-modules/resolve/main/diff_control_sd15_mlsd_fp16.safetensors, https://huggingface.co/kohya-ss/ControlNet-diff-modules/resolve/main/diff_control_sd15_normal_fp16.safetensors, https://huggingface.co/kohya-ss/ControlNet-diff-modules/resolve/main/diff_control_sd15_openpose_fp16.safetensors, https://huggingface.co/kohya-ss/ControlNet-diff-modules/resolve/main/diff_control_sd15_scribble_fp16.safetensors, https://huggingface.co/kohya-ss/ControlNet-diff-modules/resolve/main/diff_control_sd15_seg_fp16.safetensors, "
if controlnet == "t2i":
  model_url+="https://huggingface.co/webui/ControlNet-modules-safetensors/resolve/main/t2iadapter_canny-fp16.safetensors, https://huggingface.co/webui/ControlNet-modules-safetensors/resolve/main/t2iadapter_color-fp16.safetensors, https://huggingface.co/webui/ControlNet-modules-safetensors/resolve/main/t2iadapter_depth-fp16.safetensors, https://huggingface.co/webui/ControlNet-modules-safetensors/resolve/main/t2iadapter_keypose-fp16.safetensors, https://huggingface.co/webui/ControlNet-modules-safetensors/resolve/main/t2iadapter_openpose-fp16.safetensors, https://huggingface.co/webui/ControlNet-modules-safetensors/resolve/main/t2iadapter_seg-fp16.safetensors, https://huggingface.co/webui/ControlNet-modules-safetensors/resolve/main/t2iadapter_sketch-fp16.safetensors, https://huggingface.co/webui/ControlNet-modules-safetensors/resolve/main/t2iadapter_style-fp16.safetensors,"
if controlnet == "v1.1":
  model_url+="https://huggingface.co/comfyanonymous/ControlNet-v1-1_fp16_safetensors/resolve/main/control_v11e_sd15_ip2p_fp16.safetensors, https://huggingface.co/comfyanonymous/ControlNet-v1-1_fp16_safetensors/resolve/main/control_v11e_sd15_shuffle_fp16.safetensors, https://huggingface.co/comfyanonymous/ControlNet-v1-1_fp16_safetensors/resolve/main/control_v11f1p_sd15_depth_fp16.safetensors, https://huggingface.co/comfyanonymous/ControlNet-v1-1_fp16_safetensors/resolve/main/control_v11p_sd15_canny_fp16.safetensors, https://huggingface.co/comfyanonymous/ControlNet-v1-1_fp16_safetensors/resolve/main/control_v11p_sd15_inpaint_fp16.safetensors, https://huggingface.co/comfyanonymous/ControlNet-v1-1_fp16_safetensors/resolve/main/control_v11p_sd15_lineart_fp16.safetensors, https://huggingface.co/comfyanonymous/ControlNet-v1-1_fp16_safetensors/resolve/main/control_v11p_sd15_mlsd_fp16.safetensors, https://huggingface.co/comfyanonymous/ControlNet-v1-1_fp16_safetensors/resolve/main/control_v11p_sd15_normalbae_fp16.safetensors, https://huggingface.co/comfyanonymous/ControlNet-v1-1_fp16_safetensors/resolve/main/control_v11p_sd15_openpose_fp16.safetensors, https://huggingface.co/comfyanonymous/ControlNet-v1-1_fp16_safetensors/resolve/main/control_v11p_sd15_scribble_fp16.safetensors, https://huggingface.co/comfyanonymous/ControlNet-v1-1_fp16_safetensors/resolve/main/control_v11p_sd15_seg_fp16.safetensors, https://huggingface.co/comfyanonymous/ControlNet-v1-1_fp16_safetensors/resolve/main/control_v11p_sd15_softedge_fp16.safetensors, https://huggingface.co/comfyanonymous/ControlNet-v1-1_fp16_safetensors/resolve/main/control_v11p_sd15s2_lineart_anime_fp16.safetensors, https://huggingface.co/comfyanonymous/ControlNet-v1-1_fp16_safetensors/resolve/main/control_v11u_sd15_tile_fp16.safetensors, "

# Image outputs to drive (part 1)
if output_to_drive or configs_in_drive:
  if not os.path.exists('/content/gdrive'):
    drive.mount('/content/gdrive')

# Unpack Repo, Dependencies, Caches
if not os.path.exists("/content/sdw"):
  start_install = int(time.time())
  print("🚀 Unpacking... Please do not stop this process at all cost...", end='')
  with capture.capture_output() as cap:
    # !rm -rf /usr/local/lib/python3.10/dist-packages/scipy /usr/local/lib/python3.10/dist-packages/scipy-*.dist-info/ /usr/local/lib/python3.10/dist-packages/scipy.libs

    !wget https://huggingface.co/NoCrypt/fast-repo/resolve/main/ubuntu_deps.zip && unzip ubuntu_deps.zip -d ./deps && dpkg -i ./deps/* && rm -rf ubuntu_deps.zip /content/deps/
    # !apt install liblz4-tool aria2
    # !pip uninstall -q -y huggingface_hub
    # !{'curl -LO https://github.com/BurntSushi/ripgrep/releases/download/13.0.0/ripgrep_13.0.0_amd64.deb && dpkg -i ripgrep_13.0.0_amd64.deb && rm -rf ripgrep_13.0.0_amd64.deb'}
    
    # !aria2c --summary-interval=10 -c -x 16 -k 1M -s 16 -d /content -Z \
    #   https://huggingface.co/NoCrypt/fast-repo/resolve/main/dep.tar.lz4 \
    #   https://huggingface.co/NoCrypt/fast-repo/resolve/main/repo.tar.lz4 \
    #   https://huggingface.co/NoCrypt/fast-repo/resolve/main/cache.tar.lz4
    
    # !aria2c -d /content -o dep.tar.lz4 --summary-interval=10 -c -x 16 -k 1M -s 16 https://huggingface.co/NoCrypt/fast-repo/resolve/main/dep.tar.lz4
    # !aria2c -d /content -o repo.tar.lz4 --summary-interval=10 -c -x 16 -k 1M -s 16 https://huggingface.co/NoCrypt/fast-repo/resolve/main/repo.tar.lz4
    # !aria2c -d /content -o cache.tar.lz4 --summary-interval=10 -c -x 16 -k 1M -s 16 https://huggingface.co/NoCrypt/fast-repo/resolve/main/cache.tar.lz4

    !echo -e "https://huggingface.co/NoCrypt/fast-repo/resolve/main/dep.tar.lz4\n\tout=dep.tar.lz4\nhttps://huggingface.co/NoCrypt/fast-repo/resolve/main/repo.tar.lz4\n\tout=repo.tar.lz4\nhttps://huggingface.co/NoCrypt/fast-repo/resolve/main/cache.tar.lz4\n\tout=cache.tar.lz4\n" \
      | aria2c -i- -j5 -x16 -s16 -k1M -c 
    
    !tar -xI lz4 -f dep.tar.lz4 --overwrite-dir --directory=/usr/local/lib/python3.10/dist-packages/ #(manual dir)
    !tar -xI lz4 -f repo.tar.lz4 --directory=/ #/content/sdw/ (auto dir)
    !tar -xI lz4 -f cache.tar.lz4 --directory=/ #/root/.cache/huggingface (auto dir)

    !rm -rf /content/dep.tar.lz4 /content/repo.tar.lz4 /content/cache.tar.lz4
    os.environ["SAFETENSORS_FAST_GPU"]='1'
    os.environ["colab_url"] = eval_js("google.colab.kernel.proxyPort(7860, {'cache': false})")
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    os.environ["PYTHONWARNINGS"] = "ignore"
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = "garbage_collection_threshold:0.9,max_split_size_mb:512"
    # !apt -y update -qq
    # !apt install -qq libunwind8-dev
    %env LD_PRELOAD=libtcmalloc.so
    !rm /content/sdw/extensions/sd-webui-tunnels/id_rsa.pub /content/sdw/extensions/sd-webui-tunnels/id_rsa
    del cap
  if not os.path.exists(hypernetworks_dir):
    os.makedirs(hypernetworks_dir)
  # if not 'T4' in gpu_name:  # [For colab makers out there, facebook's xformers pypi already support almost all gpu now]
  #   !pip uninstall -y xformers
  install_time = timedelta(seconds=time.time()-start_install)
  print("\r🚀 Finished unpacking. Took","%02d:%02d:%02d ⚡\n" % (install_time.seconds / 3600, (install_time.seconds / 60) % 60, install_time.seconds % 60), end='', flush=True)
  # Colab 🤝 Gradio (Colab timer integration for gradio)
  !echo -n {start_colab} > /content/sdw/static/colabTimer.txt
  print("🤝 Colab timer integration complete! You can see your colab time inside webui.")

  # Update using git pull
  if latest_webui and not stable_build:
    !git config --global user.email "you@example.com"
    !git config --global user.name "Your Name"
    print('⌚ Pulling latest changes...', end="")
    with capture.capture_output() as cap:
      %cd /content/sdw
      !git pull -X theirs --rebase --autostash
      del cap
    print('\r🪄 \033[96mYou are currently using latest version of webui. Please use commit_hash if there is error', flush=True)

  # Update extensions
  if latest_extensions:
    print('⌚ Updating extensions (might take a while)...', end="")
    with capture.capture_output() as cap:
      !{'for dir in /content/sdw/extensions/*/; do cd "$dir" && git fetch origin && git pull; done'}
    del cap
    print('\r🪄 \033[96mInbuilt extensions are updated to its latest versions', flush=True)
else:
  print("🚀 Already unpacked... Skipping.")
  time_since_start = timedelta(seconds=time.time()-start_colab)
  print("⌚ You've been running this colab for","%02d:%02d:%02d" % (time_since_start.seconds / 3600, (time_since_start.seconds / 60) % 60, time_since_start.seconds % 60))

# Additional Extensions
os.makedirs(patches_dir, exist_ok=True)
with capture.capture_output() as cap:
  if umiai_ext:
    model_url+="https://github.com/Klokinator/Umi-AI, "
    !rm -rf /content/sdw/extensions/sd-dynamic-prompts
  else:
    if os.path.exists("/content/sdw/extensions/Umi-AI"):
      !rm -rf /content/sdw/extensions/Umi-AI


# Revert changes using time-machine (git reset)
if commit_hash:
  print('⌚ Activating time machine...', end="")
  with capture.capture_output() as cap:
    %cd /content/sdw
    !git config --global user.email "you@example.com"
    !git config --global user.name "Your Name"
    # !git stash
    !git reset --hard {commit_hash}
    # !git stash apply
    # !rm -rf /content/sdw/embeddings/*
    del cap
  print('\r⌚ Time machine activated, you\'re on commit', commit_hash, flush=True)
  # print('✌️ Embeddings have been deleted for time machine support')

# Colab patches for quality of life improvements
with capture.capture_output() as cap:
  # RAM Patch Code by ddPn08: https://github.com/ddPn08/automatic1111-colab/commit/81431d6bd66f0ef96fe638c2743623522e1bc797
  !wget https://raw.githubusercontent.com/ddPn08/automatic1111-colab/main/patches/stablediffusion-lowram.patch -P {patches_dir}  -c
  !wget https://gist.github.com/neggles/75eaacb3f49c209636be61fa96ca95ca/raw/f8c6382f0af65038149fd4258f8462697b698073/01-add-DPMPP-2M-V2.patch -P {patches_dir}  -c
  if ram_patch_for_sd2: 
    !cd /content/sdw/repositories/stable-diffusion-stability-ai && git apply {patches_dir}/stablediffusion-lowram.patch
  else:
    !cd /content/sdw/repositories/stable-diffusion-stability-ai && git apply -R {patches_dir}/stablediffusion-lowram.patch
  
  if dpmpp_v2: 
    !cd /content/sdw/ && git apply --whitespace=fix {patches_dir}/01-add-DPMPP-2M-V2.patch
  else:
    !cd /content/sdw/ && git apply --whitespace=fix -R {patches_dir}/01-add-DPMPP-2M-V2.patch

  # Colab Optimizations by TheLastBen. Included load in vram + gradio queue with high concurrency_count
  if colab_optimizations:
    !sed -i "s@os.path.splitext(checkpoint_.*@os.path.splitext(checkpoint_file); map_location='cuda'@" /content/sdw/modules/sd_models.py
    !sed -i 's@ui.create_ui().*@ui.create_ui();shared.demo.queue(concurrency_count=999999,status_update_rate=0.1)@' /content/sdw/webui.py
  else:
    !sed -i "s@os.path.splitext(checkpoint_.*@os.path.splitext(checkpoint_file)@" /content/sdw/modules/sd_models.py
    !sed -i 's@ui.create_ui().*@ui.create_ui()@' /content/sdw/webui.py

  # Merge in vram: self-explainotory 
  if merge_in_vram:
    !sed -i "s@'cpu'@'cuda'@" /content/sdw/modules/extras.py
  else:
    !sed -i "s@'cuda'@'cpu'@" /content/sdw/modules/extras.py

  # Remove custom theme (since it's not for everyone)
  if no_custom_theme:
    !rm -rf "/content/sdw/extensions/catppuccin_theme/style.css"
  else:
    if not os.path.exists('/content/sdw/extensions/catppuccin_theme/style.css'):
      !cp "/content/sdw/extensions/catppuccin_theme/flavors/mocha.css" "/content/sdw/extensions/catppuccin_theme/style.css"

  #rename v1.1 cn models to remove _fp16
  if controlnet == "v1.1":
    !for file in /content/sdw/extensions/sd-webui-controlnet/models/*_fp16*; do mv -- "$file" "${file/_fp16/}"; done
    !for file in /content/sdw/extensions/sd-webui-controlnet/models/*_fp16*; do rm "$file"; done


# Ngrok stuff goes here
if ngrok_token or ngrok_auto_save_load:
  if ngrok_auto_save_load:
    if not os.path.exists('/content/gdrive'):
      drive.mount('/content/gdrive')
    if ngrok_token:
      if not os.path.exists("/content/gdrive/MyDrive/WebUI/ngrokToken.txt"):
        !mkdir -p /content/gdrive/MyDrive/WebUI/
        !touch /content/gdrive/MyDrive/WebUI/ngrokToken.txt
      f = open("/content/gdrive/MyDrive/WebUI/ngrokToken.txt", "w+")
      f.write(ngrok_token+","+ngrok_region)
      f.close()
    elif os.path.exists('/content/gdrive/MyDrive/WebUI/ngrokToken.txt'):
      ngrok_token,ngrok_region = getoutput("cat /content/gdrive/MyDrive/WebUI/ngrokToken.txt").split(",",2)
    else:
      print("warning: ngrok token not detected")
  commandline_arguments += ' --ngrok ' + ngrok_token + ' --ngrok-region ' + ngrok_region
  commandline_arguments = commandline_arguments.replace("--share","")

# Configs in drive
if configs_in_drive:
  config_dir = drive_config_dir+"config.json"
  if not os.path.exists(drive_config_dir):
    !mkdir -p {drive_config_dir}
    !cp /content/sdw/styles.csv /content/sdw/ui-config.json /content/sdw/config.json {drive_config_dir}
  commandline_arguments += ' --ui-config-file ' + drive_config_dir+"ui-config.json"
  commandline_arguments += ' --ui-settings-file ' + drive_config_dir+"config.json"
  commandline_arguments += ' --styles-file ' + drive_config_dir+"styles.csv"
  
# Image outputs to drive (part 2)
if output_to_drive:
  !sed -i 's@"outdir_txt2img_samples": "outputs/txt2img-images"@"outdir_txt2img_samples": "/content/gdrive/MyDrive/WebUI/outputs/txt2img-images"@' {config_dir}
  !sed -i 's@"outdir_img2img_samples": "outputs/img2img-images"@"outdir_img2img_samples": "/content/gdrive/MyDrive/WebUI/outputs/img2img-images"@' {config_dir}
  !sed -i 's@"outdir_extras_samples": "outputs/extras-images"@"outdir_extras_samples": "/content/gdrive/MyDrive/WebUI/outputs/extras-images"@' {config_dir}
  !sed -i 's@"outdir_txt2img_grids": "outputs/txt2img-grids"@"outdir_txt2img_grids": "/content/gdrive/MyDrive/WebUI/outputs/txt2img-grids"@' {config_dir}
  !sed -i 's@"outdir_img2img_grids": "outputs/img2img-grids"@"outdir_img2img_grids": "/content/gdrive/MyDrive/WebUI/outputs/img2img-grids"@' {config_dir}
  !sed -i 's@"outdir_save": "log/images"@"outdir_save": "/content/gdrive/MyDrive/WebUI/outputs/log/images"@' {config_dir}
else: 
  if '/gdrive/' in getoutput('cat '+config_dir):
    !sed -i 's@"outdir_txt2img_samples": "outputs/txt2img-images"@"outdir_txt2img_samples": "outputs/txt2img-images"@' {config_dir}
    !sed -i 's@"outdir_img2img_samples": "outputs/img2img-images"@"outdir_img2img_samples": "outputs/img2img-images"@' {config_dir}
    !sed -i 's@"outdir_extras_samples": "outputs/extras-images"@"outdir_extras_samples": "outputs/extras-images"@' {config_dir}
    !sed -i 's@"outdir_txt2img_grids": "outputs/txt2img-grids"@"outdir_txt2img_grids": "outputs/txt2img-grids"@' {config_dir}
    !sed -i 's@"outdir_img2img_grids": "outputs/img2img-grids"@"outdir_img2img_grids": "outputs/img2img-grids"@' {config_dir}
    !sed -i 's@"outdir_save": "log/images"@"outdir_save": "log/images"@' {config_dir}


# Install models from model_url, oh boi it's getting bigger
extension_repo = []
prefixes = [
  "config:",
  "ui-config:",
  "styles:",
  "lora:",
  "hypernetwork:",
  "locon:",
  "lycoris:",
  "model:",
  "vae:",
  "control:",
  "clone:",
  "gfpgan:",
  "ersgan:",
  "swinr:",
  "ldsr:",
  "repo:",
  "embeddings:"
]
token = optional_huggingface_token if optional_huggingface_token else "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"
user_header = f"\"Authorization: Bearer {token}\""
print('📦 Downloading models and stuff...', end='')
def handle_manual(url):
  if url.startswith("config:"):
    manual_download(url, "/content/sdw/config.json")
  elif url.startswith("ui-config:"):
    manual_download(url, "/content/sdw/ui-config.json")
  elif url.startswith("styles:"):
    manual_download(url, "/content/sdw/styles.csv")
  elif url.startswith("lora:") or url.startswith("locon:"):
    manual_download(url, loras_dir)
  elif url.startswith("lycoris:"):
    manual_download(url, lycoris_dir)
  elif url.startswith("hypernetwork:"):
    manual_download(url, hypernetworks_dir)
  elif url.startswith("model:"):
    manual_download(url, models_dir)
  elif url.startswith("vae:"):
    manual_download(url, vaes_dir)
  elif url.startswith("control:"):
    manual_download(url, control_dir)
  elif url.startswith("gfpgan:"):
    manual_download(url, "/content/sdw/models/GFPGAN")
  elif url.startswith("ersgan:"):
    manual_download(url, "/content/sdw/models/ERSGAN")
  elif url.startswith("swinr:"):
    manual_download(url, "/content/sdw/models/SwinR")
  elif url.startswith("ldsr:"):
    manual_download(url, "/content/sdw/models/LDSR")
  elif url.startswith("embeddings:"):
    manual_download(url, embeddings_dir)
  elif url.startswith("extension:"):
    extension_repo.append(url)
  elif url.startswith("clone:") or url.startswith("repo:"): 
    !cd /content/.downloaded && git clone $url 

def manual_download(url, dst):
  url = url[url.find(':')+1:]
  if ".json" in url or ".csv" in url:
    !wget "{url}" -O {dst} -c
  elif '.yaml' in url or '.yml' in url or 'discord' in url:
    !wget "{url}" -P {dst} -c
  elif 'drive.google' in url:
    if 'folders' in url:
      !gdown --folder "{url}" -O {dst} --fuzzy -c
    else:
      !gdown "{url}" -O {dst} --fuzzy -c
  elif 'huggingface' in url:
    if '/blob/' in url:
      url = url.replace('/blob/', '/resolve/')
    parsed_link = '\n{}\n\tout={}'.format(url,unquote(url.split('/')[-1]))
    !echo -e "{parsed_link}" | aria2c --header={user_header} --console-log-level=error --summary-interval=10 -i- -j5 -x16 -s16 -k1M -c -d "{dst}" 
  elif 'http' in url or 'magnet' in url:
    parsed_link = '"{}"'.format(url)
    !aria2c --optimize-concurrent-downloads --console-log-level=error --summary-interval=10 -j5 -x16 -s16 -k1M -c -d {dst} -Z {parsed_link}

def download(url):
  try:
    have_drive_link
  except:
    if "drive.google.com" in url:
      # I'm sorry drive ID enjoyer, this will make ID useless :(
      !pip install -U gdown
      have_drive_link = True
  links_and_paths = url.split(',')
  !mkdir -p {destination_dir} {models_dir} {vaes_dir} {hypernetworks_dir} {embeddings_dir} {loras_dir} {lycoris_dir}
  http_links = []
  huggingface_links = []
  for link_or_path in links_and_paths:
    link_or_path = link_or_path.strip()
    if not link_or_path:
      continue

    if any(link_or_path.startswith(prefix.lower()) for prefix in prefixes):
      handle_manual(link_or_path)
      continue

    if 'github.com' in link_or_path and ( '.git' in link_or_path or not '.' in link_or_path.split('/')[-1] ):
      extension_repo.append(link_or_path)
      continue
      
    if '.yaml' in link_or_path or '.yml' in link_or_path or 'discord' in link_or_path:
      !wget {link_or_path} -P {destination_dir} -c
    elif 'drive.google' in link_or_path:
      if 'folders' in link_or_path:
        !gdown --folder {link_or_path} -O {destination_dir} --fuzzy -c
      else:
        !gdown {link_or_path} -O {destination_dir} --fuzzy -c
    elif 'huggingface' in link_or_path:
      if '/blob/' in link_or_path:
        link_or_path = link_or_path.replace('/blob/', '/resolve/')
      huggingface_links.append(link_or_path)
    elif 'http' in link_or_path or 'magnet' in link_or_path:
      http_links.append(link_or_path)
    elif '/' in link_or_path:
      if not os.path.exists('/content/gdrive/MyDrive'):
        print('Looks like there\'s a path in your url. You need to mount your drive first.')
        from google.colab import drive
        drive.mount('/content/gdrive')
      !rsync -avr --progress /content/gdrive/MyDrive/{link_or_path} {destination_dir}
    else:
      !gdown {link_or_path} -O {destination_dir} --fuzzy -c
  if http_links:
    links_string = ' '.join(['"{}"'.format(x) for x in http_links])
    !aria2c --optimize-concurrent-downloads --console-log-level=error --summary-interval=10 -j5 -x16 -s16 -k1M -c -d {destination_dir}  -Z {links_string}
    del links_string
  if huggingface_links:
    # links_string = ' '.join(['"{}"'.format(x) for x in huggingface_links])
    # !aria2c --summary-interval=10 --header={user_header} -c -x 16 -k 1M -s 16 -d {destination_dir} -Z {links_string}
    # for link in huggingface_links:
    #   !aria2c --summary-interval=10 --header={user_header} -c -x 16 -k 1M -s 16 -d {destination_dir} -o {link.split('/')[-1]} {link}
    links_string = '\n'.join(['{}\n\tout={}'.format(x,unquote(x.split('/')[-1])) for x in huggingface_links])  
    !echo -e "{links_string}" | aria2c --header={user_header} --optimize-concurrent-downloads --console-log-level=error --summary-interval=10 -i- -j5 -x16 -s16 -k1M -c -d {destination_dir} 

if verbose_download:
  download(model_url)
else:
  with capture.capture_output() as cap:
    download(model_url)
    del cap

print('\r🏁 Download finished.', flush=True)

if len(extension_repo) > 0:
  print('✨ Installing custom extensions...', end='')
  with capture.capture_output() as cap:
    for repo in extension_repo:
      repo_name = repo.split('/')[-1]
      !cd {extensions_dir} \
        && git clone "{repo}" \
        && cd {repo_name} \
        && git fetch
  print('\r🏁 Installed',len(extension_repo),'custom extensions.', flush=True)

print('\n')
# Link all files by filtering accoridng to their type
with capture.capture_output() as cap:
  # files = os.listdir(destination_dir)
  files = [os.path.join(dp,f) for dp, dn, fn in os.walk(destination_dir) for f in fn] # Thanks Aojiru!
  for file in files:
    name, file_extension = os.path.splitext(file)
    if '.aria2' in file:
      continue
    file_path = os.path.join(destination_dir, file)
    file_size = os.path.getsize(file_path)
    if "control_" in name or "t2iadapter_" in name or file_extension == ".pth":
      !ln "{file_path}" {control_dir}
    elif file_extension in ['.yaml', '.yml'] or file_size > 1_500_000_000: 
      !ln "{file_path}" {models_dir}
    elif "kl-f8" in name or "vae_" in file or "vae." in file or "vae-" in file or file_size > 380_000_000: 
      !ln "{file_path}" {vaes_dir}
    elif getoutput('if rg -q -o "lora_unet" "'+file_path+'"; then echo 1; else echo 0; fi') == "1":
      !ln "{file_path}" {loras_dir}
    elif (file_extension == '.pt' or file_extension == '.safetensors') and file_size < 10_000_000:
      !ln "{file_path}" {embeddings_dir}
    else:
      !ln "{file_path}" {hypernetworks_dir}
  del cap

# Automatically loads vae for first run, if it exists.
if auto_vae:
  if '.vae.pt' in os.listdir(vaes_dir) or '/vae' in model_url:
    commandline_arguments+=' --vae-path $(readlink -f $(find '+vaes_dir+' \( -name "*.vae.pt" -or -name "*.ckpt" \) -print -quit))'

# Configure Alternatives Tunnels (Colab native, localtunnel, cloudflared, bore with auth)
if alternative_tunnels:
  commandline_arguments = commandline_arguments.replace("--share","")
  commandline_arguments += " --multiple"
  print("⌚ \033[95m\033[1mGenerating alternative tunnels...", end='')
  with capture.capture_output() as cap:
    %cd /content
    if not os.path.exists('/tools/node/bin/lt'):
      !npm install -g localtunnel
    if not os.path.exists('/usr/bin/cloudflared'):
      !curl -Lo /usr/bin/cloudflared https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 && chmod +x /usr/bin/cloudflared
    del cap
  !true > /content/nohup.out
  !nohup lt --port 7860 > /content/nohup.out 2>&1 &
  !nohup cloudflared tunnel --url localhost:7860 > /content/nohup.out 2>&1 &
  if with_bore:
    if not os.path.exists('/usr/bin/bore'):
      !curl -Ls https://github.com/ekzhang/bore/releases/download/v0.4.0/bore-v0.4.0-x86_64-unknown-linux-musl.tar.gz | tar zx -C /usr/bin
    !nohup bore local 7860 --to bore.pub > /content/nohup.out 2>&1 &
    if not "--gradio-auth" in commandline_arguments:
      import random
      import string
      gradio_password = ''.join(random.choice(string.ascii_lowercase) for i in range(5))
      commandline_arguments+=" --gradio-auth {}:{}".format("ncpt", gradio_password)
    else:
      gradio_password = False
  !sleep 4
  print("\r💡 \033[95m\033[1mUse one of these alternative tunnels after the loading is finished: ", flush=True)
  from google.colab.output import serve_kernel_port_as_window
  # serve_kernel_port_as_window(7860, anchor_text="https://th15f4k3l1nkofcn0tr34ll0l-7860-colab.googleusercontent.com/")
  !cat /content/nohup.out | rg -a -o "https[^ ]*.*\.trycloudflare\.com|https[^ ]*.*\.loca\.lt|bore.pub:[^ ]*" | sed 's@bore.pub@http://bore.pub@'
  print("\n")
  if with_bore:
    if gradio_password:
      print("\r🔐 \033[0m\033[1mLooks like you're using bore without --gradio-auth huh... ")
      print("For security, I've enforced to use gradio auth, so use this account to login:")
      print("👉⚠️ Username: ncpt")
      print("👉⚠️ Password:", gradio_password,"\n\n")

# If no xformers installed, remove --xformers from arg to avoid using old builtin xformers 
# if not os.path.exists("/usr/local/lib/python3.10/dist-packages/xformers"):
#   commandline_arguments = commandline_arguments.replace("--xformers","")

# Krita extension support (adding --api automatically) + toggleable.
with capture.capture_output() as cap: 
  if krita_paint_ext:
    # Add api if no api (if lazy y'know)
    if not "--api" in commandline_arguments:
      commandline_arguments+=" --api"
    %cd /content/sdw
    if os.path.exists('/content/sdw/extensions/auto-sd-paint-ext'):
      !cd ./extensions/auto-sd-paint-ext && git fetch && git merge # Deflecting FETCH_HEAD not found bug
    else:
      !git clone https://github.com/Interpause/auto-sd-paint-ext extensions/auto-sd-paint-ext
  else:
    if os.path.exists('/content/sdw/extensions/auto-sd-paint-ext'):
      !rm -rf /content/sdw/extensions/auto-sd-paint-ext
  # Remove junks
  !find /content/sdw/ -name ".ipynb_checkpoints" -type d -exec rm -r {} \;


# Print all files in every important directory
print("\033[96mCan't see your files in here? Activate verbose_download to debug!\n")
print("\033[92m\033[1m╭-📦 Models + Configs\033[96m")
!find {models_dir}/ -mindepth 1 ! -name '*.txt' -printf '%f\n' 
print("\n\033[92m\033[1m╭-📦 VAEs\033[96m")
!find {vaes_dir}/ -mindepth 1 ! -name '*.txt' -printf '%f\n'
print("\n\033[92m\033[1m╭-📦 Custom Embeddings (inbuilt is hidden)\033[96m")
!find {embeddings_dir}/ -mindepth 1 -maxdepth 1 -name '*.pt' -or -name '*.safetensors' -printf '%f\n'
print("\n\033[92m\033[1m╭-📦 LoRAs\033[96m")
!find {loras_dir}/ -mindepth 1 ! -name '*.keep' -printf '%f\n'
print("\n\033[92m\033[1m╭-📦 LyCORIS\033[96m")
!find {lycoris_dir}/ -mindepth 1 ! -name '*.keep' -printf '%f\n'
print("\n\033[92m\033[1m╭-📦 Hypernetworks\033[96m")
!find {hypernetworks_dir}/ -mindepth 1 ! -name '*.txt' -printf '%f\n'
print("\n\033[92m\033[1m╭-📦 Extensions\033[96m")
!find {extensions_dir}/ -mindepth 1 -maxdepth 1 ! -name '*.txt' -printf '%f\n'
print("\n\n\033[0m")

# Start the webui
%cd /content/sdw

if null_model:
  print("\033[91m ⚠️ Null model will be loaded, if you don't understand please uncheck the model \033[0m")
  !wget https://huggingface.co/ckpt/null/resolve/main/nullModelzeros.ckpt -P {models_dir} -c
  commandline_arguments+=" --ckpt "+models_dir+"/nullModelzeros.ckpt "

from IPython.display import Audio, display
display(Audio("/content/sdw/notification.mp3", autoplay=True))

if fast_start:
  # commandline_arguments += " --skip-install"
  print("\033[91m ⚠️ Fast start is active, please disable it if you have any problem! \033[0m")
  !python webui.py $commandline_arguments
else:
  !COMMANDLINE_ARGS="{commandline_arguments}" REQS_FILE="requirements_versions.txt" python launch.py 
time_since_start = timedelta(seconds=time.time()-start_colab)
print("\n\n\033[96m⌚ You've been running this colab for","%02d:%02d:%02d" % (time_since_start.seconds / 3600, (time_since_start.seconds / 60) % 60, time_since_start.seconds % 60))
print("\n\n")