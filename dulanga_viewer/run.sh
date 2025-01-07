#!/bin/bash

# Register voice instructions
gnome-terminal --working-directory=/home/user/Projects/VGGlass/hl2ss/viewer -- bash -c '

# Source the virtual environment
source ~/Projects/VGGlass/vgglass/bin/activate

# Add your commands here

# Example command: Run the viewer script
python client_ipc_vi.py
'

#Speech Recognizer
gnome-terminal --working-directory=/home/user/Projects/VGGlass/hl2ss/viewer -- bash -c '

# Source the virtual environment
source ~/Projects/VGGlass/vgglass/bin/activate

# Add your commands here

# Example command: Run the viewer script
python voice_transcribe.py
'

# # sample RGB and gesture recognition
# gnome-terminal --working-directory=/home/user/Projects/VGGlass/hl2ss/viewer -- bash -c '

# # Source the virtual environment
# source ~/Projects/VGGlass/vgglass/bin/activate

# # Add your commands here

# # Example command: Run the viewer script
# python sample_si_pv_v2.py
# '


# sample RGB and gesture recognition v2
gnome-terminal --working-directory=/home/user/Projects/liloc-demo -- bash -c '

# Source the virtual environment
source ~/Projects/VGGlass/vgglass/bin/activate

# Add your commands here

# Example command: Run the viewer script
python hololens_v3.py #--use_grounding
'

# # Run VLM model
# gnome-terminal --working-directory=/home/user/Projects/LLM/CogVLM -- bash -c '

# # Source the virtual environment
# source /home/user/Projects/LLM/TigerBot/venv/bin/activate

# # Add your commands here

# # Example command: Run the viewer script
# python vlm_vgglass.py --from_pretrained cogvlm-chat-v1.1 --version chat --english --bf16
# '
