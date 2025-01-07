#------------------------------------------------------------------------------
# This script registers voice commands on the HoloLens and continously checks
# if any of the registered commands has been heard.
# Press esc to stop.
#------------------------------------------------------------------------------

from pynput import keyboard

import hl2ss
import hl2ss_lnm

from pymemcache.client import base
import hl2ss_rus

import time
# Settings --------------------------------------------------------------------

# HoloLens address
host = '192.168.10.149'

# Position in camera space (x, y, z)
position = [0,0, 0.5]

# Rotation in camera space (x, y, z, w) as a quaternion
rotation = [0, 0, 0, 1]

# Scale (x, y, z) in meters
scale = [0.15, 0.15, 1]

# Voice commands
strings = ['start','done']

mem_client = base.Client(('localhost', 11211))
mem_client.set('state', 0)  
mem_client.set('process', 0)
mem_client.set('instruction', '')
mem_client.set('response', '')

previous_instruction = 'Done'

ipc = hl2ss_lnm.ipc_umq(host, hl2ss.IPCPort.UNITY_MESSAGE_QUEUE)
ipc.open()


#------------------------------------------------------------------------------

enable = True
texture_file = '/home/user/Projects/LLM/CogVLM/output.png'

def say_command(command):
    display_list = hl2ss_rus.command_buffer()
    display_list.say(command)
    ipc.push(display_list) # Send command to server
    results = ipc.pull(display_list) # Get result from server
    # print(f'Response: {results[0]}')

def show_output():
    key = 0
    with open(texture_file, mode='rb') as file:
        texture = file.read()

        display_list = hl2ss_rus.command_buffer()
        display_list.begin_display_list() # Begin command sequence
        display_list.remove_all() # Remove all objects that were created remotely
        display_list.create_primitive(hl2ss_rus.PrimitiveType.Quad) # Create a quad, server will return its id
        display_list.set_target_mode(hl2ss_rus.TargetMode.UseLast) # Set server to use the last created object as target, this avoids waiting for the id of the quad
        display_list.set_local_transform(key, position, rotation, scale) # Set the local transform of the cube
        display_list.set_texture(key, texture) # Set the texture of the quad
        display_list.set_active(key, hl2ss_rus.ActiveState.Active) # Make the quad visible
        display_list.set_target_mode(hl2ss_rus.TargetMode.UseID) # Restore target mode
        display_list.end_display_list() # End command sequence
        ipc.push(display_list) # Send commands to server
        results = ipc.pull(display_list) # Get results from server

def on_press(key):
    global enable
    enable = key != keyboard.Key.esc
    mem_client.set('state', 0)
    return enable

def get_word(strings, index):
    if ((index < 0) or (index >= len(strings))):
        return '_UNKNOWN_'
    else:
        return strings[index]

listener = keyboard.Listener(on_press=on_press)
listener.start()

client = hl2ss_lnm.ipc_vi(host, hl2ss.IPCPort.VOICE_INPUT)
client.open()

# See
# https://learn.microsoft.com/en-us/windows/mixed-reality/develop/native/voice-input-in-directx
# for details
state = 0 # not listening
client.create_recognizer()
if (client.register_commands(True, strings)):
    print('Ready. Try saying any of the commands you defined.')
    client.start()    
    while (enable):
        events = client.pop()
        for event in events:
            event.unpack()
            # See
            # https://learn.microsoft.com/en-us/uwp/api/windows.media.speechrecognition.speechrecognitionresult?view=winrt-22621
            # for result details
            # print(f'Event: Command={get_word(strings, event.index)} Index={event.index} Confidence={event.confidence} Duration={event.phrase_duration} Start={event.phrase_start_time} RawConfidence={event.raw_confidence}')
            if event.index == 0:
                print('Start')
                if previous_instruction == 'Start':
                    say_command('Please stop first')
                    continue
                else:
                    previous_instruction = 'Start'
                mem_client.set('state', 1)
                state = 1
                say_command('Yes')
            elif event.index == 1:
                print('Done')
                if previous_instruction == 'Done':
                    say_command('Please start first')
                    continue
                else:
                    previous_instruction = 'Done'
                mem_client.set('state', 0)  
                state = 0
                say_command('Okay')
                while True:
                    if mem_client.get('response') == b'':
                        continue
                    elif mem_client.get('response') == b'bbox':
                        say_command('See the location in image')
                        show_output()
                        mem_client.set('response', '')
                        break
                    else:
                        say_command(mem_client.get('response').decode('utf-8'))
                        mem_client.set('response', '')
                        break        


            # if mem_client.get('response') == b'':
            #     continue
            # elif mem_client.get('response') == b'bbox':
            #     say_command('See the location in image')
            #     show_output()
            #     mem_client.set('response', '')
            #     break
            # else:
            #     say_command(mem_client.get('response').decode('utf-8'))
            #     mem_client.set('response', '')
            #     break  
    
    display_list = hl2ss_rus.command_buffer()
    display_list.begin_display_list() # Begin command sequence
    display_list.remove_all() # Remove all objects that were created remotely
    display_list.end_display_list() # End command sequence
    ipc.push(display_list) # Send commands to server
    results = ipc.pull(display_list) # Get results from server

    client.stop()
    client.clear()

client.close()

listener.join()
