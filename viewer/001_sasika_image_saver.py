import multiprocessing as mp 
import os
import time
import cv2
import hl2ss
import hl2ss_lnm
import hl2ss_mp

import config

# Settings --------------------------------------------------------------------

# HoloLens address
host = config.HOLOLENS_IP

# Ports
ports = [
    hl2ss.StreamPort.PERSONAL_VIDEO,
    hl2ss.StreamPort.MICROPHONE,
]

# PV parameters
pv_width     = 760
pv_height    = 428
pv_framerate = 30

# Maximum number of frames in buffer
buffer_elements = 150

# Image save interval in seconds
save_interval = 20  # Save frame every 20 seconds

#------------------------------------------------------------------------------ 

if __name__ == '__main__':

    # Start PV Subsystem if PV is selected ------------------------------------
    if (hl2ss.StreamPort.PERSONAL_VIDEO in ports):
        hl2ss_lnm.start_subsystem_pv(host, hl2ss.StreamPort.PERSONAL_VIDEO)

    # Start streams -----------------------------------------------------------
    producer = hl2ss_mp.producer()
    producer.configure(hl2ss.StreamPort.PERSONAL_VIDEO, hl2ss_lnm.rx_pv(host, hl2ss.StreamPort.PERSONAL_VIDEO, width=pv_width, height=pv_height, framerate=pv_framerate))
    producer.configure(hl2ss.StreamPort.MICROPHONE, hl2ss_lnm.rx_microphone(host, hl2ss.StreamPort.MICROPHONE))

    consumer = hl2ss_mp.consumer()
    manager = mp.Manager()
    sinks = {}

    for port in ports:
        producer.initialize(port, buffer_elements)
        producer.start(port)
        sinks[port] = consumer.create_sink(producer, port, manager, None)
        sinks[port].get_attach_response()
        while (sinks[port].get_buffered_frame(0)[0] != 0):
            pass
        print(f'Started {port}')        
        
    # Create Display Map ------------------------------------------------------
    def display_pv(port, payload):
        if (payload.image is not None and payload.image.size > 0):
            # Save the frame as sample.jpg every 'save_interval' seconds
            timestamp = time.time()
            if not hasattr(display_pv, "last_saved_time"):
                display_pv.last_saved_time = 0  # Initialize the variable

            if timestamp - display_pv.last_saved_time >= save_interval:
                cv2.imwrite(os.path.join("sasika_stream", "sample.jpg"), payload.image)
                display_pv.last_saved_time = timestamp

            cv2.imshow(hl2ss.get_port_name(port), payload.image)

    def display_null(port, payload):
        pass

    DISPLAY_MAP = {
        hl2ss.StreamPort.PERSONAL_VIDEO       : display_pv,
        hl2ss.StreamPort.MICROPHONE           : display_null,
    }

    # Main loop ---------------------------------------------------------------
    while True:
        for port in ports:
            _, data = sinks[port].get_most_recent_frame()
            if (data is not None):
                DISPLAY_MAP[port](port, data.payload)
        cv2.waitKey(1)

    # Stop streams ------------------------------------------------------------
    for port in ports:
        sinks[port].detach()
        producer.stop(port)
        print(f'Stopped {port}')

    # Stop PV Subsystem if PV is selected -------------------------------------
    if (hl2ss.StreamPort.PERSONAL_VIDEO in ports):
        hl2ss_lnm.stop_subsystem_pv(host, hl2ss.StreamPort.PERSONAL_VIDEO)
