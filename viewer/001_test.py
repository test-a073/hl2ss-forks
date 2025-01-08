import multiprocessing as mp
import os
import time
import cv2
import hl2ss
import hl2ss_lnm
import hl2ss_mp
import config
import sys
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('hololens_capture.log'),
        logging.StreamHandler()
    ]
)

# Settings --------------------------------------------------------------------
# HoloLens address
host = config.HOLOLENS_IP
# Ports
ports = [
    hl2ss.StreamPort.PERSONAL_VIDEO,
]  # Removed microphone as we only need video
# PV parameters
pv_width = 760
pv_height = 428
pv_framerate = 30
# Maximum number of frames in buffer
buffer_elements = 150
# Image save interval in seconds
save_interval = 20
# Output folder for saving images
output_folder = "sasika_stream"
os.makedirs(output_folder, exist_ok=True)

if __name__ == '__main__':
    try:
        # Start PV Subsystem
        logging.info("Starting PV subsystem...")
        hl2ss_lnm.start_subsystem_pv(host, hl2ss.StreamPort.PERSONAL_VIDEO)

        # Start streams
        producer = hl2ss_mp.producer()
        producer.configure(
            hl2ss.StreamPort.PERSONAL_VIDEO,
            hl2ss_lnm.rx_pv(host, hl2ss.StreamPort.PERSONAL_VIDEO, width=pv_width, height=pv_height, framerate=pv_framerate),
        )

        consumer = hl2ss_mp.consumer()
        manager = mp.Manager()
        sinks = {}

        logging.info("Initializing video stream...")
        producer.initialize(hl2ss.StreamPort.PERSONAL_VIDEO, buffer_elements)
        producer.start(hl2ss.StreamPort.PERSONAL_VIDEO)
        sinks[hl2ss.StreamPort.PERSONAL_VIDEO] = consumer.create_sink(producer, hl2ss.StreamPort.PERSONAL_VIDEO, manager, None)
        sinks[hl2ss.StreamPort.PERSONAL_VIDEO].get_attach_response()
        logging.info("Video stream started successfully")

        # Image capture function
        last_saved_time = 0
        
        logging.info(f"Starting capture loop. Saving images every {save_interval} seconds...")
        while True:
            _, data = sinks[hl2ss.StreamPort.PERSONAL_VIDEO].get_most_recent_frame()
            if data is not None and data.payload.image is not None and data.payload.image.size > 0:
                timestamp = time.time()
                if timestamp - last_saved_time >= save_interval:
                    filename = f"frame_{int(timestamp)}.jpg"
                    filepath = os.path.join(output_folder, filename)
                    cv2.imwrite(filepath, data.payload.image)
                    last_saved_time = timestamp
                    logging.info(f"Frame saved: {filepath}")

    except Exception as e:
        logging.error(f"An error occurred: {e}")
    finally:
        # Cleanup
        logging.info("Shutting down...")
        if hl2ss.StreamPort.PERSONAL_VIDEO in sinks:
            sinks[hl2ss.StreamPort.PERSONAL_VIDEO].detach()
            producer.stop(hl2ss.StreamPort.PERSONAL_VIDEO)
            logging.info("Video stream stopped")
        
        hl2ss_lnm.stop_subsystem_pv(host, hl2ss.StreamPort.PERSONAL_VIDEO)
        logging.info("PV subsystem stopped")
