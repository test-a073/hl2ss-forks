import multiprocessing as mp
import os
import cv2
import hl2ss
import hl2ss_lnm
import hl2ss_mp
import config
import logging
import time
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('hololens_capture.log'),
        logging.StreamHandler()
    ]
)

# Settings
host = config.HOLOLENS_IP
pv_width = 760
pv_height = 428
pv_framerate = 30
buffer_elements = 30
output_folder = "sasika_stream"
os.makedirs(output_folder, exist_ok=True)

if __name__ == '__main__':
    try:
        # Start PV Subsystem
        logging.info("Starting PV subsystem...")
        hl2ss_lnm.start_subsystem_pv(host, hl2ss.StreamPort.PERSONAL_VIDEO)

        # Start stream
        producer = hl2ss_mp.producer()
        producer.configure(
            hl2ss.StreamPort.PERSONAL_VIDEO,
            hl2ss_lnm.rx_pv(host, hl2ss.StreamPort.PERSONAL_VIDEO, width=pv_width, height=pv_height, framerate=pv_framerate),
        )

        consumer = hl2ss_mp.consumer()
        manager = mp.Manager()

        logging.info("Initializing video stream...")
        producer.initialize(hl2ss.StreamPort.PERSONAL_VIDEO, buffer_elements)
        producer.start(hl2ss.StreamPort.PERSONAL_VIDEO)
        sink = consumer.create_sink(producer, hl2ss.StreamPort.PERSONAL_VIDEO, manager, None)
        sink.get_attach_response()
        logging.info("Video stream started successfully")

        # Add delay to allow stream to initialize
        time.sleep(2)
        logging.info("Waiting for frame...")

        # Try to get frame multiple times if needed
        max_attempts = 5
        attempt = 0
        while attempt < max_attempts:
            _, data = sink.get_most_recent_frame()
            
            if data is not None and data.payload.image is not None and data.payload.image.size > 0:
                filename = f"frame_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
                filename = "sample.jpg"
                filepath = os.path.join(output_folder, filename)
                cv2.imwrite(filepath, data.payload.image)
                logging.info(f"Frame saved: {filepath}")
                break
            else:
                attempt += 1
                logging.warning(f"Attempt {attempt}/{max_attempts}: No valid frame received")
                time.sleep(1)
        
        if attempt >= max_attempts:
            logging.error("Failed to capture valid frame after multiple attempts")

    except Exception as e:
        logging.error(f"An error occurred: {e}")
    finally:
        # Cleanup
        logging.info("Shutting down...")
        sink.detach()
        producer.stop(hl2ss.StreamPort.PERSONAL_VIDEO)
        logging.info("Video stream stopped")
        
        hl2ss_lnm.stop_subsystem_pv(host, hl2ss.StreamPort.PERSONAL_VIDEO)
        logging.info("PV subsystem stopped")
